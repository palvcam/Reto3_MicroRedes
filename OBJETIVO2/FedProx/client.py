import torch
import torch.nn as nn
import flwr as fl
from collections import OrderedDict
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import PVModel
import pandas as pd
import glob
import sys

# CONFIG
FEATURES = [
    "POA irradiance CMP22 pyranometer (W/m2)",
    "PV module back surface temperature (degC)",
    "Dry bulb temperature (degC)",
    "Relative humidity (%RH)",
    "Atmospheric pressure (mb)",
    "Precipitation (mm) accumulated daily total",
    "Direct normal irradiance (W/m2)",
    "Global horizontal irradiance (W/m2)",
    "Diffuse horizontal irradiance (W/m2)"
]

LOCAL_EPOCHS = 5
LR = 0.001
MU = 0.01

# CLIENTE
class PVClient(fl.client.NumPyClient):
    
    def __init__(self, parque):
        self.model     = PVModel()
        self.criterion = nn.MSELoss()
        
        # CARGA DATOS DEL PARQUE
        ruta = f"../PV_MaximumPowerPredictor/{parque}_*.csv"
        dfs  = []
        
        for archivo in glob.glob(ruta):
            df = pd.read_csv(archivo)
            df = df.drop(columns=["Time Stamp (local standard time) yyyy-mm-ddThh:mm:ss"], errors="ignore")
            dfs.append(df)
        
        df_parque = pd.concat(dfs, ignore_index=True)
        
        X_p = df_parque[FEATURES]
        y_p = df_parque["Pmp (W)"]
        
        # SPLIT
        X_train, X_val, y_train, y_val = train_test_split(
            X_p, y_p, test_size=0.2, random_state=42
        )
        
        # NORMALIZACIÓN
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)
        
        # TENSORES
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
        y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
        y_val_t   = torch.tensor(y_val.values,   dtype=torch.float32).view(-1, 1)
        
        self.train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=1024, shuffle=True)
        self.val_loader   = DataLoader(TensorDataset(X_val_t,   y_val_t),   batch_size=1024)
        
        print(f"Cliente {parque}: {len(self.train_loader.dataset)} train | {len(self.val_loader.dataset)} val")
    
    # PARAMETROS
    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]
    
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict  = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    
    # TRAIN (FEDPROX)
    def fit(self, parameters, config):
        self.set_parameters(parameters)

        # Guardamos parámetros globales
        global_params = [
            torch.tensor(p, dtype=torch.float32)
            for p in parameters
        ]

        optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)

        self.model.train()

        for epoch in range(LOCAL_EPOCHS):
            for X_batch, y_batch in self.train_loader:
                
                optimizer.zero_grad()

                output = self.model(X_batch)
                loss = self.criterion(output, y_batch)

                # FedProx
                prox_term = 0.0
                for w, w_global in zip(self.model.parameters(), global_params):
                    prox_term += torch.norm(w - w_global) ** 2

                loss = loss + (MU / 2) * prox_term

                loss.backward()
                optimizer.step()

        return self.get_parameters(config={}), len(self.train_loader.dataset), {}
    
    # Evaluate
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        
        total_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                total_loss += self.criterion(self.model(X_batch), y_batch).item()
        
        loss = total_loss / len(self.val_loader)
        return loss, len(self.val_loader.dataset), {"val_loss": loss}


if __name__ == "__main__":
    parque = sys.argv[1]
    print(f"Iniciando cliente: {parque}")
    
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=PVClient(parque=parque)
    )