import torch
import torch.nn as nn
import flwr as fl
from collections import OrderedDict
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from model import PVModel
import pandas as pd
import glob
import sys

# Variables que se van a utilizar de cada cliente
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


LOCAL_EPOCHS = 10  # número de épocas que entrena cada cliente por ronda

class PVClient(fl.client.NumPyClient): # Crear el cliente
    def __init__(self, parque):
        self.model     = PVModel() 
        self.criterion = nn.MSELoss()
        
        ruta = f"../PV_MaximumPowerPredictor/{parque}_*.csv" # Carga todos los archivos CSV asociados al parque (cliente)
        dfs  = []
        for archivo in glob.glob(ruta):
            df = pd.read_csv(archivo)
            df = df.drop(columns=["Time Stamp (local standard time) yyyy-mm-ddThh:mm:ss"], errors="ignore") # Elimina columna de fecha
            dfs.append(df)
        
        # Combina todos los archivos del parque en un solo DataFrame
        df_parque = pd.concat(dfs, ignore_index=True)
        
        X_p = df_parque[FEATURES] # Las demás variables
        y_p = df_parque["Pmp (W)"] # Variable objetivo
        
        # Divide los datos en entrenamiento y validación
        X_train, X_val, y_train, y_val = train_test_split(
            X_p, y_p, test_size=0.2, random_state=42
        )
        
        # Normaliza las variables
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)
        
        # Convierte los datos a tensores de PyTorch
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
        y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
        y_val_t   = torch.tensor(y_val.values,   dtype=torch.float32).view(-1, 1)
        
        # Crea DataLoaders
        self.train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=1024, shuffle=True)
        self.val_loader   = DataLoader(TensorDataset(X_val_t,   y_val_t),   batch_size=1024)
        
        print(f"Cliente {parque}: {len(self.train_loader.dataset)} train | {len(self.val_loader.dataset)} val")
    
    ### Devuelve los parámetros actuales del modelo
    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]
    
    ### Establece los parámetros del modelo (local) con los recibidos del servidor
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict  = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    ### Entrenamiento local
    def fit(self, parameters, config):
        self.set_parameters(parameters) # Actualiza los parámetros con los del servidor
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001) # Define optimizador
        
        self.model.train()
        # Entrena varias épocas
        for epoch in range(LOCAL_EPOCHS):  
            for X_batch, y_batch in self.train_loader:
                optimizer.zero_grad()
                loss = self.criterion(self.model(X_batch), y_batch)
                loss.backward()
                optimizer.step()
        # Devuelve pesos actualizados y tamaño del conjunto de entrenamiento
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    ### Evaluación 
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                preds = self.model(X_batch)
                all_preds.append(preds)
                all_targets.append(y_batch)
        
        all_preds   = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()
        
        # Métricas
        mse  = float(((all_targets - all_preds) ** 2).mean())
        rmse = mse ** 0.5
        r2   = r2_score(all_targets, all_preds)
        
        print(f"  MSE: {mse:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")
        
        return float(mse), len(self.val_loader.dataset), {
            "val_mse":  mse,
            "val_rmse": rmse,
            "val_r2":   r2
        }


if __name__ == "__main__":
    parque = sys.argv[1] # Se recibe el nombre del parque
    print(f"Iniciando cliente: {parque}")
    
    # Conecta este cliente al servidor Flower en localhost:8080
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=PVClient(parque=parque)
    )