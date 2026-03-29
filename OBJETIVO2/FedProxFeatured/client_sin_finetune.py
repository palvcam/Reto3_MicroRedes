import torch
import torch.nn as nn
import flwr as fl
from collections import OrderedDict
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from model import PVModel
import pandas as pd
from sklearn.metrics import r2_score
import numpy as np
import glob
import sys

# =========================
# CONFIG
# =========================
FEATURES = [
    "POA irradiance CMP22 pyranometer (W/m2)",
    "PV module back surface temperature (degC)",
    "k_panel",
    "physical_model"
]

LOCAL_EPOCHS = 20
LR = 0.001
MU = 0.005


# =========================
# CLIENTE FEDPROX
# =========================
class PVClient(fl.client.NumPyClient):

    def __init__(self, parque):
        self.model = PVModel(
            input_size=len(FEATURES),
            layers_sizes=[128, 64, 32]
        )
        self.criterion = nn.MSELoss()

        # =========================
        # CARGA DATOS
        # =========================
        ruta = f"../PV_MaximumPowerPredictor/{parque}_*.csv"
        dfs = []

        for archivo in glob.glob(ruta):
            df = pd.read_csv(archivo)

            df = df.drop(
                columns=["Time Stamp (local standard time) yyyy-mm-ddThh:mm:ss"],
                errors="ignore"
            )

            dfs.append(df)

        df_parque = pd.concat(dfs, ignore_index=True)

        # =========================
        # ASEGURAR panel_id
        # =========================
        if "panel_id" not in df_parque.columns:
            print("panel_id no encontrado → usando k global")
            df_parque["panel_id"] = 0  # fallback

        # =========================
        # CALCULAR k POR PANEL
        # =========================
        k_por_panel = {}

        for panel_id, df_panel in df_parque.groupby("panel_id"):

            G = df_panel["POA irradiance CMP22 pyranometer (W/m2)"].values
            T = df_panel["PV module back surface temperature (degC)"].values
            P = df_panel["Pmp (W)"].values

            # evitar paneles con pocos datos
            if len(df_panel) < 10:
                continue

            y_k = P / (G + 1e-6)
            X_k = (T - 25).reshape(-1, 1)

            model_k = LinearRegression()
            model_k.fit(X_k, y_k)

            k_por_panel[panel_id] = float(-model_k.coef_[0])

        # =========================
        # k GLOBAL (fallback)
        # =========================
        if len(k_por_panel) > 0:
            k_global = np.mean(list(k_por_panel.values()))
        else:
            k_global = 0.004  # valor típico

        print(f"k_global: {k_global:.6f}")

        # =========================
        # ASIGNAR k_panel (ESCALABLE)
        # =========================
        df_parque["k_panel"] = df_parque["panel_id"].map(k_por_panel)

        # fallback → panel nuevo o sin datos
        df_parque["k_panel"] = df_parque["k_panel"].fillna(k_global)

        # =========================
        # FEATURE FÍSICA
        # =========================
        df_parque["physical_model"] = (
            df_parque["POA irradiance CMP22 pyranometer (W/m2)"] *
            (1 - df_parque["k_panel"] *
             (df_parque["PV module back surface temperature (degC)"] - 25))
        )

        # =========================
        # INPUT / TARGET
        # =========================
        X_p = df_parque[FEATURES]
        y_p = df_parque["Pmp (W)"]

        # =========================
        # SPLIT
        # =========================
        X_train, X_val, y_train, y_val = train_test_split(
            X_p, y_p, test_size=0.2, random_state=42
        )

        # =========================
        # ESCALADO
        # =========================
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

        X_train = self.x_scaler.fit_transform(X_train)
        X_val   = self.x_scaler.transform(X_val)

        y_train = self.y_scaler.fit_transform(y_train.values.reshape(-1, 1))
        y_val   = self.y_scaler.transform(y_val.values.reshape(-1, 1))

        # =========================
        # TENSORES
        # =========================
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
        y_val_t   = torch.tensor(y_val,   dtype=torch.float32)

        self.train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=256,
            shuffle=True
        )

        self.val_loader = DataLoader(
            TensorDataset(X_val_t, y_val_t),
            batch_size=256
        )

        print(f"Cliente {parque}: {len(self.train_loader.dataset)} train | {len(self.val_loader.dataset)} val")

    # =========================
    # PARAMETROS
    # =========================
    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    # =========================
    # TRAIN (FEDPROX)
    # =========================
    def fit(self, parameters, config):
        self.set_parameters(parameters)

        global_params = [
            torch.tensor(p, dtype=torch.float32)
            for p in parameters
        ]

        optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.model.train()

        for epoch in range(LOCAL_EPOCHS):
            for X_batch, y_batch in self.train_loader:

                optimizer.zero_grad()

                preds = self.model(X_batch)
                loss = self.criterion(preds, y_batch)

                prox_term = 0.0
                for w, w_global in zip(self.model.parameters(), global_params):
                    prox_term += torch.norm(w - w_global) ** 2

                loss = loss + (MU / 2) * prox_term

                loss.backward()
                optimizer.step()

        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    # =========================
    # EVALUACIÓN
    # =========================
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

        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()

        all_preds = self.y_scaler.inverse_transform(all_preds)
        all_targets = self.y_scaler.inverse_transform(all_targets)

        mse = float(((all_targets - all_preds) ** 2).mean())
        rmse = mse ** 0.5
        r2 = r2_score(all_targets, all_preds)

        print(f"  MSE: {mse:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")

        return float(mse), len(self.val_loader.dataset), {
            "val_mse": mse,
            "val_rmse": rmse,
            "val_r2": r2
        }


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    parque = sys.argv[1]
    print(f"Iniciando cliente: {parque}")

    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=PVClient(parque=parque).to_client()
    )