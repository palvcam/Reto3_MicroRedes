# =========================
# IMPORTS
# =========================
import torch
import torch.nn as nn
import flwr as fl
import numpy as np
from collections import OrderedDict
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import PVModel
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
import glob
import sys
import os
import re

# =========================
# CONFIG
# =========================
FEATURES = [
    "POA irradiance CMP22 pyranometer (W/m2)",
    "PV module back surface temperature (degC)",
    "Dry bulb temperature (degC)",
    "Relative humidity (%RH)",
    "Atmospheric pressure (mb)",
    "Precipitation (mm) accumulated daily total",
    "Direct normal irradiance (W/m2)",
    "Global horizontal irradiance (W/m2)",
    "Diffuse horizontal irradiance (W/m2)",
    "modelo_panel_enc"
]

EPOCH_CONFIGS = [1, 2, 3, 5, 8]
MU_CONFIGS = [0.01, 0.1, 0.5, 1.0]
LR = 0.001


def extraer_modelo(nombre_archivo):
    base = os.path.basename(nombre_archivo).replace(".csv", "")
    panel_id = "_".join(base.split("_")[1:])
    match = re.match(r'^([a-zA-Z]+)', panel_id)
    return match.group(1) if match else panel_id


# =========================
# CLIENTE
# =========================
class PVClient(fl.client.NumPyClient):

    def __init__(self, parque):
        self.model = PVModel(input_size=10)
        self.criterion = nn.MSELoss()

        # =========================
        # FEDEX INICIALIZACIÓN
        # =========================

        # Probabilidades iniciales uniformes (todas las configs igual de probables)
        self.epoch_probs = np.ones(len(EPOCH_CONFIGS)) / len(EPOCH_CONFIGS)
        self.mu_probs = np.ones(len(MU_CONFIGS)) / len(MU_CONFIGS)

        self.fedex_lr = 0.1

        # BASELINE DINÁMICO (CORRECTO)
        # ---------------------------------
        # En el paper el baseline NO es constante.
        # Es una media de losses anteriores para reducir varianza.
        self.baseline = None

        # Factor de suavizado (tipo media exponencial)
        self.beta = 0.9

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

            df["modelo_panel"] = extraer_modelo(archivo)
            dfs.append(df)

        df_parque = pd.concat(dfs, ignore_index=True)

        # Encoding del modelo de panel
        le = LabelEncoder()
        df_parque["modelo_panel_enc"] = le.fit_transform(df_parque["modelo_panel"])

        X_p = df_parque[FEATURES]
        y_p = df_parque["Pmp (W)"]

        # SPLIT
        X_train, X_val, y_train, y_val = train_test_split(
            X_p, y_p, test_size=0.2, random_state=42
        )

        # ESCALADO
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # TENSORES
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
        y_val_t = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

        self.train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=1024, shuffle=True)
        self.val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=1024)

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
    # SAMPLE FEDEX
    # =========================
    def sample_fedex_config(self):
        epoch_idx = np.random.choice(len(EPOCH_CONFIGS), p=self.epoch_probs)
        mu_idx = np.random.choice(len(MU_CONFIGS), p=self.mu_probs)
        return EPOCH_CONFIGS[epoch_idx], MU_CONFIGS[mu_idx], epoch_idx, mu_idx

    # =========================
    # TRAIN (FEDPROX + FEDEX)
    # =========================
    def fit(self, parameters, config):

        self.set_parameters(parameters)

        global_params = [torch.tensor(p.copy(), dtype=torch.float32) for p in parameters]

        # Sample hiperparámetros
        epochs, mu, epoch_idx, mu_idx = self.sample_fedex_config()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.model.train()

        # TRAIN LOCAL
        for _ in range(epochs):
            for X_batch, y_batch in self.train_loader:
                optimizer.zero_grad()

                output = self.model(X_batch)
                loss = self.criterion(output, y_batch)

                # FedProx
                prox_term = sum(torch.norm(w - wg) ** 2 for w, wg in zip(self.model.parameters(), global_params))
                loss += (mu / 2) * prox_term

                loss.backward()
                optimizer.step()

        # =========================
        # VALIDACIÓN LOCAL
        # =========================
        self.model.eval()
        mse = 0

        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                output = self.model(X_batch)
                mse += ((output - y_batch) ** 2).mean().item()

        mse /= len(self.val_loader)

        # =========================
        # BASELINE DINÁMICO
        # =========================

        # Primera iteración → baseline inicial
        if self.baseline is None:
            self.baseline = mse
        else:
            # Media exponencial (suaviza ruido entre rondas)
            self.baseline = self.beta * self.baseline + (1 - self.beta) * mse

        # Gradiente estilo policy gradient
        grad = mse - self.baseline

        # =========================
        # UPDATE FEDEX (solo índice elegido)
        # =========================
        self.epoch_probs[epoch_idx] *= np.exp(-self.fedex_lr * grad)
        self.mu_probs[mu_idx] *= np.exp(-self.fedex_lr * grad)

        # Normalizar (muy importante)
        self.epoch_probs /= self.epoch_probs.sum()
        self.mu_probs /= self.mu_probs.sum()

        return self.get_parameters(config={}), len(self.train_loader.dataset), {
            "val_mse": mse
        }

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

        mse = float(((all_targets - all_preds) ** 2).mean())
        rmse = mse ** 0.5
        r2 = r2_score(all_targets, all_preds)

        print(f"MSE: {mse:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")

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

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=PVClient(parque=parque)
    )