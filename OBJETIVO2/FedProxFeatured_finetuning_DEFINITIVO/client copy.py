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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from model import PVModel
import pandas as pd
from sklearn.metrics import r2_score
import glob
import sys
import os
import re

# =========================
# CONFIG
# =========================

# FEATURES FINALES (se crean DESPUÉS del split)
FEATURES = [
    "POA irradiance CMP22 pyranometer (W/m2)",
    "PV module back surface temperature (degC)",
    "physical_model",
    "Dry bulb temperature (degC)",
    "T_diff",
    "G_T"
]
# FEATURES BASE (ANTES del split → sin leakage)
BASE_FEATURES = [
    "POA irradiance CMP22 pyranometer (W/m2)",
    "PV module back surface temperature (degC)",
    "modelo_panel_enc",
    "panel_id"
]

# Grid Search
EPOCH_CONFIGS = [1, 2, 3, 5, 8]
MU_CONFIGS = [0.01, 0.1, 0.5, 1.0]
LR = 0.001

# Extraer el modelo de la placa del csv (para el coeficiente de la placa)
def extraer_modelo(nombre_archivo):
    base = os.path.basename(nombre_archivo).replace(".csv", "")
    partes = base.split("_")                    # ['parqueA', 'panel1', 'cleaned']
    panel_id = partes[1] if len(partes) > 1 else base   # 'panel1'
    match = re.match(r'^([a-zA-Z]+)', panel_id)          # extrae 'panel'
    return match.group(1) if match else panel_id


# =========================
# CLIENTE
# =========================
class PVClient(fl.client.NumPyClient):

    def __init__(self, parque):
        self.model = PVModel(input_size=len(FEATURES), layers_sizes=[128, 32, 16])
        self.criterion = nn.MSELoss()

        # =========================
        # FEDEX
        # =========================
        self.epoch_probs = np.ones(len(EPOCH_CONFIGS)) / len(EPOCH_CONFIGS)
        self.mu_probs = np.ones(len(MU_CONFIGS)) / len(MU_CONFIGS)
        k_epochs = len(EPOCH_CONFIGS) 
        k_mu = len(MU_CONFIGS)
        self.eta0_epochs = np.sqrt(2 * np.log(k_epochs))
        self.eta0_mu = np.sqrt(2 * np.log(k_mu)) 
        self.baseline = None # Este valor es la referencia para MU y epochs, que se calcula con la media exponencial de los losses
        self.gamma = 0.9 
        self.loss_history = []

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

        # =========================
        # LIMPIEZA (-9999)
        # =========================
        df_parque = df_parque.replace(-9999, np.nan).dropna()

        # =========================
        # ENCODING PANEL
        # =========================
        le = LabelEncoder()
        df_parque["modelo_panel_enc"] = le.fit_transform(df_parque["modelo_panel"])

        # =========================
        # ASEGURAR panel_id
        # =========================
        if "panel_id" not in df_parque.columns:
            df_parque["panel_id"] = 0

        # =========================
        # FILTRO IRRADIANCIA
        # =========================
        df_parque = df_parque[
            df_parque["POA irradiance CMP22 pyranometer (W/m2)"] > 50
        ]

        # =========================
        # SPLIT (SIN LEAKAGE)
        # =========================
        train_val_df, test_df = train_test_split(
            df_parque, test_size=0.2, random_state=42
        )

        train_df, val_df = train_test_split(
            train_val_df, test_size=0.2, random_state=42
        )

        self.test_df= test_df
        # =========================
        # CALCULAR k_panel SOLO EN TRAIN
        # =========================
        k_por_panel = {}

        for panel_id, df_panel in train_df.groupby("panel_id"):

            if len(df_panel) < 10:
                continue # si hay pocos datos se ignora, porque no es fiable

            G = df_panel["POA irradiance CMP22 pyranometer (W/m2)"].values
            T = df_panel["PV module back surface temperature (degC)"].values
            P = df_panel["Pmp (W)"].values

            y_k = P / (G + 1e-6)
            X_k = (T - 25).reshape(-1, 1)

            model_k = LinearRegression()
            model_k.fit(X_k, y_k)

            k_por_panel[panel_id] = float(-model_k.coef_[0])

        k_global = np.mean(list(k_por_panel.values())) if k_por_panel else 0.004

        # =========================
        # APLICAR k_panel
        # =========================
        train_df["k_panel"] = train_df["panel_id"].map(k_por_panel).fillna(k_global)
        val_df["k_panel"]   = val_df["panel_id"].map(k_por_panel).fillna(k_global)
        test_df["k_panel"] = test_df["panel_id"].map(k_por_panel).fillna(k_global)

        # =========================
        # MODELO FÍSICO (SIN LEAKAGE)
        # =========================
        for df_ in [train_df, val_df, test_df]:

            df_["k_panel"] = k_global

            df_["physical_model"] = (
                df_["POA irradiance CMP22 pyranometer (W/m2)"] *
                (1 - df_["k_panel"] *
                     (df_["PV module back surface temperature (degC)"] - 25))
            )

            df_["T_diff"] = df_["PV module back surface temperature (degC)"] - 25

            df_["G_T"] = (
                df_["POA irradiance CMP22 pyranometer (W/m2)"] *
                df_["PV module back surface temperature (degC)"]
            )

            # 🔥 RESIDUAL
            df_["residual"] = df_["Pmp (W)"] - df_["physical_model"]

        # =========================
        # INPUT FINAL
        # =========================
        X_train = train_df[FEATURES]
        y_train = train_df["Pmp (W)"]

        X_val = val_df[FEATURES]
        y_val = val_df["Pmp (W)"]

        X_test = test_df[FEATURES]
        y_test = test_df["Pmp (W)"]

        # =========================
        # ESCALADO
        # =========================
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

        X_train = self.x_scaler.fit_transform(X_train)
        X_val   = self.x_scaler.transform(X_val)
        X_test   = self.x_scaler.transform(X_test)

        y_train = self.y_scaler.fit_transform(y_train.values.reshape(-1, 1))
        y_val   = self.y_scaler.transform(y_val.values.reshape(-1, 1))
        y_test   = self.y_scaler.transform(y_test.values.reshape(-1, 1))

        # =========================
        # TENSORES
        # =========================
        self.train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                          torch.tensor(y_train, dtype=torch.float32)),
            batch_size=256,
            shuffle=True
        )

        self.val_loader = DataLoader(
            TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                          torch.tensor(y_val, dtype=torch.float32)),
            batch_size=256
        )

        self.test_loader = DataLoader(
            TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                          torch.tensor(y_test, dtype=torch.float32)),
            batch_size=256
        )

    # =========================
    # PARAMS
    # =========================
    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    # =========================
    # FEDEX SAMPLING
    # =========================
    def sample_fedex_config(self):
        epoch_idx = np.random.choice(len(EPOCH_CONFIGS), p=self.epoch_probs)
        mu_idx = np.random.choice(len(MU_CONFIGS), p=self.mu_probs)
        return EPOCH_CONFIGS[epoch_idx], MU_CONFIGS[mu_idx], epoch_idx, mu_idx

    # =========================
    # TRAIN
    # =========================
    def fit(self, parameters, config):

        self.set_parameters(parameters)
        global_params = [torch.tensor(p.copy(), dtype=torch.float32) for p in parameters]
        # recibe λₜ del servidor — paper Algorithm 2, página 7
        lambda_t = config.get("lambda_t", -1.0)

        epochs, mu, epoch_idx, mu_idx = self.sample_fedex_config()
        print(f"[FedEx] epochs={epochs}, mu={mu}")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.model.train()

        for _ in range(epochs):
            for X_batch, y_batch in self.train_loader:
                optimizer.zero_grad()

                preds = self.model(X_batch)
                loss = self.criterion(preds, y_batch)

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
                preds = self.model(X_batch)
                mse += ((preds - y_batch) ** 2).mean().item()

        mse /= len(self.val_loader)

        # =========================
        # BASELINE (λₜ) — paper Apéndice C.3
        # =========================
        if lambda_t < 0:
        # primera ronda: λₜ aún no disponible
            grad = 0.0
        else:
            grad = mse - lambda_t

        # =========================
        # UPDATE FEDEX (schedule agresivo)
        # =========================
        eps = 1e-8

        eta_epochs = self.eta0_epochs / (abs(grad) + eps)  # ηₜ = √(2 log k) / |∇̃ₜ|
        eta_mu     = self.eta0_mu     / (abs(grad) + eps)

        self.epoch_probs[epoch_idx] *= np.exp(-eta_epochs * grad)  # θₜ₊₁ = θₜ ⊙ exp(−ηₜ·∇̃)
        self.mu_probs[mu_idx]       *= np.exp(-eta_mu     * grad)

        self.epoch_probs /= self.epoch_probs.sum()  # renormaliza
        self.mu_probs    /= self.mu_probs.sum()

        return self.get_parameters({}), len(self.train_loader.dataset), {
            "val_mse":      mse,
            "val_samples":  float(len(self.val_loader.dataset)),
            "fedex_epochs": float(epochs),   # la config sampleada esta ronda
            "fedex_mu":     float(mu),
        }
    
    # =========================
    # EVALUACIÓN
    # =========================
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()

        preds, targets = [], []

        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                preds.append(self.model(X_batch))
                targets.append(y_batch)

        preds = self.y_scaler.inverse_transform(torch.cat(preds).numpy())
        targets = self.y_scaler.inverse_transform(torch.cat(targets).numpy())


        # 🔥 reconstrucción Pmp
        preds_final = preds + self.physical_val
        targets_final = targets + self.physical_val

        mse = float(((targets_final - preds_final) ** 2).mean())
        rmse = mse ** 0.5
        r2 = r2_score(targets_final, preds_final)

        print(f"MSE: {mse:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")


        # =========================
        # TEST (añadido)
        # =========================
        test_preds, test_targets = [], []

        with torch.no_grad():
            for X, y in self.test_loader:
                test_preds.append(self.model(X))
                test_targets.append(y)

        test_preds = self.y_scaler.inverse_transform(torch.cat(test_preds).numpy())
        test_targets = self.y_scaler.inverse_transform(torch.cat(test_targets).numpy())

        test_mse = float(((test_targets - test_preds) ** 2).mean())
        test_rmse = test_mse ** 0.5
        test_r2 = r2_score(test_targets, test_preds)

        return mse, len(self.val_loader.dataset), {
            "val_mse": mse,
            "val_rmse": rmse,
            "val_r2": r2,

            "test_mse": test_mse,
            "test_rmse": test_rmse,
            "test_r2": test_r2
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
