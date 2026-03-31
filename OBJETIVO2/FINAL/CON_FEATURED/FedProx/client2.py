# =========================
# IMPORTS
# =========================
import torch
import torch.nn as nn
import flwr as fl
import numpy as np
from collections import OrderedDict
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from model import PVModel
import pandas as pd
from sklearn.metrics import r2_score
import glob
import sys
import os

# =========================
# CONFIG
# =========================
FEATURES = [
    "POA irradiance CMP22 pyranometer (W/m2)",
    "PV module back surface temperature (degC)",
    "Dry bulb temperature (degC)",
    "T_diff",
    "k_panel",
    "physical_model",
    "poa_ghi_ratio",
    "dni_ghi_ratio",
    "dhi_ghi_ratio",
    "cloud_index",
    "temp_diff_air",
    "poa_temp",
    "ghi_temp",
]

LR = 0.001
LOCAL_EPOCHS = 10
MU = 0.1

class PVClient(fl.client.NumPyClient):
    def __init__(self, parque):

        self.model = PVModel(input_size=len(FEATURES), layers_sizes=[64, 32])
        self.criterion = nn.MSELoss()

        # =========================
        # CARGA DATOS
        # =========================
        ruta = f"../../../PV_MaximumPowerPredictor/{parque}_*.csv"
        dfs = []

        for archivo in glob.glob(ruta):
            df = pd.read_csv(archivo)
            df["panel_id"] = "".join(os.path.basename(archivo).replace(".csv","").split("_")[1:])
            dfs.append(df)

        df = pd.concat(dfs, ignore_index=True)
        df = df.replace(-9999, np.nan).dropna()

        # =========================
        # SPLIT
        # =========================
        train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42)

        # =========================
        # k_panel
        # =========================
        k_por_panel = {}
        for pid, grupo in train_df.groupby("panel_id"):
            if len(grupo) < 10:
                continue

            G = grupo["POA irradiance CMP22 pyranometer (W/m2)"].values
            T = grupo["PV module back surface temperature (degC)"].values
            P = grupo["Pmp (W)"].values

            y_k = P / (G + 1e-6)
            X_k = (T - 25).reshape(-1, 1)

            m = LinearRegression()
            m.fit(X_k, y_k)

            k_por_panel[pid] = float(-m.coef_[0])

        k_global = np.mean(list(k_por_panel.values())) if k_por_panel else 0.004

        # =========================
        # FEATURES
        # =========================
        for df_ in [train_df, val_df, test_df]:

            df_["k_panel"] = df_["panel_id"].map(k_por_panel).fillna(k_global)
            df_["physical_model"] = (
                1 - df_["k_panel"] *
                (df_["PV module back surface temperature (degC)"] - 25)
            )
            df_["T_diff"] = df_["PV module back surface temperature (degC)"] - 25
            df_["poa_ghi_ratio"] = df_["POA irradiance CMP22 pyranometer (W/m2)"] / (df_["Global horizontal irradiance (W/m2)"] + 1e-6)
            df_["dni_ghi_ratio"] = df_["Direct normal irradiance (W/m2)"] / (df_["Global horizontal irradiance (W/m2)"] + 1e-6)
            df_["dhi_ghi_ratio"] = df_["Diffuse horizontal irradiance (W/m2)"] / (df_["Global horizontal irradiance (W/m2)"] + 1e-6)
            df_["cloud_index"] = df_["Diffuse horizontal irradiance (W/m2)"] / (df_["Direct normal irradiance (W/m2)"] + 1e-6)
            df_["temp_diff_air"] = df_["PV module back surface temperature (degC)"] - df_["Dry bulb temperature (degC)"]
            df_["poa_temp"] = df_["POA irradiance CMP22 pyranometer (W/m2)"] * df_["PV module back surface temperature (degC)"]
            df_["ghi_temp"] = df_["Global horizontal irradiance (W/m2)"] * df_["Dry bulb temperature (degC)"]

        X_train = train_df[FEATURES]
        y_train = train_df["Pmp (W)"]

        X_val = val_df[FEATURES]
        y_val = val_df["Pmp (W)"]

        X_test = test_df[FEATURES]
        y_test = test_df["Pmp (W)"]

        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

        X_train = self.x_scaler.fit_transform(X_train)
        X_val   = self.x_scaler.transform(X_val)
        X_test  = self.x_scaler.transform(X_test)

        y_train = self.y_scaler.fit_transform(y_train.values.reshape(-1,1))
        y_val   = self.y_scaler.transform(y_val.values.reshape(-1,1))
        y_test  = self.y_scaler.transform(y_test.values.reshape(-1,1))

        self.train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                                     torch.tensor(y_train, dtype=torch.float32)),
                                       batch_size=256, shuffle=True)

        self.val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                                   torch.tensor(y_val, dtype=torch.float32)), batch_size=256)

        self.test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                                    torch.tensor(y_test, dtype=torch.float32)), batch_size=256)

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        self.model.load_state_dict(OrderedDict({k: torch.tensor(v) for k,v in params_dict}))

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        global_params = [torch.tensor(p, dtype=torch.float32) for p in parameters]
        optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)

        self.model.train()
        for _ in range(LOCAL_EPOCHS):
            for X,y in self.train_loader:
                optimizer.zero_grad()
                loss = self.criterion(self.model(X), y)

                prox = sum(torch.norm(w - w0)**2 for w, w0 in zip(self.model.parameters(), global_params))
                loss = loss + (MU/2)*prox

                loss.backward()
                optimizer.step()

        return self.get_parameters({}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()

        preds, targets = [], []

        with torch.no_grad():
            for X, y in self.val_loader:
                preds.append(self.model(X))
                targets.append(y)

        preds   = self.y_scaler.inverse_transform(torch.cat(preds).numpy()).flatten()
        targets = self.y_scaler.inverse_transform(torch.cat(targets).numpy()).flatten()

        val_mse  = float(((targets - preds) ** 2).mean())
        val_rmse = float(np.sqrt(((targets - preds) ** 2).mean()))  # ← float()
        val_r2   = float(r2_score(targets, preds))                  # ← float()

        test_preds, test_targets = [], []

        with torch.no_grad():
            for X, y in self.test_loader:
                test_preds.append(self.model(X))
                test_targets.append(y)

        test_preds   = self.y_scaler.inverse_transform(torch.cat(test_preds).numpy()).flatten()
        test_targets = self.y_scaler.inverse_transform(torch.cat(test_targets).numpy()).flatten()

        test_mse  = float(((test_targets - test_preds) ** 2).mean())
        test_rmse = float(np.sqrt(((test_targets - test_preds) ** 2).mean()))  # ← float()
        test_r2   = float(r2_score(test_targets, test_preds))                  # ← float()

        print(f"[VAL]  MSE: {val_mse:.4f} | RMSE: {val_rmse:.4f} | R²: {val_r2:.4f}")
        print(f"[TEST] MSE: {test_mse:.4f} | RMSE: {test_rmse:.4f} | R²: {test_r2:.4f}")

        return 0.0, len(self.val_loader.dataset), {
            "val_mse":   val_mse,
            "val_rmse":  val_rmse,
            "val_r2":    val_r2,
            "test_mse":  test_mse,
            "test_rmse": test_rmse,
            "test_r2":   test_r2,
        }

# MAIN
if __name__ == "__main__":
    parque = sys.argv[1]

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=PVClient(parque=parque)
    )