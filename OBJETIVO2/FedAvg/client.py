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
from sklearn.linear_model import LinearRegression
from model import PVModel
import pandas as pd
from sklearn.metrics import r2_score
import glob
import sys

# =========================
# CONFIG
# =========================

FEATURES = [
    "POA irradiance CMP22 pyranometer (W/m2)",
    "PV module back surface temperature (degC)",
    "Dry bulb temperature (degC)",

    "T_diff",

    "poa_ghi_ratio",
    "dni_ghi_ratio",
    "dhi_ghi_ratio",
    "cloud_index",
    "temp_diff_air",
    "physical_model",

    "poa_temp",
    "ghi_temp",

    # temporales
    "sin_hour",
    "cos_hour"
]

LAGS = [1, 2, 3]

LOCAL_EPOCHS = 10


class PVClient(fl.client.NumPyClient):
    def __init__(self, parque):

        self.model = PVModel(
            input_size=len(FEATURES) * (len(LAGS) + 1),
            layers_sizes=[128, 32, 16]
        )
        self.criterion = nn.MSELoss()

        # =========================
        # CARGA DATOS
        # =========================
        ruta = f"../PV_MaximumPowerPredictor/{parque}_*.csv"
        dfs = []

        for archivo in glob.glob(ruta):
            df = pd.read_csv(archivo)
            dfs.append(df)

        df = pd.concat(dfs, ignore_index=True)

        # =========================
        # ORDEN TEMPORAL
        # =========================
        df["Time Stamp (local standard time) yyyy-mm-ddThh:mm:ss"] = pd.to_datetime(
            df["Time Stamp (local standard time) yyyy-mm-ddThh:mm:ss"]
        )
        df = df.sort_values("Time Stamp (local standard time) yyyy-mm-ddThh:mm:ss")
        df = df.reset_index(drop=True)

        # =========================
        # LIMPIEZA
        # =========================
        df = df.replace(-9999, np.nan).dropna()

        # =========================
        # VARIABLES TEMPORALES
        # =========================
        df["hour"] = df["Time Stamp (local standard time) yyyy-mm-ddThh:mm:ss"].dt.hour
        df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)

        # =========================
        # SPLIT
        # =========================
        train_val_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)
        train_df, val_df = train_test_split(train_val_df, test_size=0.2, shuffle=False)

        # =========================
        # k GLOBAL
        # =========================
        G = train_df["POA irradiance CMP22 pyranometer (W/m2)"].values
        T = train_df["PV module back surface temperature (degC)"].values
        P = train_df["Pmp (W)"].values

        y_k = P / (G + 1e-6)
        X_k = np.column_stack([(T - 25), G, G * (T - 25)])

        model_k = LinearRegression()
        model_k.fit(X_k, y_k)

        k_global = float(-model_k.coef_[0])

        # =========================
        # CREAR FEATURES
        # =========================
        for df_ in [train_df, val_df, test_df]:

            df_["k_panel"] = k_global

            df_["physical_model"] = (
                df_["POA irradiance CMP22 pyranometer (W/m2)"] *
                (1 - df_["k_panel"] *
                     (df_["PV module back surface temperature (degC)"] - 25))
            )

            df_["T_diff"] = df_["PV module back surface temperature (degC)"] - 25

            df_["poa_ghi_ratio"] = df_["POA irradiance CMP22 pyranometer (W/m2)"] / (
                df_["Global horizontal irradiance (W/m2)"] + 1e-6
            )

            df_["dni_ghi_ratio"] = df_["Direct normal irradiance (W/m2)"] / (
                df_["Global horizontal irradiance (W/m2)"] + 1e-6
            )

            df_["dhi_ghi_ratio"] = df_["Diffuse horizontal irradiance (W/m2)"] / (
                df_["Global horizontal irradiance (W/m2)"] + 1e-6
            )

            df_["cloud_index"] = df_["Diffuse horizontal irradiance (W/m2)"] / (
                df_["Direct normal irradiance (W/m2)"] + 1e-6
            )

            df_["temp_diff_air"] = (
                df_["PV module back surface temperature (degC)"] -
                df_["Dry bulb temperature (degC)"]
            )

            df_["poa_temp"] = df_["POA irradiance CMP22 pyranometer (W/m2)"] * \
                             df_["PV module back surface temperature (degC)"]

            df_["ghi_temp"] = df_["Global horizontal irradiance (W/m2)"] * \
                             df_["Dry bulb temperature (degC)"]

        # =========================
        # LAGS
        # =========================
        for df_ in [train_df, val_df, test_df]:
            for lag in LAGS:
                for col in FEATURES:
                    df_[f"{col}_lag{lag}"] = df_[col].shift(lag)

        train_df = train_df.dropna()
        val_df   = val_df.dropna()
        test_df  = test_df.dropna()

        # =========================
        # FEATURES FINALES
        # =========================
        ALL_FEATURES = FEATURES.copy()

        for lag in LAGS:
            for col in FEATURES:
                ALL_FEATURES.append(f"{col}_lag{lag}")

        # =========================
        # INPUT / TARGET
        # =========================
        X_train = train_df[ALL_FEATURES]
        y_train = train_df["Pmp (W)"]

        X_val = val_df[ALL_FEATURES]
        y_val = val_df["Pmp (W)"]

        X_test = test_df[ALL_FEATURES]
        y_test = test_df["Pmp (W)"]

        # =========================
        # ESCALADO
        # =========================
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

        X_train = self.x_scaler.fit_transform(X_train)
        X_val   = self.x_scaler.transform(X_val)
        X_test  = self.x_scaler.transform(X_test)

        y_train = self.y_scaler.fit_transform(y_train.values.reshape(-1, 1))
        y_val   = self.y_scaler.transform(y_val.values.reshape(-1, 1))
        y_test  = self.y_scaler.transform(y_test.values.reshape(-1, 1))

        # =========================
        # DATALOADERS
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

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.model.train()
        for _ in range(LOCAL_EPOCHS):
            for X_batch, y_batch in self.train_loader:
                optimizer.zero_grad()
                loss = self.criterion(self.model(X_batch), y_batch)
                loss.backward()
                optimizer.step()

        return self.get_parameters({}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()

        preds, targets = [], []

        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                preds.append(self.model(X_batch))
                targets.append(y_batch)

        preds = self.y_scaler.inverse_transform(torch.cat(preds).numpy()).flatten()
        targets = self.y_scaler.inverse_transform(torch.cat(targets).numpy()).flatten()

        mse = float(((targets - preds) ** 2).mean())
        rmse = mse ** 0.5
        r2 = r2_score(targets, preds)

        print(f"[VAL] RMSE: {rmse:.4f} | R²: {r2:.4f}")

        # =========================
        # TEST
        # =========================
        test_preds, test_targets = [], []

        with torch.no_grad():
            for X, y in self.test_loader:
                test_preds.append(self.model(X))
                test_targets.append(y)

        test_preds = self.y_scaler.inverse_transform(torch.cat(test_preds).numpy()).flatten()
        test_targets = self.y_scaler.inverse_transform(torch.cat(test_targets).numpy()).flatten()

        test_mse = float(((test_targets - test_preds) ** 2).mean())
        test_rmse = test_mse ** 0.5
        test_r2 = r2_score(test_targets, test_preds)

        print(f"[TEST] RMSE: {test_rmse:.4f} | R²: {test_r2:.4f}")

        return mse, len(self.val_loader.dataset), {
            "val_mse": mse,
            "val_rmse": rmse,
            "val_r2": r2,
            "test_mse": test_mse,
            "test_rmse": test_rmse,
            "test_r2": test_r2
        }


if __name__ == "__main__":
    parque = sys.argv[1]

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=PVClient(parque=parque)
    )