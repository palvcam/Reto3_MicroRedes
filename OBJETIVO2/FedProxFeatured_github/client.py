# Cargar librerias
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
import re

# CONFIGURACIÓN
# Variables que vamos a utilizar
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
    "poa_temp",
    "ghi_temp",
    "k_panel",
    "factor_panel",
]

# Congiguraciones de FedEx
EPOCH_CONFIGS = [3, 5, 8, 10] # Número de épocas locales por ronda
MU_CONFIGS    = [0.0, 0.01, 0.1, 0.5] # Parámetro de regularización FedProx (0.0 = FedAvg)
LR            = 0.001 # Learning rate del optimizador Adam

# CLIENTE
class PVClient(fl.client.NumPyClient):
    def __init__(self, parque):
        self.model     = PVModel(input_size=len(FEATURES), layers_sizes=[128, 64, 32]) # Modelo
        self.criterion = nn.MSELoss() # Función de pérdida

        # CARGA DATOS
        ruta = f"../PV_MaximumPowerPredictor/{parque}_*.csv" # Buscar todos los CSV del parque
        dfs  = []

        for archivo in glob.glob(ruta):
            df = pd.read_csv(archivo)
            # Extraer panel_id del nombre del archivo
            df["panel_id"]     = "_".join(os.path.basename(archivo).replace(".csv","").split("_")[1:])
            dfs.append(df)

        # Combinar todos los paneles del parque en un único DataFrame
        df_parque = pd.concat(dfs, ignore_index=True)
        # Eliminar timestamp
        df_parque = df_parque.drop(columns=["Time Stamp (local standard time) yyyy-mm-ddThh:mm:ss"], errors="ignore") 
        # Quitamos los outliers
        df_parque = df_parque.replace(-9999, np.nan).dropna()

        # FILTRO IRRADIANCIA (Eliminar registros con irradiancia muy baja — el panel no genera potencia estable
        # por debajo de 50 W/m² (amanecer, anochecer, sombras))
        df_parque = df_parque[
            df_parque["POA irradiance CMP22 pyranometer (W/m2)"] > 50
        ].reset_index(drop=True)

        # SPLIT (60% train / 20% val / 20% test)
        train_val_df, test_df = train_test_split(df_parque, test_size=0.2, random_state=42)
        train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42)

        # ESTIMACIÓN DE k_panel (SOLO EN TRAIN)
        # Estimación de k_panel mediante regresión lineal
        # Si dividimos ambos lados del modelo físico por G:
        #   P/G = factor × (1 + k × (T-25))
        #   P/G ≈ factor + factor×k × (T-25)
        # Como factor es aproximadamente constante por panel,
        # la relación P/G ~ (T-25) es lineal con pendiente ≈ factor×k
        # → k = -pendiente / factor
        # El signo negativo es porque a más temperatura, menos potencia
        k_por_panel = {}
        for pid, grupo in train_df.groupby("panel_id"):
            if len(grupo) < 10: # Ignorar paneles con pocos datos — regresión no fiable
                continue
            G   = grupo["POA irradiance CMP22 pyranometer (W/m2)"].values
            T   = grupo["PV module back surface temperature (degC)"].values
            P   = grupo["Pmp (W)"].values
            y_k = P / (G + 1e-6) # Potencia específica (W per W/m²)
            X_k = (T - 25).reshape(-1, 1) # Desviación respecto a temperatura STC
            m   = LinearRegression()
            m.fit(X_k, y_k)
            k_por_panel[pid] = float(-m.coef_[0]) # Signo negativo: más calor = menos potencia

        # Valor global para paneles sin suficientes datos o paneles nuevos
        k_global = np.mean(list(k_por_panel.values())) if k_por_panel else 0.004

        # ESTIMACIÓN DE factor_panel (SOLO EN TRAIN)
        # factor_panel es la eficiencia nominal del panel: cuántos W produce por W/m² de irradiancia
        # Equivale a η×A del modelo físico (eficiencia × área)
        # Regresión lineal sin intercept: Pmp = factor × POA
        # Sin intercept porque si no hay irradiancia no hay potencia (pasa por el origen)
        factor_por_panel = {}
        for pid, grupo in train_df.groupby("panel_id"):
            if len(grupo) < 10: # Ignorar paneles con pocos datos — regresión no fiable
                continue
            G = grupo["POA irradiance CMP22 pyranometer (W/m2)"].values
            P = grupo["Pmp (W)"].values
            m = LinearRegression(fit_intercept=False) # Recta que pasa por el origen
            m.fit(G.reshape(-1, 1), P)
            factor_por_panel[pid] = float(m.coef_[0])
        # Valor global para paneles sin suficientes datos o paneles nuevos
        factor_global = np.mean(list(factor_por_panel.values())) if factor_por_panel else 0.15

        # APLICAR k_panel y factor_panel
        # Los parámetros estimados en train se aplican a val y test
        for df in [train_df, val_df, test_df]:
            df["k_panel"]      = df["panel_id"].map(k_por_panel).fillna(k_global)
            df["factor_panel"] = df["panel_id"].map(factor_por_panel).fillna(factor_global)


        # FEATURES DE INGENIERÍA
        for df in [train_df, val_df, test_df]:
            # Desviación de temperatura respecto a condiciones estándar STC (25°C)
            df["T_diff"]        = df["PV module back surface temperature (degC)"] - 25
            # Ratios de irradiancia — capturan condiciones de cielo y ángulo solar
            df["poa_ghi_ratio"] = df["POA irradiance CMP22 pyranometer (W/m2)"] / (df["Global horizontal irradiance (W/m2)"] + 1e-6)
            df["dni_ghi_ratio"] = df["Direct normal irradiance (W/m2)"]          / (df["Global horizontal irradiance (W/m2)"] + 1e-6)
            df["dhi_ghi_ratio"] = df["Diffuse horizontal irradiance (W/m2)"]     / (df["Global horizontal irradiance (W/m2)"] + 1e-6)
            # Índice de nubosidad
            df["cloud_index"]   = df["Diffuse horizontal irradiance (W/m2)"]     / (df["Direct normal irradiance (W/m2)"]     + 1e-6)
            # Diferencia térmica panel-aire
            df["temp_diff_air"] = df["PV module back surface temperature (degC)"] - df["Dry bulb temperature (degC)"]
            # Términos de interacción
            df["poa_temp"]      = df["POA irradiance CMP22 pyranometer (W/m2)"]  * df["PV module back surface temperature (degC)"]
            df["ghi_temp"]      = df["Global horizontal irradiance (W/m2)"]      * df["Dry bulb temperature (degC)"]


        # INPUT FINAL
        X_train = train_df[FEATURES] # Features de entrenamiento
        y_train = train_df["Pmp (W)"] # Target: potencia máxima real
        X_val   = val_df[FEATURES]
        y_val   = val_df["Pmp (W)"]
        X_test  = test_df[FEATURES]
        y_test  = test_df["Pmp (W)"]

        # ESCALADO
        self.x_scaler = StandardScaler() # Escalador de features
        self.y_scaler = StandardScaler() # Escalador del target

        X_train = self.x_scaler.fit_transform(X_train)
        X_val   = self.x_scaler.transform(X_val)
        X_test  = self.x_scaler.transform(X_test)

        y_train = self.y_scaler.fit_transform(y_train.values.reshape(-1, 1))
        y_val   = self.y_scaler.transform(y_val.values.reshape(-1, 1))
        y_test  = self.y_scaler.transform(y_test.values.reshape(-1, 1))

        # DATALOADERS
        self.train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                          torch.tensor(y_train, dtype=torch.float32)),
            batch_size=256, shuffle=True
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

    # PARÁMETROS DE LOS MODELOS
    def get_parameters(self,config=None):
        # Devuelve los pesos del modelo como lista
        # El servidor los usa para agregar los modelos de todos los clientes
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        # Carga los pesos globales agregados por el servidor en el modelo local
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict  = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    # TRAIN
    def fit(self, parameters, config):
        self.set_parameters(parameters) # Cargar pesos globales del servidor antes de entrenar
        # Calcular loss de validación ANTES del entrenamiento local
        # FedEx usa este valor para medir cuánto mejora cada configuración
        self.model.eval()
        before_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                before_loss += self.criterion(self.model(X_batch), y_batch).item()
        before_loss /= len(self.val_loader)

        # Leer configuración enviada por FedEx desde el servidor
        epochs       = int(config.get("epochs", 1)) # Épocas locales a entrenar
        mu           = float(config.get("mu", 0.0)) # Fuerza de regularización FedProx
        print(f"\n[FedEx] epochs={epochs} | mu={mu} | before_loss={before_loss:.4f}")
        # Guardar copia de los pesos globales
        global_params = [torch.tensor(p.copy(), dtype=torch.float32) for p in parameters]
        optimizer    = torch.optim.Adam(self.model.parameters(), lr=LR)

        # Bucle de entrenamiento local
        self.model.train()
        for _ in range(epochs):
            for X_batch, y_batch in self.train_loader:
                optimizer.zero_grad()
                preds = self.model(X_batch)
                loss  = self.criterion(preds, y_batch)
                # Término proximal FedProx: penaliza desviación de los pesos globales
                prox  = sum(torch.norm(w - wg)**2 for w, wg in zip(self.model.parameters(), global_params))
                loss += (mu / 2) * prox
                loss.backward() # Backpropagation
                optimizer.step() # Actualización de pesos

        # Calcular métricas de validación
        self.model.eval()
        val_loss, val_preds, val_targets = 0.0, [], []
        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                val_loss += self.criterion(self.model(X_batch), y_batch).item()
                val_preds.append(self.model(X_batch).cpu())
                val_targets.append(y_batch.cpu())
        val_loss   /= len(self.val_loader)
        val_preds   = torch.cat(val_preds).flatten()
        val_targets = torch.cat(val_targets).flatten()
        val_rmse    = torch.sqrt(torch.mean((val_preds - val_targets)**2)).item()
        ss_res      = torch.sum((val_targets - val_preds)**2)
        ss_tot      = torch.sum((val_targets - val_targets.mean())**2)
        val_r2      = (1 - ss_res / ss_tot).item() if ss_tot > 0 else 0.0

        # Calcular métricas de test
        test_loss, test_preds, test_targets = 0.0, [], []
        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                test_loss += self.criterion(self.model(X_batch), y_batch).item()
                test_preds.append(self.model(X_batch).cpu())
                test_targets.append(y_batch.cpu())
        test_loss    /= len(self.test_loader)
        test_preds    = torch.cat(test_preds).flatten()
        test_targets  = torch.cat(test_targets).flatten()
        test_rmse     = torch.sqrt(torch.mean((test_preds - test_targets)**2)).item()
        test_ss_res   = torch.sum((test_targets - test_preds)**2)
        test_ss_tot   = torch.sum((test_targets - test_targets.mean())**2)
        test_r2       = (1 - test_ss_res / test_ss_tot).item() if test_ss_tot > 0 else 0.0


        # Métricas de train
        train_loss_log, train_preds_log, train_targets_log = 0.0, [], []
        with torch.no_grad():
            for X_batch, y_batch in self.train_loader:
                train_loss_log += self.criterion(self.model(X_batch), y_batch).item()
                train_preds_log.append(self.model(X_batch).cpu())
                train_targets_log.append(y_batch.cpu())
        train_loss_log   /= len(self.train_loader)
        train_preds_log   = torch.cat(train_preds_log).flatten()
        train_targets_log = torch.cat(train_targets_log).flatten()
        ss_res_tr = torch.sum((train_targets_log - train_preds_log)**2)
        ss_tot_tr = torch.sum((train_targets_log - train_targets_log.mean())**2)
        train_r2_log = (1 - ss_res_tr / ss_tot_tr).item() if ss_tot_tr > 0 else 0.0

        print(f"\n{'='*50}")
        print(f"  TRAIN  | MSE: {train_loss_log:.4f} | R²: {train_r2_log:.4f}")
        print(f"  VAL    | MSE: {val_loss:.4f}  | RMSE: {val_rmse:.4f} | R²: {val_r2:.4f}")
        print(f"  TEST   | MSE: {test_loss:.4f}  | RMSE: {test_rmse:.4f} | R²: {test_r2:.4f}")
        print(f"{'='*50}\n")
        # Devolver pesos actualizados, número de muestras y métricas al servidor
        # FedEx usa val_mse_before y val_mse para calcular la mejora de cada config
        return (self.get_parameters({}),
                len(self.train_loader.dataset),
                {
                    "val_mse":        val_loss,
                    "val_mse_before": before_loss,
                    "val_rmse":       val_rmse,
                    "val_r2":         val_r2,
                    "val_samples":    float(len(self.val_loader.dataset)),
                    "test_mse":       test_loss,
                    "test_rmse":      test_rmse,
                    "test_r2":        test_r2,
                })

    # EVALUACIÓN
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()

        preds, targets = [], []
        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                preds.append(self.model(X_batch))
                targets.append(y_batch)

        preds   = self.y_scaler.inverse_transform(torch.cat(preds).numpy())
        targets = self.y_scaler.inverse_transform(torch.cat(targets).numpy())

        mse  = float(((targets - preds) ** 2).mean())
        rmse = mse ** 0.5
        r2   = r2_score(targets, preds)

        test_preds, test_targets = [], []
        with torch.no_grad():
            for X, y in self.test_loader:
                test_preds.append(self.model(X))
                test_targets.append(y)

        test_preds   = self.y_scaler.inverse_transform(torch.cat(test_preds).numpy())
        test_targets = self.y_scaler.inverse_transform(torch.cat(test_targets).numpy())

        test_mse  = float(((test_targets - test_preds) ** 2).mean())
        test_rmse = test_mse ** 0.5
        test_r2   = r2_score(test_targets, test_preds)

        return mse, len(self.val_loader.dataset), {
            "val_mse":   mse,
            "val_rmse":  rmse,
            "val_r2":    r2,
            "test_mse":  test_mse,
            "test_rmse": test_rmse,
            "test_r2":   test_r2,
        }

# MAIN
if __name__ == "__main__":
    parque = sys.argv[1] # Nombre del parque pasado como argumento
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080", # Dirección del servidor federado
        client=PVClient(parque=parque)
    )