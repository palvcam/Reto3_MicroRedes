import os
import sys
import warnings
import random
from pathlib import Path

import optuna
import pandas as pd
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback

# Añadir el directorio padre al path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuración de rutas
ruta_script = Path(__file__).parent
ruta_padre = ruta_script.parent.parent
sys.path.append(str(ruta_padre))

from custom_env_continuous_v2 import CustomEnvContinuousv2
from pymgrid.modules import GridModule, BatteryModule, LoadModule, RenewableModule
from pymgrid import Microgrid

warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")
warnings.filterwarnings("ignore", message=".*Training and eval env are not of the same type.*")

# =====================================================================
# CALLBACK PERSONALIZADO PARA OPTUNA (¡El que faltaba!)
# =====================================================================
class TrialEvalCallback(EvalCallback):
    """Callback que evalúa al agente y le chiva a Optuna si debe cancelar (podar) el entrenamiento."""
    def __init__(
        self,
        eval_env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10950,
        deterministic: bool = True,
        verbose: int = 0,
        best_model_save_path: str = None,
        log_path: str = None,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
            best_model_save_path=best_model_save_path,
            log_path=log_path
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        continue_training = super()._on_step()
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            if self.trial.should_prune():
                self.is_pruned = True
                return False  # Cortamos el entrenamiento
        return continue_training

# =====================================================================
# CONFIGURACIÓN DE DATOS
# =====================================================================
ruta_precios = ruta_padre / "OBJETIVO1" / 'data' / 'external' / 'precio2025-peninsula.csv'
ruta_load = ruta_padre / "OBJETIVO1" / 'data' / 'raw' / 'load' / 'RefBldgFullServiceRestaurantNew2004_v1.3_7.1_6A_USA_MN_MINNEAPOLIS.csv'
ruta_pv = ruta_padre / "OBJETIVO1" / 'data' / 'processed' / 'pv_generacion_corregida_kw.csv'

df_precios = pd.read_csv(ruta_precios, sep=';')
precios_kwh = df_precios.sort_values('datetime')['value'].values / 1000.0
load_series = pd.read_csv(ruta_load).iloc[:, -1].values
pv_series = pd.read_csv(ruta_pv).iloc[:, -1].values

min_len = min(len(precios_kwh), len(load_series), len(pv_series), 8760)
precios_kwh, load_series, pv_series = precios_kwh[:min_len], load_series[:min_len], pv_series[:min_len]

# =====================================================================
# HELPERS DE ENTORNO
# =====================================================================
def make_env(seed=0):
    def _init():
        grid_ts = pd.DataFrame({'import_price': precios_kwh, 'export_price': precios_kwh * 0.5, 'co2_per_kwh': 0.0})
        grid = GridModule(max_import=200.0, max_export=200.0, time_series=grid_ts)
        battery = BatteryModule(min_capacity=10.0, max_capacity=200.0, max_charge=50.0, max_discharge=50.0, efficiency=0.9, init_soc=0.5)
        load = LoadModule(time_series=load_series)
        pv = RenewableModule(time_series=pv_series)
        mg = Microgrid([('grid', grid), ('battery', battery), ('load', load), ('pv', pv)])
        
        env = CustomEnvContinuousv2(
            pymgrid_network=mg, horizon=8760, reward_scale_C=91.88,
            low_soc_penalty=2.0, low_soc_threshold=0.20,
            net_load_min=-40.64, net_load_max=62.45,
            price_min=0.02, price_max=0.425
        )
        return env
    return _init

# =====================================================================
# OBJETIVO DE OPTUNA (1 SEMILLA - MODO RÁPIDO)
# =====================================================================
def objective(trial: optuna.Trial):
    # 1. Muestreo de hiperparámetros
    kwargs = {
        "n_steps": trial.suggest_categorical("n_steps", [512, 1024, 2048]),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "ent_coef": trial.suggest_float("ent_coef", 1e-8, 0.01, log=True),
        "clip_range": trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3]),
        "n_epochs": trial.suggest_categorical("n_epochs", [5, 10, 20]),
        "gamma": trial.suggest_float("gamma", 0.98, 0.999),
    }

    # ¡SOLO 1 SEMILLA PARA ACELERAR!
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 2. Configurar directorios para este trial
    ruta_logs = ruta_script / "logs" / "logs_optuna_PPO_v1_fast"
    log_dir = ruta_logs / f"trial_{trial.number}"
    os.makedirs(log_dir, exist_ok=True)

    # 3. Entornos
    train_env = make_vec_env(make_env(seed), n_envs=4, vec_env_cls=SubprocVecEnv)
    eval_env = make_vec_env(make_env(seed), n_envs=1)

    tensorboard_dir = ruta_script / "logs" / "tensorboard_logs_v1"
    model = PPO(
            "MlpPolicy", 
            train_env, 
            **kwargs, 
            tensorboard_log=str(tensorboard_dir), # ¡Añadido de nuevo!
            verbose=0
        )
    
    # 4. Callback de poda (Corta los trials malos a medias, ahorrando MUCHO tiempo)
    eval_callback = TrialEvalCallback(
        eval_env,
        trial,
        best_model_save_path=log_dir,
        log_path=log_dir,
        n_eval_episodes=5,        
        eval_freq=10950,          
        deterministic=True
    )

    try:
        model.learn(total_timesteps=525600, callback=eval_callback)
    except (AssertionError, ValueError):
        # Si los parámetros son tan malos que el modelo colapsa, podamos
        train_env.close()
        eval_env.close()
        raise optuna.exceptions.TrialPruned()
    finally:
        train_env.close()
        eval_env.close()

    # Si Optuna decidió podarlo internamente
    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_callback.best_mean_reward

# =====================================================================
# EJECUCIÓN
# =====================================================================
if __name__ == "__main__":
    # He cambiado el nombre de la DB para que no se mezcle con las pruebas anteriores
    study_name = "ppo_microgrid_v1"
    storage_name = f"sqlite:///{ruta_script}/optuna_ppo_v1.db"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="maximize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,       # Primeros 5 trials completan al 100%
            n_warmup_steps=26280      # No poda antes de 3 años simulados (3 * 8760)
        )
    )

    study.optimize(objective, n_trials=50)

    print("\nOPTIMIZACIÓN FINALIZADA")
    print(f"Mejor Reward: {study.best_value:.2f}")
    print(f"Mejores Parámetros: {study.best_params}")