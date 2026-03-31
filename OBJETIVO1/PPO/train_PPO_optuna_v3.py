import os
import optuna
import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO
import sys 
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
#from optuna_integration.sb3 import TrialEvalCallback  # CORRECTO (Versiones nuevas)
# Le decimos a Python que también busque archivos en la carpeta padre (OBJETIVO1)
ruta_padre = Path(__file__).parent.parent
sys.path.append(str(ruta_padre))

# 1. IMPORTAR TUS CLASES Y FUNCIONES
from custom_env_continuous_v2 import CustomEnvContinuousv2
from pymgrid import Microgrid
from pymgrid.modules import GridModule, BatteryModule, LoadModule, RenewableModule

from stable_baselines3.common.callbacks import EvalCallback

import warnings

# Esto filtrará los mensajes molestos del gym antiguo
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")
warnings.filterwarnings("ignore", message=".*Training and eval env are not of the same type.*")

# =====================================================================
# CALLBACK PERSONALIZADO PARA OPTUNA
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
        # Ejecutamos la evaluación estándar de SB3
        continue_training = super()._on_step()
        
        # Si acaba de ocurrir una evaluación...
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.eval_idx += 1
            # Le mandamos el resultado medio a Optuna
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Le preguntamos a Optuna si este trial es un desastre y debemos cortarlo
            if self.trial.should_prune():
                self.is_pruned = True
                return False  # Cortamos el entrenamiento
                
        return continue_training

# =====================================================================
# CONFIGURACIÓN DE RUTAS Y DATOS (Ajusta según tu estructura)
# =====================================================================
ruta_padre = Path(__file__).parent.parent.parent
ruta_precios = ruta_padre / "OBJETIVO1" / 'data' / 'external' / 'precio2025-peninsula.csv'
ruta_load = ruta_padre / "OBJETIVO1" / 'data' / 'raw' / 'load' / 'RefBldgFullServiceRestaurantNew2004_v1.3_7.1_6A_USA_MN_MINNEAPOLIS.csv'
ruta_pv = ruta_padre / "OBJETIVO1" / 'data' / 'processed' / 'pv_generacion_corregida_kw.csv'


assert ruta_precios.exists(), f"ERROR: No se encontró el archivo {ruta_precios}"
assert ruta_load.exists(), f"ERROR: No se encontró el archivo {ruta_load}"
assert ruta_pv.exists(), f"ERROR: No se encontró el archivo {ruta_pv}"


# Cargar datos (igual que en tus tests)
df_precios = pd.read_csv(ruta_precios, sep=';')
df_precios['datetime'] = pd.to_datetime(df_precios['datetime'], utc=True)
df_precios = df_precios.sort_values('datetime').reset_index(drop=True)
precios_kwh = df_precios['value'].values / 1000.0

df_load = pd.read_csv(ruta_load)
df_pv = pd.read_csv(ruta_pv)
load_series = df_load.iloc[:, -1].values
pv_series = df_pv.iloc[:, -1].values

min_len = min(len(precios_kwh), len(load_series), len(pv_series), 8760)
precios_kwh = precios_kwh[:min_len]
load_series = load_series[:min_len]
pv_series = pv_series[:min_len]

# =====================================================================
# FUNCIONES CREADORAS DE ENTORNOS
# =====================================================================
def build_microgrid():
    """Crea una nueva instancia fresca de la microrred."""
    grid_ts = pd.DataFrame({
        'import_price': precios_kwh,
        'export_price': precios_kwh * 0.5,
        'co2_per_kwh': 0.0
    })
    grid = GridModule(max_import=200.0, max_export=200.0, time_series=grid_ts)
    battery = BatteryModule(min_capacity=10.0, max_capacity=200.0, max_charge=50.0, max_discharge=50.0, efficiency=0.9, init_soc=0.5)
    load = LoadModule(time_series=load_series)
    pv = RenewableModule(time_series=pv_series)
    return Microgrid([('grid', grid), ('battery', battery), ('load', load), ('pv', pv)])

def make_env():
    """Instancia tu entorno custom con los parámetros calibrados."""
    mg = build_microgrid()
    return CustomEnvContinuousv2(
        pymgrid_network=mg,
        horizon=8760,
        reward_scale_C=253.59,      # ¡El valor que calculamos!
        low_soc_penalty=0.2,       
        low_soc_threshold=0.20,
        net_load_min=-40.64,
        net_load_max=62.45,
        price_min=0.02,
        price_max=0.425,
    )

# =====================================================================
# OPTIMIZACIÓN CON OPTUNA
# =====================================================================
def sample_ppo_params(trial: optuna.Trial):
    """Define el espacio de búsqueda de hiperparámetros para PPO."""
    
    # Batch size debe ser divisor de (n_steps * n_envs). Limitamos a opciones seguras.
    # Como n_envs = 4, opciones de n_steps más moderadas (para no esperar demasiado antes de actualizar)
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    
    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "ent_coef": trial.suggest_float("ent_coef", 0.0000001, 0.1, log=True),
        "clip_range": trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3]),
        "n_epochs": trial.suggest_categorical("n_epochs", [5, 10, 20]),
        "gamma": trial.suggest_float("gamma", 0.9, 0.9999, log=True),
    }

def objective(trial: optuna.Trial):
    """Función objetivo que Optuna intentará maximizar."""

    # Carpeta donde está el script que se está ejecutando
    ruta_script = Path(__file__).parent

    # Carpeta logs_PPO dentro del directorio del script
    ruta_logs = ruta_script / "logs" / "logs_optuna_PPO_v3"

    # Carpeta específica para este trial
    log_dir = ruta_logs / f"trial_{trial.number}"

    # Crear directorios
    os.makedirs(log_dir, exist_ok=True)

    
    # 2. Muestrear hiperparámetros
    kwargs = sample_ppo_params(trial)
    
    # 3. Crear entorno de entrenamiento y entorno de evaluación
    # ¡AQUÍ ESTÁ LA MAGIA! Entrenamos con 4 entornos en paralelo
    env = make_vec_env(make_env, n_envs=4, vec_env_cls=SubprocVecEnv)
    
    # El entorno de evaluación lo dejamos en 1. 
    # El callback TrialEvalCallback espera un solo entorno para hacer las validaciones correctas.
    eval_env = make_vec_env(make_env, n_envs=1)
    
    tensorboard_dir = ruta_script /"logs"/ "tensorboard_logs_v3"
    # 4. Crear el modelo PPO
    model = PPO(
        "MlpPolicy",
        env,
        **kwargs,
        tensorboard_log=str(tensorboard_dir),
        verbose=0
    )
    
    # 5. Configurar el Pruning (Poda)
    # Evalúa el modelo cada 10,000 steps. Si es muy malo, cancela el trial.
    eval_callback = TrialEvalCallback(
        eval_env,
        trial,
        best_model_save_path=log_dir,
        log_path=log_dir,
        n_eval_episodes=1,     # Evaluar durante 1 episodio (1 año entero)
        eval_freq=10950, # Evalúa cada 5 años en lugar de cada año
        deterministic=True
    )
    
    # 6. Entrenar el modelo
    # Probamos durante ~5 años simulados por cada hiperparámetro (aprox 40k steps)
    # Puedes subir esto a 100k o 200k steps si tienes buena máquina.
    TOTAL_TIMESTEPS = 525600
    
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback, tb_log_name=f"trial_{trial.number}")
        model.env.close()
        eval_env.close()
    except (AssertionError, ValueError) as e:
        # PPO a veces lanza errores si los hiperparámetros son horribles y la red colapsa
        model.env.close()
        eval_env.close()
        raise optuna.exceptions.TrialPruned()

    # ¡NUEVO! Comprobamos si el callback decidió podar el modelo
    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_callback.best_mean_reward

# =====================================================================
# EJECUCIÓN PRINCIPAL
# =====================================================================
if __name__ == "__main__":
    print("Iniciando la optimización con Optuna...")
    
    # Crear o conectar a la base de datos persistente SQLite
    study_name = "ppo_microgrid_study_v3"
    ruta_script = Path(__file__).parent
    storage_path = ruta_script / "optuna_microgrid_v3.db"

    storage_name = f"sqlite:///{storage_path}"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="maximize",
        load_if_exists=True,
        # 4. Ajuste del Pruner:
        # - n_startup_trials=5: Deja que los primeros 5 trials terminen al 100% para crear una buena línea base.
        # - n_warmup_steps=8760*5: No poda a ningún agente antes de que haya entrenado al menos 5 años.
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, 
            n_warmup_steps= 3
        )
    )
    
    # Ejecutar 50 trials (pruebas de combinaciones distintas)
    study.optimize(objective, n_trials=50)
    
    print("¡Optimización finalizada!")
    print("Mejores hiperparámetros encontrados:")
    print(study.best_params)
    print(f"Mejor reward medio: {study.best_value}")