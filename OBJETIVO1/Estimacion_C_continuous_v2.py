from custom_env_continuous_v2 import CustomEnvContinuousv2
from pymgrid import Microgrid
from pymgrid.modules import GridModule, BatteryModule, LoadModule, RenewableModule
from pathlib import Path
import pandas as pd
import numpy as np

ruta_padre=Path(__file__).parent.parent
# PRECIOS DE LA RED (Península 2025)
ruta_precios = ruta_padre /"OBJETIVO1" / 'data' / 'external'/ 'precio2025-peninsula.csv'
df_precios = pd.read_csv(ruta_precios, sep=';')
df_precios['datetime'] = pd.to_datetime(df_precios['datetime'], utc=True)
df_precios = df_precios.sort_values('datetime').reset_index(drop=True)
# Convertir de €/MWh a €/kWh
precios_kwh = df_precios['value'].values / 1000.0

# DEMANDA (LOAD) Y GENERACIÓN SOLAR (PV)
ruta_load = ruta_padre/ "OBJETIVO1" / 'data'/ 'raw'/ 'load'/ 'RefBldgFullServiceRestaurantNew2004_v1.3_7.1_6A_USA_MN_MINNEAPOLIS.csv'
ruta_pv = ruta_padre / "OBJETIVO1" / 'data'/ 'processed'/ 'pv_generacion_corregida_kw.csv'

df_load = pd.read_csv(ruta_load)
df_pv = pd.read_csv(ruta_pv)

load_series = df_load.iloc[:, -1].values
pv_series = df_pv.iloc[:, -1].values

# Asegurarnos de que todas las series tienen la misma longitud
min_len = min(len(precios_kwh), len(load_series), len(pv_series), 8760)
precios_kwh = precios_kwh[:min_len]
load_series = load_series[:min_len]
pv_series = pv_series[:min_len]

print(f"Datos cargados correctamente. Longitud de las series: {min_len} horas.")

# RED ELÉCTRICA (GridModule)
# En las versiones recientes de pymgrid, los datos de la red se pasan como un DataFrame
def build_microgrid(precios_kwh, load_series, pv_series):
    grid_ts = pd.DataFrame({
        'import_price': precios_kwh,
        'export_price': precios_kwh * 0.5,
        'co2_per_kwh': 0.0
    })

    grid = GridModule(
        max_import=200.0,
        max_export=200.0,
        time_series=grid_ts
    )

    battery = BatteryModule(
        min_capacity=10.0,
        max_capacity=200.0,
        max_charge=50.0,
        max_discharge=50.0,
        efficiency=0.9,
        init_soc=0.5
    )

    load = LoadModule(time_series=load_series)
    pv = RenewableModule(time_series=pv_series)

    modules = [
        ('grid', grid),
        ('battery', battery),
        ('load', load),
        ('pv', pv)
    ]

    return Microgrid(modules)


print("=== INICIANDO ESTIMACIÓN CON AGENTE ALEATORIO ===")

# Construimos la microrred y el entorno (usando C=1.0 para ver los valores brutos)
mg_est = build_microgrid(precios_kwh, load_series, pv_series)
env_est = CustomEnvContinuousv2(
    pymgrid_network=mg_est,
    horizon=24 * 365,
    reward_scale_C=1,        # <- Mantenemos 1.0 para extraer el coste real
    low_soc_penalty=0.0,       # <- Apagamos la penalización para no ensuciar el coste económico
    low_soc_threshold=0.20,
    net_load_min=-40.64,
    net_load_max=62.45,
    price_min=0.02,
    price_max=0.425,
)

obs, info = env_est.reset()
raw_rewards = []
soc_history = []

# Simulamos 1 año entero (o hasta que acabe el episodio)
done = False
steps = 0

while not done:
    # Acción 100% aleatoria
    action = env_est.action_space.sample()
    
    obs, reward, terminated, truncated, step_info = env_est.step(action)
    
    # Guardamos el reward bruto (que equivale al coste económico negativo de pymgrid)
    raw_rewards.append(step_info["mg_reward"])
    soc_history.append(step_info["soc_after"])
    
    done = terminated or truncated
    steps += 1

raw_rewards = np.array(raw_rewards)

# Estadísticas
media = np.mean(raw_rewards)
desviacion_std = np.std(raw_rewards)
minimo = np.min(raw_rewards)
maximo = np.max(raw_rewards)
p05 = np.percentile(raw_rewards, 5) # El 5% de las peores horas (anomalías/balanceo)

print(f"\nResultados tras {steps} steps aleatorios:")
print(f"  Media del reward bruto: {media:.2f}")
print(f"  Desviación Estándar (σ): {desviacion_std:.2f}")
print(f"  Mínimo (Peor coste): {minimo:.2f}")
print(f"  Percentil 5 (Peor coste sin contar anomalías extremas): {p05:.2f}")

# --- PROPUESTA DE CALIBRACIÓN ---
C_propuesta = desviacion_std

print("\n=== CALIBRACIÓN RECOMENDADA ===")
print(f"1. Fija tu constante C (reward_scale_C) en: {C_propuesta:.2f}")
print("   (Esto hará que el ~68% de tus recompensas económicas caigan entre [-1, 1])")

print("\n2. Configuración para 'low_soc_penalty':")
print("   - Si quieres un AVISO LEVE (equivalente a media desviación estándar económica):")
print(f"     low_soc_penalty = 0.5")
print("   - Si quieres un CASTIGO SEVERO (equivalente a una hora económicamente desastrosa):")
print(f"     low_soc_penalty = 2.0")
print("   - Si quieres un CASTIGO EXTREMO (equivalente a las peores penalizaciones de balanceo):")
print(f"     low_soc_penalty = {abs(p05 / C_propuesta):.2f} (Aprox)")

env_est.close()