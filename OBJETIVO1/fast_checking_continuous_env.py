from custom_env_continuous import CustomEnvContinuous 
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

# Construimos la microrred
mg = build_microgrid(precios_kwh, load_series, pv_series)

env = CustomEnvContinuous(
    pymgrid_network=mg,
    horizon=24 * 365,
    reward_scale_C=1.0,
    low_soc_penalty=0.0,
    low_soc_threshold=0.20,
    net_load_min=-40.64,
    net_load_max=62.45,
    price_min=0.02,
    price_max=0.425,
)


print("=== VERIFICACIÓN PREVIA DE DATOS Y ESCALAS ===")

print("len(precios_kwh):", len(precios_kwh))
print("len(load_series):", len(load_series))
print("len(pv_series):", len(pv_series))
print("len(mg):", len(mg))

net_load_series = load_series - pv_series

print("\n--- NET LOAD ---")
print("min:", net_load_series.min())
print("max:", net_load_series.max())
print("mean:", net_load_series.mean())
print("p1:", np.percentile(net_load_series, 1))
print("p5:", np.percentile(net_load_series, 5))
print("p95:", np.percentile(net_load_series, 95))
print("p99:", np.percentile(net_load_series, 99))

print("\n--- PRICE ---")
print("min:", precios_kwh.min())
print("max:", precios_kwh.max())
print("mean:", precios_kwh.mean())

print("\n--- LOAD ---")
print("min:", load_series.min())
print("max:", load_series.max())
print("mean:", load_series.mean())

print("\n--- PV ---")
print("min:", pv_series.min())
print("max:", pv_series.max())
print("mean:", pv_series.mean())

print("\n--- COMPARACIÓN CON LOS PARÁMETROS DEL ENTORNO ---")
print("net_load_min del entorno:", -40.64)
print("net_load_max del entorno:", 62.45)
print("price_min del entorno:", 0.02)
print("price_max del entorno:", 0.425)

obs, info = env.reset()

print(obs)
print(env.observation_space.contains(obs))

print("=== TEST EXTRA: consistencia de reset ===")

for i in range(5):
    obs, info = env.reset()
    print(f"Reset {i+1}: soc={info['soc']}, obs={obs}")

obs, info = env.reset()


print("=== TEST 1: estructura básica del entorno ===")

print("action_space:", env.action_space)
print("observation_space:", env.observation_space)
print("obs shape:", obs.shape)
print("obs dtype:", obs.dtype)
print("obs:", obs)
print("info:", info)

print("\n¿obs pertenece a observation_space?:", env.observation_space.contains(obs))


print("=== TEST 2: longitud temporal ===")
print("len(mg):", len(mg))
print("len(load_series):", len(load_series))
print("len(pv_series):", len(pv_series))
print("len(precios_kwh):", len(precios_kwh))

print("=== TEST 3: estado físico inicial ===")

obs, info = env.reset()

print("current_step:", env.current_step)
print("load:", env._get_current_load())
print("pv:", env._get_current_pv())
print("net_load:", env._get_current_load() - env._get_current_pv())
print("soc:", env._get_current_soc())
print("import_price:", env._get_current_import_price())
print("export_price:", env._get_current_export_price())
print("obs:", obs)

print("=== TEST 4: mapeo interno de acciones ===")

test_actions = [
    np.array([-1.0, -1.0], dtype=np.float32),
    np.array([-1.0,  0.0], dtype=np.float32),
    np.array([-1.0,  1.0], dtype=np.float32),
    np.array([ 0.0, -1.0], dtype=np.float32),
    np.array([ 0.0,  0.0], dtype=np.float32),
    np.array([ 0.0,  1.0], dtype=np.float32),
    np.array([ 1.0, -1.0], dtype=np.float32),
    np.array([ 1.0,  0.0], dtype=np.float32),
    np.array([ 1.0,  1.0], dtype=np.float32),
]

for a in test_actions:
    print("agent action:", a, "-> control_dict:", env._get_control_dict(a))



print("=== TEST 5: un step aislado ===")

obs, info = env.reset()
action = np.array([0.0, 0.0], dtype=np.float32)   # neutra en espacio simétrico

next_obs, reward, terminated, truncated, step_info = env.step(action)

print("action:", action)
print("mapped control:", step_info["control_dict"])
print("reward:", reward)
print("terminated:", terminated)
print("truncated:", truncated)
print("next_obs:", next_obs)
print("next_obs shape:", next_obs.shape)
print("info keys:", list(step_info.keys()))
print("soc_before:", step_info["soc_before"])
print("soc_after:", step_info["soc_after"])
print("cost:", step_info["cost"])
print("mg_reward:", step_info["mg_reward"])

print("=== TEST 6: varios steps aleatorios ===")

obs, info = env.reset()
print("obs inicial válida:", env.observation_space.contains(obs))

for t in range(10):
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)

    valid_obs = env.observation_space.contains(next_obs)

    print(f"\nStep {t+1}")
    print("action:", action)
    print("reward:", reward)
    print("terminated:", terminated, "| truncated:", truncated)
    print("obs válida:", valid_obs)
    print("soc_before:", info["soc_before"], "| soc_after:", info["soc_after"])
    print("control_dict:", info["control_dict"])

    if terminated or truncated:
        print("Fin de episodio detectado")
        break

print("=== TEST 7: semántica de la batería ===")

battery_test_actions = [-1.0, 0.0, 1.0]   # en espacio simétrico
grid_neutral = 0.0                         # se mapeará a 0.5

for a_batt in battery_test_actions:
    obs, info = env.reset()

    soc_before = env._get_current_soc()
    action = np.array([a_batt, grid_neutral], dtype=np.float32)

    next_obs, reward, terminated, truncated, step_info = env.step(action)

    print("\n----------------------------------")
    print("agent action:", action)
    print("mapped control:", step_info["control_dict"])
    print("soc_before:", soc_before)
    print("soc_after :", step_info["soc_after"])
    print("reward:", reward)
    print("cost:", step_info["cost"])

print("=== TEST 8: semántica de la red ===")

grid_test_actions = [-1.0, 0.0, 1.0]   # espacio simétrico
battery_neutral = 0.0                  # se mapeará a 0.5

for a_grid in grid_test_actions:
    obs, info = env.reset()

    action = np.array([battery_neutral, a_grid], dtype=np.float32)

    next_obs, reward, terminated, truncated, step_info = env.step(action)

    print("\n----------------------------------")
    print("agent action:", action)
    print("mapped control:", step_info["control_dict"])
    print("reward:", reward)
    print("cost:", step_info["cost"])
    print("soc_before:", step_info["soc_before"])
    print("soc_after :", step_info["soc_after"])
    print("mg_info:", step_info["mg_info"])

print("=== TEST 9: rejilla 3x3 de acciones extremas ===")

candidate_vals = [-1.0, 0.0, 1.0]
results = []

for a_batt in candidate_vals:
    for a_grid in candidate_vals:
        obs, info = env.reset()

        action = np.array([a_batt, a_grid], dtype=np.float32)
        next_obs, reward, terminated, truncated, step_info = env.step(action)

        results.append({
            "agent_battery": a_batt,
            "agent_grid": a_grid,
            "mapped_battery": step_info["control_dict"]["battery"][0],
            "mapped_grid": step_info["control_dict"]["grid"][0],
            "reward": reward,
            "cost": step_info["cost"],
            "soc_before": step_info["soc_before"],
            "soc_after": step_info["soc_after"],
            "terminated": terminated,
            "truncated": truncated,
        })

for row in results:
    print(row)


print("=== TEST 10: consistencia temporal de la observación ===")

obs, info = env.reset()
print("step 0 obs:", obs)

for t in range(5):
    action = np.array([0.0, 0.0], dtype=np.float32)  # acción neutra
    obs, reward, terminated, truncated, info = env.step(action)

    print(f"\nstep {t+1}")
    print("current_step:", env.current_step)
    print("obs:", obs)
    print("hour_sin, hour_cos:", obs[3], obs[4])
    print("day_sin, day_cos :", obs[5], obs[6])

    if terminated or truncated:
        break

print("=== TEST 11: truncación por horizonte ===")

mg_short = build_microgrid(precios_kwh, load_series, pv_series)

env_short = CustomEnvContinuous(
    pymgrid_network=mg_short,
    horizon=3,
    reward_scale_C=1.0,
    low_soc_penalty=0.0,
    low_soc_threshold=0.20,
    net_load_min=-40.64,
    net_load_max=62.45,
    price_min=0.02,
    price_max=0.425,
)

obs, info = env_short.reset()

for t in range(5):
    action = np.array([0.0, 0.0], dtype=np.float32)
    obs, reward, terminated, truncated, step_info = env_short.step(action)

    print(f"\nStep {t+1}")
    print("terminated:", terminated)
    print("truncated:", truncated)
    print("obs:", obs)
    print("obs válida:", env_short.observation_space.contains(obs))
    print("terminal_observation en info?:", "terminal_observation" in step_info)

    if terminated or truncated:
        print("Fin detectado")
        print("terminal_observation:", step_info.get("terminal_observation", None))
        break

print("=== TEST 12: reset tras final de episodio ===")

obs, info = env_short.reset()

done = False
while not done:
    action = np.array([0.0, 0.0], dtype=np.float32)
    obs, reward, terminated, truncated, step_info = env_short.step(action)
    done = terminated or truncated

print("Episodio terminado. current_step:", env_short.current_step)

obs_reset, info_reset = env_short.reset()

print("Tras reset, current_step:", env_short.current_step)
print("obs_reset:", obs_reset)
print("obs_reset válida:", env_short.observation_space.contains(obs_reset))
print("info_reset:", info_reset)

print("=== TEST 13: inspección de penalización low SoC ===")

mg_pen = build_microgrid(precios_kwh, load_series, pv_series)

env_pen = CustomEnvContinuous(
    pymgrid_network=mg_pen,
    horizon=20,
    reward_scale_C=1.0,
    low_soc_penalty=0.2,
    low_soc_threshold=0.20,
    net_load_min=-40.64,
    net_load_max=62.45,
    price_min=0.02,
    price_max=0.425,
)

obs, info = env_pen.reset()

for t in range(20):
    action = np.array([1.0, 0.0], dtype=np.float32)

    obs, reward, terminated, truncated, step_info = env_pen.step(action)

    print(f"\nStep {t+1}")
    print("soc_after:", step_info["soc_after"])
    print("low_soc_penalty_applied:", step_info["low_soc_penalty_applied"])
    print("reward:", reward)

    if terminated or truncated:
        break

from stable_baselines3.common.env_checker import check_env

print("=== TEST 14: check_env ===")
mg_check = build_microgrid(precios_kwh, load_series, pv_series)

env_check = CustomEnvContinuous(
    pymgrid_network=mg_check,
    horizon=24 * 365,
    reward_scale_C=1.0,
    low_soc_penalty=0.0,
    low_soc_threshold=0.20,
    net_load_min=-40.64,
    net_load_max=62.45,
    price_min=0.02,
    price_max=0.425,
)

check_env(env_check, warn=True)
print("check_env superado")