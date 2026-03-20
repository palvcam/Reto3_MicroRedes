import pandas as pd
import numpy as np
from pathlib import Path

def convert_lux_to_pv_kw(
    input_csv,
    output_csv,
    lux_col="GH illum (lx)",
    lux_per_wm2=120.0,
    pv_rated_kw=100.0,
    clip_to_rated=True
):
    """
    Convierte una serie de iluminancia (lux) a:
      1) irradiancia estimada (W/m²)
      2) potencia fotovoltaica estimada (kW)

    Suposiciones:
    - 120 lux ≈ 1 W/m² para luz solar exterior
    - STC FV: 1000 W/m² corresponde a la potencia pico del generador
    """

    df = pd.read_csv(input_csv)

    if lux_col not in df.columns:
        raise ValueError(f"No se encontró la columna '{lux_col}' en el CSV.")

    lux = df[lux_col].astype(float).to_numpy()

    # 1) lux -> irradiancia
    irradiance_wm2 = lux / lux_per_wm2
    irradiance_wm2 = np.maximum(irradiance_wm2, 0.0)

    # 2) irradiancia -> potencia FV
    pv_kw = pv_rated_kw * (irradiance_wm2 / 1000.0)

    if clip_to_rated:
        pv_kw = np.clip(pv_kw, 0.0, pv_rated_kw)

    out = pd.DataFrame({
        "GH_illum_lx": lux,
        "irradiance_wm2": irradiance_wm2,
        "pv_kw": pv_kw
    })

    out.to_csv(output_csv, index=False)
    return out


if __name__ == "__main__":

    # 1. Directorio del archivo actual (src/utils)
    CURRENT_DIR = Path(__file__).resolve().parent

    # 2. Directorio raíz del proyecto (subimos dos niveles)
    PROJECT_ROOT = CURRENT_DIR.parent.parent

    # 3. Construimos rutas a los datos
    RAW_FILE = PROJECT_ROOT / "OBJETIVO1" / "data" / "raw" / "pv" / "SanFrancisco_724940TYA.csv"
    PROCESSED_FILE = PROJECT_ROOT / "OBJETIVO1" / "data" / "processed" / "pv_kw_aprox_timeseries.csv"

    print("Leyendo:", RAW_FILE)
    print("Guardando:", PROCESSED_FILE)


    out = convert_lux_to_pv_kw(
        input_csv=RAW_FILE,
        output_csv=PROCESSED_FILE,
        lux_col="GH illum (lx)",
        lux_per_wm2=120.0,
        pv_rated_kw=100.0
    )

    print(out.head())
    print(out.describe())