import pandas as pd
from pathlib import Path

def convertir_datos_fotovoltaicos(p_nominal_kw=100.0):
    # 1. Localizar la ruta del script actual (OBJETIVO1/src/utils)
    ruta_script = Path(__file__).resolve().parent
    
    # 2. Definir la raíz del proyecto (subir 2 niveles desde utils -> src -> OBJETIVO1)
    raiz_proyecto = ruta_script.parent.parent

    # 3. Definir rutas relativas a la raíz
    ruta_raw = raiz_proyecto / "OBJETIVO1" / "data" / "raw" / "pv"
    ruta_processed = raiz_proyecto / "OBJETIVO1" /  "data" / "processed"
    
    # Crear carpeta processed si no existe
    ruta_processed.mkdir(parents=True, exist_ok=True)

    # 4. Definir archivos
    archivo_entrada = ruta_raw / "SanFrancisco_724940TYA.csv"
    archivo_salida = ruta_processed / "pv_generacion_corregida_kw.csv"


    if not archivo_entrada.exists():
        print(f"Error: No se encuentra el archivo en {archivo_entrada}")
        return

    print(f"Leyendo datos de: {archivo_entrada.name}...")
    
    # 1. Cargar el CSV
    df = pd.read_csv(archivo_entrada)

    # 2. Renombrar columna para evitar confusión (Originalmente dice 'lx' pero es 'W/m2')
    # Usamos iloc para seleccionar la primera columna sin importar el nombre exacto
    df.columns = ['irradiancia_wm2']

    # 3. Aplicar modelo físico corregido
    # P (kW) = P_nominal * (G / 1000)
    # No dividimos por 120 porque los datos ya son irradiancia.
    df['pv_kw'] = p_nominal_kw * (df['irradiancia_wm2'] / 1000.0)

    # Opcional: Limitar la potencia al máximo nominal (clipping)
    df['pv_kw'] = df['pv_kw'].clip(upper=p_nominal_kw)

    # 4. Guardar el nuevo CSV
    df.to_csv(archivo_salida, index=False)
    print(f"Conversión finalizada. Archivo guardado en: {archivo_salida}")

if __name__ == "__main__":
    # Puedes ajustar aquí la potencia de tu instalación en kW
    POTENCIA_INSTALACION = 75.0 
    convertir_datos_fotovoltaicos(POTENCIA_INSTALACION)