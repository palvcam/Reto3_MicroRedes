import pytest
import pandas as pd
import sys
import os
import numpy as np
# Ruta absoluta al directorio raíz del proyecto
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Añadir al path
sys.path.append(ROOT)

from FINAL.CON_FEATURED.FedProxFedEx.client_ch_testing import apply_feature_engineering


@pytest.fixture
def dummy_dataframe():
    """
    Fixture de Pytest que crea un DataFrame de prueba con 2 filas.
    Usamos valores exactos para poder calcular mentalmente el resultado esperado.
    """
    data = {
        "panel_id": ["panel_1", "panel_2"],
        "POA irradiance CMP22 pyranometer (W/m2)": [800.0, 1000.0],
        "PV module back surface temperature (degC)": [35.0, 25.0], # T_diff será 10 y 0
        "Global horizontal irradiance (W/m2)": [1000.0, 1000.0],
        "Direct normal irradiance (W/m2)": [700.0, 800.0],
        "Diffuse horizontal irradiance (W/m2)": [300.0, 200.0],
        "Dry bulb temperature (degC)": [30.0, 20.0],
    }
    return pd.DataFrame(data)

def test_apply_feature_engineering_math(dummy_dataframe):
    """
    Test para verificar que las fórmulas físicas no fallan y 
    calculan correctamente las variables derivadas.
    """
    # Arrange: Diccionarios mockeados para k_panel
    mock_k_por_panel = {"panel_1": 0.005}
    mock_k_global = 0.004

    # Act: Aplicamos la función
    result_df = apply_feature_engineering(dummy_dataframe, mock_k_por_panel, mock_k_global)

    # Assert: Comprobamos T_diff (Temperatura del panel - 25)
    # Fila 0: 35 - 25 = 10
    # Fila 1: 25 - 25 = 0
    assert result_df.loc[0, "T_diff"] == 10.0, "Fallo en el cálculo de T_diff"
    assert result_df.loc[1, "T_diff"] == 0.0, "Fallo en el cálculo de T_diff"

    # Assert: Comprobamos la asignación de k_panel (el panel_2 debe coger el global)
    assert result_df.loc[0, "k_panel"] == 0.005, "No se asignó bien el k_panel específico"
    assert result_df.loc[1, "k_panel"] == 0.004, "No se asignó bien el k_global (fallback)"

    # Assert: Comprobamos el modelo físico: 1 - k * (T - 25)
    # Fila 0: 1 - 0.005 * 10 = 0.95
    # Fila 1: 1 - 0.004 * 0 = 1.0
    np.testing.assert_almost_equal(result_df.loc[0, "physical_model"], 0.95)
    np.testing.assert_almost_equal(result_df.loc[1, "physical_model"], 1.0)

    # Assert: Comprobamos el cloud_index (Diffuse / Direct)
    # Fila 0: 300 / 700 = 0.4285...
    np.testing.assert_almost_equal(result_df.loc[0, "cloud_index"], 300.0 / 700.0, decimal=5)