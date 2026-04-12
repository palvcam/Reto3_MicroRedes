import pytest
import pandas as pd
import sys
import os
import numpy as np

# Ruta absoluta al directorio raíz del proyecto
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Añadir al path
sys.path.append(ROOT)

from federated_docker.client.client import apply_feature_engineering, PVClient, FEATURES


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

@pytest.fixture
def dummy_data_env(tmp_path, monkeypatch):
    """
    Fixture que crea archivos CSV sintéticos en una carpeta temporal 
    y redirige el DATA_PATH del cliente hacia ella.

    IMPORTANTE:
    - Los datos NO son aleatorios puros
    - Se construye una relación física realista entre inputs y target (Pmp)
    - Esto permite que el modelo realmente aprenda durante los tests
    """
    import federated_docker.client.client as client_module

    # Redirigir la ruta de datos
    monkeypatch.setattr(client_module, "DATA_PATH", str(tmp_path))
    parque_name = "testpark"

    np.random.seed(42)
    n_rows = 100

    # Variables físicas base
    poa = np.random.uniform(200, 1000, n_rows)
    temp_panel = np.random.uniform(15, 60, n_rows)
    ghi = np.random.uniform(200, 1000, n_rows)
    dni = np.random.uniform(100, 800, n_rows)
    dhi = np.random.uniform(50, 300, n_rows)
    temp_air = np.random.uniform(10, 40, n_rows)

    # ⚡ Generación de target realista (aprendible)
    # Modelo físico simplificado de panel FV
    pmp = 0.18 * poa * (1 - 0.004 * (temp_panel - 25)) \
          + np.random.normal(0, 5, n_rows)

    # Introducir valores erróneos para testear limpieza
    pmp[5] = -9999
    pmp[15] = -9999

    data = {
        "Time Stamp (local standard time) yyyy-mm-ddThh:mm:ss": ["2023-01-01T12:00:00"] * n_rows,
        "POA irradiance CMP22 pyranometer (W/m2)": poa,
        "PV module back surface temperature (degC)": temp_panel,
        "Global horizontal irradiance (W/m2)": ghi,
        "Direct normal irradiance (W/m2)": dni,
        "Diffuse horizontal irradiance (W/m2)": dhi,
        "Dry bulb temperature (degC)": temp_air,
        "Pmp (W)": pmp
    }

    df = pd.DataFrame(data)

    # Crear dos CSVs (simulando dos paneles)
    df.to_csv(tmp_path / f"{parque_name}_panel_1.csv", index=False)
    df.to_csv(tmp_path / f"{parque_name}_panel_2.csv", index=False)

    return parque_name

# ==========================================
# TESTS MATEMÁTICOS (Unit Tests)
# ==========================================

def test_apply_feature_engineering_math(dummy_dataframe):
    """
    Test para verificar que las fórmulas físicas no fallan y 
    calculan correctamente las variables derivadas con la nueva lógica.
    """
    # Arrange: Diccionarios mockeados para k_panel y factor_panel
    mock_k_por_panel = {"panel_1": 0.005}
    mock_k_global = 0.004
    
    mock_factor_por_panel = {"panel_1": 0.15} # 15% de eficiencia media
    mock_factor_global = 0.16 # 16% de eficiencia media global

    # Act: Aplicamos la función con todos los argumentos requeridos
    result_df = apply_feature_engineering(
        dummy_dataframe, 
        mock_k_por_panel, 
        mock_k_global, 
        mock_factor_por_panel, 
        mock_factor_global
    )

    # Assert: Comprobamos T_diff (Temperatura del panel - 25)
    # Fila 0: 35 - 25 = 10
    # Fila 1: 25 - 25 = 0
    assert result_df.loc[0, "T_diff"] == 10.0, "Fallo en el cálculo de T_diff"
    assert result_df.loc[1, "T_diff"] == 0.0, "Fallo en el cálculo de T_diff"

    # Assert: Comprobamos la asignación de mapas y fallbacks (el panel_2 debe coger los globales)
    assert result_df.loc[0, "k_panel"] == 0.005, "No se asignó bien el k_panel específico"
    assert result_df.loc[1, "k_panel"] == 0.004, "No se asignó bien el k_global (fallback)"
    assert result_df.loc[0, "factor_panel"] == 0.15, "No se asignó bien el factor_panel específico"
    assert result_df.loc[1, "factor_panel"] == 0.16, "No se asignó bien el factor_global (fallback)"

    # Assert: Comprobamos T_correccion: 1 - k * (T - 25)
    # Fila 0: 1 - 0.005 * 10 = 0.95
    # Fila 1: 1 - 0.004 * 0 = 1.0
    np.testing.assert_almost_equal(result_df.loc[0, "T_correccion"], 0.95)
    np.testing.assert_almost_equal(result_df.loc[1, "T_correccion"], 1.0)

    # Assert: Comprobamos el nuevo physical_model: G * factor * T_correccion
    # Fila 0: 800.0 * 0.15 * 0.95 = 114.0
    # Fila 1: 1000.0 * 0.16 * 1.0 = 160.0
    np.testing.assert_almost_equal(result_df.loc[0, "physical_model"], 114.0)
    np.testing.assert_almost_equal(result_df.loc[1, "physical_model"], 160.0)

    # Assert: Comprobamos el cloud_index (Diffuse / Direct)
    # Fila 0: 300 / 700 = 0.4285...
    np.testing.assert_almost_equal(result_df.loc[0, "cloud_index"], 300.0 / 700.0, decimal=5)

    # NUEVO: Comprobacion de que no aparecen NaNs y que las columnas esperadas existen
    expected_cols = ["T_diff", "k_panel", "factor_panel", "T_correccion", "physical_model", "cloud_index"]
    for col in expected_cols:
        assert col in result_df.columns
    assert not result_df[expected_cols].isna().any().any()

# ==========================================
# TESTS DE INTEGRACIÓN DEL PIPELINE
# ==========================================

def test_client_data_pipeline(dummy_data_env):
    """
    Verifica: Carga de CSV, limpieza de datos, split, features y DataLoaders.
    """
    client = PVClient(parque=dummy_data_env)
    
    # 1. Comprobar que los splits tienen datos (se cargaron correctamente)
    assert len(client.train_loader.dataset) > 0, "Train loader vacío"
    assert len(client.val_loader.dataset) > 0, "Val loader vacío"
    assert len(client.test_loader.dataset) > 0, "Test loader vacío"
    
    # 2. Comprobar que la limpieza de datos eliminó los -9999
    # Había 200 filas totales (100 por panel). 4 tenían -9999. Quedan 196.
    total_samples = (len(client.train_loader.dataset) + 
                     len(client.val_loader.dataset) + 
                     len(client.test_loader.dataset))
    assert total_samples == 196, f"La limpieza falló, muestras totales: {total_samples}"

    # 3. Comprobar el escalado (StandardScaler)
    # Extraemos el primer batch de entrenamiento
    X_batch, y_batch = next(iter(client.train_loader))
    
    # Comprobar dimensiones (batch_size o restante, num_features)
    assert X_batch.shape[1] == len(FEATURES), "El número de features no coincide con el vector de entrada"
    
    # En datos escalados, la media debería estar cerca de 0 y std cerca de 1 (a nivel de todo el dataset, en batch es aproximado)
    mean_val = X_batch.mean().item()
    std_val = X_batch.std().item()
    assert -0.5 < mean_val < 0.5, "Los datos parecen no estar centrados en 0"
    assert 0.5 < std_val < 1.5, "La desviación estándar parece no estar en 1"

    X_train_all = client.train_loader.dataset.tensors[0].numpy()

    assert X_train_all.shape[1] == len(FEATURES), "El número de features no coincide con el vector de entrada"

    mean_val = X_train_all.mean()
    std_val = X_train_all.std()

    assert -0.2 < mean_val < 0.2, "Los datos de train no parecen estar centrados en 0"
    assert 0.8 < std_val < 1.2, "La desviación estándar global no parece cercana a 1"

def test_client_get_set_parameters(dummy_data_env):
    """
    Verifica la correcta serialización y deserialización de los parámetros del modelo.

    Se comprueba que:
    1. get_parameters devuelve una lista de numpy arrays
    2. set_parameters actualiza correctamente TODOS los pesos del modelo
    3. Los nuevos pesos coinciden exactamente con los inyectados
    """

    client = PVClient(parque=dummy_data_env)

    # 1. Extraer parámetros originales
    original_params = client.get_parameters({})

    assert isinstance(original_params, list), \
        "get_parameters debe devolver una lista de arrays"

    assert isinstance(original_params[0], np.ndarray), \
        "Los parámetros internos deben ser numpy arrays"

    # 2. Crear parámetros dummy robustos (todos unos)
    # Esto evita problemas con pesos iniciales en 0 (bias)
    dummy_params = [np.ones_like(p) for p in original_params]

    # 3. Inyectar nuevos parámetros
    client.set_parameters(dummy_params)

    # 4. Recuperar parámetros tras la actualización
    new_params = client.get_parameters({})

    # 5. Comprobaciones
    for i, (p_new, p_expected) in enumerate(zip(new_params, dummy_params)):

        # Verificar que la forma no cambia
        assert p_new.shape == p_expected.shape, \
            f"Shape incorrecto en parámetro {i}: esperado {p_expected.shape}, obtenido {p_new.shape}"

        # Verificar igualdad exacta
        np.testing.assert_array_almost_equal(
            p_new,
            p_expected,
            err_msg=f"set_parameters falló en el parámetro {i}: los valores no coinciden"
        )

    # 6. Comprobación extra: los parámetros realmente cambiaron
    changed = any(not np.allclose(p0, p1) for p0, p1 in zip(original_params, new_params))

    assert changed, \
        "set_parameters no produjo ningún cambio en los pesos del modelo"

def test_client_fit(dummy_data_env):
    """
    Verifica el bucle de entrenamiento local (1 época de prueba).
    """
    client = PVClient(parque=dummy_data_env)
    initial_params = client.get_parameters({})
    
    # Llamamos al fit pidiendo solo 1 época y sin penalización proxy (mu=0)
    config = {"epochs": 1, "mu": 0.0}
    updated_params, num_samples, metrics = client.fit(initial_params, config)
    
    assert len(updated_params) == len(initial_params), "La estructura de pesos cambió tras el fit"
    assert num_samples > 0, "No se devolvió el número de muestras usadas"
    
    # Comprobar que se devolvió el diccionario de métricas
    expected_metrics = ["val_mse", "val_r2", "real_val_rmse", "real_test_mse"]
    for m in expected_metrics:
        assert m in metrics, f"Falta la métrica {m} en la respuesta de fit()"
    #NUEVO: 
    #Verificamos cambio real (actualizacion) de los pesos
    changed = any(not np.allclose(p0, p1) for p0, p1 in zip(initial_params, updated_params))
    assert changed, "fit() no actualizó los pesos del modelo"
    #Verificamos que las metricas sean finitas
    for key, value in metrics.items():
        assert np.isfinite(value), f"La métrica {key} no es finita"

def test_client_evaluate(dummy_data_env):
    """
    Verifica el ciclo de evaluación del cliente y el formato de salida de métricas.
    """
    client = PVClient(parque=dummy_data_env)
    current_params = client.get_parameters({})
    
    loss, num_samples, metrics = client.evaluate(current_params, {})
    
    assert isinstance(loss, float), "El loss debe ser un float"
    assert num_samples == len(client.val_loader.dataset), "El num_samples devuelto en evaluate debe ser el de validación"
    
    expected_metrics = ["val_rmse", "test_mse", "test_r2"]
    for m in expected_metrics:
        assert m in metrics, f"Falta la métrica {m} en la respuesta de evaluate()"
    #NUEVO:
    #Verificamos que la perdida sea finita
    assert np.isfinite(loss), "La loss no es finita"
    #verificamos que metricas son finitas
    for key, value in metrics.items():
        assert np.isfinite(value), f"La métrica {key} no es finita"
    #Verificamos que mse mayor o igual que cero 
    assert metrics["test_mse"] >= 0