import pytest
import sys
import os
from unittest.mock import MagicMock
import numpy as np

# Apuntar a la carpeta superior
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from federated_docker.server.server import fedex_aggregate_metrics, history, FlowerFedExServer


def test_fedex_aggregate_metrics():
    """Prueba que el servidor calcula bien la media ponderada de las métricas."""
    # Arrange: Limpiamos el historial por si otras pruebas lo han ensuciado
    for key in history:
        history[key].clear()

    # Añadidas las claves 'test_*' y 'real_*' necesarias para el nuevo script del servidor
    metricas_simuladas = [
        (100, {"val_mse": 10.0, "val_rmse": 3.16, "val_r2": 0.8, 
               "test_mse": 15.0, "test_rmse": 3.87, "test_r2": 0.7,
               "real_val_mse": 100.0, "real_val_rmse": 10.0, "real_test_mse": 150.0, "real_test_rmse": 12.2}),
        (200, {"val_mse": 20.0, "val_rmse": 4.47, "val_r2": 0.7,
               "test_mse": 25.0, "test_rmse": 5.0, "test_r2": 0.6,
               "real_val_mse": 200.0, "real_val_rmse": 14.1, "real_test_mse": 250.0, "real_test_rmse": 15.8}),
        (100, {"val_mse": 30.0, "val_rmse": 5.47, "val_r2": 0.6,
               "test_mse": 35.0, "test_rmse": 5.9, "test_r2": 0.5,
               "real_val_mse": 300.0, "real_val_rmse": 17.3, "real_test_mse": 350.0, "real_test_rmse": 18.7})
    ]

    # Act: Ejecutamos la función de agregación
    resultado = fedex_aggregate_metrics(metricas_simuladas)

    # Assert: Comprobamos resultados matemáticos (100*10 + 200*20 + 100*30) / 400 = 20.0
    assert resultado["mse_val"] == 20.0, "Error en el cálculo ponderado del MSE"
    np.testing.assert_almost_equal(resultado["r2_val"], 0.7)
    
    # Comprobar historial (se guardan las métricas reales/sin escalar)
    # (100*100 + 200*200 + 100*300) / 400 = 200.0
    assert history["mse_val"][0] == 200.0, "El valor sin escalar guardado en historial es incorrecto"


# ==========================================
# TESTS DE LÓGICA DEL SERVIDOR (MOCKS)
# ==========================================

def test_parse_fit_results():
    """
    Verifica que el servidor extrae correctamente las métricas 
    necesarias para el optimizador FedEx.
    """
    # Arrange: Mockeamos los resultados de dos clientes
    res1 = MagicMock()
    res1.metrics = {"val_mse_before": 15.0, "val_mse": 10.0, "val_samples": 100}
    
    res2 = MagicMock()
    res2.metrics = {"val_mse_before": 18.0, "val_mse": 12.0, "val_samples": 200}
    
    # Act
    before, after, weights = FlowerFedExServer._parse_fit_results([res1, res2])
    
    # Assert
    np.testing.assert_array_equal(before, [15.0, 18.0])
    np.testing.assert_array_equal(after, [10.0, 12.0])
    np.testing.assert_array_equal(weights, [100, 200])

def test_communication_round():
    """
    Verifica la orquestación completa de una ronda de entrenamiento.
    Simulamos el client_manager y los clientes para probar solo la lógica del servidor.
    """
    # Arrange: Mocks
    mock_manager = MagicMock()
    mock_client = MagicMock()
    # Simulamos que hay 3 clientes disponibles
    mock_manager.sample.return_value = [mock_client, mock_client, mock_client]

    # Mock de la respuesta get_parameters (ronda 1)
    mock_init_res = MagicMock()
    mock_init_res.parameters = "pesos_globales_iniciales"
    mock_client.get_parameters.return_value = mock_init_res

    # Mock de la respuesta de entrenamiento (fit)
    mock_fit_res = MagicMock()
    mock_fit_res.num_examples = 100
    mock_fit_res.metrics = {
        "val_mse_before": 15.0, "val_mse": 10.0, "val_samples": 100,
        "val_rmse": 3.1, "val_r2": 0.8,
        "test_mse": 12.0, "test_rmse": 3.4, "test_r2": 0.7,
        "real_val_mse": 100.0, "real_val_rmse": 10.0,
        "real_test_mse": 120.0, "real_test_rmse": 11.0
    }
    mock_client.fit.return_value = mock_fit_res

    # Instanciar el servidor
    server = FlowerFedExServer(client_manager=mock_manager)
    # Mockear la estrategia (FedAvg) para evitar la matemática real de agregación de tensores
    server.strategy = MagicMock()
    server.strategy.aggregate_fit.return_value = ("pesos_agregados", {})

    # Función ficticia para simular FedEx devolviendo una config
    def dummy_get_config():
        return {"epochs": 5, "mu": 0.1}

    # Act
    before, after, weights = server.communication_round(dummy_get_config)

    # Assert
    assert server._round == 1, "El contador de rondas no se incrementó"
    assert mock_client.get_parameters.call_count == 1, "Debió pedir pesos iniciales solo al primer cliente"
    assert mock_client.fit.call_count == 3, "Debió llamar al fit de los 3 clientes"
    assert server.global_parameters == "pesos_agregados", "No se actualizaron los pesos tras la agregación"
    
    # Comprobar la salida a FedEx
    np.testing.assert_array_equal(before, [15.0, 15.0, 15.0])
    np.testing.assert_array_equal(after, [10.0, 10.0, 10.0])

def test_full_evaluation():
    """
    Verifica que en la evaluación final el servidor llama a evaluate()
    en todos los clientes con la mejor configuración.
    """
    # Arrange
    mock_manager = MagicMock()
    mock_client = MagicMock()
    mock_manager.sample.return_value = [mock_client, mock_client, mock_client]

    mock_eval_res = MagicMock()
    mock_eval_res.num_examples = 150
    # Simulamos que la métrica de test es 8.5
    mock_eval_res.metrics = {"val_mse": 9.0, "test_mse": 8.5}
    mock_client.evaluate.return_value = mock_eval_res

    server = FlowerFedExServer(client_manager=mock_manager)
    server.global_parameters = "pesos_finales_entrenados"

    def dummy_get_best_config():
        return {"epochs": 8, "mu": 0.0}

    # Act
    before, after, weights = server.full_evaluation(dummy_get_best_config)

    # Assert
    assert mock_client.evaluate.call_count == 3, "Debió evaluar a los 3 clientes"
    np.testing.assert_array_equal(before, [9.0, 9.0, 9.0])
    np.testing.assert_array_equal(after, [8.5, 8.5, 8.5])
    np.testing.assert_array_equal(weights, [150.0, 150.0, 150.0])