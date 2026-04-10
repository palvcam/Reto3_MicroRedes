import pytest
import sys
import os

# Apuntar a la carpeta superior
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from FINAL.CON_FEATURED.FedProxFedEx.server import fedex_aggregate_metrics, history

def test_fedex_aggregate_metrics():
    """Prueba que el servidor calcula bien la media ponderada de las métricas."""
    # Arrange: Limpiamos el historial por si otras pruebas lo han ensuciado
    history["round"].clear()
    history["mse_val"].clear()
    history["rmse_val"].clear()
    history["r2_val"].clear()

    # Simulamos 3 clientes con diferentes pesos (número de muestras)
    # Cliente 1: 100 muestras, MSE = 10
    # Cliente 2: 200 muestras, MSE = 20
    # Cliente 3: 100 muestras, MSE = 30
    # Cálculo mental esperado para MSE: (100*10 + 200*20 + 100*30) / 400 = 8000 / 400 = 20.0
    metricas_simuladas = [
        (100, {"val_mse": 10.0, "val_rmse": 3.16, "val_r2": 0.8}),
        (200, {"val_mse": 20.0, "val_rmse": 4.47, "val_r2": 0.7}),
        (100, {"val_mse": 30.0, "val_rmse": 5.47, "val_r2": 0.6})
    ]

    # Act: Ejecutamos la función de agregación
    resultado = fedex_aggregate_metrics(metricas_simuladas)

    # Assert: Comprobamos resultados matemáticos
    assert resultado["mse_val"] == 20.0, "Error en el cálculo ponderado del MSE"
    # Cálculo para R2: (100*0.8 + 200*0.7 + 100*0.6)/400 = (80+140+60)/400 = 280/400 = 0.7
    np_test = pytest.importorskip("numpy")
    np_test.testing.assert_almost_equal(resultado["r2_val"], 0.7)

    # Assert: Comprobamos que el historial se ha actualizado
    assert len(history["round"]) == 1, "El historial de rondas no se ha actualizado"
    assert history["mse_val"][0] == 20.0, "El valor guardado en el historial es incorrecto"