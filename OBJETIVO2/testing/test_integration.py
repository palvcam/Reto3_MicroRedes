import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from flwr.common import (
    EvaluateIns,
    FitIns,
    GetParametersIns,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

# Ajusta esta ruta si tu estructura cambia
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from federated_docker.client.client import PVClient
import federated_docker.client.client as client_module
import federated_docker.server.server as server_module


# =========================================================
# FIXTURE DE DATOS REALES PEQUEÑOS PARA PROBAR CLIENTE+MODELO
# =========================================================

@pytest.fixture
def dummy_data_env(tmp_path, monkeypatch):
    """
    Crea dos CSVs sintéticos para que PVClient pueda ejecutarse de extremo a extremo.

    El target Pmp está construido con una relación física simple y aprendible:
    más irradiancia -> más potencia
    más temperatura -> ligera pérdida

    Además, se introducen dos -9999 por archivo para verificar la limpieza.
    """
    monkeypatch.setattr(client_module, "DATA_PATH", str(tmp_path))
    parque_name = "testpark"

    rng = np.random.default_rng(42)
    n_rows = 120

    poa = rng.uniform(200, 1000, n_rows)
    temp_panel = rng.uniform(15, 60, n_rows)
    ghi = rng.uniform(200, 1000, n_rows)
    dni = rng.uniform(100, 800, n_rows)
    dhi = rng.uniform(50, 300, n_rows)
    temp_air = rng.uniform(10, 40, n_rows)

    # Target aprendible
    pmp = 0.18 * poa * (1 - 0.004 * (temp_panel - 25)) + rng.normal(0, 4, n_rows)

    # Inyectar errores deliberados para probar limpieza
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
        "Pmp (W)": pmp,
    }

    df = pd.DataFrame(data)

    # Dos archivos para simular dos paneles del mismo parque
    df.to_csv(tmp_path / f"{parque_name}_panel_1.csv", index=False)
    df.to_csv(tmp_path / f"{parque_name}_panel_2.csv", index=False)

    return parque_name


# =========================================================
# HELPERS PARA TESTS DE SERVIDOR CON CLIENTES FAKE
# =========================================================

def make_parameters(fill_value: float, shapes=None):
    """
    Crea parámetros Flower falsos, pero consistentes, con las shapes indicadas.
    """
    if shapes is None:
        shapes = [(4, 3), (4,), (1, 4), (1,)]

    ndarrays = [np.full(shape, fill_value, dtype=np.float32) for shape in shapes]
    return ndarrays_to_parameters(ndarrays)


class FakeClient:
    """
    Cliente fake que imita la interfaz que FlowerFedExServer espera.

    - get_parameters devuelve pesos iniciales
    - fit devuelve pesos actualizados + métricas
    - evaluate devuelve métricas de validación/test
    """

    def __init__(
        self,
        client_id: str,
        init_value: float,
        fit_value: float,
        val_before: float,
        val_after: float,
        val_samples: int,
        test_mse: float,
    ):
        self.client_id = client_id
        self._init_parameters = make_parameters(init_value)
        self._fit_parameters = make_parameters(fit_value)

        self._val_before = float(val_before)
        self._val_after = float(val_after)
        self._val_samples = int(val_samples)
        self._test_mse = float(test_mse)

        self.get_parameters_called = 0
        self.fit_called = 0
        self.evaluate_called = 0
        self.last_fit_config = None
        self.last_eval_config = None

    def get_parameters(self, ins: GetParametersIns, timeout=None, group_id=None):
        self.get_parameters_called += 1
        return SimpleNamespace(parameters=self._init_parameters)

    def fit(self, ins: FitIns, timeout=None, group_id=None):
        self.fit_called += 1
        self.last_fit_config = ins.config

        return SimpleNamespace(
            parameters=self._fit_parameters,
            num_examples=50,
            metrics={
                "val_mse_before": self._val_before,
                "val_mse": self._val_after,
                "val_rmse": self._val_after ** 0.5,
                "val_r2": 0.8,
                "val_samples": float(self._val_samples),
                "test_mse": self._test_mse,
                "test_rmse": self._test_mse ** 0.5,
                "test_r2": 0.75,
                "real_val_mse": self._val_after * 100.0,
                "real_val_rmse": (self._val_after ** 0.5) * 10.0,
                "real_test_mse": self._test_mse * 100.0,
                "real_test_rmse": (self._test_mse ** 0.5) * 10.0,
            },
        )

    def evaluate(self, ins: EvaluateIns, timeout=None, group_id=None):
        self.evaluate_called += 1
        self.last_eval_config = ins.config

        return SimpleNamespace(
            loss=self._val_after,
            num_examples=self._val_samples,
            metrics={
                "val_mse": self._val_after,
                "val_rmse": self._val_after ** 0.5,
                "val_r2": 0.81,
                "test_mse": self._test_mse,
                "test_rmse": self._test_mse ** 0.5,
                "test_r2": 0.77,
            },
        )


class FakeClientManager:
    """
    Client manager mínimo para pruebas de integración del servidor.
    """

    def __init__(self, clients):
        self.clients = clients
        self.wait_for_called = 0
        self.sample_called = 0
        self.last_wait_for_args = None
        self.last_sample_args = None

    def wait_for(self, num_clients, timeout):
        self.wait_for_called += 1
        self.last_wait_for_args = {"num_clients": num_clients, "timeout": timeout}
        return True

    def sample(self, num_clients):
        self.sample_called += 1
        self.last_sample_args = {"num_clients": num_clients}
        return self.clients[:num_clients]


class FakeStrategy:
    """
    Estrategia fake que agrega parámetros haciendo media aritmética simple.
    """

    def __init__(self):
        self.aggregate_fit_called = 0

    def aggregate_fit(self, server_round, results, failures):
        self.aggregate_fit_called += 1

        fit_results = [r for _, r in results]
        all_ndarrays = [parameters_to_ndarrays(r.parameters) for r in fit_results]

        averaged = []
        for tensors_same_position in zip(*all_ndarrays):
            averaged.append(np.mean(np.stack(tensors_same_position, axis=0), axis=0))

        aggregated_parameters = ndarrays_to_parameters(averaged)
        aggregated_metrics = {}
        return aggregated_parameters, aggregated_metrics


# =========================================================
# TEST 1: INTEGRACIÓN CLIENTE + MODELO
# =========================================================

def test_client_model_end_to_end_train_then_evaluate(dummy_data_env):
    """
    Test de integración cliente + modelo.

    Verifica que:
    1. el cliente puede inicializarse con datos reales en CSV,
    2. puede entrenar localmente con fit(),
    3. puede evaluar con evaluate(),
    4. los parámetros actualizados producidos por fit() son utilizables por evaluate().
    """
    client = PVClient(parque=dummy_data_env)

    initial_params = client.get_parameters({})
    updated_params, num_samples, fit_metrics = client.fit(
        initial_params,
        {"epochs": 2, "mu": 0.0},
    )

    assert isinstance(updated_params, list), (
        "fit() debe devolver una lista de parámetros actualizados"
    )
    assert num_samples > 0, (
        "fit() debe devolver un número de muestras mayor que cero"
    )
    assert "val_mse" in fit_metrics and "test_mse" in fit_metrics, (
        "fit() debe incluir al menos val_mse y test_mse en sus métricas"
    )

    loss, eval_samples, eval_metrics = client.evaluate(updated_params, {})

    assert np.isfinite(loss), (
        "La loss devuelta por evaluate() debe ser un número finito"
    )
    assert eval_samples == len(client.val_loader.dataset), (
        "evaluate() debe devolver como num_samples el tamaño del conjunto de validación"
    )
    assert "val_mse" in eval_metrics and "test_mse" in eval_metrics, (
        "evaluate() debe incluir al menos val_mse y test_mse en sus métricas"
    )
    assert np.isfinite(eval_metrics["test_mse"]), (
        "La métrica test_mse devuelta por evaluate() debe ser finita"
    )


# =========================================================
# TEST 2: INTEGRACIÓN SERVIDOR + CLIENTES FAKE EN EVALUACIÓN
# =========================================================

def test_server_full_evaluation_with_three_fake_clients(monkeypatch):
    """
    Test de integración servidor + clientes fake para full_evaluation().

    Verifica que:
    1. el servidor espera a 3 clientes,
    2. samplea 3 clientes,
    3. llama a evaluate() en cada uno,
    4. devuelve arrays before/after/weights con longitud correcta.
    """
    clients = [
        FakeClient("c1", init_value=0.0, fit_value=1.0, val_before=10.0, val_after=8.0, val_samples=20, test_mse=7.0),
        FakeClient("c2", init_value=0.0, fit_value=2.0, val_before=12.0, val_after=9.0, val_samples=30, test_mse=8.0),
        FakeClient("c3", init_value=0.0, fit_value=3.0, val_before=11.0, val_after=7.0, val_samples=25, test_mse=6.0),
    ]
    manager = FakeClientManager(clients)
    server = server_module.FlowerFedExServer(manager)

    # Necesita parámetros globales previos para evaluar
    server.global_parameters = make_parameters(5.0)

    before, after, weights = server.full_evaluation(
        lambda: {"epochs": 3, "mu": 0.01}
    )

    assert manager.wait_for_called == 1, (
        "El servidor debería esperar a que haya clientes disponibles antes de evaluar"
    )
    assert manager.sample_called == 1, (
        "El servidor debería samplear clientes exactamente una vez en full_evaluation()"
    )
    assert len(before) == 3 and len(after) == 3 and len(weights) == 3, (
        "full_evaluation() debe devolver tres arrays de longitud 3"
    )
    assert np.allclose(before, np.array([8.0, 9.0, 7.0])), (
        "El array 'before' debería contener los val_mse devueltos por los clientes fake"
    )
    assert np.allclose(after, np.array([7.0, 8.0, 6.0])), (
        "El array 'after' debería contener los test_mse devueltos por los clientes fake"
    )
    assert np.allclose(weights, np.array([20.0, 30.0, 25.0])), (
        "El array 'weights' debería contener los num_examples devueltos por los clientes fake"
    )

    for client in clients:
        assert client.evaluate_called == 1, (
            f"El cliente {client.client_id} debería haber recibido exactamente una llamada a evaluate()"
        )


# =========================================================
# TEST 3: RONDA FEDERADA SIMULADA CON 3 CLIENTES FALSOS
# =========================================================

def test_server_communication_round_updates_global_model(monkeypatch):
    """
    Test de integración de una ronda federada simulada.

    Verifica que:
    1. el servidor pide pesos iniciales al primer cliente si global_parameters es None,
    2. llama a fit() en los 3 clientes,
    3. agrega los modelos locales con la estrategia,
    4. actualiza global_parameters con el promedio agregado,
    5. devuelve before/after/weights para FedEx.
    """
    clients = [
        FakeClient("c1", init_value=0.0, fit_value=1.0, val_before=10.0, val_after=8.0, val_samples=20, test_mse=7.0),
        FakeClient("c2", init_value=0.0, fit_value=2.0, val_before=12.0, val_after=9.0, val_samples=30, test_mse=8.0),
        FakeClient("c3", init_value=0.0, fit_value=3.0, val_before=11.0, val_after=7.0, val_samples=25, test_mse=6.0),
    ]
    manager = FakeClientManager(clients)
    server = server_module.FlowerFedExServer(manager)

    fake_strategy = FakeStrategy()
    server.strategy = fake_strategy

    # Limpiar historial global del módulo servidor para que el test no dependa de otros
    for key in server_module.history:
        server_module.history[key].clear()

    before, after, weights = server.communication_round(
        lambda: {"epochs": 2, "mu": 0.1}
    )

    assert manager.wait_for_called == 1, (
        "communication_round() debería esperar a que haya 3 clientes disponibles"
    )
    assert manager.sample_called == 1, (
        "communication_round() debería samplear clientes exactamente una vez"
    )
    assert server._round == 1, (
        "El contador interno de rondas debería incrementarse a 1 tras la primera ronda"
    )

    # Solo el primer cliente debería proporcionar los parámetros iniciales
    assert clients[0].get_parameters_called == 1, (
        "El primer cliente debería haber proporcionado los parámetros iniciales del modelo global"
    )
    assert clients[1].get_parameters_called == 0 and clients[2].get_parameters_called == 0, (
        "Solo el primer cliente debería recibir get_parameters() en la primera ronda"
    )

    for client in clients:
        assert client.fit_called == 1, (
            f"El cliente {client.client_id} debería haber recibido exactamente una llamada a fit()"
        )
        assert client.last_fit_config == {"epochs": 2, "mu": 0.1}, (
            f"El cliente {client.client_id} recibió una configuración incorrecta en fit()"
        )

    assert fake_strategy.aggregate_fit_called == 1, (
        "La estrategia de agregación debería ejecutarse exactamente una vez"
    )

    assert np.allclose(before, np.array([10.0, 12.0, 11.0])), (
        "El array 'before' debería contener los val_mse_before de los clientes"
    )
    assert np.allclose(after, np.array([8.0, 9.0, 7.0])), (
        "El array 'after' debería contener los val_mse de los clientes"
    )
    assert np.allclose(weights, np.array([20.0, 30.0, 25.0])), (
        "El array 'weights' debería contener los val_samples de los clientes"
    )

    # Comprobar que el modelo global se ha actualizado a la media de [1, 2, 3] = 2
    assert server.global_parameters is not None, (
        "Tras la agregación, global_parameters no debería seguir siendo None"
    )

    aggregated_ndarrays = parameters_to_ndarrays(server.global_parameters)
    for i, arr in enumerate(aggregated_ndarrays):
        assert np.allclose(arr, 2.0), (
            f"El tensor agregado {i} no coincide con la media esperada de los clientes fake"
        )

    # Además, la agregación de métricas debió actualizar el historial
    assert len(server_module.history["round"]) == 1, (
        "El historial del servidor debería registrar una ronda tras communication_round()"
    )