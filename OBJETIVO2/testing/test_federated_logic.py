import pytest
import torch
import numpy as np
from collections import OrderedDict
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from FINAL.CON_FEATURED.FedProxFedEx.model import PVModel

def test_get_and_set_parameters():
    """
    Verifica la lógica de serialización de pesos del cliente federado.
    Extrae los pesos a NumPy y los vuelve a inyectar comprobando que son idénticos.
    """
    # Arrange: Creamos dos modelos (simulando que uno es Cliente A y otro es Servidor/Cliente B)
    model_origen = PVModel(input_size=13, layers_sizes=[64, 32])
    model_destino = PVModel(input_size=13, layers_sizes=[64, 32])
    
    # Act 1 (Simula get_parameters): Extraemos pesos a listas de NumPy
    parametros_extraidos = [val.cpu().numpy() for val in model_origen.state_dict().values()]
    
    # Assert 1: Validamos formatos de salida
    assert isinstance(parametros_extraidos, list), "Los parámetros deben enviarse como una lista"
    assert isinstance(parametros_extraidos[0], np.ndarray), "Los pesos internos deben ser arrays de NumPy"
    
    # Act 2 (Simula set_parameters): Los recargamos en el modelo destino
    params_dict = zip(model_destino.state_dict().keys(), parametros_extraidos)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model_destino.load_state_dict(state_dict, strict=True)
    
    # Assert 2: Verificamos que los pesos del modelo origen y destino ahora son matemáticamente idénticos
    for param_origen, param_destino in zip(model_origen.parameters(), model_destino.parameters()):
        assert torch.equal(param_origen, param_destino), "Error al recargar los parámetros: los pesos no coinciden"