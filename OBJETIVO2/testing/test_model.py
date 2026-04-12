#verificar que la clase PVModel se construye bien, 
# que no explota con diferentes tamaños de batch
# y siempre devuelve un tensor con la forma correcta.

import pytest
import torch
import sys
import os
import torch.nn as nn 

# Apuntamos a la carpeta superior para encontrar model.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from federated_docker.client.model import PVModel

def test_pvmodel_initialization():
    """Verifica que el modelo se puede instanciar sin errores con diferentes configuraciones."""
    model = PVModel(input_size=13, layers_sizes=[256, 128, 64], dropout=0.2)
    assert model is not None, "El modelo no se ha instanciado correctamente"

def test_pvmodel_forward_pass():
    """Verifica que el modelo procesa tensores correctamente y devuelve la forma (batch, 1)."""
    input_size = 13
    batch_size = 32 # Simulamos un batch de 32 filas de datos
    model = PVModel(input_size=input_size, layers_sizes=[128, 64, 32])
    
    # Arrange: Creamos un tensor de entrada simulado (ruido aleatorio)
    dummy_input = torch.randn(batch_size, input_size)
    
    # Act: Pasamos los datos por la red
    output = model(dummy_input)
    
    # Assert: Comprobamos dimensiones y ausencia de NaNs
    assert output.shape == (batch_size, 1), f"Forma incorrecta. Se esperaba (32, 1), se obtuvo {output.shape}"
    assert not torch.isnan(output).any(), "La red neuronal ha devuelto valores NaN (Not a Number)"

def test_pvmodel_eval_vs_train_mode():
    """
    Verifica que el Dropout funciona: 
    - En .train() la misma entrada da salidas distintas.
    - En .eval() la misma entrada da salidas idénticas.
    """
    input_size = 13
    model = PVModel(input_size=input_size, dropout=0.5)
    dummy_input = torch.randn(10, input_size)

    # Modo Entrenamiento (Dropout Activo)
    model.train()
    out1_train = model(dummy_input)
    out2_train = model(dummy_input)
    assert not torch.allclose(out1_train, out2_train), "El Dropout no está funcionando en modo train"

    # Modo Evaluación (Dropout Apagado)
    model.eval()
    out1_eval = model(dummy_input)
    out2_eval = model(dummy_input)
    assert torch.allclose(out1_eval, out2_eval), "El modelo no es determinista en modo eval"

def test_pvmodel_backward_pass():
    """
    Verifica que la red no tiene grafos desconectados y que 
    los gradientes fluyen hasta la primera capa.
    """
    input_size = 13
    model = PVModel(input_size=input_size)
    criterion = nn.MSELoss()
    
    dummy_input = torch.randn(16, input_size)
    dummy_target = torch.randn(16, 1) # Target ficticio

    # Forward
    output = model(dummy_input)
    loss = criterion(output, dummy_target)
    
    # Backward
    loss.backward()

    # Extraemos la primera capa lineal (índice 0 en nuestro Sequential)
    first_layer = model.model[0]
    
    # Assert: Si el gradiente es None, la capa está desconectada del loss
    assert first_layer.weight.grad is not None, "Los gradientes no llegan a la primera capa (backward roto)"
    assert not torch.all(first_layer.weight.grad == 0), "Los gradientes son exactamente cero"

def test_pvmodel_single_sample_inference():
    """
    Verifica que el modelo soporta la inferencia de un solo dato (batch_size=1).
    El BatchNorm1d falla con batch_size=1 en modo train, pero debe funcionar en eval.
    """
    input_size = 13
    model = PVModel(input_size=input_size)
    model.eval() # Fundamental para que BatchNorm no explote
    
    dummy_input = torch.randn(1, input_size)
    
    try:
        output = model(dummy_input)
        assert output.shape == (1, 1), "Fallo en la dimensionalidad con batch_size=1"
    except ValueError as e:
        pytest.fail(f"El modelo falló al hacer inferencia de un solo dato: {e}")