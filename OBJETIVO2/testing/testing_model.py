#verificar que la clase PVModel se construye bien, 
# que no explota con diferentes tamaños de batch
# y siempre devuelve un tensor con la forma correcta.

import pytest
import torch
import sys
import os

# Apuntamos a la carpeta superior para encontrar model.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from FINAL.CON_FEATURED.FedProxFedEx.model import PVModel

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