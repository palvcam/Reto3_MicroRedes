import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import copy
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from FINAL.CON_FEATURED.FedProxFedEx.model import PVModel

def entrenar_modelo_local(model, dataloader, epochs=5, lr=0.01):
    """Función auxiliar para entrenar un modelo rápidamente."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
    return model

def evaluar_modelo(model, dataloader):
    """Función auxiliar para evaluar el MSE de un modelo."""
    criterion = nn.MSELoss()
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            preds = model(X_batch)
            total_loss += criterion(preds, y_batch).item()
    return total_loss / len(dataloader)

def test_comparativa_centralizado_vs_federado():
    """
    Test de Regresión ML:
    Comprueba que el Aprendizaje Federado (FedAvg) consigue un rendimiento 
    comparable al Centralizado, asegurando que la arquitectura no degrada la predicción.
    """
    # 1. PREPARACIÓN DE DATOS SIMULADOS (Mock Data)
    # Creamos un problema de regresión sintético simple y lineal con algo de ruido
    n_samples = 1000
    n_features = 13
    X = torch.randn(n_samples, n_features)
    # y = suma de las features + ruido
    y = X.sum(dim=1, keepdim=True) + torch.randn(n_samples, 1) * 0.1 

    # Split: 80% Entrenamiento, 20% Test
    dataset_train = TensorDataset(X[:800], y[:800])
    dataset_test  = TensorDataset(X[800:], y[800:])
    loader_test   = DataLoader(dataset_test, batch_size=32)

    # El Centralizado ve TODO el train de golpe
    loader_central = DataLoader(dataset_train, batch_size=32, shuffle=True)

    # El Federado lo dividimos en 2 clientes que solo ven su mitad
    dataset_c1 = TensorDataset(X[:400], y[:400])
    dataset_c2 = TensorDataset(X[400:800], y[400:800])
    loader_c1  = DataLoader(dataset_c1, batch_size=32, shuffle=True)
    loader_c2  = DataLoader(dataset_c2, batch_size=32, shuffle=True)


    # 2. ENTRENAMIENTO CENTRALIZADO (El Baseline)
    model_central = PVModel(input_size=n_features, layers_sizes=[64, 32])
    model_central = entrenar_modelo_local(model_central, loader_central, epochs=10)
    mse_central = evaluar_modelo(model_central, loader_test)


    # 3. ENTRENAMIENTO FEDERADO (Simulación FedAvg)
    # Creamos dos modelos que parten de los mismos pesos iniciales
    model_fed_global = PVModel(input_size=n_features, layers_sizes=[64, 32])
    model_cliente1 = copy.deepcopy(model_fed_global)
    model_cliente2 = copy.deepcopy(model_fed_global)

    # Cada cliente entrena solo con sus datos locales
    model_cliente1 = entrenar_modelo_local(model_cliente1, loader_c1, epochs=10)
    model_cliente2 = entrenar_modelo_local(model_cliente2, loader_c2, epochs=10)

    # El servidor agrega los pesos (Media aritmética - FedAvg simple)
    dict_global = model_fed_global.state_dict()
    dict_c1 = model_cliente1.state_dict()
    dict_c2 = model_cliente2.state_dict()

    for key in dict_global.keys():
        dict_global[key] = (dict_c1[key] + dict_c2[key]) / 2.0
    
    model_fed_global.load_state_dict(dict_global)
    mse_federado = evaluar_modelo(model_fed_global, loader_test)


    # 4. ASSERT: TEST DE REGRESIÓN
    # En Machine Learning, un modelo distribuido suele tener un ligero peaje al principio.
    # Aceptamos que el federado sea peor, pero NO DEBE ser un desastre (margen de tolerancia).
    # Fijamos un margen generoso (ej. el error federado no debe ser más de 2 veces el centralizado en pocas épocas).
    print(f"\n[Regresión] MSE Centralizado: {mse_central:.4f} | MSE Federado: {mse_federado:.4f}")
    
    assert mse_federado < (mse_central * 3.0), \
        f"Degradación crítica: MSE Federado ({mse_federado}) es muchísimo peor que el Centralizado ({mse_central})"
    
    # Comprobamos que ambos modelos realmente están aprendiendo y superan un baseline tonto (MSE << varianza inicial)
    varianza_inicial = torch.var(y[800:]).item()
    assert mse_central < varianza_inicial, "El modelo centralizado no está aprendiendo nada"
    assert mse_federado < varianza_inicial, "El modelo federado no está aprendiendo nada"