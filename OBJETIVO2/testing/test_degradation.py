import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import copy
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from federated_docker.client.model import PVModel

def entrenar_modelo_local(model, dataloader, epochs=5, lr=0.01):
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
    criterion = nn.MSELoss()
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            preds = model(X_batch)
            total_loss += criterion(preds, y_batch).item()
    return total_loss / len(dataloader)

def test_centralizado_vs_federado_restricted_features():
    """
    Test de Integración ML (Realista):
    Comprueba el escenario del proyecto: El modelo Centralizado tiene acceso a TODAS 
    las características (ej. 15), mientras que el Federado usa un subconjunto local (ej. 10).
    
    Afirmaciones:
    1. Ambos modelos deben aprender (MSE < Varianza inicial).
    2. El Centralizado DEBE ser mejor que el Federado (por tener más información).
    3. El Federado no debe degradarse hasta el punto de ser inútil.
    """
    # 1. PREPARACIÓN DE DATOS SIMULADOS
    n_samples = 1000
    features_central = 15
    features_federado = 10  # El federado tiene 5 variables menos
    
    # Creamos los datos base completos
    X_full = torch.randn(n_samples, features_central)
    
    # El target (y) depende fuertemente de TODAS las variables para penalizar al federado
    y = X_full.sum(dim=1, keepdim=True) + torch.randn(n_samples, 1) * 0.1 

    # Datos para Centralizado (Todas las features)
    dataset_train_cent = TensorDataset(X_full[:800], y[:800])
    loader_train_cent  = DataLoader(dataset_train_cent, batch_size=32, shuffle=True)
    loader_test_cent   = DataLoader(TensorDataset(X_full[800:], y[800:]), batch_size=32)

    # Datos para Federado (Solo las primeras 10 features)
    X_restricted = X_full[:, :features_federado]
    dataset_c1 = TensorDataset(X_restricted[:400], y[:400])
    dataset_c2 = TensorDataset(X_restricted[400:800], y[400:800])
    loader_c1  = DataLoader(dataset_c1, batch_size=32, shuffle=True)
    loader_c2  = DataLoader(dataset_c2, batch_size=32, shuffle=True)
    loader_test_fed = DataLoader(TensorDataset(X_restricted[800:], y[800:]), batch_size=32)

    # 2. ENTRENAMIENTO CENTRALIZADO (15 features)
    model_central = PVModel(input_size=features_central, layers_sizes=[64, 32])
    model_central = entrenar_modelo_local(model_central, loader_train_cent, epochs=10)
    mse_central = evaluar_modelo(model_central, loader_test_cent)

    # 3. ENTRENAMIENTO FEDERADO (10 features)
    model_fed_global = PVModel(input_size=features_federado, layers_sizes=[64, 32])
    model_cliente1 = copy.deepcopy(model_fed_global)
    model_cliente2 = copy.deepcopy(model_fed_global)

    model_cliente1 = entrenar_modelo_local(model_cliente1, loader_c1, epochs=10)
    model_cliente2 = entrenar_modelo_local(model_cliente2, loader_c2, epochs=10)

    # Agregación FedAvg
    dict_global = model_fed_global.state_dict()
    dict_c1 = model_cliente1.state_dict()
    dict_c2 = model_cliente2.state_dict()

    for key in dict_global.keys():
        dict_global[key] = (dict_c1[key] + dict_c2[key]) / 2.0
    
    model_fed_global.load_state_dict(dict_global)
    mse_federado = evaluar_modelo(model_fed_global, loader_test_fed)

    # 4. ASERCIONES (La lógica de tu proyecto)
    print(f"\nMSE Centralizado (15 feat): {mse_central:.4f}")
    print(f"MSE Federado (10 feat): {mse_federado:.4f}")
    
    varianza_inicial = torch.var(y[800:]).item()
    
    # A. Ambos deben aprender algo útil
    assert mse_central < varianza_inicial, "El centralizado no aprende"
    assert mse_federado < varianza_inicial, "El federado no aprende con features reducidas"
    
    # B. El Centralizado debe ganar (o al menos empatar estadísticamente)
    assert mse_central <= mse_federado, "¡Anomalía! El federado con menos datos superó al centralizado."