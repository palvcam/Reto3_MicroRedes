print("Arrancando servidor...")  # Mensaje de inicio del servidor FedEx

import flwr as fl                    # Framework Federated Learning
import numpy as np                   # Arrays numéricos y operaciones matemáticas
import matplotlib.pyplot as plt      # Generación de gráficos de evolución
import torch                         # Tensores PyTorch para modelo NN
from model import PVModel            # Arquitectura del modelo PV (5→32→16)
from flwr.common import parameters_to_ndarrays  # Convierte parámetros Flower a numpy

# =========================
# VARIABLES GLOBALES - Monitoreo rendimiento
# =========================
history = {                          # Diccionario para historial métricas globales
    "round": [],                     # Número de ronda (1, 2, 3, ..., 30)
    "mse_val": [],                       # MSE global agregado por cliente
    "rmse_val": [],                      # RMSE global agregado por cliente  
    "r2_val": [],                         # R² global agregado por cliente
    "mse_test": [],                       # MSE global agregado por cliente
    "rmse_test": [],                      # RMSE global agregado por cliente  
    "r2_test": []   
}

best_mse = float("inf")              # MSE mínimo visto hasta ahora (inicial ∞)
latest_parameters = None             # Últimos parámetros globales agregados (FedAvg)

# =========================
# UTILIDAD - Persistencia mejor modelo global
# =========================
def save_model(parameters, path="best_model_AVG.pth"):
    model = PVModel(input_size=5, layers_sizes=[128, 32, 16])  # Crea modelo vacío (5 feats)
    
    # Convierte parámetros Flower → numpy arrays
    ndarrays = parameters_to_ndarrays(parameters)
    
    # Zip nombres tensores → valores numpy
    params_dict = zip(model.state_dict().keys(), ndarrays)
    
    # Convierte numpy → torch tensors
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    
    # Carga parámetros en modelo y guarda
    model.load_state_dict(state_dict)
    torch.save(model.state_dict(), path)
    
    print("Modelo guardado:", path)    # Log confirmación guardado

def weighted_average(metrics):
    total_samples = sum(n for n, _ in metrics)
    
    # MSE global ponderado: Σ(n_i * mse_i) / Σn_i
    mse_val  = sum(n * m["val_mse"]   for n, m in metrics) / total_samples
    rmse_val = sum(n * m["val_rmse"]  for n, m in metrics) / total_samples
    r2_val   = sum(n * m["val_r2"]    for n, m in metrics) / total_samples

    # TEST
    mse_test  = sum(n * m["test_mse"]   for n, m in metrics) / total_samples
    rmse_test = sum(n * m["test_rmse"]  for n, m in metrics) / total_samples
    r2_test   = sum(n * m["test_r2"]    for n, m in metrics) / total_samples
    
    # Actualiza historial para trazas de evolución
    current_round = len(history["round"]) + 1
    history["round"].append(current_round)    # Ronda actual
    history["mse_val"].append(mse_val)               # MSE global ronda t
    history["rmse_val"].append(rmse_val)             # RMSE global ronda t
    history["r2_val"].append(r2_val)                 # R² global ronda t

    history["mse_test"].append(mse_test)               # MSE global ronda t
    history["rmse_test"].append(rmse_test)             # RMSE global ronda t
    history["r2_test"].append(r2_test)                 # R² global ronda t
    
    # Log métricas globales
    print(f"\n[GLOBAL VAL] MSE: {mse_val:.1f} | RMSE: {rmse_val:.1f} | R²: {r2_val:.3f}")
    print(f"\n[GLOBAL TEST] MSE: {mse_test:.1f} | RMSE: {rmse_test:.1f} | R²: {r2_test:.3f}")
    
    # Chequea si es nuevo mejor modelo global
    if latest_parameters is not None and mse_val < best_mse:
        best_mse = mse_val                       # Actualiza mejor MSE
        save_model(latest_parameters)        # Persiste mejor modelo
        print("Nuevo mejor modelo!")         # Log mejora
    
    # Retorna dict para server history (no usado directamente)
    return {"mse_val": mse_val, "rmse_val": rmse_val, "r2_val": r2_val}

# Define cómo el servidor va a agregar los modelos de los clientes
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0, # usa el 100% de los clientes disponibles para entrenar en cada ronda
    fraction_evaluate=1.0, # usa el 100% de los clientes para evaluar el modelo global
    min_fit_clients=3, # mínimo 3 clientes deben participar en el entrenamiento
    min_evaluate_clients=3, # mínimo 3 clientes para evaluar
    min_available_clients=3, # el servidor espera hasta que haya 3 clientes conectados antes de empezar
    evaluate_metrics_aggregation_fn=weighted_average
)


if __name__ == "__main__":
    print("Iniciando servidor FL...")
    
    # Arranca el servidor
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=50), # ejecuta 20 rondas de FL
        strategy=strategy, # usa la estrategia definida
    )
    
    # =========================
    # VISUALIZACIÓN - Post-entrenamiento
    # =========================
    
    # Gráfico 1: Evolución MSE por ronda
    plt.figure(figsize=(10, 6))
    plt.plot(history["round"], history["mse_val"], 'b-o', linewidth=3, markersize=8)
    plt.title('FedEx Global MSE Evolution VAL')      # Título gráfico
    plt.xlabel('Round')                          # Eje X
    plt.ylabel('MSE')                            # Eje Y
    plt.grid(True)                               # Grid fondo
    plt.savefig('fedex_mse_val.png')                 # Guarda PNG
    plt.show()                                   # Muestra gráfico
    
    # Gráfico 2: Evolución RMSE por ronda
    plt.figure(figsize=(10, 6))
    plt.plot(history["round"], history["rmse_val"], 'r-o', linewidth=3, markersize=8)
    plt.title('FedEx Global RMSE Evolution VAL')
    plt.xlabel('Round')
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.savefig('fedex_rmse_val.png')
    plt.show()
    
    # Gráfico 3: Evolución R² por ronda
    plt.figure(figsize=(10, 6))
    plt.plot(history["round"], history["r2_val"], 'g-o', linewidth=3, markersize=8)
    plt.title('FedEx Global R² Evolution VAL')
    plt.xlabel('Round')
    plt.ylabel('R²')
    plt.grid(True)
    plt.savefig('fedex_r2_val.png')
    plt.show()
    
    # Resumen final métricas (inicial → final)
    print("\n RESULTADOS FINALES VAL:")
    print(f"MSE inicial:  {history['mse_val'][0]:.1f} → final: {history['mse_val'][-1]:.1f}")
    print(f"RMSE inicial: {history['rmse_val'][0]:.1f} → final: {history['rmse_val'][-1]:.1f}")
    print(f"R² inicial:   {history['r2_val'][0]:.3f} → final: {history['r2_val'][-1]:.3f}")

    #### TEST
    # Gráfico 1: Evolución MSE por ronda
    plt.figure(figsize=(10, 6))
    plt.plot(history["round"], history["mse_test"], 'b-o', linewidth=3, markersize=8)
    plt.title('FedEx Global MSE Evolution TEST')      # Título gráfico
    plt.xlabel('Round')                          # Eje X
    plt.ylabel('MSE')                            # Eje Y
    plt.grid(True)                               # Grid fondo
    plt.savefig('fedex_mse_test.png')                 # Guarda PNG
    plt.show()                                   # Muestra gráfico
    
    # Gráfico 2: Evolución RMSE por ronda
    plt.figure(figsize=(10, 6))
    plt.plot(history["round"], history["rmse_test"], 'r-o', linewidth=3, markersize=8)
    plt.title('FedEx Global RMSE Evolution TEST')
    plt.xlabel('Round')
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.savefig('fedex_rmse_test.png')
    plt.show()
    
    # Gráfico 3: Evolución R² por ronda
    plt.figure(figsize=(10, 6))
    plt.plot(history["round"], history["r2_test"], 'g-o', linewidth=3, markersize=8)
    plt.title('FedEx Global R² Evolution TEST')
    plt.xlabel('Round')
    plt.ylabel('R²')
    plt.grid(True)
    plt.savefig('fedex_r2_test.png')
    plt.show()
    
    # Resumen final métricas (inicial → final)
    print("\n RESULTADOS FINALES TEST:")
    print(f"MSE inicial:  {history['mse_test'][0]:.1f} → final: {history['mse_test'][-1]:.1f}")
    print(f"RMSE inicial: {history['rmse_test'][0]:.1f} → final: {history['rmse_test'][-1]:.1f}")
    print(f"R² inicial:   {history['r2_test'][0]:.3f} → final: {history['r2_test'][-1]:.3f}")