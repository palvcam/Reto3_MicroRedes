print("Arrancando servidor...")

import flwr as fl
import numpy as np
import matplotlib.pyplot as plt

# ← Solo MSE history
history = {"round": [], "mse": []}

def fedex_aggregate_metrics(metrics):
    total_samples = sum(n for n, _ in metrics)
    
    mse  = sum(n * m["val_mse"]  for n, m in metrics) / total_samples
    rmse = sum(n * m["val_rmse"] for n, m in metrics) / total_samples
    r2   = sum(n * m["val_r2"]   for n, m in metrics) / total_samples
    
    # ← FedEx métricas (solo logging, no gráfico)
    epochs_avg = np.mean([m.get('fedex_epochs', 5) for _, m in metrics])
    mu_avg = np.mean([m.get('fedex_mu', 0.1) for _, m in metrics])
    
    # ← Solo guardar MSE para gráfico
    history["round"].append(len(history["round"])+1)
    history["mse"].append(mse)
    
    print(f"\n[GLOBAL] MSE: {mse:.1f} | RMSE: {rmse:.1f} | R²: {r2:.3f}")
    print(f"[FedEx] Epochs: {epochs_avg:.1f} | MU: {mu_avg:.3f}")
    
    return {"mse": mse, "rmse": rmse, "r2": r2}

# Estrategia FedEx
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=3,
    evaluate_metrics_aggregation_fn=fedex_aggregate_metrics
)

if __name__ == "__main__":
    print("Iniciando servidor FL con FedEx...")
    
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=30),
        strategy=strategy,
    )
    
    # ← GRÁFICO SOLO MSE (claro y directo)
    print("\n MSE Evolution:")
    plt.figure(figsize=(10, 6))
    plt.plot(history["round"], history["mse"], 'b-o', linewidth=3, markersize=8)
    plt.title('FedEx Global MSE Evolution', fontsize=16, fontweight='bold')
    plt.xlabel('Communication Round', fontsize=12)
    plt.ylabel('MSE', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig('fedex_mse.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ← Summary claro
    print(f"\n RESULTADOS FINALES:")
    print(f"MSE inicial: {history['mse'][0]:.1f}")
    print(f"MSE final:   {history['mse'][-1]:.1f}")
    print(f"Mejora:     {((history['mse'][0]-history['mse'][-1])/history['mse'][0]*100):.1f}%")

