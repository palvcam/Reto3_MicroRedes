print("Arrancando servidor...")

import flwr as fl

def weighted_average(metrics):
    total_samples = sum(n for n, _ in metrics)
    
    mse  = sum(n * m["val_mse"]  for n, m in metrics) / total_samples
    rmse = sum(n * m["val_rmse"] for n, m in metrics) / total_samples
    r2   = sum(n * m["val_r2"]   for n, m in metrics) / total_samples
    
    print(f"\n[GLOBAL] MSE: {mse:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}\n")
    
    return {"mse": mse, "rmse": rmse, "r2": r2}

# Estrategia (FedAvg, FedProx se implementa en cliente)
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=3,
    evaluate_metrics_aggregation_fn=weighted_average
)

if __name__ == "__main__":
    print("Iniciando servidor FL...")

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=20),
        strategy=strategy,
    )