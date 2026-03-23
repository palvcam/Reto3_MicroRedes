print("Arrancando servidor...")
import flwr as fl

def weighted_average(metrics):
    total_samples = sum(n for n, _ in metrics)
    
    mse  = sum(n * m["val_mse"]  for n, m in metrics) / total_samples
    rmse = sum(n * m["val_rmse"] for n, m in metrics) / total_samples
    r2   = sum(n * m["val_r2"]   for n, m in metrics) / total_samples
    
    print(f"\n[GLOBAL] MSE: {mse:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}\n")
    
    return {"mse": mse, "rmse": rmse, "r2": r2}

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