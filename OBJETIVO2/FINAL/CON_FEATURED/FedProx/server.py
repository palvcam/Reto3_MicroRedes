print("Arrancando servidor...")

import flwr as fl
import matplotlib.pyplot as plt

history = {"round": [], "mse": [], "rmse": [], "r2": []}

def weighted_average(metrics):
    total_samples = sum(n for n, _ in metrics)
    
    mse  = sum(n * m["val_mse"]  for n, m in metrics) / total_samples
    rmse = sum(n * m["val_rmse"] for n, m in metrics) / total_samples
    r2   = sum(n * m["val_r2"]   for n, m in metrics) / total_samples
    
    # Guardar métricas por ronda
    current_round = len(history["round"]) + 1
    history["round"].append(current_round)
    history["mse"].append(mse)
    history["rmse"].append(rmse)
    history["r2"].append(r2)

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
    # =========================
    # GRÁFICAS DE EVOLUCIÓN
    # =========================
    rounds = history["round"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Evolución métricas globales — Aprendizaje Federado", fontsize=14)

    # MSE
    axes[0].plot(rounds, history["mse"], marker="o", color="steelblue", linewidth=2)
    axes[0].set_title("MSE (validación)")
    axes[0].set_xlabel("Ronda")
    axes[0].set_ylabel("MSE")
    axes[0].grid(True, alpha=0.3)

    # RMSE
    axes[1].plot(rounds, history["rmse"], marker="o", color="darkorange", linewidth=2)
    axes[1].set_title("RMSE (validación)")
    axes[1].set_xlabel("Ronda")
    axes[1].set_ylabel("RMSE")
    axes[1].grid(True, alpha=0.3)

    # R²
    axes[2].plot(rounds, history["r2"], marker="o", color="seagreen", linewidth=2)
    axes[2].set_title("R² (validación)")
    axes[2].set_xlabel("Ronda")
    axes[2].set_ylabel("R²")
    axes[2].set_ylim([0, 1])
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("evolucion_metricas.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Gráfica guardada en evolucion_metricas.png")