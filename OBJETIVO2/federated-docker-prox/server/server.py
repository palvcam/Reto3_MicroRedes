print("Iniciando servidor puerto 8080...")

# =========================
# IMPORTS
# =========================
import flwr as fl
import numpy as np
import os
import flwr.common as flwr_common
from flwr.server.client_manager import SimpleClientManager
from flwr.server.strategy import FedAvg
from flwr.server.superlink.fleet.grpc_bidi.grpc_server import start_grpc_server
from flwr.common import GetParametersIns, FitIns, EvaluateIns
import matplotlib.pyplot as plt
import time


# =========================
# CHECKPOINT STRATEGY
# =========================
class CheckpointFedAvg(FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            aggregated_ndarrays = flwr_common.parameters_to_ndarrays(
                aggregated_parameters
            )

            save_path = "modelos_guardados"
            os.makedirs(save_path, exist_ok=True)

            np.savez(f"{save_path}/modelo_global_final.npz", *aggregated_ndarrays)

            print(f"[Servidor] Modelo global (ronda {server_round}) guardado")

        return aggregated_parameters, aggregated_metrics


# =========================
# HISTÓRICO
# =========================
history = {
    "round": [],
    "mse_val": [],
    "rmse_val": [],
    "r2_val": [],
    "mse_test": [],
    "rmse_test": [],
    "r2_test": [],
}


# =========================
# MÉTRICAS
# =========================
def aggregate_metrics(metrics):
    total_samples = sum(n for n, _ in metrics)

    mse_val = sum(n * m["val_mse"] for n, m in metrics) / total_samples
    rmse_val = sum(n * m["val_rmse"] for n, m in metrics) / total_samples
    r2_val = sum(n * m["val_r2"] for n, m in metrics) / total_samples

    mse_test = sum(n * m["test_mse"] for n, m in metrics) / total_samples
    rmse_test = sum(n * m["test_rmse"] for n, m in metrics) / total_samples
    r2_test = sum(n * m["test_r2"] for n, m in metrics) / total_samples

    # SIN ESCALAR
    real_mse_val   = sum(n * m["real_val_mse"]   for n, m in metrics) / total_samples
    real_rmse_val  = sum(n * m["real_val_rmse"]  for n, m in metrics) / total_samples
    real_mse_test  = sum(n * m["real_test_mse"]  for n, m in metrics) / total_samples
    real_rmse_test = sum(n * m["real_test_rmse"] for n, m in metrics) / total_samples



    history["round"].append(len(history["round"]) + 1)
    history["mse_val"].append(real_mse_val)       
    history["rmse_val"].append(real_rmse_val) 
    history["r2_val"].append(r2_val)
    history["mse_test"].append(real_mse_test)   
    history["rmse_test"].append(real_rmse_test) 
    history["r2_test"].append(r2_test)

    rnd = len(history["round"])
    print(f"\nGLOBAL R{rnd} VAL:  MSE={real_mse_val:.4f} W²  RMSE={real_rmse_val:.4f} W  R²={r2_val:.4f}")
    print(f"GLOBAL R{rnd} TEST: MSE={real_mse_test:.4f} W²  RMSE={real_rmse_test:.4f} W  R²={r2_test:.4f}")

    return {"mse_val": mse_val, "rmse_val": real_rmse_val, "r2_val": r2_val}


# =========================
# FEDPROX CONFIG
# =========================
MU = 0.1
EPOCHS = 5


# =========================
# SERVIDOR (ANTES FlowerFedExServer)
# =========================
class FlowerFedProxServer:
    def __init__(self, client_manager):
        self.client_manager = client_manager
        self.strategy = CheckpointFedAvg(
            fraction_fit=1.0,
            min_fit_clients=3,
            min_available_clients=3,
            fraction_evaluate=1.0,
            min_evaluate_clients=3,
        )
        self.global_parameters = None
        self._round = 0

    def communication_round(self):
        self._round += 1

        self.client_manager.wait_for(num_clients=3, timeout=300.0)
        clients = self.client_manager.sample(num_clients=3)

        fit_results = []

        for client in clients:

            fit_config = {
                "epochs": EPOCHS,
                "mu": MU,
            }

            print(f"  config -> {fit_config}")

            if self.global_parameters is None:
                init_res = client.get_parameters(
                    ins=GetParametersIns(config={}),
                    timeout=300.0,
                    group_id=0,
                )
                self.global_parameters = init_res.parameters

            fit_res = client.fit(
                ins=FitIns(parameters=self.global_parameters, config=fit_config),
                timeout=10000,
                group_id=0,
            )

            fit_results.append(fit_res)

        pairs = [(None, r) for r in fit_results]
        agg_result = self.strategy.aggregate_fit(self._round, pairs, [])

        if agg_result is not None and agg_result[0] is not None:
            self.global_parameters = agg_result[0]

        metrics_for_log = [(int(r.num_examples), r.metrics) for r in fit_results]
        aggregate_metrics(metrics_for_log)

    def full_evaluation(self):
        self.client_manager.wait_for(num_clients=3, timeout=300.0)
        clients = self.client_manager.sample(num_clients=3)

        eval_config = {
            "epochs": EPOCHS,
            "mu": MU,
        }

        for client in clients:
            client.evaluate(
                ins=EvaluateIns(parameters=self.global_parameters, config=eval_config),
                timeout=300.0,
                group_id=0,
            )


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    NUM_ROUNDS = 30
    ADDRESS = "0.0.0.0:8080"
    MAX_MSG = 1024 * 1024 * 1024

    round_times = []
    total_start = time.time()

    client_manager = SimpleClientManager()

    grpc_server = start_grpc_server(
        client_manager=client_manager,
        server_address=ADDRESS,
        max_message_length=MAX_MSG,
        certificates=None,
    )

    print(f"gRPC escuchando en {ADDRESS}")
    print("Esperando 3 clientes...")

    client_manager.wait_for(num_clients=3, timeout=600.0)

    print("3 clientes conectados. Iniciando...")

    server = FlowerFedProxServer(client_manager)

    # =========================
    # LOOP FEDERADO
    # =========================
    for rnd in range(NUM_ROUNDS):
        print(f"\n=== Round {rnd + 1}/{NUM_ROUNDS} ===")
        round_start = time.time()

        server.communication_round()

        round_time = time.time() - round_start
        round_times.append(round_time)

        print(f"  Tiempo ronda: {round_time:.1f}s")

    total_time = time.time() - total_start

    print(f"\nTiempo total: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Tiempo medio: {np.mean(round_times):.1f}s ± {np.std(round_times):.1f}s")

    server.full_evaluation()

    grpc_server.stop(grace=3000)

        # =========================
    # GRÁFICAS DE EVOLUCIÓN
    # =========================
    rounds = history["round"]

    # -------- VALIDACIÓN --------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Evolución métricas globales VALIDACIÓN", fontsize=14)

    axes[0].plot(rounds, history["mse_val"], marker="o", linewidth=2)
    axes[0].set_title("MSE (validación)")
    axes[0].set_xlabel("Ronda")
    axes[0].set_ylabel("MSE (W²)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(rounds, history["rmse_val"], marker="o", linewidth=2)
    axes[1].set_title("RMSE (validación)")
    axes[1].set_xlabel("Ronda")
    axes[1].set_ylabel("RMSE (W)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(rounds, history["r2_val"], marker="o", linewidth=2)
    axes[2].set_title("R² (validación)")
    axes[2].set_xlabel("Ronda")
    axes[2].set_ylabel("R²")
    axes[2].set_ylim([0, 1])
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("evolucion_metricas_val.png", dpi=150)
    plt.show()

    print("Gráfica VALIDACIÓN guardada")

    # -------- TEST --------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Evolución métricas globales TEST", fontsize=14)

    axes[0].plot(rounds, history["mse_test"], marker="o", linewidth=2)
    axes[0].set_title("MSE (test)")
    axes[0].set_xlabel("Ronda")
    axes[0].set_ylabel("MSE (W²)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(rounds, history["rmse_test"], marker="o", linewidth=2)
    axes[1].set_title("RMSE (test)")
    axes[1].set_xlabel("Ronda")
    axes[1].set_ylabel("RMSE (W)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(rounds, history["r2_test"], marker="o", linewidth=2)
    axes[2].set_title("R² (test)")
    axes[2].set_xlabel("Ronda")
    axes[2].set_ylabel("R²")
    axes[2].set_ylim([0, 1])
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("evolucion_metricas_test.png", dpi=150)
    plt.show()

    print("Gráfica TEST guardada")