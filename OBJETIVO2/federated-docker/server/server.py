print("Iniciando servidor puerto 8080...")
# Importar librerias
import flwr as fl
import numpy as np
import os
import flwr.common as flwr_common
from hyper import FedEx
from flwr.server.client_manager import SimpleClientManager
from flwr.server.strategy import FedAvg
from flwr.server.superlink.fleet.grpc_bidi.grpc_server import start_grpc_server
from flwr.common import GetParametersIns, FitIns, EvaluateIns
import matplotlib.pyplot as plt
import time

class CheckpointFedAvg(FedAvg):
    """Estrategia que hereda de FedAvg para guardar el modelo tras cada ronda."""
    def aggregate_fit(self, server_round, results, failures):
        # 1. Ejecuta la agregación normal matemática de FedAvg
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        # 2. Guarda los pesos si la agregación fue exitosa
        if aggregated_parameters is not None:
            # Descomprime el formato de Flower a Arrays de NumPy
            aggregated_ndarrays = flwr_common.parameters_to_ndarrays(aggregated_parameters)
            
            # Ruta absoluta dentro del contenedor Docker
            save_path = "/app/modelos_guardados"
            os.makedirs(save_path, exist_ok=True)
            
            # Sobrescribe el archivo. Al final, contendrá el último modelo.
            np.savez(f"{save_path}/modelo_global_final.npz", *aggregated_ndarrays)
            print(f"[Servidor] Checkpoint: Modelo global (ronda {server_round}) guardado en disco.")
            
        return aggregated_parameters, aggregated_metrics

history = {"round": [], "mse_val": [], "rmse_val": [], "r2_val": [], "mse_test": [], "rmse_test": [], "r2_test": []} # Diccionario que acumula las métricas globales de cada ronda

# Agrega las métricas de los 3 clientes en una métrica global ponderada por número de muestras
def fedex_aggregate_metrics(metrics):
    total_samples = sum(n for n, _ in metrics)
    mse_val  = sum(n * m["val_mse"]  for n, m in metrics) / total_samples
    rmse_val = sum(n * m["val_rmse"] for n, m in metrics) / total_samples
    r2_val   = sum(n * m["val_r2"]   for n, m in metrics) / total_samples

    mse_test  = sum(n * m["test_mse"]  for n, m in metrics) / total_samples
    rmse_test = sum(n * m["test_rmse"] for n, m in metrics) / total_samples
    r2_test   = sum(n * m["test_r2"]   for n, m in metrics) / total_samples
    # Se guardan las métricas en el historial
    history["round"].append(len(history["round"]) + 1) 
    history["mse_val"].append(mse_val)
    history["rmse_val"].append(rmse_val)
    history["r2_val"].append(r2_val)
    history["mse_test"].append(mse_test)
    history["rmse_test"].append(rmse_test)
    history["r2_test"].append(r2_test)
    print(f"\nGLOBAL R{len(history['round'])} VAL: MSE={mse_val:.4f}  RMSE={rmse_val:.4f}  R2={r2_val:.4f}")
    print(f"\nGLOBAL R{len(history['round'])} TEST: MSE={mse_test:.4f}  RMSE={rmse_test:.4f}  R2={r2_test:.4f}")
    return {"mse_val": mse_val, "rmse_val": rmse_val, "r2_val": r2_val}

# FedAvg define cómo el servidor va a agregar los modelos de los clientes
# Lo sustituimos por CheckpointFedAvg, que hereda de FedAvg pero además guarda el modelo global en disco tras cada ronda
strategy = CheckpointFedAvg(
    fraction_fit=1.0, # usa el 100% de los clientes disponibles para entrenar en cada ronda
    min_fit_clients=3, # mínimo 3 clientes deben participar en el entrenamiento
    min_available_clients=3, # el servidor espera hasta que haya 3 clientes conectados antes de empezar
    fraction_evaluate=1.0, # usa el 100% de los clientes para evaluar el modelo global
    min_evaluate_clients=3, # mínimo 3 clientes para evaluar
    evaluate_metrics_aggregation_fn=fedex_aggregate_metrics # Función de agregación de métricas
)


class FlowerFedExServer:
    def __init__(self, client_manager):
        self.client_manager = client_manager # Gestiona los clientes conectados
        self.strategy = strategy # Estrategia FedAvg para agregar pesos
        self.global_parameters = None # Pesos globales del modelo (None hasta la ronda 1)
        self._round = 0 # Contador de rondas

    @staticmethod
    def _parse_fit_results(fit_results):
        # Extrae de los resultados de fit que necesita FedEx para actualizar las probabilidades de cada configuración
        before  = np.array([r.metrics["val_mse_before"] for r in fit_results]) # Loss antes de entrenar
        after   = np.array([r.metrics["val_mse"]        for r in fit_results]) # Loss después de entrenar
        weights = np.array([r.metrics["val_samples"]    for r in fit_results]) # Nº muestras de validación
        return before, after, weights

    def communication_round(self, get_config):
        """
        Ejecuta una ronda federada completa. Llamado por fedex.step() en cada ronda.
        1. Asigna una config (epochs, mu) a cada cliente via FedEx
        2. Entrena cada cliente localmente con esa config
        3. Agrega los modelos locales con FedAvg
        4. Devuelve losses antes/después a FedEx para actualizar probabilidades
        """
        self._round += 1 # Incrementar contador de rondas
        self.client_manager.wait_for(num_clients=3, timeout=300.0) # Esperar a que haya 3 clientes disponibles antes de continuar
        clients = self.client_manager.sample(num_clients=3) # Seleccionar 3 clientes para la ronda

        fit_results = [] # Lista para acumular resultados de cada cliente
        for client in clients:
            # FedEx genera una config (epochs, mu) para cada cliente
            config_dict = get_config()
            fit_config = {
                "epochs": int(config_dict["epochs"]), # Épocas de entrenamiento local
                "mu":     float(config_dict["mu"]), # Parámetro FedProx
            }
            print(f"  config -> {fit_config}")

            # En la primera ronda no hay pesos globales todavía
            # Se solicitan los pesos iniciales al primer cliente
            if self.global_parameters is None:
                init_res = client.get_parameters(
                    ins=GetParametersIns(config={}), # Solicitud de parámetros iniciales
                    timeout=300.0,
                    group_id=0, 
                )
                self.global_parameters = init_res.parameters

            # Enviar pesos globales y config al cliente para que entrene localmente
            fit_res = client.fit(
                            ins=FitIns(parameters=self.global_parameters, config=fit_config),
                            timeout=10000,
                            group_id=0,
                        )
            fit_results.append(fit_res) # Guardar resultado del cliente

        # Agregar los modelos locales con FedAvg para obtener el nuevo modelo global
        pairs = [(None, r) for r in fit_results]
        agg_result = self.strategy.aggregate_fit(self._round, pairs, [])
        # Actualizar pesos globales con el modelo agregado
        if agg_result is not None and agg_result[0] is not None:
            self.global_parameters = agg_result[0]

        # Loguear métricas globales de la ronda (n_muestras, métricas)
        metrics_for_log = [(int(r.num_examples), r.metrics) for r in fit_results]
        fedex_aggregate_metrics(metrics_for_log)

        # Devolver losses antes/después y pesos a FedEx para actualizar probabilidades
        return self._parse_fit_results(fit_results)

    def full_evaluation(self, get_config):
        """
        Evaluación final del modelo. Llamado por fedex.test() al final del entrenamiento.
        Usa la mejor config encontrada (mle=True) para evaluar en test.
        """
        self.client_manager.wait_for(num_clients=3, timeout=300.0)
        clients = self.client_manager.sample(num_clients=3)

        # Obtener la mejor config encontrada por FedEx
        config_dict = get_config()
        eval_config = {
            "epochs": int(config_dict["epochs"]),
            "mu":     float(config_dict["mu"]),
        }
        before_list, after_list, weights_list = [], [], []
        for client in clients:
            # Llamar a evaluate() en cada cliente con los pesos globales finales
            eval_res = client.evaluate(
                        ins=EvaluateIns(parameters=self.global_parameters, config=eval_config),
                        timeout=300.0,
                        group_id=0,
                    )
            # "before" = val_mse
            # "after"  = test_mse
            before_list.append(eval_res.metrics.get("val_mse", eval_res.metrics["test_mse"]))
            after_list.append(eval_res.metrics["test_mse"])
            weights_list.append(eval_res.num_examples)
        return (
            np.array(before_list),
            np.array(after_list),
            np.array(weights_list, dtype=np.float64),
        )


if __name__ == "__main__":
    CONFIGS    = {"epochs": [3, 5, 8, 10], "mu": [0.0, 0.01, 0.1, 0.5]} # Grid de hiperparámetros
    NUM_ROUNDS = 30 # Número de rondas de entrenamiento federado
    ADDRESS    = "0.0.0.0:8080" # Dirección del servidor
    MAX_MSG    = 1024 * 1024 * 1024 # Tamaño máximo de mensaje: 1GB

    ## TIEMPO
    round_times = []
    total_start = time.time()

    # Inicializar gestor de clientes
    client_manager = SimpleClientManager()

    # Arrancar servidor
    grpc_server = start_grpc_server(
        client_manager=client_manager,
        server_address=ADDRESS,
        max_message_length=MAX_MSG,
        certificates=None,
    )
    print(f"gRPC escuchando en {ADDRESS}")

    # Bloquear hasta que los 3 clientes se conecten (máximo 10 minutos)
    print("Esperando 3 clientes...")
    client_manager.wait_for(num_clients=3, timeout=600.0)
    print("3 clientes conectados. Iniciando...")

    flower_server = FlowerFedExServer(client_manager)
    # Inicializar FedEx con el grid de configuraciones
    fedex = FedEx(
        server=flower_server,
        configs=CONFIGS,
        eta0="auto", # Paso de gradiente exponenciado: sqrt(2*log(k)) automático
        sched="auto", # Schedule: eta0 / sqrt(t) — decrece con el número de rondas
        baseline=0.2, # Factor de descuento del historial (0=solo ronda actual, 1=media total)
        diff=True, # Usar diferencia before/after como señal en lugar del valor absoluto
    )

    # Usar diferencia before/after como señal en lugar del valor absoluto
    for rnd in range(NUM_ROUNDS):
        print(f"\n=== Round {rnd + 1}/{NUM_ROUNDS} ===")
        round_start = time.time()
        fedex.step() # Una ronda federada completa: entrenar + agregar + actualizar FedEx
        # Entropía: alta = FedEx explorando, baja = FedEx convergiendo a mejor config
        # MLE: probabilidad de la mejor config — sube cuando FedEx converge
        round_time = time.time() - round_start
        round_times.append(round_time)
        print(f"  Tiempo ronda: {round_time:.1f}s")

    total_time = time.time() - total_start
    print(f"\nTiempo total entrenamiento: {total_time:.1f}s  ({total_time/60:.1f} min)")
    print(f"Tiempo medio por ronda: {np.mean(round_times):.1f}s ± {np.std(round_times):.1f}s")

    # Mejor configuración encontrada por FedEx
    fedex.test(mle=True) # mle=True = usar la config más probable
    best = fedex.sample(mle=True) # Imprimir la mejor configuración encontrada
    print(f"\nMejor config: epochs={best['epochs']}  mu={best['mu']}")
    grpc_server.stop(grace=3000) # Parar el servidor

    # =========================
    # GRÁFICAS DE EVOLUCIÓN
    # =========================
    rounds = history["round"]

    ## VAL
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Evolución métricas globales VALIDACIÓN", fontsize=14)

    # MSE
    axes[0].plot(rounds, history["mse_val"], marker="o", color="steelblue", linewidth=2)
    axes[0].set_title("MSE (validación)")
    axes[0].set_xlabel("Ronda")
    axes[0].set_ylabel("MSE")
    axes[0].grid(True, alpha=0.3)

    # RMSE
    axes[1].plot(rounds, history["rmse_val"], marker="o", color="darkorange", linewidth=2)
    axes[1].set_title("RMSE (validación)")
    axes[1].set_xlabel("Ronda")
    axes[1].set_ylabel("RMSE")
    axes[1].grid(True, alpha=0.3)

    # R²
    axes[2].plot(rounds, history["r2_val"], marker="o", color="seagreen", linewidth=2)
    axes[2].set_title("R² (validación)")
    axes[2].set_xlabel("Ronda")
    axes[2].set_ylabel("R²")
    axes[2].set_ylim([0, 1])
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    #plt.savefig("/app/output/evolucion_metricas_val.png", dpi=150, bbox_inches="tight")
    plt.savefig("evolucion_metricas_val.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Gráfica guardada en aws evolucion_metricas (validaión).png")

    ### TEST
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Evolución métricas globales TEST", fontsize=14)

    # MSE
    axes[0].plot(rounds, history["mse_test"], marker="o", color="steelblue", linewidth=2)
    axes[0].set_title("MSE (test)")
    axes[0].set_xlabel("Ronda")
    axes[0].set_ylabel("MSE")
    axes[0].grid(True, alpha=0.3)

    # RMSE
    axes[1].plot(rounds, history["rmse_test"], marker="o", color="darkorange", linewidth=2)
    axes[1].set_title("RMSE (test)")
    axes[1].set_xlabel("Ronda")
    axes[1].set_ylabel("RMSE")
    axes[1].grid(True, alpha=0.3)

    # R²
    axes[2].plot(rounds, history["r2_test"], marker="o", color="seagreen", linewidth=2)
    axes[2].set_title("R² (test)")
    axes[2].set_xlabel("Ronda")
    axes[2].set_ylabel("R²")
    axes[2].set_ylim([0, 1])
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    #plt.savefig("/app/output/evolucion_metricas_test.png", dpi=150, bbox_inches="tight")
    plt.savefig("evolucion_metricas_test.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Gráfica guardada en aws evolucion_metricas (test).png")
    