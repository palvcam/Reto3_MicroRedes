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
# HIPERPARÁMETROS FEDEX - Appendix C.3 paper [file:1]
# =========================
GAMMA = 0.9                          # Factor descuento EMA baseline λ_t (Eq. C.3)

# =========================
# UTILIDAD - Persistencia mejor modelo global
# =========================
def save_model(parameters, path="best_model_EX.pth"):
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

# =========================
# AGREGACIÓN MÉTRICAS GLOBALES - evaluate_metrics_aggregation_fn [file:1]
# =========================
def fedex_aggregate_metrics(metrics):
    """Agrega métricas evaluate() de todos los clientes: Σ(n_i * m_i)/Σn_i."""
    global best_mse, latest_parameters
    
    # total_samples = Σ n_i (train samples por cliente)
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

# =========================
# ESTRATEGIA SERVIDOR FEDEX - Algorithm 2 + Appendix C.3 [file:1]
# =========================
class CustomStrategy(fl.server.strategy.FedAvg):
    """Override FedAvg: agrega baseline λ_t para FedEx exp-grad en clientes."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)           # Inicializa FedAvg base
        # Estado interno FedEx servidor — Algorithm 2 línea 3
        self.lambda_t = None                 # b_t: baseline actual (Appendix C.3)
        self.loss_history = []               # L_1, L_2, ..., L_{t-1} losses globales

    def configure_fit(self, server_round, parameters, client_manager):
        """Algorithm 2 línea 4: Envía w_t + config{λ_t} a clientes."""
        # Primera ronda λ_t=None → -1.0 (cliente ignora baseline)
        config = {
            "lambda_t": float(self.lambda_t) if self.lambda_t is not None else -1.0
        }
        # Empaqueta parámetros + config para fit()
        fit_ins = fl.common.FitIns(parameters, config)
        # Samplea clientes (min_fit_clients=3 fijos)
        clients = client_manager.sample(
            num_clients=self.min_fit_clients,
            min_num_clients=self.min_available_clients
        )
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(self, server_round, results, failures):
        """Algorithm 2 líneas 6-10: FedAvg(w) + update baseline λ_{t+1}."""
        global latest_parameters
        
        # 1. Agregación FedAvg estándar: w_{t+1} = Agg(w_t, w_ti)
        aggregated = super().aggregate_fit(server_round, results, failures)
        
        # Guarda parámetros globales para save_model
        if aggregated is not None:
            parameters, _ = aggregated
            latest_parameters = parameters      # w_{t+1} global

        # 2. COMPUTA L_t ≡ Σ|V_i|*mse_i/Σ|V_i| (Alg. 2 línea 8)
        # |V_i| = val_samples del cliente * num_examples (train size proxy)
        total_val_samples = sum(
            r.metrics.get("val_samples", 1) * r.num_examples for _, r in results
        )
        # L_t = weighted MSE global actual
        weighted_mse = sum(
            r.metrics.get("val_mse", 0) * r.metrics.get("val_samples", 1) * r.num_examples
            for _, r in results
        ) / max(total_val_samples, 1)

        # 3. UPDATE HISTORIAL (después de usar λ_t anterior en clientes)
        self.loss_history.append(weighted_mse)   # Agrega L_t
        self.loss_history = self.loss_history[-10:]  # Buffer fijo 10 (estabilidad)

        # 4. COMPUTA λ_{t+1} = b_{t+1} (Appendix C.3 Eq. C.3)
        t = len(self.loss_history)
        if t == 1:
            # Ronda 1: b_1 = L_1 (sin EMA)
            self.lambda_t = weighted_mse
        else:
            # EMA baseline: b_{t+1} = Σ γ^(t-1-s) * L_s / Z
            weights = np.array([GAMMA ** (t - 1 - s) for s in range(t - 1)])
            self.lambda_t = np.dot(weights, self.loss_history[:-1]) / weights.sum()

        # Log evolución baseline para debug
        print(f"[FedEx servidor] λₜ = {self.lambda_t:.4f} (weighted_mse={weighted_mse:.4f})")

        return aggregated                     # Retorna w_{t+1}, num_examples

# =========================
# INSTANCIACIÓN ESTRATEGIA - Configuración 3 clientes fijos
# =========================
strategy = CustomStrategy(
    fraction_fit=1.0,                    # 100% clientes disponibles cada ronda
    fraction_evaluate=1.0,               # Eval global en todos los clientes
    min_fit_clients=3,                   # Mínimo 3 clientes para fit()
    min_evaluate_clients=3,              # Mínimo 3 clientes para evaluate()
    min_available_clients=3,             # Espera conectados los 3 parques
    evaluate_metrics_aggregation_fn=fedex_aggregate_metrics  # Agregador métricas
)

# =========================
# MAIN - Loop entrenamiento FedEx 30 rondas
# =========================
if __name__ == "__main__":
    print("Iniciando servidor FL con FedEx...")  # Log inicio servidor
    
    # Inicia servidor Flower (bloquea hasta 30 rondas completas)
    fl.server.start_server(
        server_address="0.0.0.0:8080",       # Escucha localhost:8080
        config=fl.server.ServerConfig(num_rounds=30),  # 30 rondas fijas
        strategy=strategy,                   # Estrategia FedEx custom
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
