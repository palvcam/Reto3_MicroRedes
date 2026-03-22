print("Arrancando servidor...")
import flwr as fl

# Define cómo el servidor va a agregar los modelos de los clientes
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0, # usa el 100% de los clientes disponibles para entrenar en cada ronda
    fraction_evaluate=1.0, # usa el 100% de los clientes para evaluar el modelo global
    min_fit_clients=3, # mínimo 3 clientes deben participar en el entrenamiento
    min_evaluate_clients=3, # mínimo 3 clientes para evaluar
    min_available_clients=3, # el servidor espera hasta que haya 3 clientes conectados antes de empezar
)


if __name__ == "__main__":
    print("Iniciando servidor FL...")
    
    # Arranca el servidor
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=50), # ejecuta 20 rondas de FL
        strategy=strategy, # usa la estrategia definida
    )