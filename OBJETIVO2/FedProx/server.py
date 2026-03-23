print("Arrancando servidor...")

import flwr as fl

# Estrategia (FedAvg, FedProx se implementa en cliente)
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=3,
)

if __name__ == "__main__":
    print("Iniciando servidor FL...")

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=20),
        strategy=strategy,
    )