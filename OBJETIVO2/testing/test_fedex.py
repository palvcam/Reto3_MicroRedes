import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from FINAL.CON_FEATURED.FedProxFedEx.hyper import FedEx

class MockServerFantasma:
    """Simula ser el FlowerFedExServer para no tener que arrancar gRPC."""
    def communication_round(self, get_config):
        # Simulamos 3 clientes pidiendo configuraciones
        configs = [get_config() for _ in range(3)]
        
        before = [100.0, 100.0, 100.0] # Loss inicial
        after = []
        weights = [500, 500, 500]      # 500 muestras cada uno
        
        # TRUCO: Engañamos a FedEx. Le decimos que si usa 10 épocas, 
        # el error baja drásticamente (a 50), si usa 3 épocas, apenas baja (a 90).
        for cfg in configs:
            if cfg["epochs"] == 10:
                after.append(50.0) 
            else:
                after.append(90.0)
                
        return before, after, weights

def test_fedex_aprende_y_reduce_entropia():
    """Verifica que el algoritmo FedEx es capaz de converger a la mejor configuración."""
    # Arrange
    servidor_simulado = MockServerFantasma()
    configs_de_prueba = {"epochs": [3, 10]} # Simplificamos el grid para el test
    
    # Instanciamos FedEx
    fedex = FedEx(server=servidor_simulado, configs=configs_de_prueba, diff=True)
    entropia_inicial = fedex.entropy()
    probabilidad_mejor_inicial = fedex.trace('mle')[-1]
    
    # Act: Simulamos 8 rondas de entrenamiento
    for _ in range(8):
        fedex.step()
        
    entropia_final = fedex.entropy()
    probabilidad_mejor_final = fedex.trace('mle')[-1]
    
    # Assert
    assert entropia_final < entropia_inicial, "La entropía debería disminuir (indicando convergencia)"
    assert probabilidad_mejor_final > probabilidad_mejor_inicial, "La probabilidad MLE (Maximum Likelihood Estimate) de la mejor config debería aumentar"