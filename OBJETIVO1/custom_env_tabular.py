import gym
from gym import spaces
import numpy as np

class CustomEnvTabular(gym.Env):
    """
    Entorno de Microrred para Q-Learning (Tabular).
    Integra la red de pymgrid, recibe una serie temporal externa de precios
    y discretiza los estados internamente (incluyendo la hora del día).
    """
    
    def __init__(self, pymgrid_network, hourly_prices, horizon=24*365):
        super(CustomEnvTabular, self).__init__()
        
        # 1. Parámetros del entorno
        self.mg = pymgrid_network
        self.hourly_prices = hourly_prices  # Lista o array con los precios externos
        self.horizon = horizon
        self.current_step = 0
        
        # ESPACIO DE ACCIONES
        # 9 acciones discretas
        # 0 a 3: Cargar (100%, 75%, 50%, 25%)
        # 4: Neutro (0%)
        # 5 a 8: Descargar (25%, 50%, 75%, 100%)
        self.action_space = spaces.Discrete(9)
        
        # ESPACIO DE OBSERVACIÓN
        # Definimos los bins para las variables continuas
        self.net_load_bins = [-5.0, 0.0, 5.0, 10.0] # CAMBIANDO...
        self.price_bins = [0.10, 0.20, 0.30] # CAMBIANDO...
        
        # Estado: [Carga_Neta, Batería_SoC, Precio_Actual, Hora_del_Día]
        self.observation_space = spaces.MultiDiscrete([
            len(self.net_load_bins) + 1,  # 5 estados (índices 0 a 4)
            11,                           # 11 estados (0% a 100%, índices 0 a 10)
            len(self.price_bins) + 1,     # 4 estados (índices 0 a 3)
            24                            # 24 estados (horas 0 a 23)
        ])

    def _get_obs(self):
        """Extrae la información, la discretiza y devuelve el estado actual."""
        # 1. Extraer valores físicos puros
        net_load_raw = self.mg.load - self.mg.pv
        soc_raw = self.mg.battery.soc
        
        # Extraer el precio usando el paso actual (protegido por si nos pasamos de longitud)
        if self.current_step < len(self.hourly_prices):
            grid_price_raw = self.hourly_prices[self.current_step]
        else:
            grid_price_raw = self.hourly_prices[-1] 
            
        # Calcular la hora del día (0 a 23)
        hora_del_dia = self.current_step % 24
            
        # 2. Discretizar en "cajones"
        d_net_load = int(np.digitize(net_load_raw, bins=self.net_load_bins))
        
        d_soc = int(np.round(soc_raw, 1) * 10)
        d_soc = max(0, min(10, d_soc)) # Asegurar límites entre 0 y 10
        
        d_price = int(np.digitize(grid_price_raw, bins=self.price_bins))
        
        # 3. Devolver el array del estado discreto
        return np.array([d_net_load, d_soc, d_price, hora_del_dia], dtype=np.int32)

    def _get_control_dict(self, action):
        """Traduce la acción elegida por el agente a órdenes físicas para pymgrid."""
        pv = self.mg.pv
        load = self.mg.load
        
        # Mapear acción a porcentajes
        charge_pct, discharge_pct = 0.0, 0.0
        if action == 0: charge_pct = 1.0
        elif action == 1: charge_pct = 0.75
        elif action == 2: charge_pct = 0.50
        elif action == 3: charge_pct = 0.25
        elif action == 4: pass # Neutro
        elif action == 5: discharge_pct = 0.25
        elif action == 6: discharge_pct = 0.50
        elif action == 7: discharge_pct = 0.75
        elif action == 8: discharge_pct = 1.0

        # Calcular energía real a mover, respetando límites de la batería
        p_charge = min(self.mg.battery.p_charge_max * charge_pct, self.mg.battery.capa_to_charge)
        p_discharge = min(self.mg.battery.p_discharge_max * discharge_pct, self.mg.battery.capa_to_discharge)

        # Balance de energía en el nodo principal
        net_energy_needed = load + p_charge - pv - p_discharge

        if net_energy_needed > 0:
            grid_import, grid_export = net_energy_needed, 0.0
        else:
            grid_import, grid_export = 0.0, abs(net_energy_needed)

        return {
            'pv_consummed': min(pv, load + p_charge),
            'battery_charge': p_charge,
            'battery_discharge': p_discharge,
            'grid_import': grid_import,
            'grid_export': grid_export
        }

    def step(self, action):
        """Avanza la simulación 1 paso (1 hora) y devuelve resultados."""
        # 1. Generar órdenes y ejecutarlas en el simulador
        control_dict = self._get_control_dict(action)
        status = self.mg.run(control_dict)
        
        # 2. Obtener el precio actual para la factura eléctrica
        if self.current_step < len(self.hourly_prices):
            current_price = self.hourly_prices[self.current_step]
        else:
            current_price = self.hourly_prices[-1]
            
        # 3. Calcular el coste real del paso actual
        # Asumimos que la energía exportada se paga a la mitad del precio de compra (ajustable)
        cost = (control_dict['grid_import'] * current_price) - (control_dict['grid_export'] * current_price * 0.5)
        
        # Queremos maximizar la recompensa, usamos el coste negativo
        reward = -cost
        
        # 4. REWARD SHAPING: Penalización por dejar la batería vacía (<5%)
        if self.mg.battery.soc < 0.05:
            reward -= 10.0
            
        # 5. Actualizar el reloj
        self.current_step += 1
        done = bool(self.current_step >= self.horizon)
        
        # 6. Observar el nuevo estado (si no ha terminado)
        if not done:
            obs = self._get_obs()
        else:
            # Estado dummy o vacío al terminar el episodio
            obs = np.array([0, 0, 0, 0], dtype=np.int32) 
            
        info = {'cost': cost, 'status': status}
        
        return obs, reward, done, info

    def reset(self):
        """Reinicia la microrred y el reloj al comienzo del episodio."""
        self.mg.reset()
        self.current_step = 0
        return self._get_obs()