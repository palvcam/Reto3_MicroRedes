import gymnasium as gym
from gymnasium import spaces
import numpy as np


class CustomEnvTabular(gym.Env):
    """
    Entorno de microrred para Q-Learning tabular.

    La microred (pymgrid) ya incorpora:
    - LoadModule con la serie de carga
    - RenewableModule con la serie PV
    - GridModule con precios de import/export
    - BatteryModule

    Este entorno:
    - observa el estado actual desde pymgrid
    - discretiza el estado
    - transforma una acción discreta (solo batería) en una acción normalizada
    - usa mg.run(..., normalized=True)
    - calcula el comportamiento físico de la red a posteriori para monitorización
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        pymgrid_network,
        horizon=24 * 365,
        low_soc_penalty=50.0,
        low_soc_threshold=0.2,
        battery_max_capacity=200.0
    ):
        super().__init__()

        self.mg = pymgrid_network
        self.low_soc_penalty = float(low_soc_penalty)
        self.low_soc_threshold = float(low_soc_threshold)
        self.battery_max_capacity = float(battery_max_capacity)

        self.horizon = min(horizon, len(self.mg))
        self.current_step = 0

        # ---------------------------------------------------------
        # ESPACIO DE ACCIONES (solo batería)
        # ---------------------------------------------------------
        # 9 acciones discretas.
        self.action_space = spaces.Discrete(9)

        self.battery_action_map = {
            0: 1.0,    # Descarga máxima
            1: 0.75,
            2: 0.5,
            3: 0.25,
            4: 0.0,    # Mantener (Neutro)
            5: -0.25,
            6: -0.5,
            7: -0.75,
            8: -1.0     # Carga máxima
        }
        
        # ---------------------------------------------------------
        # ESPACIO DE OBSERVACIÓN
        # ---------------------------------------------------------
        self.net_load_bins = [-426.61, -75.55, 15.02, 38.4]
        self.price_bins = [0.097, 0.134, 0.164]

        self.observation_space = spaces.MultiDiscrete([
            len(self.net_load_bins) + 1,  # 5 bins de carga neta
            11,                           # SoC discretizado 0..10
            len(self.price_bins) + 1,     # 4 bins de precio
            24                            # hora del día
        ])

    def _get_current_load(self):
        # Entramos en 'load' y cogemos el primer módulo [0]
        return float(self.mg.modules["load"][0].current_load)

    def _get_current_pv(self):
        # ¡Ojo! La llave es 'renewable', no 'pv'
        return float(self.mg.modules["renewable"][0].current_renewable)

    def _get_current_soc(self):
        return float(self.mg.modules["battery"][0].soc)

    def _get_current_import_price(self):
        return float(self.mg.modules["grid"][0].import_price[0])

    def _get_current_export_price(self):
        return float(self.mg.modules["grid"][0].export_price[0])
    
    def _get_obs(self):
        """Lee el estado actual desde pymgrid y lo discretiza."""
        load_raw = self._get_current_load()
        pv_raw = self._get_current_pv()
        net_load_raw = load_raw - pv_raw

        soc_raw = self._get_current_soc()
        grid_price_raw = self._get_current_import_price()

        hora_del_dia = self.current_step % 24

        d_net_load = int(np.digitize(net_load_raw, bins=self.net_load_bins))
        d_soc = int(np.round(soc_raw * 10))
        d_soc = max(0, min(10, d_soc))
        d_price = int(np.digitize(grid_price_raw, bins=self.price_bins))

        return np.array([d_net_load, d_soc, d_price, hora_del_dia], dtype=np.int32)

    def _get_info(self):
        """Información auxiliar útil"""
        return {
            "current_step": self.current_step,
            "current_load": self._get_current_load(),
            "current_pv": self._get_current_pv(),
            "current_import_price": self._get_current_import_price(),
            "current_export_price": self._get_current_export_price(),
            "soc": self._get_current_soc(),
            "accion_red_real": 1 # Por defecto neutro en el paso 0
        }

    def _get_control_dict(self, action):
        """
        El agente solo decide la acción de la batería. 
        El entorno (física) adapta la red como flex module automáticamente.
        """
        # 1. El agente decide sobre la batería (ej. -1.0 a 1.0)
        battery_action_norm = self.battery_action_map[action]
        battery_kw = battery_action_norm * 50.0  # max_charge/discharge
        
        # 2. Física: ¿Cuánta energía necesita/sobra en el edificio?
        net_load_kw = self._get_current_load() - self._get_current_pv()
        
        # 3. La red se ADAPTA a la demanda restante (Actúa como Flex)
        p_red_kw = net_load_kw - battery_kw
        
        # 4. Le damos a Pymgrid las órdenes explícitas porque su arquitectura lo exige
        return {
            "battery": [float(battery_kw)],
            "grid": [float(p_red_kw)]
        }

    def step(self, action):
        """Avanza la simulación un paso y extrae la física real para monitorización."""
        
        # 1. GUARDAR ESTADO PREVIO (Para calcular qué hizo la física realmente)
        soc_previo = self._get_current_soc()
        current_load = self._get_current_load()
        current_pv = self._get_current_pv()
        net_load = current_load - current_pv

        # 2. EJECUTAR ACCIÓN 
        # (Solo mandamos la batería, Pymgrid usa la red como Flex Module automáticamente)
        control_dict = self._get_control_dict(action)
        mg_obs, mg_reward, mg_done, mg_info = self.mg.run(
            control_dict,
            normalized=False
        )

        # 3. CÁLCULOS FÍSICOS A POSTERIORI (Lo que realmente ocurrió)
        soc_nuevo = self._get_current_soc()

        # Variación de energía química real en la batería (kWh). 
        # Lo que realmente subió/bajó el nivel teniendo en cuenta eficiencias.
        delta_energia_bat = (soc_nuevo - soc_previo) * self.battery_max_capacity
        carga_bat = max(0, delta_energia_bat)
        descarga_bat = max(0, -delta_energia_bat)

        # Reconstruimos la orden eléctrica exacta que le dimos a la red
        # (Net Load - Acción de la batería)
        battery_kw = self.battery_action_map[action] * 50.0  
        p_red_real = net_load - battery_kw

        # Separamos el balance de red en métricas limpias
        importacion_red = max(0, p_red_real)
        exportacion_red = max(0, -p_red_real)

        # 4. RECOMPENSAS Y PENALIZACIONES
        reward = float(mg_reward)
        cost = -float(mg_reward)

        if soc_nuevo < self.low_soc_threshold:
            violacion = self.low_soc_threshold - soc_nuevo
            penalizacion_aplicada = (violacion / self.low_soc_threshold) * self.low_soc_penalty
            reward -= penalizacion_aplicada

        # 5. ACTUALIZAR PASO Y ESTADO
        self.current_step += 1
        terminated = bool(mg_done)
        truncated = bool(self.current_step >= self.horizon)

        if not (terminated or truncated):
            obs = self._get_obs()
        else:
            obs = np.array([0, 0, 0, 0], dtype=np.int32)

        # 6. GUARDAR TODA LA INFORMACIÓN DETALLADA EN INFO
        info = {
            "step": self.current_step,
            "cost_eur": cost,                     # Gasto económico real en este paso
            "reward_rl": reward,                  # Recompensa que recibe el algoritmo (con penalizaciones)
            "soc": soc_nuevo,                     # % de batería
            "load_kw": current_load,              # Consumo del edificio
            "pv_kw": current_pv,                  # Generación solar
            "net_load_kw": net_load,              # Carga neta (Load - PV)
            "bat_charge_kw": carga_bat,           # Lo que la batería CARGÓ realmente
            "bat_discharge_kw": descarga_bat,     # Lo que la batería DESCARGÓ realmente
            "grid_import_kw": importacion_red,    # Lo que se COMPRÓ de la red
            "grid_export_kw": exportacion_red,    # Lo que se VENDIÓ a la red
            "grid_balance_kw": p_red_real,        # Balance total (Positivo=Import, Negativo=Export)
            "action_chosen": action,               # La acción discreta (0-8) que eligió el agente
            "current_import_price": self._get_current_import_price(),
            "current_export_price": self._get_current_export_price()
        }

        return obs, reward, terminated, truncated, info
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.mg.reset()
        self.current_step = 0

        obs = self._get_obs()
        info = self._get_info()

        return obs, info