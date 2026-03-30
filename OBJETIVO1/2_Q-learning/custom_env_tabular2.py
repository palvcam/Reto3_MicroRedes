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
            0: 0.0,    # Descarga máxima
            1: 0.125,
            2: 0.25,
            3: 0.375,
            4: 0.5,    # Mantener (Neutro)
            5: 0.625,
            6: 0.75,
            7: 0.875,
            8: 1.0     # Carga máxima
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
        return float(self.mg.modules.load.item().current_load)

    def _get_current_pv(self):
        return float(self.mg.pv.item().current_renewable)

    def _get_current_soc(self):
        return float(self.mg.battery.item().soc)

    def _get_current_import_price(self):
        # import_price devuelve array: [actual, forecast...]
        return float(self.mg.grid.item().import_price[0])

    def _get_current_export_price(self):
        return float(self.mg.grid.item().export_price[0])

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
        """Traduce la acción discreta al diccionario que espera pymgrid.     """
       
        battery_action = self.battery_action_map[action]        
        return {
            "battery": [float(battery_action)],
            "grid": [0.5]  # la física se encargará de calcular lo que realmente se importa/exporta según el balance de la red
        }

    def step(self, action):
        """Avanza la simulación un paso y extrae la física real de la red."""
        
        # 1. ESTADO PREVIO (Para cálculos físicos)
        soc_previo  = self._get_current_soc()
        current_load = self._get_current_load()
        current_pv = self._get_current_pv()
        net_load = current_load - current_pv

        # 2. EJECUTAR ACCIÓN
        control_dict = self._get_control_dict(action)
        mg_obs, mg_reward, mg_done, mg_info = self.mg.run(
            control_dict,
            normalized=True
        )

        # 3. CÁLCULOS DE LA RED REAL (Ecuación de balance)
        soc_nuevo = self._get_current_soc()

        # Variación de energía en la batería (kWh)
        delta_energia_bat = (soc_nuevo - soc_previo) * self.battery_max_capacity

        # Balance de la red: Lo que falta del edificio + lo que entra/sale de la batería
        p_red_real = net_load + delta_energia_bat

        # Clasificamos la acción de la red real
        tolerancia = 0.05 # Margen para errores de coma flotante
        if p_red_real > tolerancia:
            accion_red_real = 0 # Importar de la red
        elif p_red_real < -tolerancia:
            accion_red_real = 2 # Exportar a la red
        else:
            accion_red_real = 1 # Neutro

        # 4. RECOMPENSAS
        reward = float(mg_reward)
        cost = -float(mg_reward)

        if soc_nuevo < self.low_soc_threshold:
                violacion = self.low_soc_threshold - soc_nuevo
                penalizacion_aplicada = (violacion / self.low_soc_threshold) * self.low_soc_penalty
                reward -= penalizacion_aplicada

        self.current_step += 1
        terminated = bool(mg_done)
        truncated = bool(self.current_step >= self.horizon)

        if not (terminated or truncated):
            obs = self._get_obs()
        else:
            obs = np.array([0, 0, 0, 0], dtype=np.int32)

        info = {
            "cost": cost,
            "mg_reward": float(mg_reward),
            "mg_done": bool(mg_done),
            "mg_obs": mg_obs,
            "mg_info": mg_info,
            "control_dict": control_dict,
            "curren_load": current_load,
            "current_pv": current_pv,
            "current_import_price": self._get_current_import_price(),
            "current_export_price": self._get_current_export_price(),
            "soc": soc_nuevo,
            "accion_red_real": accion_red_real,
            "p_red_kw": p_red_real
        }

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.mg.reset()
        self.current_step = 0

        obs = self._get_obs()
        info = self._get_info()

        return obs, info