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
    - transforma una acción discreta en una acción normalizada
      para los módulos controlables de pymgrid ('battery' y 'grid')
    - usa mg.run(..., normalized=True)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        pymgrid_network,
        horizon=24 * 365,
        low_soc_penalty=50.0,
        low_soc_threshold=0.2
    ):
        super().__init__()

        self.mg = pymgrid_network
        self.low_soc_penalty = float(low_soc_penalty)
        self.low_soc_threshold = float(low_soc_threshold)

        # Horizon del episodio
        # len(self.mg) usa la longitud de los módulos temporales de pymgrid
        self.horizon = min(horizon, len(self.mg))
        self.current_step = 0

        # ---------------------------------------------------------
        # ESPACIO DE ACCIONES
        # ---------------------------------------------------------
        # 9 acciones discretas.
        # Se mapean a una acción normalizada para batería.
        # Grid se deja de momento en valor "neutro" fijo.
        self.action_space = spaces.MultiDiscrete([9, 3])

        self.battery_action_map = {
            0: -1.0,  # Descarga máxima
            1: -0.75,
            2: -0.5,
            3: -0.25,
            4:  0.0,  # Mantener
            5:  0.25,
            6:  0.5,
            7:  0.75,
            8:  1.0   # Carga máxima
        }
        # Acción fija para grid en esta primera versión
        self.grid_action_map = {
            0: 0.0,  # Importar el máximo posible
            1: 0.5,  # Neutral (Ni importa ni exporta, o lo hace lo mínimo)
            2: 1.0   # Exportar el máximo posible
        }
        # ---------------------------------------------------------
        # ESPACIO DE OBSERVACIÓN
        # ---------------------------------------------------------
        # Estado discreto:
        # [net_load_bin, soc_bin, price_bin, hour]
        self.net_load_bins = [-426.61, -75.55, 15.02, 38.4]
        self.price_bins = [0.097, 0.134, 0.164]

        self.observation_space = spaces.MultiDiscrete([
            len(self.net_load_bins) + 1,  # 5 bins de carga neta
            11,                           # SoC discretizado 0..10
            len(self.price_bins) + 1,     # 4 bins de precio
            24                            # hora del día
        ])

    # =========================================================
    # HELPERS DE ACCESO AL ESTADO ACTUAL DE LA MICRORED
    # =========================================================
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

    # =========================================================
    # OBSERVACIÓN
    # =========================================================
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
        """Información auxiliar útil para debug/análisis."""
        return {
            "current_step": self.current_step,
            "current_load": self._get_current_load(),
            "current_pv": self._get_current_pv(),
            "current_import_price": self._get_current_import_price(),
            "current_export_price": self._get_current_export_price(),
            "soc": self._get_current_soc(),
        }

    # =========================================================
    # ACCIÓN
    # =========================================================
    def _get_control_dict(self, action):
        """
        Traduce la acción MultiDiscrete de Gymnasium al diccionario 
        de acciones normalizadas que espera pymgrid.
        """
        # action es ahora un array, ej: [4, 1]
        battery_idx = action[0]
        grid_idx = action[1]
        
        # Obtenemos los valores físicos/normalizados
        battery_action = self.battery_action_map[battery_idx]
        grid_action = self.grid_action_map[grid_idx]
        
        return {
            "battery": [float(battery_action)],
            "grid": [float(grid_action)],
        }

    # =========================================================
    # STEP
    # =========================================================
    def step(self, action):
        """Avanza la simulación un paso."""
        control_dict = self._get_control_dict(action)

        # IMPORTANTE:
        # mg.sample_action() devolvió acciones normalizadas en [0,1],
        # así que aquí usamos normalized=True.
        mg_obs, mg_reward, mg_done, mg_info = self.mg.run(
            control_dict,
            normalized=True
        )

        # Reward base de pymgrid:
        # positivo = ingreso / negativo = coste
        reward = float(mg_reward)

        cost = -float(mg_reward)

        # Penalización proporcional si el SoC baja del umbral
        current_soc = self._get_current_soc()
        if current_soc < self.low_soc_threshold:
            violacion = self.low_soc_threshold - current_soc # cuánto nos hemos pasado del límite.
            penalizacion_aplicada = (violacion / self.low_soc_threshold) * self.low_soc_penalty
            
            reward -= penalizacion_aplicada

        self.current_step += 1

        # Terminación natural del simulador
        terminated = bool(mg_done)

        # Truncado por horizonte externo del episodio
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
            "current_load": self._get_current_load() if not (terminated or truncated) else None,
            "current_pv": self._get_current_pv() if not (terminated or truncated) else None,
            "current_import_price": self._get_current_import_price() if not (terminated or truncated) else None,
            "current_export_price": self._get_current_export_price() if not (terminated or truncated) else None,
            "soc": self._get_current_soc() if not (terminated or truncated) else None,
        }

        return obs, reward, terminated, truncated, info

    # =========================================================
    # RESET
    # =========================================================
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.mg.reset()
        self.current_step = 0

        obs = self._get_obs()
        info = self._get_info()

        return obs, info