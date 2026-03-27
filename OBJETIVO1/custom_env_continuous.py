import gymnasium as gym
from gymnasium import spaces
import numpy as np


class CustomEnvContinuous(gym.Env):
    """
    Entorno continuo de microrred para algoritmos Deep-RL tipo A2C/PPO.

    Filosofía:
    - misma microred de pymgrid
    - misma lógica económica
    - misma idea de control de batería
    - misma penalización por SoC demasiado bajo

    Cambios respecto al entorno tabular:
    - acción continua en lugar de discreta
    - observación continua en lugar de discretizada
    - compatible con Stable Baselines3
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        pymgrid_network,
        horizon=24 * 365,
        reward_scale_C=1.0,
        low_soc_penalty=0.3,
        low_soc_threshold=0.2,
        net_load_min=-40.64,
        net_load_max=62.45,
        price_min=0.0206,
        price_max=0.42315,
    ):
        super().__init__()

        self.mg = pymgrid_network
        self.horizon = min(horizon, len(self.mg))
        self.current_step = 0

        # Normalización reward
        self.reward_scale_C = float(reward_scale_C)

        # Penalización por batería baja
        self.low_soc_penalty = float(low_soc_penalty)
        self.low_soc_threshold = float(low_soc_threshold)

        # Rangos aproximados para observación continua
        self.net_load_min = float(net_load_min)
        self.net_load_max = float(net_load_max)
        self.price_min = float(price_min)
        self.price_max = float(price_max)

        # ---------------------------------------------------------
        # ESPACIO DE ACCIONES
        # ---------------------------------------------------------
        # Acción continua simétrica 2D en [-1, 1]:
        #   action[0] -> batería
        #   action[1] -> red
        #
        # Internamente, estas acciones se reescalan a [0, 1] porque
        # pymgrid.run(..., normalized=True) espera valores normalizados.
        #
        # Mapeo interno:
        #   -1.0 -> 0.0
        #    0.0 -> 0.5
        #    1.0 -> 1.0
        #
        # La semántica física exacta de 0/0.5/1 para batería y red
        # debe verificarse empíricamente con tests manuales.
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )



        # ---------------------------------------------------------
        # ESPACIO DE OBSERVACIÓN
        # ---------------------------------------------------------
        # Observación continua:
        # [net_load_norm, soc, import_price_norm, hour_sin, hour_cos, day_sin, day_cos]
        #
        # - net_load_norm: carga neta (load - pv) normalizada a [-1, 1]
        # - soc: estado de carga de la batería en [0, 1]
        # - import_price_norm: precio de importación normalizado a [0, 1]
        # - hour_sin, hour_cos: codificación cíclica de la hora del día
        # - day_sin, day_cos: codificación cíclica del día del año
        self.observation_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([ 1.0, 1.0, 1.0,  1.0,  1.0,  1.0,  1.0], dtype=np.float32),
            dtype=np.float32
        )

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
        return float(self.mg.grid.item().import_price[0])

    def _get_current_export_price(self):
        return float(self.mg.grid.item().export_price[0])

    # =========================================================
    # NORMALIZACIONES AUXILIARES
    # =========================================================
    def _normalize_net_load(self, net_load):
        """
        Normaliza net_load usando min-max y lo lleva aproximadamente a [-1, 1].
        Aplica clipping por seguridad
        """
        denom = self.net_load_max - self.net_load_min
        if denom <= 0:
            return 0.0

        x01 = (net_load - self.net_load_min) / denom
        x11 = 2.0 * x01 - 1.0
        return float(np.clip(x11, -1.0, 1.0))

    def _normalize_price(self, price):
        """
        Normalizar price a [0, 1] mediante min-max.
        Aplica clipping por seguridad. 
        """
        denom = self.price_max - self.price_min
        if denom <= 0:
            return 0.0
        x = (price - self.price_min) / denom
        return float(np.clip(x, 0.0, 1.0))

    def _encode_hour_cyclic(self, hour):
        """
        Codificación cíclica de la hora del día.
        """
        angle = 2.0 * np.pi * hour / 24.0
        return float(np.sin(angle)), float(np.cos(angle))

    def _encode_day_of_year_cyclic(self, day_of_year):
        """
        Codificación cíclica del día del año.
        """
        angle = 2.0 * np.pi * day_of_year / 365.0
        return float(np.sin(angle)), float(np.cos(angle))

    # =========================================================
    # OBSERVACIÓN
    # =========================================================
    def _get_obs(self):
        load_raw = self._get_current_load()
        pv_raw = self._get_current_pv()
        net_load_raw = load_raw - pv_raw

        soc_raw = np.clip(self._get_current_soc(), 0.0, 1.0)
        price_raw = self._get_current_import_price()

        hour = self.current_step % 24
        hour_sin, hour_cos = self._encode_hour_cyclic(hour)

        day_of_year = (self.current_step // 24) % 365
        day_sin, day_cos = self._encode_day_of_year_cyclic(day_of_year)

        net_load_norm = self._normalize_net_load(net_load_raw)
        price_norm = self._normalize_price(price_raw)

        obs = np.array([
            net_load_norm,
            soc_raw,
            price_norm,
            hour_sin,
            hour_cos,
            day_sin,
            day_cos
        ], dtype=np.float32)

        return obs

    def _get_info(self):
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
    def _map_symmetric_action_to_unit_interval(self, x):
            """
            Mapea una acción de [-1, 1] a [0, 1].
            """
            x = float(np.clip(x, -1.0, 1.0))
            return 0.5 * (x + 1.0)

    def _get_control_dict(self, action):
        """
        Convierte una acción continua simétrica del agente (shape (2,))
        en una acción normalizada en [0, 1] para pymgrid.

        action:
            action[0] -> batería, en [-1, 1]
            action[1] -> red, en [-1, 1]

        pymgrid.run(..., normalized=True) espera:
            {"battery": [x], "grid": [y]} con x,y en [0,1]
        """
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        if action.shape != (2,):
            raise ValueError(f"Se esperaba acción con shape (2,), recibido {action.shape}")

        battery_action = self._map_symmetric_action_to_unit_interval(action[0])
        grid_action = self._map_symmetric_action_to_unit_interval(action[1])

        return {
            "battery": [battery_action],
            "grid": [grid_action],
        }


    # =========================================================
    # STEP
    # =========================================================
    def step(self, action):
        # Última observación válida antes de aplicar la acción
        last_valid_observation = self._get_obs()
        

        control_dict = self._get_control_dict(action)
        soc_before = self._get_current_soc()

        mg_obs, mg_reward, mg_done, mg_info = self.mg.run(
            control_dict,
            normalized=True
        )

        raw_reward = float(mg_reward)
        cost = -raw_reward
        reward = raw_reward / self.reward_scale_C

        terminated = bool(mg_done)

        current_load = None
        current_pv = None
        current_import_price = None
        current_export_price = None
        current_soc = None
        terminal_observation = None

        self.current_step += 1

        if not terminated:
            obs = self._get_obs()

            current_load = self._get_current_load()
            current_pv = self._get_current_pv()
            current_import_price = self._get_current_import_price()
            current_export_price = self._get_current_export_price()
            current_soc = self._get_current_soc()
        else:
            # Si pymgrid termina de forma natural, no asumimos que exista
            # una "siguiente observación" física válida.
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            current_soc = soc_before
            terminal_observation = last_valid_observation.copy()

        low_soc_penalty_applied = 0.0
        if (current_soc is not None) and (current_soc < self.low_soc_threshold):
            soc_deficit_ratio = (self.low_soc_threshold - current_soc) / self.low_soc_threshold
            low_soc_penalty_applied = self.low_soc_penalty * soc_deficit_ratio
            reward -= low_soc_penalty_applied

        
        truncated = bool(self.current_step >= self.horizon)

        if truncated and not terminated:
            terminal_observation = obs.copy()
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        info = {
            "cost": cost,
            "mg_reward": raw_reward,
            "reward_normalized": reward,
            "mg_done": bool(mg_done),
            "mg_obs": mg_obs,
            "mg_info": mg_info,
            "control_dict": control_dict,
            "current_load": current_load,
            "current_pv": current_pv,
            "current_import_price": current_import_price,
            "current_export_price": current_export_price,
            "soc_before": soc_before,
            "soc_after": current_soc,
            "soc": current_soc,
            "low_soc_penalty_applied": low_soc_penalty_applied,
        }

        if terminal_observation is not None:
            info["terminal_observation"] = terminal_observation

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