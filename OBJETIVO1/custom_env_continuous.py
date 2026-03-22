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
        reward_scale_C=787.0511991711403,
        low_soc_penalty=0.3,
        low_soc_threshold=0.05,
        net_load_min=-426.61,
        net_load_max=38.4,
        price_min=0.0,
        price_max=0.30,
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
        # Una acción continua:
        #   action[0] in [0,1]
        #
        # Según lo observado en pymgrid:
        #   1.0 -> descarga fuerte
        #   0.0 -> carga fuerte
        #   0.5 -> zona intermedia / casi neutra
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Acción fija para grid en esta versión
        self.grid_action_neutral = 0.50

        # ---------------------------------------------------------
        # ESPACIO DE OBSERVACIÓN
        # ---------------------------------------------------------
        # Observación continua:
        # [net_load_norm, soc, import_price_norm, hour_sin, hour_cos]
        #
        # - net_load_norm: carga neta normalizada aproximadamente a [-1,1]
        # - soc: en [0,1]
        # - import_price_norm: precio normalizado aproximadamente a [0,1]
        # - hour_sin, hour_cos: codificación cíclica de la hora
        self.observation_space = spaces.Box(
            low=np.array([-1.5, 0.0, 0.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.5, 1.0, 1.5, 1.0, 1.0], dtype=np.float32),
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
        """
        denom = self.net_load_max - self.net_load_min
        if denom <= 0:
            return 0.0

        x01 = (net_load - self.net_load_min) / denom
        x11 = 2.0 * x01 - 1.0
        return float(x11)

    def _normalize_price(self, price):
        """
        Normaliza el precio aproximadamente a [0,1].
        """
        denom = self.price_max - self.price_min
        if denom <= 0:
            return 0.0

        x = (price - self.price_min) / denom
        return float(x)

    def _encode_hour_cyclic(self, hour):
        """
        Codificación cíclica de la hora del día.
        """
        angle = 2.0 * np.pi * hour / 24.0
        return float(np.sin(angle)), float(np.cos(angle))

    # =========================================================
    # OBSERVACIÓN
    # =========================================================
    def _get_obs(self):
        load_raw = self._get_current_load()
        pv_raw = self._get_current_pv()
        net_load_raw = load_raw - pv_raw

        soc_raw = self._get_current_soc()
        price_raw = self._get_current_import_price()

        hour = self.current_step % 24
        hour_sin, hour_cos = self._encode_hour_cyclic(hour)

        net_load_norm = self._normalize_net_load(net_load_raw)
        price_norm = self._normalize_price(price_raw)

        obs = np.array([
            net_load_norm,
            soc_raw,
            price_norm,
            hour_sin,
            hour_cos,
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
    def _get_control_dict(self, action):
        """
        Convierte una acción continua del agente en una acción normalizada
        para pymgrid.

        action: np.array shape (1,)
        mg.run(..., normalized=True) espera:
            {"battery": [x], "grid": [y]}
        """
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        if action.shape != (1,):
            raise ValueError(f"Se esperaba acción con shape (1,), recibido {action.shape}")

        battery_action = float(np.clip(action[0], 0.0, 1.0))
        grid_action = float(self.grid_action_neutral)

        return {
            "battery": [battery_action],
            "grid": [grid_action],
        }

    # =========================================================
    # STEP
    # =========================================================
    def step(self, action):
        control_dict = self._get_control_dict(action)

        mg_obs, mg_reward, mg_done, mg_info = self.mg.run(
            control_dict,
            normalized=True
        )

        raw_reward = float(mg_reward)
        cost = -raw_reward

        # Reward normalizada
        reward = raw_reward / self.reward_scale_C

        # Penalización suave si SoC demasiado bajo
        current_soc = self._get_current_soc()
        if current_soc < self.low_soc_threshold:
            soc_deficit_ratio = (self.low_soc_threshold - current_soc) / self.low_soc_threshold
            reward -= self.low_soc_penalty * soc_deficit_ratio

        self.current_step += 1

        terminated = bool(mg_done)
        truncated = bool(self.current_step >= self.horizon)

        if not (terminated or truncated):
            obs = self._get_obs()
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        info = {
            "cost": cost,
            "mg_reward": raw_reward,
            "reward_normalized": reward,
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