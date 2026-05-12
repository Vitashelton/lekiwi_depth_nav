"""
Velocity controller with first-order low-pass filter and acceleration limits.

Models real motor dynamics: the commanded velocity is not achieved
instantaneously, preventing unrealistic step-response behavior that
causes sim-to-real gap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VelocityControllerConfig:
    max_linear_accel: float = 0.5     # m/s²
    max_angular_accel: float = 180.0  # deg/s²
    lowpass_alpha: float = 0.3        # 1.0 = instant, 0.0 = frozen
    dt: float = 0.05                  # control period (s)

    @classmethod
    def from_dict(cls, d: dict) -> "VelocityControllerConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class VelocityController:
    """Filters and rate-limits velocity commands to simulate actuator dynamics.

    Applies in order:
      1. Acceleration clamping (prevents instant speed changes)
      2. First-order low-pass filter (smooths output)
    """

    def __init__(self, config: VelocityControllerConfig) -> None:
        self.cfg = config
        self._current_vx: float = 0.0
        self._current_vy: float = 0.0
        self._current_omega: float = 0.0

    def step(self, target: np.ndarray) -> np.ndarray:
        """Apply dynamics and return the actual velocity achieved this step.

        Args:
            target: (3,) desired [vx, vy, omega_deg].

        Returns:
            (3,) actual achieved velocity.
        """
        dt = self.cfg.dt
        target_vx, target_vy, target_omega = float(target[0]), float(target[1]), float(target[2])

        # --- Acceleration clamping ---
        max_dv = self.cfg.max_linear_accel * dt
        max_dw = self.cfg.max_angular_accel * dt

        dvx = np.clip(target_vx - self._current_vx, -max_dv, max_dv)
        dvy = np.clip(target_vy - self._current_vy, -max_dv, max_dv)
        dw = np.clip(target_omega - self._current_omega, -max_dw, max_dw)

        desired_vx = self._current_vx + dvx
        desired_vy = self._current_vy + dvy
        desired_omega = self._current_omega + dw

        # --- First-order low-pass filter ---
        alpha = self.cfg.lowpass_alpha
        self._current_vx = alpha * desired_vx + (1 - alpha) * self._current_vx
        self._current_vy = alpha * desired_vy + (1 - alpha) * self._current_vy
        self._current_omega = alpha * desired_omega + (1 - alpha) * self._current_omega

        return np.array([self._current_vx, self._current_vy, self._current_omega],
                        dtype=np.float32)

    def reset(self) -> None:
        self._current_vx = 0.0
        self._current_vy = 0.0
        self._current_omega = 0.0
