"""
Domain randomization wrapper for sim-to-real transfer.

Randomizes: obstacle positions, friction coefficients, scan noise,
mass properties, and start/goal perturbations each episode.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np


@dataclass
class DomainRandConfig:
    obstacle_pos_std: float = 0.1      # ±10cm position jitter (m)
    friction_range: tuple[float, float] = (0.4, 0.9)
    mass_scale_range: tuple[float, float] = (0.8, 1.2)
    scan_noise_scale: float = 1.0
    start_goal_jitter: float = 0.2      # m
    enable: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "DomainRandConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class DomainRandomizationWrapper(gym.Wrapper):
    """Randomizes environment properties each episode for robust learning."""

    def __init__(self, env: gym.Env, config: DomainRandConfig) -> None:
        super().__init__(env)
        self.dr_cfg = config
        self._rng = np.random.RandomState()

    def reset(self, **kwargs):
        if self.dr_cfg.enable and hasattr(self.unwrapped, "_apply_domain_rand"):
            self.unwrapped._apply_domain_rand(self.dr_cfg, self._rng)
        return self.env.reset(**kwargs)
