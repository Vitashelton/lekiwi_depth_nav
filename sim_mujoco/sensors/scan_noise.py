"""
Configurable scan noise models for pseudo-LiDAR / depth sensors.

Supports: Gaussian range noise, dropout (return 0), max-range missing
returns, random angular bias.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ScanNoiseConfig:
    """Noise configuration for ray-cast scans.

    All probabilities are per-ray, per-step.
    """
    gaussian_std: float = 0.02          # std of additive Gaussian noise (m)
    dropout_prob: float = 0.01          # probability of returning 0
    max_range_miss_prob: float = 0.02   # probability of returning max_range
    angular_bias_std: float = 0.0       # std of angular bias per ray (rad)
    enable: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "ScanNoiseConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class ScanNoiseModel:
    """Applies noise to a clean ray-cast scan array."""

    def __init__(self, config: ScanNoiseConfig, seed: int = 0) -> None:
        self.cfg = config
        self._rng = np.random.RandomState(seed)

    def apply(self, scan_m: np.ndarray, max_range: float = 5.0) -> np.ndarray:
        """Apply noise in-place to a metric scan array.

        Args:
            scan_m: (N,) clean metric ranges in meters.
            max_range: sensor maximum range for missing-return simulation.

        Returns:
            Noisy scan_m of same shape.
        """
        if not self.cfg.enable:
            return scan_m

        out = scan_m.copy()

        # 1. Gaussian additive noise
        if self.cfg.gaussian_std > 0:
            out += self._rng.normal(0, self.cfg.gaussian_std, size=out.shape)
            out = np.clip(out, 0.0, max_range)

        # 2. Dropout (return 0 = blind spot)
        if self.cfg.dropout_prob > 0:
            mask = self._rng.rand(len(out)) < self.cfg.dropout_prob
            out[mask] = 0.0

        # 3. Max-range missing return
        if self.cfg.max_range_miss_prob > 0:
            mask = self._rng.rand(len(out)) < self.cfg.max_range_miss_prob
            out[mask] = max_range

        return out

    def seed(self, seed: int) -> None:
        self._rng = np.random.RandomState(seed)


def sector_min_distance(
    scan_m: np.ndarray, sector: str = "front", fov_deg: float = 90.0
) -> float:
    """Return minimum distance in a sector of the scan.

    Args:
        scan_m: (N,) metric ranges.
        sector: "front" (center 1/3), "left" (left 1/3), "right" (right 1/3).
        fov_deg: full FOV in degrees.

    Returns:
        Minimum range in that sector.
    """
    N = len(scan_m)
    third = N // 3
    if sector == "front":
        start, end = third, 2 * third
    elif sector == "left":
        start, end = 0, third
    elif sector == "right":
        start, end = 2 * third, N
    else:
        start, end = 0, N
    return float(scan_m[start:end].min())
