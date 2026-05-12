"""
MuJoCo ray-cast pseudo-LiDAR sensor.

Emulates the D435i depth-to-scan pipeline by casting N rays from the
camera_mount site and returning metric + normalized scan arrays.

Output format matches raspberry_pi/depth_to_scan.py so that trained
models can be deployed directly on real hardware.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import mujoco
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False

from sim_mujoco.sensors.scan_noise import ScanNoiseConfig, ScanNoiseModel


@dataclass
class RayScanConfig:
    """Configuration for the ray-cast pseudo-LiDAR sensor."""
    num_bins: int = 64
    fov_deg: float = 90.0
    min_range: float = 0.15
    max_range: float = 5.0
    ray_height: float = 0.10           # height of ray origin (m)
    noise: ScanNoiseConfig | None = None

    def __post_init__(self) -> None:
        if self.noise is None:
            self.noise = ScanNoiseConfig()

    @classmethod
    def from_dict(cls, d: dict) -> "RayScanConfig":
        noise_cfg = ScanNoiseConfig.from_dict(d.get("noise", {}))
        return cls(
            num_bins=d.get("num_bins", 64),
            fov_deg=d.get("fov_deg", 90.0),
            min_range=d.get("min_range", 0.15),
            max_range=d.get("max_range", 5.0),
            ray_height=d.get("ray_height", 0.10),
            noise=noise_cfg,
        )


class RayScanSensor:
    """Casts N rays from a MuJoCo site and returns pseudo-LiDAR scans.

    Output is compatible with DepthToScan from raspberry_pi/depth_to_scan.py:
      - scan_m:  (N,) metric ranges in meters, [min_range, max_range]
      - scan_norm: (N,) normalized ranges in [0, 1]
    """

    def __init__(
        self,
        model: "mujoco.MjModel",
        data: "mujoco.MjData",
        config: RayScanConfig,
        site_name: str = "camera_mount",
        seed: int = 0,
    ) -> None:
        if not HAS_MUJOCO:
            raise ImportError("mujoco is required for RayScanSensor")

        self._model = model
        self._data = data
        self.cfg = config
        self._site_id = model.site(site_name).id

        # Precompute ray directions in the site's local frame
        half_fov = np.deg2rad(config.fov_deg / 2.0)
        angles = np.linspace(-half_fov, half_fov, config.num_bins)
        self._ray_dirs = np.column_stack([
            np.cos(angles),    # x (forward)
            np.sin(angles),    # y (lateral)
            np.zeros_like(angles),  # z (horizontal rays)
        ])

        self._noise_model = ScanNoiseModel(config.noise, seed=seed)

    def get_scan(self) -> tuple[np.ndarray, np.ndarray]:
        """Cast rays and return (scan_norm, scan_m).

        Uses MuJoCo mj_ray to cast from the camera_mount site.
        Each ray is checked against all geoms in the scene.
        """
        # Get ray origin in world frame
        site_pos = self._data.site_xpos[self._site_id].copy()
        site_mat = self._data.site_xmat[self._site_id].reshape(3, 3)

        scan_m = np.full(self.cfg.num_bins, self.cfg.max_range, dtype=np.float32)

        for i in range(self.cfg.num_bins):
            # Transform ray direction from local to world frame
            local_dir = self._ray_dirs[i]
            world_dir = site_mat @ local_dir

            # Ray origin slightly above ground to avoid self-collision
            origin = site_pos + np.array([0, 0, self.cfg.ray_height])

            # Cast ray using MuJoCo
            geom_id = -1
            dist = self.cfg.max_range
            try:
                geom_id, dist = mujoco.mj_ray(
                    self._model, self._data,
                    origin, world_dir,
                    geom_group=None,      # check all geom groups
                    flg_static=True,       # include static geoms
                    bodyexclude=-1,        # exclude no bodies
                )
            except Exception:
                pass

            if geom_id >= 0 and dist > 0:
                scan_m[i] = float(dist)
            else:
                scan_m[i] = self.cfg.max_range

        # Clip to valid range
        scan_m = np.clip(scan_m, self.cfg.min_range, self.cfg.max_range)

        # Apply noise
        scan_m = self._noise_model.apply(scan_m, self.cfg.max_range)

        # Normalize
        scan_norm = (scan_m - self.cfg.min_range) / (
            self.cfg.max_range - self.cfg.min_range
        )

        return scan_norm.astype(np.float32), scan_m.astype(np.float32)

    def seed(self, seed: int) -> None:
        self._noise_model.seed(seed)

    @property
    def bin_angles_deg(self) -> np.ndarray:
        """Center angles of each bin in degrees."""
        half_fov = self.cfg.fov_deg / 2.0
        return np.linspace(-half_fov, half_fov, self.cfg.num_bins)
