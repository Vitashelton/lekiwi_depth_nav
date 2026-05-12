"""
Omnidirectional kinematics for the LeKiwi 3-wheel base.

Converts (vx, vy, omega) velocity commands to individual wheel angular
velocities using the standard inverse kinematics matrix.

Wheel mounting angles: [150°, -90°, 30°] (lerobot convention).
omega is in deg/s (lerobot convention).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OmniKinematicsConfig:
    base_radius: float = 0.125     # center-to-wheel distance (m)
    wheel_radius: float = 0.05      # wheel radius (m)
    max_linear_vel: float = 0.3     # m/s
    max_angular_vel: float = 90.0   # deg/s (lerobot convention)
    wheel_angles_deg: tuple[float, ...] = (150.0, -90.0, 30.0)
    # Offset for lerobot convention (subtract 90° from standard)
    angle_offset_deg: float = -90.0

    @classmethod
    def from_dict(cls, d: dict) -> "OmniKinematicsConfig":
        return cls(
            base_radius=d.get("base_radius", 0.125),
            wheel_radius=d.get("wheel_radius", 0.05),
            max_linear_vel=d.get("max_linear_vel", 0.3),
            max_angular_vel=d.get("max_angular_vel", 90.0),
            wheel_angles_deg=tuple(d.get("wheel_angles_deg", (150.0, -90.0, 30.0))),
            angle_offset_deg=d.get("angle_offset_deg", -90.0),
        )


class OmniKinematics:
    """Forward and inverse kinematics for the LeKiwi 3-omniwheel base."""

    def __init__(self, config: OmniKinematicsConfig) -> None:
        self.cfg = config
        self._M_inv: np.ndarray = self._build_inverse_matrix()

    def _build_inverse_matrix(self) -> np.ndarray:
        """Build the 3×3 inverse kinematics matrix.

        M[i] = [cos(αᵢ), sin(αᵢ), base_radius]
        wheel_vel = M_inv @ [vx, vy, omega_rad]
        """
        R = self.cfg.base_radius
        r = self.cfg.wheel_radius
        angles_rad = np.deg2rad([
            a + self.cfg.angle_offset_deg for a in self.cfg.wheel_angles_deg
        ])

        M = np.zeros((3, 3))
        for i, alpha in enumerate(angles_rad):
            M[i, 0] = np.cos(alpha)
            M[i, 1] = np.sin(alpha)
            M[i, 2] = R

        # Scale by wheel radius
        M_inv = np.linalg.pinv(M) / r
        return M_inv.astype(np.float32)

    def inverse(self, vx: float, vy: float, omega_deg: float) -> np.ndarray:
        """Convert (vx, vy, omega_deg) to wheel angular velocities (rad/s).

        Args:
            vx: forward velocity (m/s).
            vy: lateral velocity (m/s).
            omega_deg: angular velocity (deg/s, lerobot convention).

        Returns:
            (3,) wheel velocities in rad/s.
        """
        omega_rad = np.deg2rad(omega_deg)
        cmd = np.array([vx, vy, omega_rad], dtype=np.float32)
        return self._M_inv @ cmd

    def forward(self, wheel_vels: np.ndarray) -> tuple[float, float, float]:
        """Convert wheel angular velocities to (vx, vy, omega_deg)."""
        # Not strictly needed for control but useful for odometry
        R = self.cfg.base_radius
        r = self.cfg.wheel_radius
        angles_rad = np.deg2rad([
            a + self.cfg.angle_offset_deg for a in self.cfg.wheel_angles_deg
        ])

        # Approximate forward kinematics
        sum_cos = sum(np.cos(a) for a in angles_rad)
        sum_sin = sum(np.sin(a) for a in angles_rad)
        vx = r * np.dot(wheel_vels, [np.cos(a) for a in angles_rad]) / max(sum_cos, 0.001)
        vy = r * np.dot(wheel_vels, [np.sin(a) for a in angles_rad]) / max(sum_sin, 0.001)
        omega_rad = r * np.dot(wheel_vels, np.ones(3)) / (3 * R)
        omega_deg = np.rad2deg(omega_rad)
        return float(vx), float(vy), float(omega_deg)

    def clip_action(self, action: np.ndarray) -> np.ndarray:
        """Clip (vx, vy, omega_deg) to valid ranges."""
        action[0] = np.clip(action[0], -self.cfg.max_linear_vel, self.cfg.max_linear_vel)
        action[1] = np.clip(action[1], -self.cfg.max_linear_vel, self.cfg.max_linear_vel)
        action[2] = np.clip(action[2], -self.cfg.max_angular_vel, self.cfg.max_angular_vel)
        return action
