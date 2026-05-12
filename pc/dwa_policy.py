"""
Simple Dynamic Window Approach (DWA) local planner for omnidirectional robot.

Outputs (vx [m/s], vy [m/s], omega [deg/s]) matching lerobot convention.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class DWAConfig:
    max_linear_vel: float = 0.3      # m/s
    max_angular_vel: float = 90.0    # deg/s
    linear_accel: float = 0.5        # m/s^2
    angular_accel: float = 180.0     # deg/s^2
    dt: float = 0.1                  # prediction time step (s)
    predict_steps: int = 15
    heading_weight: float = 0.3
    clearance_weight: float = 1.0
    velocity_weight: float = 0.1
    obstacle_cost_gain: float = 1.0
    num_samples: int = 50


class DWAPlanner:
    """
    DWA local planner for omnidirectional robot with 64-D scan input.

    The omega values are in deg/s throughout to match lerobot convention.
    """

    def __init__(
        self,
        config: DWAConfig,
        scan_bins: int = 64,
        fov_deg: float = 90.0,
    ):
        self.cfg = config
        half_fov = np.deg2rad(fov_deg / 2)
        self._bin_angles = np.linspace(-half_fov, half_fov, scan_bins)
        self._prev_vx = 0.0
        self._prev_vy = 0.0
        self._prev_omega = 0.0  # deg/s

    def __call__(
        self, scan_m: np.ndarray, goal_heading: float = 0.0
    ) -> tuple[float, float, float]:
        """
        Args:
            scan_m: metric scan (N,) in meters.
            goal_heading: desired heading (rad).

        Returns:
            (vx [m/s], vy [m/s], omega [deg/s])
        """
        # --- Dynamic window: feasible velocities ---
        vx_min = max(-self.cfg.max_linear_vel,
                     self._prev_vx - self.cfg.linear_accel * self.cfg.dt)
        vx_max = min(self.cfg.max_linear_vel,
                     self._prev_vx + self.cfg.linear_accel * self.cfg.dt)
        vy_min = max(-self.cfg.max_linear_vel,
                     self._prev_vy - self.cfg.linear_accel * self.cfg.dt)
        vy_max = min(self.cfg.max_linear_vel,
                     self._prev_vy + self.cfg.linear_accel * self.cfg.dt)
        omega_min = max(-self.cfg.max_angular_vel,
                        self._prev_omega - self.cfg.angular_accel * self.cfg.dt)
        omega_max = min(self.cfg.max_angular_vel,
                        self._prev_omega + self.cfg.angular_accel * self.cfg.dt)

        # --- Sample candidates ---
        best_score = float("-inf")
        best_vx, best_vy, best_omega = 0.0, 0.0, 0.0

        rng = np.random.RandomState()
        for _ in range(self.cfg.num_samples):
            vx = rng.uniform(vx_min, vx_max)
            vy = rng.uniform(vy_min, vy_max)
            omega = rng.uniform(omega_min, omega_max)  # deg/s

            score = self._evaluate_trajectory(vx, vy, omega, scan_m, goal_heading)
            if score > best_score:
                best_score = score
                best_vx, best_vy, best_omega = vx, vy, omega

        self._prev_vx = best_vx
        self._prev_vy = best_vy
        self._prev_omega = best_omega
        return best_vx, best_vy, best_omega

    def _evaluate_trajectory(
        self,
        vx: float,
        vy: float,
        omega_deg: float,
        scan_m: np.ndarray,
        goal_heading: float,
    ) -> float:
        """Score a velocity candidate by forward-simulating its trajectory."""
        x, y, theta = 0.0, 0.0, 0.0
        min_clearance = float("inf")
        dt = self.cfg.dt
        omega_rad = np.deg2rad(omega_deg)

        for _step in range(1, self.cfg.predict_steps + 1):
            x += (vx * np.cos(theta) - vy * np.sin(theta)) * dt
            y += (vx * np.sin(theta) + vy * np.cos(theta)) * dt
            theta += omega_rad * dt

            dist_to_obs = self._check_clearance(x, y, theta, scan_m)
            min_clearance = min(min_clearance, dist_to_obs)

            if dist_to_obs < 0.1:  # collision
                return -1e6

        # Heading score
        heading_error = abs(theta - goal_heading)
        heading_error = min(heading_error, 2 * np.pi - heading_error)
        heading_score = 1.0 - heading_error / np.pi

        # Clearance score
        clearance_score = np.clip(min_clearance / 1.0, 0.0, 1.0)

        # Velocity score
        vel_score = (abs(vx) + abs(vy)) / (2 * self.cfg.max_linear_vel + 1e-6)

        return (
            self.cfg.heading_weight * heading_score
            + self.cfg.clearance_weight * clearance_score
            + self.cfg.velocity_weight * vel_score
        )

    def _check_clearance(
        self, x: float, y: float, theta: float, scan_m: np.ndarray
    ) -> float:
        """Estimate obstacle clearance at predicted pose."""
        target_angle = np.arctan2(y, x) if abs(x) + abs(y) > 0.01 else 0.0
        angle_diff = np.abs(self._bin_angles - target_angle)
        closest_bin = int(np.argmin(angle_diff))
        dist = scan_m[closest_bin] - np.sqrt(x**2 + y**2)
        return max(0.0, dist)
