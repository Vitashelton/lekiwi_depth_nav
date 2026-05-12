"""
Rule-based obstacle avoidance policy.

Strategy:
  1. If all scan readings are far enough, move forward toward goal.
  2. If obstacles appear in the center, compute "repulsion" direction from
     the angular distribution of close-range readings.
  3. If obstacles are too close on all sides, rotate in place.

Outputs (vx [m/s], vy [m/s], omega [deg/s]) matching lerobot convention.
"""

from __future__ import annotations

import numpy as np


class RulePolicy:
    """
    Simple rule-based policy using pseudo-LiDAR scan.

    The policy takes a 64-D scan and a goal heading, and outputs
    (vx, vy, omega) velocity commands with omega in deg/s.
    """

    def __init__(
        self,
        safe_distance: float = 0.3,
        danger_distance: float = 0.2,
        forward_speed: float = 0.2,
        turn_gain: float = 30.0,       # deg/s per rad of heading error
        scan_bins: int = 64,
        fov_deg: float = 90.0,
        max_linear_vel: float = 0.3,
        max_angular_vel: float = 90.0,  # deg/s
    ):
        self.safe_distance = safe_distance
        self.danger_distance = danger_distance
        self.forward_speed = forward_speed
        self.turn_gain = turn_gain
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel
        self.scan_bins = scan_bins

        # Precompute bin center angles (radians)
        half_fov = np.deg2rad(fov_deg / 2)
        self._bin_angles = np.linspace(-half_fov, half_fov, scan_bins)

    def __call__(
        self, scan_m: np.ndarray, goal_heading: float = 0.0
    ) -> tuple[float, float, float]:
        """
        Args:
            scan_m: metric scan array, shape (N,), in meters.
            goal_heading: desired heading relative to robot (rad, 0=forward).

        Returns:
            (vx [m/s], vy [m/s], omega [deg/s])
        """
        # --- Danger zone check ---
        danger_mask = scan_m < self.danger_distance
        safe_mask = scan_m > self.safe_distance

        if not np.any(danger_mask) and np.all(safe_mask):
            # Clear path: go forward, turn toward goal
            vx = self.forward_speed
            vy = 0.0
            omega = self.turn_gain * goal_heading
            omega = max(-self.max_angular_vel, min(self.max_angular_vel, omega))
            return vx, vy, omega

        # --- Obstacle avoidance ---
        # Weight each direction by inverse distance
        inv_dist = 1.0 / np.maximum(scan_m, 0.05)

        # Angular repulsion vector
        rx = np.sum(inv_dist * np.cos(self._bin_angles))
        ry = np.sum(inv_dist * np.sin(self._bin_angles))

        # Normalize repulsion magnitude
        rep_mag = np.sqrt(rx**2 + ry**2)
        if rep_mag > 1e-6:
            rx /= rep_mag
            ry /= rep_mag

        # Blend: move away from obstacles, but also toward goal
        goal_x = np.cos(goal_heading)
        goal_y = np.sin(goal_heading)

        # Weight: close obstacles dominate
        min_dist = scan_m.min()
        obs_weight = np.clip(1.0 - min_dist / self.safe_distance, 0.0, 1.0)
        goal_weight = 1.0 - obs_weight

        dir_x = goal_weight * goal_x - obs_weight * rx
        dir_y = goal_weight * goal_y - obs_weight * ry
        dir_len = np.sqrt(dir_x**2 + dir_y**2)
        if dir_len > 1e-6:
            dir_x /= dir_len
            dir_y /= dir_len

        # Speed scales with clearance
        speed = self.forward_speed * np.clip(min_dist / self.safe_distance, 0.1, 1.0)
        vx = speed * dir_x
        vy = speed * dir_y

        # Angular velocity: align heading with desired direction
        desired_angle = np.arctan2(dir_y, dir_x)
        omega = self.turn_gain * desired_angle
        omega = max(-self.max_angular_vel, min(self.max_angular_vel, omega))

        return vx, vy, omega
