"""
Simple 2D navigation environment for LeKiwi omnidirectional mobile robot.

Follows the Gymnasium API. The robot is modeled as a holonomic point mass
that can independently control vx, vy, and omega.

Observation space (67-D):
  - 64-D ray scan (distances in meters, normalized to [0, 1])
  - vx (m/s)
  - vy (m/s)
  - goal_heading (radians relative to robot frame)

Action space (3-D):
  - vx (m/s, in [-max_vx, max_vx])
  - vy (m/s, in [-max_vy, max_vy])
  - omega (deg/s, in [-max_omega, max_omega], lerobot convention)
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class Simple2DNavEnv(gym.Env):
    """
    2D holonomic navigation environment with ray-based sensing.
    """

    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(
        self,
        # Map
        map_size: float = 10.0,         # meters, square arena
        num_obstacles: int = 8,
        obstacle_min_r: float = 0.2,
        obstacle_max_r: float = 0.6,
        # Scan
        num_scan_bins: int = 64,
        scan_fov_deg: float = 90.0,
        scan_range: float = 5.0,
        scan_noise_std: float = 0.02,
        # Robot
        robot_radius: float = 0.15,
        max_vx: float = 0.3,
        max_vy: float = 0.3,
        max_omega: float = 90.0,       # deg/s (lerobot convention)
        dt: float = 0.05,               # control period (s)
        # Task
        goal_tolerance: float = 0.3,    # meters
        max_episode_steps: int = 1200,
        # Reward weights
        goal_reward: float = 10.0,
        collision_penalty: float = 5.0,
        progress_weight: float = 0.5,
        clearance_weight: float = 0.1,
        smoothness_weight: float = 0.01,
        timeout_penalty: float = 0.0,
        # Random seed
        seed: int | None = None,
    ):
        super().__init__()
        self.map_size = map_size
        self.num_obstacles = num_obstacles
        self.obstacle_min_r = obstacle_min_r
        self.obstacle_max_r = obstacle_max_r
        self.num_scan_bins = num_scan_bins
        self.scan_fov_deg = scan_fov_deg
        self.scan_range = scan_range
        self.scan_noise_std = scan_noise_std
        self.robot_radius = robot_radius
        self.max_vx = max_vx
        self.max_vy = max_vy
        self.max_omega = max_omega
        self.dt = dt
        self.goal_tolerance = goal_tolerance
        self._max_episode_steps = max_episode_steps

        self.goal_reward = goal_reward
        self.collision_penalty = collision_penalty
        self.progress_weight = progress_weight
        self.clearance_weight = clearance_weight
        self.smoothness_weight = smoothness_weight
        self.timeout_penalty = timeout_penalty

        # Observation: scan (N) + vx + vy + goal_heading
        obs_dim = num_scan_bins + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # Action: vx, vy, omega (deg/s)
        self.action_space = spaces.Box(
            low=np.array([-max_vx, -max_vy, -max_omega], dtype=np.float32),
            high=np.array([max_vx, max_vy, max_omega], dtype=np.float32),
            dtype=np.float32,
        )

        # Scan ray angles (rad)
        half_fov = np.deg2rad(scan_fov_deg / 2)
        self._ray_angles = np.linspace(-half_fov, half_fov, num_scan_bins)

        # State
        self._robot_x = 0.0
        self._robot_y = 0.0
        self._robot_theta = 0.0   # rad
        self._robot_vx = 0.0
        self._robot_vy = 0.0
        self._robot_omega = 0.0   # deg/s
        self._goal_x = 0.0
        self._goal_y = 0.0
        self._obstacles: list[tuple[float, float, float]] = []  # (x, y, r)
        self._step_count = 0
        self._prev_action: Optional[np.ndarray] = None

        self.rng = np.random.RandomState(seed)
        # Internal seeds for reproducibility
        self._seed_val = seed

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self._step_count = 0
        self._prev_action = None

        # Generate obstacles
        self._obstacles = self._generate_obstacles()

        # Place robot at random free location
        self._robot_x, self._robot_y, self._robot_theta = self._random_free_pose()
        self._robot_vx = 0.0
        self._robot_vy = 0.0
        self._robot_omega = 0.0

        # Place goal at random free location away from robot
        self._goal_x, self._goal_y, _ = self._random_free_pose(
            min_dist_from=self._robot_x,
            min_dist_from_y=self._robot_y,
            min_dist=2.0,
        )

        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._step_count += 1

        # Unpack action (vx, vy, omega_deg)
        vx = float(np.clip(action[0], -self.max_vx, self.max_vx))
        vy = float(np.clip(action[1], -self.max_vy, self.max_vy))
        omega_deg = float(np.clip(action[2], -self.max_omega, self.max_omega))
        omega_rad = np.deg2rad(omega_deg)

        # Integrate
        dt = self.dt
        self._robot_theta += omega_rad * dt
        self._robot_theta = math.atan2(
            math.sin(self._robot_theta), math.cos(self._robot_theta)
        )
        # Move in world frame
        cos_t = math.cos(self._robot_theta)
        sin_t = math.sin(self._robot_theta)
        self._robot_x += (vx * cos_t - vy * sin_t) * dt
        self._robot_y += (vx * sin_t + vy * cos_t) * dt
        self._robot_vx = vx
        self._robot_vy = vy
        self._robot_omega = omega_deg

        # Observation
        obs = self._get_obs()

        # Check terminal conditions
        collision = self._check_collision()
        reached_goal = self._dist_to_goal() < self.goal_tolerance
        timed_out = self._step_count >= self._max_episode_steps

        # Reward
        reward = self._compute_reward(action, collision, reached_goal, timed_out)

        terminated = collision or reached_goal
        truncated = timed_out

        info = {
            "collision": collision,
            "reached_goal": reached_goal,
            "timed_out": timed_out,
            "dist_to_goal": self._dist_to_goal(),
            "min_scan": obs[: self.num_scan_bins].min() * self.scan_range,
            "step": self._step_count,
        }
        self._prev_action = np.array([vx, vy, omega_deg])

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        scan = self._cast_rays()
        # Normalize scan
        scan_norm = np.clip(scan / self.scan_range, 0.0, 1.0)
        # Normalize velocities
        vx_norm = self._robot_vx / (self.max_vx + 1e-6)
        vy_norm = self._robot_vy / (self.max_vy + 1e-6)
        # Goal heading relative to robot
        dx = self._goal_x - self._robot_x
        dy = self._goal_y - self._robot_y
        goal_heading = math.atan2(dy, dx) - self._robot_theta
        goal_heading = math.atan2(math.sin(goal_heading), math.cos(goal_heading))
        goal_heading_norm = goal_heading / math.pi

        return np.array(
            list(scan_norm) + [vx_norm, vy_norm, goal_heading_norm],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Ray casting
    # ------------------------------------------------------------------

    def _cast_rays(self) -> np.ndarray:
        """Cast N rays and return distances to nearest obstacle/wall."""
        ranges = np.full(self.num_scan_bins, self.scan_range, dtype=np.float32)
        for i, angle in enumerate(self._ray_angles):
            ray_angle = self._robot_theta + angle
            cos_a = math.cos(ray_angle)
            sin_a = math.sin(ray_angle)
            min_dist = self.scan_range

            # Check wall boundaries
            for edge in [0.0, self.map_size]:
                if abs(cos_a) > 1e-8:
                    t = (edge - self._robot_x) / cos_a
                    if t > 0:
                        y_hit = self._robot_y + t * sin_a
                        if 0.0 <= y_hit <= self.map_size:
                            min_dist = min(min_dist, t)
                if abs(sin_a) > 1e-8:
                    t = (edge - self._robot_y) / sin_a
                    if t > 0:
                        x_hit = self._robot_x + t * cos_a
                        if 0.0 <= x_hit <= self.map_size:
                            min_dist = min(min_dist, t)

            # Check circular obstacles
            for ox, oy, orad in self._obstacles:
                dx = ox - self._robot_x
                dy = oy - self._robot_y
                # Project ray to closest point on line
                proj = dx * cos_a + dy * sin_a
                if proj < 0:
                    continue
                closest_dist_sq = dx**2 + dy**2 - proj**2
                r_eff = orad + self.robot_radius
                if closest_dist_sq < r_eff**2:
                    # Ray intersects circle
                    dist_to_obs = proj - math.sqrt(
                        max(0, r_eff**2 - closest_dist_sq)
                    )
                    if dist_to_obs < min_dist:
                        min_dist = max(0.0, dist_to_obs)

            # Add noise
            if self.scan_noise_std > 0:
                min_dist += self.rng.normal(0, self.scan_noise_std)
            ranges[i] = max(0.0, min_dist)

        return ranges

    # ------------------------------------------------------------------
    # Collision detection
    # ------------------------------------------------------------------

    def _check_collision(self) -> bool:
        # Wall collision
        r = self.robot_radius
        if (
            self._robot_x - r < 0
            or self._robot_x + r > self.map_size
            or self._robot_y - r < 0
            or self._robot_y + r > self.map_size
        ):
            return True
        # Obstacle collision
        for ox, oy, orad in self._obstacles:
            dist = math.hypot(self._robot_x - ox, self._robot_y - oy)
            if dist < r + orad:
                return True
        return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _dist_to_goal(self) -> float:
        return math.hypot(self._goal_x - self._robot_x, self._goal_y - self._robot_y)

    def _compute_reward(
        self,
        action: np.ndarray,
        collision: bool,
        reached_goal: bool,
        timed_out: bool,
    ) -> float:
        if reached_goal:
            return self.goal_reward
        if collision:
            return -self.collision_penalty
        if timed_out:
            return -self.timeout_penalty

        # Progress reward: reduction in distance to goal
        current_dist = self._dist_to_goal()
        prev_dx = self._goal_x - (self._robot_x - self._robot_vx * self.dt)
        prev_dy = self._goal_y - (self._robot_y - self._robot_vy * self.dt)
        prev_dist = math.hypot(prev_dx, prev_dy)
        progress = prev_dist - current_dist

        # Clearance reward
        scan = self._cast_rays()
        min_scan = scan.min()
        clearance_rew = -self.clearance_weight * math.exp(-min_scan / 0.2)

        # Smoothness reward
        smooth_rew = 0.0
        if self._prev_action is not None:
            smooth_rew = -self.smoothness_weight * np.sum(
                (action - self._prev_action) ** 2
            )

        return (
            self.progress_weight * progress
            + clearance_rew
            + smooth_rew
        )

    def _generate_obstacles(self) -> list[tuple[float, float, float]]:
        obstacles = []
        margin = self.robot_radius + 0.1
        for _ in range(self.num_obstacles):
            for _attempt in range(50):
                x = self.rng.uniform(margin, self.map_size - margin)
                y = self.rng.uniform(margin, self.map_size - margin)
                r = self.rng.uniform(self.obstacle_min_r, self.obstacle_max_r)
                # Check overlap with other obstacles
                ok = True
                for ox, oy, orad in obstacles:
                    if math.hypot(x - ox, y - oy) < r + orad + 0.2:
                        ok = False
                        break
                if ok:
                    obstacles.append((x, y, r))
                    break
        return obstacles

    def _random_free_pose(
        self,
        min_dist_from: float = -1.0,
        min_dist_from_y: float = -1.0,
        min_dist: float = 0.0,
    ) -> tuple[float, float, float]:
        margin = self.robot_radius + 0.3
        for _ in range(200):
            x = self.rng.uniform(margin, self.map_size - margin)
            y = self.rng.uniform(margin, self.map_size - margin)
            theta = self.rng.uniform(0, 2 * math.pi)

            if min_dist_from >= 0:
                if math.hypot(x - min_dist_from, y - min_dist_from_y) < min_dist:
                    continue

            # Check not inside obstacle
            ok = True
            for ox, oy, orad in self._obstacles:
                if math.hypot(x - ox, y - oy) < self.robot_radius + orad + 0.1:
                    ok = False
                    break
            if ok:
                return x, y, theta
        return margin, margin, 0.0  # fallback

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self):
        """Text-based render for debugging."""
        print(
            f"Step {self._step_count:4d} | "
            f"pos=({self._robot_x:.2f}, {self._robot_y:.2f}) "
            f"theta={np.rad2deg(self._robot_theta):.1f}° | "
            f"goal=({self._goal_x:.2f}, {self._goal_y:.2f}) "
            f"dist={self._dist_to_goal():.2f}m | "
            f"min_scan={self._cast_rays().min():.2f}m"
        )
