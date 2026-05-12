"""
Gymnasium-compatible LeKiwi depth navigation environment.

Full observation space, reward shaping designed to prevent spin/stall
artifacts, and comprehensive termination conditions.

This is the PRIMARY training environment for the paper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sim_mujoco.envs.lekiwi_mujoco_env import LeKiwiMujocoEnv
from sim_mujoco.controllers.omni_kinematics import OmniKinematicsConfig
from sim_mujoco.controllers.velocity_controller import VelocityControllerConfig
from sim_mujoco.sensors.ray_scan_sensor import RayScanConfig
from sim_mujoco.wrappers.action_smoothing import ActionSmoothingWrapper
from sim_mujoco.wrappers.domain_randomization import (
    DomainRandConfig,
    DomainRandomizationWrapper,
)


@dataclass
class EnvConfig:
    """Configuration for LeKiwiDepthScanEnv."""
    world_xml: str = "lab_empty.xml"
    scan_bins: int = 64
    scan_fov_deg: float = 90.0
    scan_min_range: float = 0.15
    scan_max_range: float = 5.0
    room_w: float = 5.0
    room_h: float = 6.0
    goal_tolerance: float = 0.3
    max_episode_steps: int = 600
    dt: float = 0.05
    # Reward weights
    goal_reward: float = 20.0
    collision_penalty: float = 10.0
    timeout_penalty: float = 0.0
    progress_weight: float = 1.0
    clearance_weight: float = 0.2
    smoothness_weight: float = 0.05
    spin_penalty_weight: float = 0.5
    stagnation_penalty_weight: float = 0.3
    forward_weight: float = 0.3
    # Start/goal pairs
    start_goal_pairs: list[tuple] | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "EnvConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class LeKiwiDepthScanEnv(gym.Env):
    """Gymnasium environment for LeKiwi depth-based navigation in MuJoCo.

    Observation (72-D):
      - 64-D normalized scan [0, 1]
      - 2-D relative goal vector (dx, dy) normalized by room diagonal
      - 2-D goal heading (cos, sin)
      - 3-D current velocity (vx, vy, omega_norm)
      - 1-D dist_to_goal normalized

    Action (3-D):
      - vx (m/s) in [-max_vx, max_vx]
      - vy (m/s) in [-max_vy, max_vy]
      - omega (deg/s) in [-max_omega, max_omega]

    Reward is shaped to prevent spinning and stagnation.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        config: EnvConfig | None = None,
        render_mode: str = "rgb_array",
    ) -> None:
        super().__init__()
        self.cfg = config or EnvConfig()

        # Build MuJoCo backend
        self._mj = LeKiwiMujocoEnv(
            world_xml=self.cfg.world_xml,
            kinematics_config=OmniKinematicsConfig(),
            velocity_config=VelocityControllerConfig(dt=self.cfg.dt),
            scan_config=RayScanConfig(
                num_bins=self.cfg.scan_bins,
                fov_deg=self.cfg.scan_fov_deg,
                min_range=self.cfg.scan_min_range,
                max_range=self.cfg.scan_max_range,
            ),
            dt=self.cfg.dt,
            render_mode=render_mode,
        )

        # Observation space
        obs_dim = self.cfg.scan_bins + 2 + 2 + 3 + 1  # 64 + 2 + 2 + 3 + 1 = 72
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # Action: vx, vy, omega_deg
        self.action_space = spaces.Box(
            low=np.array([-0.3, -0.3, -90.0], dtype=np.float32),
            high=np.array([0.3, 0.3, 90.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Start-goal pairs for curriculum
        self._sg_pairs = self.cfg.start_goal_pairs or [
            ((0.5, 1.5), (4.0, 1.5)),
            ((0.5, 0.5), (4.0, 2.5)),
        ]
        self._sg_idx = 0
        self._step_count = 0
        self._prev_action: Optional[np.ndarray] = None
        self._prev_pos: Optional[np.ndarray] = None
        self._spin_steps: int = 0
        self._stagnation_steps: int = 0
        self._path_length: float = 0.0
        self._rng = np.random.RandomState()

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.RandomState(seed)
            self._mj.scan_sensor.seed(seed)

        # Cycle through start-goal pairs
        sg = self._sg_pairs[self._sg_idx % len(self._sg_pairs)]
        self._sg_idx += 1
        start, goal = sg
        self._mj.reset(start_xy=start, goal_xy=goal)

        self._step_count = 0
        self._prev_action = None
        self._prev_pos = self._mj.robot_pos[:2].copy()
        self._spin_steps = 0
        self._stagnation_steps = 0
        self._path_length = 0.0

        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._step_count += 1
        action = np.asarray(action, dtype=np.float32)

        # Step physics
        self._mj.step_physics(action)

        # Observations
        obs = self._get_obs()

        # Collision & bounds
        collision = self._mj.check_collision()
        oob = self._mj.check_out_of_bounds(self.cfg.room_w, self.cfg.room_h)

        # Goal
        reached_goal = self._mj.dist_to_goal < self.cfg.goal_tolerance
        timed_out = self._step_count >= self.cfg.max_episode_steps

        # Spin detection
        ang_vel = abs(float(self._mj.robot_ang_vel[2]))
        if ang_vel > np.deg2rad(30.0):  # spinning faster than 30 deg/s
            self._spin_steps += 1
        else:
            self._spin_steps = max(0, self._spin_steps - 1)

        # Stagnation detection
        current_pos = self._mj.robot_pos[:2]
        displacement = float(np.linalg.norm(current_pos - self._prev_pos))
        self._path_length += displacement
        if displacement < 0.01:  # less than 1cm per step
            self._stagnation_steps += 1
        else:
            self._stagnation_steps = max(0, self._stagnation_steps - 1)
        self._prev_pos = current_pos.copy()

        # Reward
        reward = self._compute_reward(action, collision, reached_goal, timed_out)

        terminated = bool(collision or reached_goal or oob)
        truncated = bool(timed_out)

        info = {
            "success": bool(reached_goal),
            "collision": bool(collision),
            "timeout": bool(timed_out),
            "out_of_bounds": bool(oob),
            "dist_to_goal": float(self._mj.dist_to_goal),
            "min_scan": float(obs[:self.cfg.scan_bins].min() * self.cfg.scan_max_range),
            "path_length": float(self._path_length),
            "spin_count": int(self._spin_steps),
            "stagnation_steps": int(self._stagnation_steps),
            "step": self._step_count,
        }
        self._prev_action = action.copy()

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        scan_norm, _ = self._mj.get_scan()

        # Relative goal vector (normalized by room diagonal)
        goal_vec = self._mj.goal_pos[:2] - self._mj.robot_pos[:2]
        diag = np.sqrt(self.cfg.room_w**2 + self.cfg.room_h**2)
        goal_vec_norm = goal_vec / diag

        # Goal heading
        goal_angle = math.atan2(goal_vec[1], goal_vec[0]) - self._mj.robot_theta
        goal_cos = math.cos(goal_angle)
        goal_sin = math.sin(goal_angle)

        # Velocity
        lin = self._mj.robot_lin_vel
        ang = self._mj.robot_ang_vel
        vx = lin[0] / 0.3
        vy = lin[1] / 0.3
        omega_norm = ang[2] / np.deg2rad(90.0)

        # Distance (normalized)
        dist_norm = self._mj.dist_to_goal / diag

        return np.concatenate([
            scan_norm,
            goal_vec_norm,
            np.array([goal_cos, goal_sin], dtype=np.float32),
            np.array([vx, vy, omega_norm], dtype=np.float32),
            np.array([dist_norm], dtype=np.float32),
        ]).astype(np.float32)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        action: np.ndarray,
        collision: bool,
        reached_goal: bool,
        timed_out: bool,
    ) -> float:
        if reached_goal:
            return self.cfg.goal_reward
        if collision:
            return -self.cfg.collision_penalty
        if timed_out:
            return -self.cfg.timeout_penalty

        reward = 0.0

        # --- Progress reward ---
        prev_dist = float(np.linalg.norm(self._prev_pos - self._mj.goal_pos[:2]))
        current_dist = self._mj.dist_to_goal
        progress = prev_dist - current_dist
        reward += self.cfg.progress_weight * progress

        # --- Clearance reward ---
        _, scan_m = self._mj.get_scan()
        min_scan = float(scan_m.min())
        reward -= self.cfg.clearance_weight * math.exp(-min_scan / 0.2)

        # --- Smoothness penalty ---
        if self._prev_action is not None:
            action_diff = np.sum((action - self._prev_action) ** 2)
            reward -= self.cfg.smoothness_weight * action_diff

        # --- Spin penalty ---
        if self._spin_steps > 5:
            reward -= self.cfg.spin_penalty_weight * (self._spin_steps / 10.0)

        # --- Stagnation penalty ---
        if self._stagnation_steps > 10:
            reward -= self.cfg.stagnation_penalty_weight * (self._stagnation_steps / 10.0)

        # --- Forward progress / goal alignment ---
        robot_dir = np.array([math.cos(self._mj.robot_theta), math.sin(self._mj.robot_theta)])
        goal_dir = self._mj.goal_pos[:2] - self._mj.robot_pos[:2]
        goal_dir_norm = goal_dir / (np.linalg.norm(goal_dir) + 1e-8)
        alignment = float(np.dot(robot_dir, goal_dir_norm))
        reward += self.cfg.forward_weight * alignment

        return float(reward)

    # ------------------------------------------------------------------
    # Render / Close
    # ------------------------------------------------------------------

    def render(self) -> Optional[np.ndarray]:
        return self._mj.render()

    def close(self) -> None:
        self._mj.close()

    # ------------------------------------------------------------------
    # Domain randomization
    # ------------------------------------------------------------------

    def _apply_domain_rand(self, dr_cfg: DomainRandConfig, rng: np.random.RandomState) -> None:
        self._mj._apply_domain_rand(dr_cfg, rng)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def seed(self, seed: int | None = None) -> list:
        if seed is not None:
            self._rng = np.random.RandomState(seed)
            self._mj.scan_sensor.seed(seed)
        return [seed] if seed is not None else []

    @property
    def num_scan_bins(self) -> int:
        return self.cfg.scan_bins

    @property
    def scan_range(self) -> float:
        return self.cfg.scan_max_range


# ── Factory ───────────────────────────────────────────────────────────────

def make_env(
    world_xml: str = "lab_empty.xml",
    scan_bins: int = 64,
    max_steps: int = 600,
    render_mode: str = "rgb_array",
    apply_dr: bool = False,
) -> LeKiwiDepthScanEnv:
    """Create a LeKiwiDepthScanEnv with specified world.

    Args:
        world_xml: world filename under sim_mujoco/worlds/.
        scan_bins: 32, 64, or 128 scan bins.
        max_steps: max episode steps.
        render_mode: "human" or "rgb_array".
        apply_dr: enable domain randomization.

    Returns:
        Configured environment.
    """
    config = EnvConfig(
        world_xml=world_xml,
        scan_bins=scan_bins,
        max_episode_steps=max_steps,
    )
    env = LeKiwiDepthScanEnv(config=config, render_mode=render_mode)

    # Wrap with action smoothing
    env = ActionSmoothingWrapper(env)

    # Optionally add domain randomization
    if apply_dr:
        dr_cfg = DomainRandConfig(enable=True)
        env = DomainRandomizationWrapper(env, dr_cfg)

    return env
