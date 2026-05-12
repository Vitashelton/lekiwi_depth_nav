"""
Base MuJoCo environment for LeKiwi omnidirectional robot.

Provides the Gymnasium-compatible foundation: loading XML worlds,
setting up the robot, sensors, controllers, and basic step logic.

This is the base class — use lekiwi_depth_scan_env.py for the
full observation/reward environment used in training.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    import mujoco
    from mujoco import MjModel, MjData, mj_name2id, mjtObj
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False

from sim_mujoco.controllers.omni_kinematics import OmniKinematics, OmniKinematicsConfig
from sim_mujoco.controllers.velocity_controller import (
    VelocityController,
    VelocityControllerConfig,
)
from sim_mujoco.sensors.ray_scan_sensor import RayScanSensor, RayScanConfig


_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
_WORLDS_DIR = Path(__file__).resolve().parent.parent / "worlds"


class LeKiwiMujocoEnv:
    """Low-level MuJoCo environment for the LeKiwi omnidirectional base.

    Manages:
      - MuJoCo model loading and stepping
      - Robot velocity control via inverse kinematics
      - Pseudo-LiDAR ray-cast sensing
      - Goal marker placement

    This is NOT a Gymnasium env — wrap with LeKiwiDepthScanEnv for the
    full Gymnasium API with observations, rewards, and termination.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        world_xml: str,
        kinematics_config: OmniKinematicsConfig | None = None,
        velocity_config: VelocityControllerConfig | None = None,
        scan_config: RayScanConfig | None = None,
        dt: float = 0.05,
        render_mode: str = "rgb_array",
    ) -> None:
        if not HAS_MUJOCO:
            raise ImportError(
                "mujoco is required. Install with: pip install mujoco"
            )

        self.dt = dt
        self.render_mode = render_mode

        # Load MuJoCo model
        if not os.path.isabs(world_xml):
            world_xml = str(_WORLDS_DIR / world_xml)
        self._model: MjModel = mujoco.MjModel.from_xml_path(world_xml)
        self._data: MjData = mujoco.MjData(self._model)

        # Override timestep
        self._model.opt.timestep = dt

        # Configs
        self.kinematics_cfg = kinematics_config or OmniKinematicsConfig()
        self.velocity_cfg = velocity_config or VelocityControllerConfig(dt=dt)
        self.scan_cfg = scan_config or RayScanConfig()

        # Kinematics & control
        self.kinematics = OmniKinematics(self.kinematics_cfg)
        self.vel_ctrl = VelocityController(self.velocity_cfg)

        # Pseudo-LiDAR sensor
        self.scan_sensor = RayScanSensor(
            self._model, self._data, self.scan_cfg, site_name="camera_mount"
        )

        # Goal
        self._goal_body_id: int = mj_name2id(self._model, mjtObj.mjOBJ_BODY, "goal_marker")
        self._start_pos: np.ndarray = self._data.qpos[:3].copy()  # x,y,z at reset
        self.goal_pos: np.ndarray = np.array([3.0, 1.5, 0.02])

        # Renderer
        self._renderer: Optional[Any] = None
        if render_mode == "rgb_array":
            self._init_renderer()

        # Collision tracking
        self._chassis_geom_id = mj_name2id(self._model, mjtObj.mjOBJ_GEOM, "chassis_body")
        self._wall_geom_names = [
            n for n in ["wall_north", "wall_south", "wall_east", "wall_west"]
            if n in {self._model.geom(i).name for i in range(self._model.ngeom)}
        ]

    def _init_renderer(self) -> None:
        try:
            self._renderer = mujoco.Renderer(self._model, 480, 480)
        except Exception:
            self._renderer = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def robot_pos(self) -> np.ndarray:
        return self._data.body("lekiwi_base").xpos.copy()

    @property
    def robot_quat(self) -> np.ndarray:
        return self._data.body("lekiwi_base").xquat.copy()

    @property
    def robot_theta(self) -> float:
        """Yaw angle in radians."""
        quat = self.robot_quat
        siny = 2 * (quat[0] * quat[3] + quat[1] * quat[2])
        cosy = 1 - 2 * (quat[2]**2 + quat[3]**2)
        return math.atan2(siny, cosy)

    @property
    def robot_lin_vel(self) -> np.ndarray:
        return self._data.body("lekiwi_base").cvel[:3].copy()

    @property
    def robot_ang_vel(self) -> np.ndarray:
        return self._data.body("lekiwi_base").cvel[3:].copy()

    @property
    def dist_to_goal(self) -> float:
        return float(np.linalg.norm(self.robot_pos[:2] - self.goal_pos[:2]))

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def set_goal(self, x: float, y: float) -> None:
        self.goal_pos = np.array([x, y, 0.02])
        self._data.mocap_pos[self._goal_body_id] = self.goal_pos
        mujoco.mj_forward(self._model, self._data)

    def set_robot_pose(self, x: float, y: float, theta_deg: float = 0.0) -> None:
        """Teleport the robot to a position (only before stepping)."""
        qpos = self._data.qpos.copy()
        qpos[0] = x
        qpos[1] = y
        qpos[2] = 0.06  # chassis height
        # Set yaw via quaternion
        half = np.deg2rad(theta_deg) / 2.0
        qpos[3] = np.cos(half)  # qw
        qpos[4:7] = 0.0
        qpos[6] = np.sin(half)  # qz
        self._data.qpos = qpos
        mujoco.mj_forward(self._model, self._data)

    def _apply_wheel_velocities(self, wheel_vels: np.ndarray) -> None:
        """Set motor velocities for the 3 wheels."""
        for i in range(3):
            actuator_id = self._model.actuator(f"motor_{i+1}").id
            self._data.ctrl[actuator_id] = float(wheel_vels[i])
            # Set velocity control mode
            self._model.actuator_gainprm[actuator_id, 0] = float(wheel_vels[i])

    def step_physics(self, action: np.ndarray) -> None:
        """Convert a (vx, vy, omega) action to wheel velocities and step MuJoCo.

        Args:
            action: (3,) array [vx (m/s), vy (m/s), omega (deg/s)].
        """
        # Clip
        action = self.kinematics.clip_action(np.asarray(action, dtype=np.float32))
        # Apply velocity dynamics
        achieved = self.vel_ctrl.step(action)
        # Inverse kinematics: (vx, vy, omega) → wheel velocities
        wheel_vels = self.kinematics.inverse(
            float(achieved[0]), float(achieved[1]), float(achieved[2])
        )
        self._apply_wheel_velocities(wheel_vels)
        mujoco.mj_step(self._model, self._data)

    # ------------------------------------------------------------------
    # Sensing
    # ------------------------------------------------------------------

    def get_scan(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (scan_norm, scan_m) from the D435i pseudo-LiDAR."""
        return self.scan_sensor.get_scan()

    def get_robot_state(self) -> dict:
        """Return a dict of robot state for logging."""
        return {
            "pos": self.robot_pos,
            "theta": self.robot_theta,
            "lin_vel": self.robot_lin_vel,
            "ang_vel": self.robot_ang_vel,
            "dist_to_goal": self.dist_to_goal,
        }

    # ------------------------------------------------------------------
    # Collision
    # ------------------------------------------------------------------

    def check_collision(self) -> bool:
        """Check if the chassis is in contact with any wall or obstacle geom."""
        for contact in self._data.contact[:self._data.ncon]:
            g1 = self._model.geom(contact.geom1).name
            g2 = self._model.geom(contact.geom2).name
            if g1 == "chassis_body" and g2 != "goal_geom" and g2 != "floor":
                return True
            if g2 == "chassis_body" and g1 != "goal_geom" and g1 != "floor":
                return True
        return False

    def check_out_of_bounds(self, room_w: float = 5.0, room_h: float = 6.0) -> bool:
        x, y = self.robot_pos[0], self.robot_pos[1]
        return x < -0.2 or x > room_w + 0.2 or y < -0.2 or y > room_h + 0.2

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "rgb_array" and self._renderer is not None:
            self._renderer.update_scene(self._data, camera="tracking")
            return self._renderer.render()
        return None

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self, start_xy: tuple[float, float] = (0.5, 1.5), goal_xy: tuple[float, float] = (4.0, 1.5)
    ) -> None:
        """Reset the simulation to initial state."""
        mujoco.mj_resetData(self._model, self._data)
        self.set_robot_pose(start_xy[0], start_xy[1], 0.0)
        self.set_goal(goal_xy[0], goal_xy[1])
        self.vel_ctrl.reset()
        mujoco.mj_forward(self._model, self._data)

    # ------------------------------------------------------------------
    # Domain randomization support
    # ------------------------------------------------------------------

    def _apply_domain_rand(self, dr_cfg, rng: np.random.RandomState) -> None:
        """Randomize obstacle positions and properties. Override in subclass."""
        # Default: add jitter to non-wall geoms
        for i in range(self._model.nbody):
            name = self._model.body(i).name
            if name in ("world", "lekiwi_base", "goal_marker", "d435i_body",
                         "wheel_1", "wheel_2", "wheel_3"):
                continue
            if "wall" in name:
                continue
            # Add small position jitter
            jid = self._model.body_jntadr[i]
            if jid >= 0:
                for d in range(3):
                    self._data.qpos[jid + d] += rng.normal(0, dr_cfg.obstacle_pos_std)
