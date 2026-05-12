"""
Base controller: Interface to LeKiwi three-omniwheel mobile robot.

Kinematics follow the HuggingFace LeRobot LeKiwi implementation:
  - Wheel mounting angles: [240, 0, 120] degrees with a -90° offset
    → effective angles = [150°, -90°, 30°] from x-axis
  - Body velocity to wheel linear speed via kinematic matrix M:
      M[i] = [cos(alpha_i), sin(alpha_i), base_radius]

The LeKiwi uses Feetech STS3215 servo motors in VELOCITY mode for the base.
Raw commands are signed 16-bit integers where 4096 counts = 360 deg/s.
"""

from __future__ import annotations

import math
import time
import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RobotConfig:
    wheel_radius: float = 0.05     # meters (lerobot default)
    base_radius: float = 0.125     # distance center→wheel, meters (lerobot default)
    max_linear_vel: float = 0.3    # m/s
    max_angular_vel: float = 90.0  # deg/s (lerobot convention)
    max_raw: int = 3000            # max raw command (out of ±32767)
    steps_per_deg: float = 4096.0 / 360.0  # Feetech STS3215 encoder resolution


def _clamp(v: float, limit: float) -> float:
    return max(-limit, min(limit, v))


class BaseController:
    """
    Convert body-frame velocity (vx [m/s], vy [m/s], theta [deg/s])
    to raw motor commands for LeKiwi's three omniwheels.

    Two modes:
      - "mock": prints wheel commands, no hardware needed (for PC testing)
      - "feetech": sends commands via FeetechMotorsBus (Raspberry Pi + real robot)

    To use real hardware, provide a FeetechMotorsBus instance via set_motors_bus().
    """

    # LeKiwi motor names (matches lerobot convention)
    MOTOR_NAMES = ["base_left_wheel", "base_back_wheel", "base_right_wheel"]

    # Wheel mounting angles: [240°, 0°, 120°] minus 90° offset
    # Effective angles in degrees: [150, -90, 30]
    WHEEL_ANGLES_DEG = [150.0, -90.0, 30.0]
    WHEEL_ANGLES_RAD = np.radians(WHEEL_ANGLES_DEG)

    # Kinematic matrix M: each row maps body velocity [vx, vy, theta_rad] to
    # wheel linear speed (m/s). theta must be in rad/s for the matrix multiply.
    # M shape: (3, 3)
    KINEMATIC_MATRIX: np.ndarray = None  # set in __init__

    def __init__(self, config: RobotConfig):
        self.cfg = config
        self._motors_bus = None  # FeetechMotorsBus, set externally for real HW
        self._current_cmd = {"vx": 0.0, "vy": 0.0, "theta": 0.0}

        # Build kinematic matrix and its inverse
        m = np.array([
            [np.cos(a), np.sin(a), self.cfg.base_radius]
            for a in self.WHEEL_ANGLES_RAD
        ])
        self._M = m           # (3, 3): body → wheel linear speed
        self._M_inv = np.linalg.inv(m)  # (3, 3): wheel linear speed → body

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_velocity(self, vx: float, vy: float, omega_deg: float) -> None:
        """
        Set body-frame velocity command.

        Args:
            vx: forward velocity (m/s, positive = forward).
            vy: lateral velocity (m/s, positive = right).
            omega_deg: rotational velocity (deg/s, positive = CCW).
        """
        vx = _clamp(vx, self.cfg.max_linear_vel)
        vy = _clamp(vy, self.cfg.max_linear_vel)
        omega_deg = _clamp(omega_deg, self.cfg.max_angular_vel)
        self._current_cmd = {"vx": vx, "vy": vy, "theta": omega_deg}

        raw_cmds = self._body_to_wheel_raw(vx, vy, omega_deg)
        self._send_raw_commands(raw_cmds)

    def stop(self) -> None:
        """Emergency stop: zero velocity on all wheels."""
        self.set_velocity(0.0, 0.0, 0.0)

    def set_motors_bus(self, bus) -> None:
        """
        Inject a FeetechMotorsBus instance for real hardware control.
        Call this before using set_velocity() on the Raspberry Pi.

        Example:
            from lerobot.motors.feetech import FeetechMotorsBus
            bus = FeetechMotorsBus(port="/dev/ttyACM0", motors={...})
            controller.set_motors_bus(bus)
        """
        self._motors_bus = bus

    # ------------------------------------------------------------------
    # Kinematics (matches lerobot LeKiwi._body_to_wheel_raw)
    # ------------------------------------------------------------------

    def _body_to_wheel_raw(
        self, vx: float, vy: float, omega_deg: float
    ) -> dict[str, int]:
        """
        Convert body velocity command to raw wheel speed integers.

        Args:
            vx, vy: m/s
            omega_deg: deg/s

        Returns:
            {"base_left_wheel": raw, "base_back_wheel": raw, "base_right_wheel": raw}
        """
        # Convert rotational velocity from deg/s to rad/s for the kinematics.
        omega_rad = omega_deg * (math.pi / 180.0)
        velocity_vector = np.array([vx, vy, omega_rad])

        # Compute each wheel's linear speed (m/s) via kinematic matrix.
        wheel_linear = self._M.dot(velocity_vector)  # (3,)

        # Convert to angular speed: rad/s → deg/s
        wheel_radps = wheel_linear / self.cfg.wheel_radius
        wheel_degps = wheel_radps * (180.0 / math.pi)

        # Scale if any command exceeds max_raw
        max_val = np.max(np.abs(wheel_degps))
        if max_val > 0:
            raw_max = self.cfg.max_raw / self.cfg.steps_per_deg
            if max_val > raw_max:
                wheel_degps = wheel_degps * (raw_max / max_val)

        raw_cmds = [self._degps_to_raw(d) for d in wheel_degps]
        return dict(zip(self.MOTOR_NAMES, raw_cmds))

    def _wheel_raw_to_body(
        self, left_raw: int, back_raw: int, right_raw: int
    ) -> tuple[float, float, float]:
        """
        Convert raw wheel feedback to body velocity (odometry estimate).

        Returns:
            (vx [m/s], vy [m/s], omega [deg/s])
        """
        raw_vals = np.array([left_raw, back_raw, right_raw])
        wheel_degps = np.array([self._raw_to_degps(r) for r in raw_vals])
        wheel_radps = wheel_degps * (math.pi / 180.0)
        wheel_linear = wheel_radps * self.cfg.wheel_radius
        velocity_vector = self._M_inv.dot(wheel_linear)
        vx, vy, omega_rad = velocity_vector
        omega_deg = omega_rad * (180.0 / math.pi)
        return vx, vy, omega_deg

    # ------------------------------------------------------------------
    # Raw value conversion
    # ------------------------------------------------------------------

    def _degps_to_raw(self, degps: float) -> int:
        """Convert wheel angular speed (deg/s) to Feetech raw integer."""
        raw_float = abs(degps) * self.cfg.steps_per_deg
        raw_int = int(round(raw_float))
        if degps < 0:
            raw_int = -raw_int
        # Clamp to signed 16-bit
        raw_int = max(-0x7FFF, min(0x7FFF, raw_int))
        return raw_int

    def _raw_to_degps(self, raw: int) -> float:
        """Convert Feetech raw integer back to deg/s."""
        return raw / self.cfg.steps_per_deg

    # ------------------------------------------------------------------
    # Motor output (mock by default, real via FeetechMotorsBus)
    # ------------------------------------------------------------------

    def _send_raw_commands(self, raw_cmds: dict[str, int]) -> None:
        """
        Send raw wheel commands to motors.
        MOCK implementation prints nothing; uncomment logger for debugging.
        """
        if self._motors_bus is not None:
            # Real hardware path: sync_write to Feetech motors in VELOCITY mode
            self._motors_bus.sync_write("Goal_Velocity", raw_cmds)
        else:
            # Mock path: silently store, log at DEBUG level if needed
            pass
