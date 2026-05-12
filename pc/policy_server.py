"""
Policy server: receives scan from Raspberry Pi, runs policy inference,
returns velocity command.

This is the main PC-side control loop running at policy frequency.

Supports three operation modes:
  --mode lerobot_raw            Pass candidate action through unchanged.
  --mode rule_shield            Replace candidate with rule-based safe action.
  --mode residual_correction    Apply learned residual correction to candidate.
"""

from __future__ import annotations

import json
import time
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
import zmq

from rule_policy import RulePolicy
from dwa_policy import DWAPlanner, DWAConfig
from mlp_policy import MLPPolicy
from geometric_risk import compute_action_projection

logger = logging.getLogger(__name__)


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class PolicyServer:
    """
    Subscribe to scan data from Pi, run policy, publish velocity commands.

    ZMQ topology:
      - SUB socket: receives scan from Pi
      - PUB socket: publishes velocity commands to Pi
    """

    def __init__(
        self,
        config_dir: str,
        mode: str = "lerobot_raw",
        residual_model_path: Optional[str] = None,
    ):
        """
        Args:
            config_dir: path to config directory.
            mode: "lerobot_raw" | "rule_shield" | "residual_correction".
            residual_model_path: path to ResidualCorrectionNet checkpoint
                (required for mode="residual_correction").
        """
        config_dir = Path(config_dir)

        # Load configs
        net_cfg = load_yaml(config_dir / "network.yaml")["network"]
        policy_cfg = load_yaml(config_dir / "policy.yaml")
        robot_cfg = load_yaml(config_dir / "robot.yaml")["robot"]

        self._mode = mode
        self._scan_bins: int = policy_cfg["scan"]["num_bins"]
        self._max_linear_vel: float = robot_cfg["max_linear_vel"]
        self._max_angular_vel: float = robot_cfg["max_angular_vel"]

        # Network setup
        transport = net_cfg.get("transport", "tcp")
        pc_ip = net_cfg["pc"]["ip"]
        pi_ip = net_cfg.get("pi_known_ip", "127.0.0.1")

        # Sub socket: connect to Pi's PUB to receive scan
        self._ctx = zmq.Context()
        self._scan_sub = self._ctx.socket(zmq.SUB)
        self._scan_sub.setsockopt(zmq.RCVHWM, 1)
        self._scan_sub.setsockopt(zmq.LINGER, 100)
        scan_sub_addr = f"{transport}://{pi_ip}:{net_cfg['pi']['scan_pub_port']}"
        self._scan_sub.connect(scan_sub_addr)
        self._scan_sub.setsockopt(
            zmq.SUBSCRIBE, net_cfg.get("scan_topic", "scan").encode()
        )

        # Pub socket: bind and send velocity commands to Pi
        self._cmd_pub = self._ctx.socket(zmq.PUB)
        self._cmd_pub.setsockopt(zmq.SNDHWM, 1)
        self._cmd_pub.setsockopt(zmq.LINGER, 100)
        cmd_pub_addr = f"{transport}://{pc_ip}:{net_cfg['pc']['cmd_pub_port']}"
        self._cmd_pub.bind(cmd_pub_addr)

        # --- Build candidate (LeRobot) policy ---
        active = policy_cfg.get("active_policy", "rule")
        scan_cfg = policy_cfg["scan"]

        if active == "rule":
            rc = policy_cfg["rule"]
            self._candidate_policy = RulePolicy(
                safe_distance=rc["safe_distance"],
                danger_distance=rc["danger_distance"],
                forward_speed=rc["forward_speed"],
                turn_gain=rc["turn_gain"],
                scan_bins=self._scan_bins,
                fov_deg=scan_cfg["fov_deg"],
                max_linear_vel=robot_cfg["max_linear_vel"],
                max_angular_vel=robot_cfg["max_angular_vel"],
            )
        elif active == "dwa":
            dc = policy_cfg["dwa"]
            self._candidate_policy = DWAPlanner(
                config=DWAConfig(
                    max_linear_vel=dc["max_linear_vel"],
                    max_angular_vel=dc["max_angular_vel"],
                    linear_accel=dc["linear_accel"],
                    angular_accel=dc["angular_accel"],
                    dt=dc["dt"],
                    predict_steps=dc["predict_steps"],
                    heading_weight=dc["heading_weight"],
                    clearance_weight=dc["clearance_weight"],
                    velocity_weight=dc["velocity_weight"],
                    num_samples=dc["num_samples"],
                ),
                scan_bins=self._scan_bins,
                fov_deg=scan_cfg["fov_deg"],
            )
        elif active == "mlp":
            mc = policy_cfg["mlp"]
            model_path = Path(config_dir).parent / mc["model_path"]
            if model_path.exists():
                self._candidate_policy = MLPPolicy.load(str(model_path))
                logger.info(f"Loaded MLP policy from {model_path}")
            else:
                logger.warning(
                    f"Model {model_path} not found. Using untrained MLP."
                )
                self._candidate_policy = MLPPolicy(
                    obs_dim=mc["obs_dim"],
                    act_dim=mc["act_dim"],
                    max_vx=mc["max_vx"],
                    max_vy=mc["max_vy"],
                    max_omega=mc["max_omega"],
                )
        else:
            raise ValueError(f"Unknown candidate policy type: {active}")

        # --- Build shield / residual policies ---
        self._shield_policy: Optional[RulePolicy] = None
        self._residual_model = None
        self._projection_rng = np.random.RandomState(42)

        if mode in ("rule_shield", "residual_correction"):
            self._shield_policy = RulePolicy(
                safe_distance=0.3,
                danger_distance=0.2,
                forward_speed=0.2,
                turn_gain=30.0,
                scan_bins=self._scan_bins,
                fov_deg=90.0,
                max_linear_vel=self._max_linear_vel,
                max_angular_vel=self._max_angular_vel,
            )

        if mode == "residual_correction":
            from residual_correction import ResidualCorrectionNet

            model_path = residual_model_path or str(
                Path(config_dir).parent / "models" / "residual_correction.pt"
            )
            if Path(model_path).exists():
                self._residual_model = ResidualCorrectionNet.load(model_path)
                logger.info(f"Loaded residual correction model from {model_path}")
            else:
                raise FileNotFoundError(
                    f"Residual model not found: {model_path}. "
                    "Train first with: python train/train_residual_correction.py"
                )

        self._running = False
        self._prev_vx: float = 0.0
        self._prev_vy: float = 0.0
        logger.info(f"PolicyServer initialized: mode={mode}, candidate={active}")

    # ------------------------------------------------------------------
    def run(self, freq_hz: float = 20.0) -> None:
        """Main control loop."""
        period = 1.0 / freq_hz
        self._running = True
        logger.info(f"PolicyServer running at {freq_hz} Hz, mode={self._mode}")

        while self._running:
            loop_start = time.perf_counter()

            # Receive scan (non-blocking, get latest)
            scan_m, goal_heading = self._recv_scan()
            if scan_m is None:
                elapsed = time.perf_counter() - loop_start
                time.sleep(max(0.0, period - elapsed))
                continue

            # Compute final action based on mode
            vx, vy, omega = self._compute_action(scan_m, goal_heading)

            # Publish command
            self._publish_cmd(vx, vy, omega)
            self._prev_vx, self._prev_vy = vx, vy

            # Maintain loop rate
            elapsed = time.perf_counter() - loop_start
            time.sleep(max(0.0, period - elapsed))

    # ------------------------------------------------------------------
    def _compute_action(
        self, scan_m: np.ndarray, goal_heading: float
    ) -> tuple[float, float, float]:
        """Compute the final action based on the active mode."""
        # Get candidate action from the LeRobot (or mock) policy
        if isinstance(self._candidate_policy, MLPPolicy):
            # MLP expects normalized observation
            scan_norm = np.clip(scan_m / 5.0, 0.0, 1.0)
            vx_norm = self._prev_vx / (self._max_linear_vel + 1e-6)
            vy_norm = self._prev_vy / (self._max_linear_vel + 1e-6)
            gh_norm = goal_heading / math.pi
            obs = np.concatenate([
                scan_norm, [vx_norm, vy_norm, gh_norm]
            ]).astype(np.float32)
            action_np = self._candidate_policy.predict(obs, deterministic=True)
            cvx, cvy, comega = float(action_np[0]), float(action_np[1]), float(action_np[2])
        else:
            cvx, cvy, comega = self._candidate_policy(scan_m, goal_heading)

        import math as _math

        if self._mode == "lerobot_raw":
            return cvx, cvy, comega

        elif self._mode == "rule_shield":
            # Override with rule-based safe action
            assert self._shield_policy is not None
            return self._shield_policy(scan_m, goal_heading)

        elif self._mode == "residual_correction":
            assert self._residual_model is not None
            candidate = np.array([cvx, cvy, comega], dtype=np.float32)
            current_vel = np.array(
                [self._prev_vx, self._prev_vy], dtype=np.float32
            )
            safe_action = self._residual_model.correct_action(
                scan_m=scan_m,
                candidate_action=candidate,
                current_velocity=current_vel,
                goal_heading=goal_heading,
                max_linear_vel=self._max_linear_vel,
                max_angular_vel=self._max_angular_vel,
            )
            return (
                float(safe_action[0]),
                float(safe_action[1]),
                float(safe_action[2]),
            )

        else:
            raise ValueError(f"Unknown mode: {self._mode}")

    # ------------------------------------------------------------------
    def _recv_scan(self) -> tuple[Optional[np.ndarray], float]:
        """Receive latest scan from Pi. Returns (scan_m, goal_heading)."""
        try:
            latest = None
            while self._scan_sub.poll(timeout=1) != 0:
                parts = self._scan_sub.recv_multipart(zmq.NOBLOCK)
                if len(parts) >= 2:
                    latest = parts
            if latest is None:
                return None, 0.0

            msg = json.loads(latest[1].decode())
            scan_m = np.array(msg.get("scan_m", msg["scan"]), dtype=np.float32)
            goal_heading = msg.get("goal_heading", 0.0)
            return scan_m, goal_heading
        except zmq.ZMQError:
            return None, 0.0

    # ------------------------------------------------------------------
    def _publish_cmd(self, vx: float, vy: float, omega: float) -> None:
        """Publish velocity command to Pi."""
        msg = {
            "vx": float(vx),
            "vy": float(vy),
            "omega": float(omega),
            "timestamp": time.time(),
        }
        try:
            self._cmd_pub.send_multipart(
                [b"cmd_vel", json.dumps(msg).encode()],
                flags=zmq.NOBLOCK,
            )
        except zmq.ZMQError:
            pass

    # ------------------------------------------------------------------
    def stop(self) -> None:
        self._running = False
        self._publish_cmd(0.0, 0.0, 0.0)
        time.sleep(0.05)
        self._scan_sub.close()
        self._cmd_pub.close()
        self._ctx.term()
        logger.info("PolicyServer stopped.")
