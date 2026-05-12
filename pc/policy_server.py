"""
Policy server: receives scan from Raspberry Pi, runs policy inference,
returns velocity command.

This is the main PC-side control loop running at policy frequency.
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

    def __init__(self, config_dir: str):
        config_dir = Path(config_dir)

        # Load configs
        net_cfg = load_yaml(config_dir / "network.yaml")["network"]
        policy_cfg = load_yaml(config_dir / "policy.yaml")
        robot_cfg = load_yaml(config_dir / "robot.yaml")["robot"]

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
        self._scan_sub.setsockopt(zmq.SUBSCRIBE, net_cfg.get("scan_topic", "scan").encode())

        # Pub socket: bind and send velocity commands to Pi
        self._cmd_pub = self._ctx.socket(zmq.PUB)
        self._cmd_pub.setsockopt(zmq.SNDHWM, 1)
        self._cmd_pub.setsockopt(zmq.LINGER, 100)
        cmd_pub_addr = f"{transport}://{pc_ip}:{net_cfg['pc']['cmd_pub_port']}"
        self._cmd_pub.bind(cmd_pub_addr)

        # Policy
        active = policy_cfg["active_policy"]
        scan_cfg = policy_cfg["scan"]
        self._scan_bins = scan_cfg["num_bins"]

        if active == "rule":
            rc = policy_cfg["rule"]
            self._policy = RulePolicy(
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
            self._policy = DWAPlanner(
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
                self._policy = MLPPolicy.load(str(model_path))
                logger.info(f"Loaded MLP policy from {model_path}")
            else:
                logger.warning(
                    f"Model {model_path} not found. Using untrained MLP. "
                    "Train first with: python sim/train_sac.py"
                )
                self._policy = MLPPolicy(
                    obs_dim=mc["obs_dim"],
                    act_dim=mc["act_dim"],
                    max_vx=mc["max_vx"],
                    max_vy=mc["max_vy"],
                    max_omega=mc["max_omega"],
                )
        else:
            raise ValueError(f"Unknown policy type: {active}")

        self._active_policy = active
        self._running = False

    def run(self, freq_hz: float = 20.0) -> None:
        """Main control loop."""
        period = 1.0 / freq_hz
        self._running = True
        logger.info(f"PolicyServer running at {freq_hz} Hz, policy={self._active_policy}")

        while self._running:
            loop_start = time.perf_counter()

            # Receive scan (non-blocking, get latest)
            scan_m, goal_heading = self._recv_scan()
            if scan_m is None:
                # No scan yet, sleep and retry
                elapsed = time.perf_counter() - loop_start
                time.sleep(max(0.0, period - elapsed))
                continue

            # Run policy
            vx, vy, omega = self._policy(scan_m, goal_heading)

            # Publish command
            self._publish_cmd(vx, vy, omega)

            # Maintain loop rate
            elapsed = time.perf_counter() - loop_start
            time.sleep(max(0.0, period - elapsed))

    def _recv_scan(self) -> tuple[Optional[np.ndarray], float]:
        """
        Receive latest scan from Pi. Returns (scan_m, goal_heading).
        scan_m is None if no data.
        """
        try:
            # Drain all pending messages, keep only the latest
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

    def stop(self) -> None:
        self._running = False
        # Send zero velocity before shutting down
        self._publish_cmd(0.0, 0.0, 0.0)
        time.sleep(0.05)
        self._scan_sub.close()
        self._cmd_pub.close()
        self._ctx.term()
        logger.info("PolicyServer stopped.")
