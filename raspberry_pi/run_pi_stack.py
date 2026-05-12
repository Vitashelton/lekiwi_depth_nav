"""
Main Raspberry Pi stack: Run camera, scan publisher, command subscriber,
and base controller in a multi-threaded loop.

The Pi stack:
  1. Reads depth frames from RealSense D435i (or mock camera)
  2. Converts depth → pseudo-LiDAR scan via Depth-to-Scan
  3. Publishes scan to PC via ZMQ PUB
  4. Receives velocity command from PC via ZMQ SUB
  5. Executes velocity on LeKiwi base (omega in deg/s, lerobot convention)

Usage:
    python run_pi_stack.py --config config/

    With mock camera for PC testing:
    python run_pi_stack.py --config config/ --mock-camera
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

import yaml
import numpy as np

from camera_node import CameraConfig, create_camera
from depth_to_scan import DepthToScan, ScanConfig
from scan_publisher import ScanPublisher
from command_subscriber import CommandSubscriber
from base_controller import BaseController, RobotConfig


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class PiStack:
    def __init__(
        self,
        config_dir: str,
        use_mock_camera: bool = False,
    ):
        config_dir = Path(config_dir)

        # Load configs
        robot_cfg = RobotConfig(**load_yaml(config_dir / "robot.yaml")["robot"])
        cam_raw = load_yaml(config_dir / "camera.yaml")["camera"]
        net_cfg = load_yaml(config_dir / "network.yaml")["network"]
        policy_cfg = load_yaml(config_dir / "policy.yaml")

        # --- Scan processor ---
        scan_raw = policy_cfg["scan"]
        self.scan_processor = DepthToScan(ScanConfig(
            num_bins=scan_raw["num_bins"],
            min_range=scan_raw["min_range"],
            max_range=scan_raw["max_range"],
            band_top=scan_raw["band_top"],
            band_bottom=scan_raw["band_bottom"],
            fov_deg=scan_raw["fov_deg"],
            fx=cam_raw["fx"],
            fy=cam_raw["fy"],
            cx=cam_raw["cx"],
            cy=cam_raw["cy"],
        ))

        # --- Camera ---
        cam_config = CameraConfig(
            depth_width=cam_raw["depth_width"],
            depth_height=cam_raw["depth_height"],
            depth_fps=cam_raw["depth_fps"],
            fx=cam_raw["fx"],
            fy=cam_raw["fy"],
            cx=cam_raw["cx"],
            cy=cam_raw["cy"],
        )
        self.camera = create_camera(cam_config, use_mock=use_mock_camera)

        # --- Network ---
        transport = net_cfg.get("transport", "tcp")

        # Pi publishes scan → PC subscribes
        scan_addr = f"{transport}://*:{net_cfg['pi']['scan_pub_port']}"
        self.scan_pub = ScanPublisher(scan_addr, topic=net_cfg.get("scan_topic", "scan"))

        # PC publishes velocity commands → Pi subscribes
        pc_ip = net_cfg.get("pc_known_ip", "127.0.0.1")
        cmd_addr = f"{transport}://{pc_ip}:{net_cfg['pc']['cmd_pub_port']}"
        self.cmd_sub = CommandSubscriber(cmd_addr, topic=net_cfg.get("cmd_topic", "cmd_vel"))

        # --- Base controller ---
        self.base_ctrl = BaseController(robot_cfg)

        # --- Safety ---
        safety_cfg = load_yaml(config_dir / "robot.yaml").get("safety", {})
        self._cmd_timeout = safety_cfg.get("max_cmd_timeout", 0.5)
        self._min_scan_dist = safety_cfg.get("min_scan_distance", 0.10)

        self._running = False
        self._current_goal_heading = 0.0  # rad, updated by brain (future)

    def start(self) -> None:
        self.camera.start()
        self._running = True
        print(f"[PiStack] Publishing scan on {self.scan_pub.address}")
        print(f"[PiStack] Subscribing to commands on tcp://...:{self.cmd_sub._address}")

    def step(self) -> bool:
        """
        One control cycle: read depth → compute scan → publish → recv cmd → execute.
        """
        depth = self.camera.get_depth_frame()
        if depth is None:
            return False

        # Depth-to-Scan
        scan_norm, scan_m = self.scan_processor(depth)

        # Publish scan (include goal heading for policies that need it)
        # The scan publisher sends both normalized and metric scans
        self.scan_pub.publish(scan_norm, scan_m)

        # Safety: emergency stop if anything is too close
        if scan_m.min() < self._min_scan_dist:
            self.base_ctrl.stop()
            return True

        # Receive velocity command from PC
        cmd = self.cmd_sub.recv(timeout_ms=5)

        # Watchdog: stop if no command for too long
        timeout = self.cmd_sub.get_time_since_last_cmd()
        if timeout > self._cmd_timeout:
            self.base_ctrl.stop()
        elif cmd is not None:
            # Command omega is in deg/s (lerobot convention)
            self.base_ctrl.set_velocity(cmd.vx, cmd.vy, cmd.omega)

        return True

    def run(self) -> None:
        """Main loop."""
        self.start()
        try:
            while self._running:
                ok = self.step()
                if not ok:
                    time.sleep(0.001)
        except KeyboardInterrupt:
            print("\n[PiStack] Interrupted.")
        finally:
            self.stop()

    def stop(self) -> None:
        self._running = False
        self.base_ctrl.stop()
        self.scan_pub.close()
        self.cmd_sub.close()
        self.camera.stop()
        print("[PiStack] Stopped.")


def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi navigation stack")
    parser.add_argument("--config", default="config", help="Path to config directory")
    parser.add_argument("--mock-camera", action="store_true", help="Use mock camera for testing")
    args = parser.parse_args()

    stack = PiStack(args.config, use_mock_camera=args.mock_camera)
    stack.run()


if __name__ == "__main__":
    main()
