"""
Camera node: Read depth frames from Intel RealSense D435i.

On platforms without a RealSense camera (e.g., PC for testing),
fall back to synthetic depth generation.
"""

from __future__ import annotations

import time
import numpy as np
import sys
from typing import Optional
from dataclasses import dataclass


@dataclass
class CameraConfig:
    depth_width: int = 848
    depth_height: int = 480
    depth_fps: int = 30
    rgb_width: int = 848
    rgb_height: int = 480
    rgb_fps: int = 30
    fx: float = 424.0
    fy: float = 424.0
    cx: float = 424.0
    cy: float = 240.0


class RealSenseCamera:
    """Read depth frames from Intel RealSense D435i using pyrealsense2."""

    def __init__(self, config: CameraConfig):
        self.cfg = config
        self._pipeline = None
        self._profile = None
        self._depth_scale: float = 0.001

    def start(self) -> None:
        """Initialize and start the RealSense pipeline."""
        try:
            import pyrealsense2 as rs
        except ImportError:
            raise ImportError(
                "pyrealsense2 not installed. "
                "Install with: pip install pyrealsense2"
            )

        self._pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(
            rs.stream.depth,
            self.cfg.depth_width,
            self.cfg.depth_height,
            rs.format.z16,
            self.cfg.depth_fps,
        )
        self._profile = self._pipe.start(cfg)
        depth_sensor = self._profile.get_device().first_depth_sensor()
        self._depth_scale = depth_sensor.get_depth_scale()
        print(f"[RealSenseCamera] Started. Depth scale: {self._depth_scale:.4f} m/unit")

    def get_depth_frame(self) -> Optional[np.ndarray]:
        """
        Returns:
            Depth image in meters as float32 numpy array (H, W),
            or None if no frame is available.
        """
        import pyrealsense2 as rs

        try:
            frames = self._pipe.wait_for_frames(timeout_ms=100)
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                return None
            depth_data = np.asanyarray(depth_frame.get_data(), dtype=np.float32)
            # Convert from uint16 scale to meters
            depth_m = depth_data * self._depth_scale
            return depth_m
        except Exception:
            return None

    def get_rgb_frame(self) -> Optional[np.ndarray]:
        """Not used in main navigation loop; provided for debugging."""
        try:
            import pyrealsense2 as rs
            frames = self._pipe.wait_for_frames(timeout_ms=100)
            color_frame = frames.get_color_frame()
            if not color_frame:
                return None
            return np.asanyarray(color_frame.get_data())
        except Exception:
            return None

    def stop(self) -> None:
        if self._pipe is not None:
            self._pipe.stop()
            print("[RealSenseCamera] Stopped.")


class MockCamera:
    """
    Synthetic depth camera for testing on PC without RealSense hardware.
    Generates simple depth images with a flat floor and a few virtual obstacles.
    """

    def __init__(self, config: CameraConfig):
        self.cfg = config

    def start(self) -> None:
        print("[MockCamera] Started (no real camera).")

    def get_depth_frame(self) -> np.ndarray:
        H, W = self.cfg.depth_height, self.cfg.depth_width
        # Flat floor: depth increases with row (simple pinhole model)
        row_idx = np.arange(H, dtype=np.float32).reshape(-1, 1)
        cy = self.cfg.cy
        fy = self.cfg.fy
        # Approximate: assuming camera height ~0.3m above ground
        camera_height = 0.3
        depth_col = np.where(
            row_idx > cy,
            camera_height * fy / (row_idx - cy + 1e-6),
            3.0,
        )
        depth = np.tile(np.clip(depth_col, 0.15, 5.0), (1, W))

        # Add a synthetic obstacle (vertical bar) in the center
        obstacle_col = int(W * 0.55)
        half_width = 20
        left = max(0, obstacle_col - half_width)
        right = min(W, obstacle_col + half_width)
        depth[:, left:right] = np.minimum(depth[:, left:right], 0.6)

        # Add a gap on the left
        depth[:, :int(W * 0.2)] = np.minimum(depth[:, :int(W * 0.2)], 0.4)

        # Add noise
        depth += np.random.normal(0, 0.01, depth.shape).astype(np.float32)
        depth = np.maximum(depth, 0.0)

        # Simulate invalid pixels (5% random dropouts)
        dropout_mask = np.random.random(depth.shape) < 0.05
        depth[dropout_mask] = 0.0

        return depth.astype(np.float32)

    def stop(self) -> None:
        print("[MockCamera] Stopped.")


def create_camera(config: CameraConfig, use_mock: bool = False):
    """Factory function to create camera based on availability."""
    if use_mock:
        return MockCamera(config)
    try:
        import pyrealsense2  # noqa: F401
        cam = RealSenseCamera(config)
        return cam
    except ImportError:
        print("[WARN] pyrealsense2 not available, using MockCamera.")
        return MockCamera(config)
