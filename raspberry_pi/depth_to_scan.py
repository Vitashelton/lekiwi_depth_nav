"""
Depth-to-Scan: Convert RealSense D435i depth image to pseudo-LiDAR scan.

Core algorithm:
  1. Extract horizontal band from depth image
  2. Column-wise min pooling (captures thin obstacles)
  3. Angular binning into N bins over specified FOV
  4. Range clipping and normalization
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class ScanConfig:
    """Configuration for depth-to-scan conversion."""
    num_bins: int = 64
    min_range: float = 0.15      # meters
    max_range: float = 5.0       # meters
    band_top: int = 220          # row index (from top of image)
    band_bottom: int = 260       # row index
    fov_deg: float = 90.0        # horizontal field of view, degrees

    # Camera intrinsics
    fx: float = 424.0
    fy: float = 424.0
    cx: float = 424.0
    cy: float = 240.0


class DepthToScan:
    """
    Convert a depth image (HxW numpy array) to a 1D pseudo-LiDAR scan.

    Two output modes:
      - scan_m: metric ranges in meters
      - scan_norm: normalized to [0, 1] for policy input
    """

    def __init__(self, config: ScanConfig):
        self.cfg = config
        self._precompute_angle_bins()

    def _precompute_angle_bins(self) -> None:
        """Precompute column-to-bin mapping based on camera intrinsics and FOV."""
        width = 848  # assumed depth image width
        self._bin_edges_rad = np.linspace(
            -np.deg2rad(self.cfg.fov_deg / 2),
            np.deg2rad(self.cfg.fov_deg / 2),
            self.cfg.num_bins + 1,
        )
        # For each column j, compute its horizontal angle
        col_indices = np.arange(width, dtype=np.float32)
        col_angles = np.arctan((col_indices - self.cfg.cx) / self.cfg.fx)

        # Build mapping: bin_index -> list of column indices
        self._bin_col_map: list[np.ndarray] = []
        for k in range(self.cfg.num_bins):
            mask = (col_angles >= self._bin_edges_rad[k]) & (
                col_angles < self._bin_edges_rad[k + 1]
            )
            cols = np.where(mask)[0]
            self._bin_col_map.append(cols)

    def __call__(self, depth_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Args:
            depth_image: np.ndarray of shape (H, W), units in meters.
                         Invalid values should be 0 or NaN.

        Returns:
            scan_norm: np.ndarray of shape (N,), range [0, 1] for policy input.
            scan_m: np.ndarray of shape (N,), range [min_range, max_range] in meters.
        """
        if depth_image.ndim != 2:
            raise ValueError(f"Expected 2D depth image, got shape {depth_image.shape}")

        H, W = depth_image.shape

        # --- Step 1: Extract horizontal band ---
        top = max(0, self.cfg.band_top)
        bottom = min(H, self.cfg.band_bottom + 1)
        band = depth_image[top:bottom, :].astype(np.float32)

        # --- Step 2: Mask invalid values ---
        valid_mask = np.isfinite(band) & (band > 0.0) & (band <= self.cfg.max_range)
        # Replace invalid with max_range so they don't affect min
        safe_band = np.where(valid_mask, band, self.cfg.max_range + 1.0)

        # --- Step 3: Column-wise minimum ---
        col_min = safe_band.min(axis=0)  # shape (W,)
        # Revert columns with no valid data back to max_range
        col_has_valid = valid_mask.any(axis=0)
        col_min = np.where(col_has_valid, col_min, self.cfg.max_range)

        # --- Step 4: Angular binning ---
        scan_m = np.full(self.cfg.num_bins, self.cfg.max_range, dtype=np.float32)
        for k in range(self.cfg.num_bins):
            cols = self._bin_col_map[k]
            if len(cols) > 0:
                bin_min = col_min[cols].min()
                scan_m[k] = bin_min

        # --- Step 5: Range clipping ---
        scan_m = np.clip(scan_m, self.cfg.min_range, self.cfg.max_range)

        # --- Step 6: Normalize ---
        scan_norm = (scan_m - self.cfg.min_range) / (
            self.cfg.max_range - self.cfg.min_range
        )
        # Invert so 1.0 = far away, 0.0 = very close (optional)
        # We keep it as is: 1.0 = max_range (safe), 0.0 = min_range (danger)

        return scan_norm.astype(np.float32), scan_m.astype(np.float32)

    def get_bin_angles_deg(self) -> np.ndarray:
        """Center angle (degrees) for each bin. Negative = left, positive = right."""
        edge_deg = (self._bin_edges_rad * 180.0 / np.pi)
        return (edge_deg[:-1] + edge_deg[1:]) / 2.0
