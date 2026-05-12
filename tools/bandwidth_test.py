"""
Bandwidth comparison: raw depth vs JPEG depth vs pseudo-LiDAR scan.

Tests:
  1. Raw depth image (848×480, 16-bit): ~814 KB/frame
  2. JPEG compressed depth (quality=50): ~40-80 KB/frame
  3. 32-D pseudo-LiDAR scan: 128 bytes/frame
  4. 64-D pseudo-LiDAR scan: 256 bytes/frame
  5. 128-D pseudo-LiDAR scan: 512 bytes/frame

Usage:
    python tools/bandwidth_test.py --config config/
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from raspberry_pi.camera_node import MockCamera, CameraConfig
from raspberry_pi.depth_to_scan import DepthToScan, ScanConfig


def main():
    parser = argparse.ArgumentParser(description="Bandwidth benchmark")
    parser.add_argument("--config", default="config")
    parser.add_argument("--num-frames", type=int, default=100)
    args = parser.parse_args()

    cam = MockCamera(CameraConfig())
    depths = [cam.get_depth_frame() for _ in range(args.num_frames)]

    print("=" * 60)
    print("BANDWIDTH COMPARISON")
    print("=" * 60)

    # 1. Raw depth (uint16 → bytes)
    raw_sizes = [d.tobytes().__len__() for d in depths]
    raw_kb = np.mean(raw_sizes) / 1024
    print(f"\n[1] Raw Depth (848×480, uint16)")
    print(f"    Size per frame: {raw_kb:.1f} KB")
    print(f"    At 30 Hz:       {raw_kb * 30:.1f} KB/s = {raw_kb * 30 / 1024:.2f} MB/s")

    # 2. JPEG compressed (simulated: convert to uint8, encode as JPEG)
    # In practice, depth should NOT be JPEG-compressed (nearest-neighbor data loss).
    # We simulate the size here for comparison.
    jpeg_sizes = []
    try:
        import cv2
        for d in depths:
            d_vis = np.clip(d / 5.0 * 255, 0, 255).astype(np.uint8)
            _, buf = cv2.imencode(".jpg", d_vis, [cv2.IMWRITE_JPEG_QUALITY, 50])
            jpeg_sizes.append(len(buf))
        jpeg_kb = np.mean(jpeg_sizes) / 1024
        print(f"\n[2] JPEG Depth (quality=50)")
        print(f"    Size per frame: {jpeg_kb:.1f} KB")
        print(f"    At 30 Hz:       {jpeg_kb * 30:.1f} KB/s = {jpeg_kb * 30 / 1024:.2f} MB/s")
    except ImportError:
        print(f"\n[2] JPEG Depth — cv2 not available, skipping")

    # 3. Pseudo-LiDAR scans
    for bins in [32, 64, 128]:
        d2s = DepthToScan(ScanConfig(num_bins=bins))
        scan_sizes = []
        for d in depths:
            scan_norm, scan_m = d2s(d)
            # JSON serialization size
            msg = {
                "scan": scan_norm.tolist(),
                "scan_m": scan_m.tolist(),
                "timestamp": 0.0,
                "seq": 0,
            }
            import json
            scan_sizes.append(len(json.dumps(msg).encode()))

        scan_kb = np.mean(scan_sizes) / 1024
        reduction = (1 - scan_kb / raw_kb) * 100
        print(f"\n[{bins}-D Pseudo-LiDAR Scan]")
        print(f"    JSON size/frame: {scan_kb:.2f} KB ({np.mean(scan_sizes):.0f} bytes)")
        print(f"    At 30 Hz:        {scan_kb * 30:.1f} KB/s")
        print(f"    vs Raw Depth:    {reduction:.1f}% reduction")

    # 4. Binary float32 (most efficient)
    print(f"\n[Binary float32 (optimal)]")
    for bins in [32, 64, 128]:
        float_size = bins * 4 * 2  # scan_norm + scan_m, each float32
        print(f"    {bins}-D: {float_size} bytes/frame, {float_size * 30 / 1024:.1f} KB/s at 30 Hz")


if __name__ == "__main__":
    main()
