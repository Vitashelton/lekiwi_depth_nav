"""
Visualize pseudo-LiDAR scan data.

Usage:
    # Plot a single scan from a log file
    python tools/plot_scan.py --input logs/scan_log.npz --index 50

    # Plot scan heatmap (all scans over time)
    python tools/plot_scan.py --input logs/scan_log.npz --heatmap

    # Live plot from simulation
    python tools/plot_scan.py --live --duration 10
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def plot_single(scan_m: np.ndarray, title: str = "Pseudo-LiDAR Scan") -> None:
    """Plot a single scan as a polar/linear plot."""
    if not HAS_MPL:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return

    bins = len(scan_m)
    fov_deg = 90.0
    half = np.deg2rad(fov_deg / 2)
    angles = np.linspace(-half, half, bins)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Linear plot
    angles_deg = np.rad2deg(angles)
    ax1.fill_between(angles_deg, 0, scan_m, alpha=0.3)
    ax1.plot(angles_deg, scan_m, "b-", linewidth=1.5)
    ax1.axhline(y=5.0, color="gray", linestyle="--", alpha=0.5, label="max range")
    ax1.axhline(y=0.15, color="r", linestyle="--", alpha=0.5, label="min range")
    ax1.set_xlabel("Angle (deg)")
    ax1.set_ylabel("Range (m)")
    ax1.set_title("Linear View")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Polar plot
    ax2 = fig.add_subplot(1, 2, 2, projection="polar")
    ax2.fill_between(angles, 0, scan_m, alpha=0.3)
    ax2.plot(angles, scan_m, "b-", linewidth=1.5)
    ax2.set_theta_zero_location("N")
    ax2.set_theta_direction(-1)
    ax2.set_thetamin(-45)
    ax2.set_thetamax(45)
    ax2.set_title("Polar View")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_heatmap(scans_m: np.ndarray, timestamps: np.ndarray | None = None) -> None:
    """Plot scan heatmap over time."""
    if not HAS_MPL:
        print("matplotlib not installed.")
        return

    if timestamps is None:
        timestamps = np.arange(len(scans_m))

    bins = scans_m.shape[1]
    fov_deg = 90.0
    angles_deg = np.linspace(-fov_deg / 2, fov_deg / 2, bins)

    fig, ax = plt.subplots(figsize=(12, 6))
    extent = [angles_deg[0], angles_deg[-1], timestamps[-1], timestamps[0]]
    im = ax.imshow(
        scans_m,
        aspect="auto",
        extent=extent,
        cmap="viridis_r",
        vmin=0,
        vmax=5.0,
    )
    ax.set_xlabel("Angle (deg)")
    ax.set_ylabel("Time (s)")
    ax.set_title("Scan Heatmap (range in meters)")
    plt.colorbar(im, ax=ax, label="Range (m)")
    plt.tight_layout()
    plt.show()


def live_plot(duration: float = 10.0, scan_bins: int = 64) -> None:
    """Live plot of scans from simulation."""
    if not HAS_MPL:
        print("matplotlib not installed.")
        return

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from sim.simple_2d_env import Simple2DNavEnv

    env = Simple2DNavEnv(num_scan_bins=scan_bins)
    obs, _ = env.reset()

    fov_deg = 90.0
    half = np.deg2rad(fov_deg / 2)
    angles = np.linspace(-half, half, scan_bins)

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6))
    line, = ax.plot(np.rad2deg(angles), obs[:scan_bins] * 5.0, "b-")
    ax.set_ylim(0, 5.5)
    ax.set_xlabel("Angle (deg)")
    ax.set_ylabel("Range (m)")
    ax.grid(True, alpha=0.3)

    t0 = time.time()
    while time.time() - t0 < duration:
        action = np.array([0.1, 0.0, np.random.uniform(-20, 20)], dtype=np.float32)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

        line.set_ydata(obs[:scan_bins] * 5.0)
        ax.set_title(f"Live Scan (t={time.time()-t0:.1f}s)")
        plt.pause(0.01)

    plt.ioff()
    plt.show()
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Plot scan data")
    parser.add_argument("--input", default=None, help="Path to .npz scan log")
    parser.add_argument("--index", type=int, default=0, help="Frame index to plot")
    parser.add_argument("--heatmap", action="store_true", help="Plot heatmap of all frames")
    parser.add_argument("--live", action="store_true", help="Live plot from simulation")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--scan-bins", type=int, default=64)
    args = parser.parse_args()

    if args.live:
        live_plot(args.duration, args.scan_bins)
    elif args.input:
        data = np.load(args.input)
        scans_m = data["scans_m"]
        if "timestamps" in data:
            timestamps = data["timestamps"]
        else:
            timestamps = np.arange(len(scans_m))

        if args.heatmap:
            plot_heatmap(scans_m, timestamps)
        else:
            idx = min(args.index, len(scans_m) - 1)
            plot_single(scans_m[idx], title=f"Scan frame {idx}")
    else:
        # Quick demo: generate a scan from mock camera and plot
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from raspberry_pi.camera_node import MockCamera, CameraConfig
        from raspberry_pi.depth_to_scan import DepthToScan, ScanConfig

        cam = MockCamera(CameraConfig())
        d2s = DepthToScan(ScanConfig())
        depth = cam.get_depth_frame()
        _, scan_m = d2s(depth)
        plot_single(scan_m, title="Mock Camera Scan")


if __name__ == "__main__":
    main()
