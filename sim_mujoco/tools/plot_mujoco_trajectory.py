"""
Plot 2D trajectory of the robot from a recorded MuJoCo episode dataset.

Reads an NPZ file (from record_mujoco_dataset.py) and produces:
  - 2D path plot with obstacles, start, and goal markers
  - Scan ray overlay at sampled waypoints
  - Velocity profile subplot (vx, vy, omega over time)
  - Heading error over time subplot

Command:
    python sim_mujoco/tools/plot_mujoco_trajectory.py \
        --input logs/mujoco_episode.npz \
        --output logs/trajectory_plot.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ── Data loading ────────────────────────────────────────────────────────────

def load_episode_data(input_path: str) -> dict[str, np.ndarray]:
    """Load recorded episode data from an NPZ file.

    Expected keys (flexible):
      - positions: (T, 2) robot x, y positions
      - velocities: (T, 3) vx, vy, omega
      - scans: (T, N) metric scan readings
      - goal: (2,) goal x, y
      - start: (2,) start x, y
      - obstacles: (M, 3) obstacle (x, y, radius)
      - headings: (T,) robot heading angle
      - timestamps: (T,) or single float

    Missing keys are filled with defaults or derived values.

    Args:
        input_path: Path to the NPZ file.

    Returns:
        Dict of arrays.
    """
    data = np.load(str(input_path), allow_pickle=True)
    result: dict[str, Any] = {}

    # Positions
    if "positions" in data:
        result["positions"] = data["positions"]
    elif "robot_pos" in data:
        result["positions"] = data["robot_pos"]
    else:
        result["positions"] = np.zeros((1, 2), dtype=np.float32)

    # Velocities
    if "velocities" in data:
        result["velocities"] = data["velocities"]
    elif "actions" in data:
        result["velocities"] = data["actions"]
    else:
        result["velocities"] = np.zeros((result["positions"].shape[0], 3), dtype=np.float32)

    # Scans
    if "scans" in data:
        result["scans"] = data["scans"]
    elif "scans_m" in data:
        result["scans"] = data["scans_m"]
    elif "scan" in data:
        result["scans"] = data["scan"]
    else:
        result["scans"] = np.zeros((result["positions"].shape[0], 64), dtype=np.float32)

    # Goal position
    if "goal" in data:
        result["goal"] = data["goal"]
    elif "goal_xy" in data:
        result["goal"] = data["goal_xy"]
    else:
        result["goal"] = np.array([4.0, 1.5], dtype=np.float32)

    # Start position
    if "start" in data:
        result["start"] = data["start"]
    elif "start_xy" in data:
        result["start"] = data["start_xy"]
    else:
        result["start"] = result["positions"][0, :2].copy()

    # Obstacles
    if "obstacles" in data:
        obstacles_val = data["obstacles"]
        # Handle object arrays from allow_pickle
        if obstacles_val.dtype == np.dtype("O"):
            result["obstacles"] = np.array(obstacles_val.item(), dtype=np.float32)
        else:
            result["obstacles"] = obstacles_val
    else:
        result["obstacles"] = np.zeros((0, 3), dtype=np.float32)

    # Headings: derive from velocity or positions if not explicit
    if "headings" in data:
        result["headings"] = data["headings"].ravel()
    else:
        pos = result["positions"]
        if pos.shape[0] > 1:
            dx = np.diff(pos[:, 0])
            dy = np.diff(pos[:, 1])
            headings = np.arctan2(dy, dx)
            result["headings"] = np.concatenate([headings[:1], headings])
        else:
            result["headings"] = np.zeros(pos.shape[0], dtype=np.float32)

    if "timestamps" in data:
        ts = data["timestamps"].ravel()
        if len(ts) == 1:
            # Single dt per step
            dt = float(ts[0])
            result["time_axis"] = np.arange(result["positions"].shape[0]) * dt
        else:
            result["time_axis"] = ts
    else:
        result["time_axis"] = np.arange(result["positions"].shape[0]) * 0.05

    return result


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_trajectory(
    episode_data: dict[str, np.ndarray],
    title: str = "MuJoCo Episode Trajectory",
    output_path: Optional[str] = None,
    scan_sample_every: int = 20,
    fov_deg: float = 90.0,
) -> None:
    """Generate full trajectory plot with 4 subplots.

    Args:
        episode_data: Dict from ``load_episode_data()``.
        title: Figure title.
        output_path: If provided, save figure to this path instead of showing.
        scan_sample_every: Plot scan rays every N steps.
        fov_deg: Scan field of view for rendering rays.
    """
    if not HAS_MPL:
        print("[Error] matplotlib is required for plotting.")
        return

    positions = episode_data["positions"]
    velocities = episode_data["velocities"]
    scans = episode_data["scans"]
    goal = episode_data["goal"][:2]
    start = episode_data["start"][:2]
    obstacles = episode_data["obstacles"]
    headings = episode_data["headings"]
    time_axis = episode_data["time_axis"]

    T = positions.shape[0]
    scan_bins = scans.shape[1]

    # Bin angles
    half_fov = np.deg2rad(fov_deg / 2.0)
    bin_angles = np.linspace(-half_fov, half_fov, scan_bins)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # ── (1) Trajectory map (main plot) ──
    ax_map = fig.add_subplot(2, 3, (1, 4))
    ax_map.set_aspect("equal")
    ax_map.set_title("2D Trajectory", fontweight="bold")

    # Room boundary
    room_w, room_h = 5.0, 6.0
    ax_map.plot([0, room_w, room_w, 0, 0], [0, 0, room_h, room_h, 0],
                "k-", lw=1.5, alpha=0.5, label="Room boundary")

    # Obstacles
    for obs in obstacles:
        ox, oy = float(obs[0]), float(obs[1])
        r = float(obs[2]) if obs.shape[0] > 2 else 0.15
        circle = plt.Circle(
            (ox, oy), r, fc="#FFCDD2", ec="#C62828", lw=0.8, alpha=0.7, zorder=3,
        )
        ax_map.add_patch(circle)

    # Start and goal
    ax_map.plot(start[0], start[1], "o", color="#2E7D32", markersize=12,
                label="Start", zorder=9, markeredgecolor="white", markeredgewidth=1.5)
    ax_map.plot(goal[0], goal[1], "*", color="#D32F2F", markersize=18,
                label="Goal", zorder=9, markeredgecolor="white", markeredgewidth=1.0)

    # Robot trajectory
    if T > 1:
        # Color by time (blue -> red)
        points = np.arange(T)
        sc = ax_map.scatter(positions[:, 0], positions[:, 1],
                           c=points, cmap="coolwarm", s=12, alpha=0.8,
                           zorder=5, edgecolors="none", label="Robot path")
        plt.colorbar(sc, ax=ax_map, label="Step", shrink=0.85)
    else:
        ax_map.plot(positions[0, 0], positions[0, 1], "bo", markersize=8)

    # Scan ray overlay at sampled points
    max_scan_range = 5.0
    sample_indices = list(range(0, T, scan_sample_every))
    if T - 1 not in sample_indices and T > 1:
        sample_indices.append(T - 1)

    for idx in sample_indices:
        px, py = positions[idx, 0], positions[idx, 1]
        theta = headings[idx]
        scan_m = scans[idx]
        bin_thetas = bin_angles + theta

        # Draw a subset of rays (every 4th bin)
        for bi in range(0, scan_bins, 4):
            dist = scan_m[bi]
            if dist <= 0.01 or dist >= max_scan_range - 0.1:
                continue
            ang = bin_thetas[bi]
            ex = px + dist * np.cos(ang)
            ey = py + dist * np.sin(ang)
            ax_map.plot([px, ex], [py, ey], color="#64B5F6", lw=0.4, alpha=0.3, zorder=2)

    ax_map.set_xlim(-0.3, room_w + 0.3)
    ax_map.set_ylim(-0.3, room_h + 0.3)
    ax_map.set_xlabel("X (m)")
    ax_map.set_ylabel("Y (m)")
    ax_map.legend(fontsize=8, loc="upper right")
    ax_map.grid(True, alpha=0.2)

    # ── (2) Velocity profile ──
    ax_vel = fig.add_subplot(2, 3, 2)
    if velocities.shape[1] >= 3:
        ax_vel.plot(time_axis, velocities[:, 0], color="#1976D2", lw=1.2, label="vx (m/s)")
        ax_vel.plot(time_axis, velocities[:, 1], color="#388E3C", lw=1.2, label="vy (m/s)")
        ax_vel.plot(time_axis, velocities[:, 2], color="#F57C00", lw=1.2, label="omega (deg/s)")
    else:
        ax_vel.plot(time_axis, velocities[:, 0], color="#1976D2", lw=1.2, label="vx (m/s)")
    ax_vel.set_xlabel("Time (s)")
    ax_vel.set_ylabel("Velocity")
    ax_vel.set_title("Velocity Profile", fontweight="bold")
    ax_vel.legend(fontsize=7)
    ax_vel.grid(True, alpha=0.3)

    # ── (3) Heading error ──
    ax_hdg = fig.add_subplot(2, 3, 3)
    goal_vec = np.array([goal[0] - positions[:, 0], goal[1] - positions[:, 1]])
    goal_angles = np.arctan2(goal_vec[1], goal_vec[0])
    heading_errors = goal_angles - headings
    # Wrap to [-pi, pi]
    heading_errors = np.arctan2(np.sin(heading_errors), np.cos(heading_errors))
    heading_errors_deg = np.rad2deg(heading_errors)

    ax_hdg.fill_between(
        time_axis, -10, 10, alpha=0.08, color="#4CAF50",
        label="Good zone (+/-10 deg)",
    )
    ax_hdg.plot(time_axis, heading_errors_deg, color="#7B1FA2", lw=1.2)
    ax_hdg.axhline(y=0, color="gray", lw=0.8, linestyle=":")
    ax_hdg.set_xlabel("Time (s)")
    ax_hdg.set_ylabel("Heading Error (deg)")
    ax_hdg.set_title("Goal Heading Error Over Time", fontweight="bold")
    ax_hdg.legend(fontsize=7)
    ax_hdg.grid(True, alpha=0.3)

    # ── (4) Distance to goal ──
    ax_dist = fig.add_subplot(2, 3, 5)
    dist_to_goal = np.sqrt(
        (positions[:, 0] - goal[0]) ** 2 + (positions[:, 1] - goal[1]) ** 2
    )
    ax_dist.plot(time_axis, dist_to_goal, color="#00838F", lw=1.5)
    ax_dist.fill_between(time_axis, 0, dist_to_goal, alpha=0.15, color="#00838F")
    ax_dist.axhline(y=0.3, color="#D32F2F", lw=0.8, linestyle="--",
                    label="Goal tolerance (0.3 m)")
    ax_dist.set_xlabel("Time (s)")
    ax_dist.set_ylabel("Distance (m)")
    ax_dist.set_title("Distance to Goal", fontweight="bold")
    ax_dist.legend(fontsize=7)
    ax_dist.grid(True, alpha=0.3)

    # ── (5) Scan overview ──
    ax_scan = fig.add_subplot(2, 3, 6)
    angles_deg = np.linspace(-fov_deg / 2, fov_deg / 2, scan_bins)
    # Show mean and min/max envelope across trajectory
    scan_mean = np.mean(scans, axis=0)
    scan_min = np.min(scans, axis=0)
    scan_max = np.max(scans, axis=0)
    ax_scan.fill_between(angles_deg, scan_min, scan_max,
                        alpha=0.2, color="#FF7043", label="Range (min-max)")
    ax_scan.plot(angles_deg, scan_mean, color="#BF360C", lw=1.5, label="Mean scan")
    ax_scan.set_xlabel("Angle (deg)")
    ax_scan.set_ylabel("Distance (m)")
    ax_scan.set_title("Scan Envelope Over Episode", fontweight="bold")
    ax_scan.legend(fontsize=7)
    ax_scan.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved to {output_path}")
    else:
        plt.show()

    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot 2D trajectory from a recorded MuJoCo episode NPZ file."
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to episode NPZ file (from record_mujoco_dataset.py).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save the plot image (e.g. logs/trajectory.png). "
             "If not provided, the plot is shown interactively.",
    )
    parser.add_argument(
        "--title", type=str, default="MuJoCo Episode Trajectory",
        help="Plot title.",
    )
    parser.add_argument(
        "--scan-sample-every", type=int, default=20,
        help="Plot scan rays every N steps (default: 20).",
    )
    parser.add_argument(
        "--fov-deg", type=float, default=90.0,
        help="Scan field of view in degrees (default: 90.0).",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Suppress interactive display (useful with --output in headless envs).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    print(f"Loading episode data from {args.input}...")
    episode_data = load_episode_data(str(input_path))

    T = episode_data["positions"].shape[0]
    print(f"  Loaded {T} steps.")
    print(f"  Start: ({episode_data['start'][0]:.2f}, {episode_data['start'][1]:.2f})")
    print(f"  Goal:  ({episode_data['goal'][0]:.2f}, {episode_data['goal'][1]:.2f})")
    print(f"  Obstacles: {episode_data['obstacles'].shape[0]}")
    print(f"  Scan bins: {episode_data['scans'].shape[1]}")

    if args.output:
        print(f"  Output: {args.output}")
    elif args.no_show:
        print("  [Warning] No --output and --no-show: nothing will be displayed.")

    if not HAS_MPL:
        print("[Error] matplotlib is required. Install with: pip install matplotlib")
        sys.exit(1)

    matplotlib.use("TkAgg" if args.output is None and not args.no_show else "Agg")

    plot_trajectory(
        episode_data=episode_data,
        title=args.title,
        output_path=args.output,
        scan_sample_every=args.scan_sample_every,
        fov_deg=args.fov_deg,
    )


if __name__ == "__main__":
    main()
