"""
Real-time simulation visualization for LeKiwi Depth Navigation.

Renders the 2D navigation environment with robot pose, obstacles, goal,
scan rays, trajectory history, and live metrics overlay.

Supports rule, DWA, random policies.

Usage:
    python tools/visualize_sim.py --policy rule --episodes 3 --show-rays
    python tools/visualize_sim.py --policy dwa --scan-bins 64
    python tools/visualize_sim.py --policy random --save-video demo.mp4
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle as MplCircle, FancyBboxPatch
    from matplotlib.lines import Line2D

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _build_policy(
    name: str, scan_bins: int
) -> Callable[[np.ndarray, float], tuple[float, float, float]]:
    """Build a policy callable by name.

    Args:
        name: "rule", "dwa", or "random".
        scan_bins: number of scan bins (for matching FOV).

    Returns:
        A callable (scan_m, goal_heading) -> (vx, vy, omega).
    """
    from pc.rule_policy import RulePolicy
    from pc.dwa_policy import DWAPlanner, DWAConfig

    if name == "rule":
        return RulePolicy(scan_bins=scan_bins, fov_deg=90.0)

    if name == "dwa":
        return DWAPlanner(
            config=DWAConfig(num_samples=50),
            scan_bins=scan_bins,
            fov_deg=90.0,
        )

    if name == "random":
        rng = np.random.RandomState()

        def _random(scan_m: np.ndarray, goal_heading: float) -> tuple[float, float, float]:
            return (
                rng.uniform(-0.3, 0.3),
                rng.uniform(-0.3, 0.3),
                rng.uniform(-90.0, 90.0),
            )

        return _random

    raise ValueError(f"Unknown policy: {name}")


def _save_video(frames: list[np.ndarray], path: str, fps: int = 20) -> None:
    """Save a list of RGBA frame arrays to an MP4 file via imageio."""
    try:
        import imageio
    except ImportError:
        print("[warn] imageio not installed. Install with: pip install imageio")
        return

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(path, fps=fps, format="FFMPEG", mode="I")
    for frame in frames:
        writer.append_data((frame * 255).astype(np.uint8))
    writer.close()
    print(f"Video saved to {path}")


class SimVisualizer:
    """Real-time matplotlib visualizer for the 2D navigation environment."""

    def __init__(
        self,
        env: Any,
        policy: Callable[[np.ndarray, float], tuple[float, float, float]],
        show_rays: bool = False,
        pause_sec: float = 0.001,
        record_frames: bool = False,
    ) -> None:
        if not HAS_MPL:
            raise RuntimeError("matplotlib is required. Install with: pip install matplotlib")

        self.env = env
        self.policy = policy
        self.show_rays = show_rays
        self.pause_sec = pause_sec
        self.record_frames = record_frames
        self.frames: list[np.ndarray] = []

        self._scan_bins: int = env.num_scan_bins
        self._scan_range: float = env.scan_range
        self._fov_half: float = np.deg2rad(env.scan_fov_deg / 2.0)
        self._ray_angles: np.ndarray = np.linspace(
            -self._fov_half, self._fov_half, self._scan_bins
        )

        # Pre-allocated state
        self._traj_x: list[float] = []
        self._traj_y: list[float] = []

        # Set up figure
        self.fig = plt.figure("LeKiwi Simulation", figsize=(10, 8))
        gs = self.fig.add_gridspec(1, 1)
        self.ax: plt.Axes = self.fig.add_subplot(gs[0, 0])
        self.ax.set_aspect("equal")

        # Artists (created once, updated in-place for performance)
        self._map_border: Optional[Line2D] = None
        self._obstacle_patches: list[MplCircle] = []
        self._goal_marker: Optional[Line2D] = None
        self._robot_body: Optional[MplCircle] = None
        self._robot_dir: Optional[Line2D] = None
        self._traj_line: Optional[Line2D] = None
        self._ray_lines: list[Line2D] = []
        self._info_text: Optional[Any] = None

        plt.ion()
        self.fig.show()

    # ------------------------------------------------------------------
    def reset_episode(self) -> None:
        """Clear per-episode state."""
        self._traj_x.clear()
        self._traj_y.clear()

    # ------------------------------------------------------------------
    def _draw_map(self) -> None:
        """Draw static map elements: border, obstacles, goal placeholder."""
        ms = self.env.map_size
        self.ax.set_xlim(-0.5, ms + 0.5)
        self.ax.set_ylim(-0.5, ms + 0.5)
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # Border
        (self._map_border,) = self.ax.plot(
            [0, ms, ms, 0, 0],
            [0, 0, ms, ms, 0],
            "k-",
            linewidth=1.5,
            zorder=2,
        )

        # Obstacles
        for ox, oy, orad in self.env._obstacles:
            patch = MplCircle(
                (ox, oy), orad, fc="#FF8A80", ec="#B71C1C", linewidth=1.0, zorder=3, alpha=0.8
            )
            self.ax.add_patch(patch)
            self._obstacle_patches.append(patch)

    # ------------------------------------------------------------------
    def update(self, obs: np.ndarray, info: dict, reward: float) -> None:
        """Redraw all dynamic elements for the current frame.

        Args:
            obs: Full observation vector [scan(64), vx, vy, goal_heading].
            info: Dict from env.step() containing robot state metrics.
            reward: Step reward.
        """
        ax = self.ax
        rx = self.env._robot_x
        ry = self.env._robot_y
        rt = self.env._robot_theta
        gx = self.env._goal_x
        gy = self.env._goal_y

        # --- First-time draw of static elements ---
        if self._map_border is None:
            self._draw_map()

        # --- Trajectory ---
        self._traj_x.append(rx)
        self._traj_y.append(ry)
        if self._traj_line is None:
            (self._traj_line,) = ax.plot(
                self._traj_x, self._traj_y, "b-", linewidth=0.8, alpha=0.5, zorder=5
            )
        else:
            self._traj_line.set_data(self._traj_x, self._traj_y)

        # --- Robot body ---
        robot_r = self.env.robot_radius
        if self._robot_body is None:
            self._robot_body = MplCircle(
                (rx, ry), robot_r, fc="#2196F3", ec="#0D47A1", linewidth=1.2, zorder=8
            )
            ax.add_patch(self._robot_body)
        else:
            self._robot_body.center = (rx, ry)

        # --- Robot heading indicator ---
        hx = rx + robot_r * 1.6 * math.cos(rt)
        hy = ry + robot_r * 1.6 * math.sin(rt)
        if self._robot_dir is None:
            (self._robot_dir,) = ax.plot(
                [rx, hx], [ry, hy], "k-", linewidth=2.0, zorder=9
            )
        else:
            self._robot_dir.set_data([rx, hx], [ry, hy])

        # --- Goal ---
        if self._goal_marker is None:
            (self._goal_marker,) = ax.plot(
                gx, gy, "r*", markersize=14, markeredgewidth=1.0,
                markeredgecolor="#B71C1C", zorder=7, label="Goal"
            )
            ax.legend(loc="upper right", fontsize=7)

        # --- Scan rays ---
        scan_m = obs[: self._scan_bins] * self._scan_range
        if self.show_rays:
            # Remove old ray lines
            for ln in self._ray_lines:
                ln.remove()
            self._ray_lines.clear()

            for i, angle in enumerate(self._ray_angles):
                ray_angle = rt + angle
                dist = scan_m[i]
                ex = rx + dist * math.cos(ray_angle)
                ey = ry + dist * math.sin(ray_angle)
                (ln,) = ax.plot(
                    [rx, ex], [ry, ey], "g-", linewidth=0.4, alpha=0.6, zorder=4
                )
                self._ray_lines.append(ln)

        # --- Info overlay ---
        lines = [
            f"Step: {info.get('step', 0)}",
            f"Dist to goal: {info.get('dist_to_goal', 0):.2f} m",
            f"Min scan: {info.get('min_scan', 0):.2f} m",
            f"Reward: {reward:+.3f}",
        ]
        if info.get("collision"):
            lines.append("COLLISION!")
        elif info.get("reached_goal"):
            lines.append("GOAL REACHED!")
        elif info.get("timed_out"):
            lines.append("TIMED OUT")

        text_str = "\n".join(lines)
        if self._info_text is None:
            self._info_text = ax.text(
                0.02, 0.98, text_str, transform=ax.transAxes,
                fontsize=9, fontfamily="monospace", verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
                zorder=20,
            )
        else:
            self._info_text.set_text(text_str)

        # --- Frame capture ---
        if self.record_frames:
            self.fig.canvas.draw()
            buf = np.asarray(self.fig.canvas.buffer_rgba())
            self.frames.append(buf)

        self.fig.canvas.flush_events()
        plt.pause(self.pause_sec)

    # ------------------------------------------------------------------
    def close(self) -> None:
        plt.ioff()
        plt.close(self.fig)

    # ------------------------------------------------------------------
    def run_episode(self) -> dict:
        """Run a single episode, updating the display each step.

        Returns:
            Episode result dict with status and steps.
        """
        self.reset_episode()
        obs, _ = self.env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0

        while not done:
            scan_norm = obs[: self._scan_bins]
            scan_m = scan_norm * self._scan_range
            goal_heading_norm = obs[-1]
            goal_heading = goal_heading_norm * math.pi

            vx, vy, omega = self.policy(scan_m, goal_heading)
            action = np.array([vx, vy, omega], dtype=np.float32)

            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_steps += 1

            self.update(obs, info, reward)

        status = "reached_goal" if info.get("reached_goal") else (
            "collision" if info.get("collision") else "timeout"
        )
        return {
            "status": status,
            "steps": ep_steps,
            "reward": ep_reward,
            "min_scan": info.get("min_scan", 0),
        }


# ═══════════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-time simulation visualization for LeKiwi Depth Navigation"
    )
    parser.add_argument("--policy", default="rule", choices=["rule", "dwa", "random"],
                        help="Policy type to use for control.")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes to run.")
    parser.add_argument("--scan-bins", type=int, default=64,
                        help="Number of scan bins (32, 64, 128).")
    parser.add_argument("--obstacles", type=int, default=8,
                        help="Number of obstacles in the map.")
    parser.add_argument("--show-rays", action="store_true",
                        help="Draw 64-D scan rays each frame.")
    parser.add_argument("--save-video", default=None, type=str,
                        help="Save rendered frames to MP4 file (e.g. demo.mp4).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not HAS_MPL:
        print("matplotlib is required. Install with: pip install matplotlib")
        sys.exit(1)

    from sim.simple_2d_env import Simple2DNavEnv

    # Build env and policy
    env = Simple2DNavEnv(
        map_size=10.0,
        num_obstacles=args.obstacles,
        num_scan_bins=args.scan_bins,
        scan_fov_deg=90.0,
        scan_range=5.0,
        scan_noise_std=0.02,
        robot_radius=0.15,
        max_vx=0.3,
        max_vy=0.3,
        max_omega=90.0,
        dt=0.05,
        goal_tolerance=0.3,
        max_episode_steps=1200,
        seed=args.seed,
    )
    policy = _build_policy(args.policy, args.scan_bins)

    record = args.save_video is not None
    viz = SimVisualizer(
        env=env,
        policy=policy,
        show_rays=args.show_rays,
        record_frames=record,
    )

    print(f"Policy: {args.policy}  |  Scan bins: {args.scan_bins}  |  Episodes: {args.episodes}")
    print("Close the figure window to stop early.\n")

    results = []
    for ep in range(args.episodes):
        result = viz.run_episode()
        results.append(result)
        print(
            f"  Episode {ep + 1}/{args.episodes}  "
            f"status={result['status']:>12s}  "
            f"steps={result['steps']:4d}  "
            f"reward={result['reward']:+.2f}"
        )

    print(f"\nSummary: {args.episodes} episodes with '{args.policy}' policy\n")

    if record:
        _save_video(viz.frames, args.save_video, fps=20)

    viz.close()
    env.close()


if __name__ == "__main__":
    main()
