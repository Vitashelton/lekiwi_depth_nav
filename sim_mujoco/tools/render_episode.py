"""
Render a MuJoCo episode with policy control to frames or video.

Supports rule-based, DWA, and random exploration policies.
Outputs an MP4 video using imageio.

Command:
    python sim_mujoco/tools/render_episode.py \
        --world lab_corridor.xml \
        --policy dwa \
        --output episode.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from sim_mujoco.envs.lekiwi_depth_scan_env import make_env, EnvConfig
from pc.rule_policy import RulePolicy
from pc.dwa_policy import DWAPlanner, DWAConfig


# ── Policy factory ──────────────────────────────────────────────────────────

def build_policy_fn(
    policy_name: str,
    scan_bins: int = 64,
    fov_deg: float = 90.0,
    seed: int = 0,
) -> tuple[Callable, str]:
    """Build a policy function for the given name.

    Args:
        policy_name: "rule", "dwa", or "random".
        scan_bins: Number of scan bins.
        fov_deg: Scan field of view (degrees).
        seed: Random seed.

    Returns:
        Tuple of (policy_callable, display_name).

    Raises:
        ValueError: If policy_name is unknown.
    """
    if policy_name == "rule":
        rule = RulePolicy(scan_bins=scan_bins, fov_deg=fov_deg)

        def fn(scan_m: np.ndarray, goal_heading: float) -> np.ndarray:
            return np.asarray(rule(scan_m, goal_heading), dtype=np.float32)

        return fn, "Rule-Based"

    elif policy_name == "dwa":
        dwa_cfg = DWAConfig()
        dwa = DWAPlanner(config=dwa_cfg, scan_bins=scan_bins, fov_deg=fov_deg)

        def fn(scan_m: np.ndarray, goal_heading: float) -> np.ndarray:
            return np.asarray(dwa(scan_m, goal_heading), dtype=np.float32)

        return fn, "DWA Planner"

    elif policy_name == "random":
        rng = np.random.RandomState(seed)

        def fn(scan_m: np.ndarray, goal_heading: float) -> np.ndarray:
            del scan_m, goal_heading
            return np.array([
                rng.uniform(-0.3, 0.3),
                rng.uniform(-0.3, 0.3),
                rng.uniform(-90.0, 90.0),
            ], dtype=np.float32)

        return fn, "Random Exploration"

    else:
        raise ValueError(
            f"Unknown policy '{policy_name}'. Use 'rule', 'dwa', or 'random'."
        )


# ── Renderer ────────────────────────────────────────────────────────────────

def render_episode(
    world_xml: str,
    policy_fn: Callable[[np.ndarray, float], np.ndarray],
    output_path: str,
    max_steps: int = 600,
    fps: int = 20,
    scan_bins: int = 64,
    seed: int = 0,
    start_xy: tuple[float, float] = (0.5, 1.5),
    goal_xy: tuple[float, float] = (4.0, 1.5),
) -> dict[str, Any]:
    """Run an episode and render frames to a video file.

    Args:
        world_xml: World XML filename.
        policy_fn: Policy function (scan_m, goal_heading) -> action.
        output_path: Output MP4 path.
        max_steps: Maximum episode steps.
        fps: Frames per second in output video.
        scan_bins: Number of scan bins.
        seed: Random seed.
        start_xy: (x, y) start position.
        goal_xy: (x, y) goal position.

    Returns:
        Episode summary dict.
    """
    env = make_env(world_xml=world_xml, max_steps=max_steps, render_mode="rgb_array")

    # Set custom start/goal
    env_cfg = env.unwrapped.cfg
    env_cfg.start_goal_pairs = [(start_xy, goal_xy)]

    obs, _ = env.reset(seed=seed)
    done = False
    frames: list[np.ndarray] = []

    while not done:
        # Render current frame
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        # Compute action
        scan_norm = obs[:scan_bins]
        scan_m = scan_norm * env.unwrapped.cfg.scan_max_range

        # Goal heading from observation
        goal_cos = float(obs[scan_bins + 2])
        goal_sin = float(obs[scan_bins + 3])
        goal_heading = np.arctan2(goal_sin, goal_cos)

        action = policy_fn(scan_m, goal_heading)

        obs, _reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

    # Render final frame
    frame = env.render()
    if frame is not None:
        frames.append(frame)

    env.close()

    # ── Write video ──
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import imageio
    except ImportError:
        raise ImportError(
            "imageio is required for video writing. "
            "Install with: pip install imageio imageio-ffmpeg"
        )

    writer = imageio.get_writer(
        str(output_path),
        fps=fps,
        codec="libx264",
        format="FFMPEG",
        macro_block_size=None,
    )

    for frame in frames:
        writer.append_data(frame)

    writer.close()
    print(f"  Saved {len(frames)} frames to {output_path}")

    return {
        "success": bool(info.get("success", False)),
        "collision": bool(info.get("collision", False)),
        "timeout": bool(info.get("timeout", False)),
        "steps": int(info.get("step", 0)),
        "path_length": float(info.get("path_length", 0.0)),
        "frames": len(frames),
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a MuJoCo navigation episode to an MP4 video."
    )
    parser.add_argument(
        "--world", type=str, required=True,
        help="World XML file (e.g. lab_corridor.xml).",
    )
    parser.add_argument(
        "--policy", type=str, default="rule",
        choices=["rule", "dwa", "random"],
        help="Policy to control the robot (default: rule).",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output video path (e.g. episode.mp4).",
    )
    parser.add_argument(
        "--max-steps", type=int, default=600,
        help="Maximum episode steps (default: 600).",
    )
    parser.add_argument(
        "--fps", type=int, default=20,
        help="Frames per second in output video (default: 20).",
    )
    parser.add_argument(
        "--scan-bins", type=int, default=64,
        help="Number of scan bins (default: 64).",
    )
    parser.add_argument(
        "--fov-deg", type=float, default=90.0,
        help="Scan field of view in degrees (default: 90.0).",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for reproducibility (default: 0).",
    )
    parser.add_argument(
        "--start-x", type=float, default=0.5,
        help="Robot start X coordinate (default: 0.5).",
    )
    parser.add_argument(
        "--start-y", type=float, default=1.5,
        help="Robot start Y coordinate (default: 1.5).",
    )
    parser.add_argument(
        "--goal-x", type=float, default=4.0,
        help="Goal X coordinate (default: 4.0).",
    )
    parser.add_argument(
        "--goal-y", type=float, default=1.5,
        help="Goal Y coordinate (default: 1.5).",
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"RENDERING EPISODE")
    print(f"{'='*60}")
    print(f"  World:    {args.world}")
    print(f"  Policy:   {args.policy}")
    print(f"  Output:   {args.output}")
    print(f"  Start:    ({args.start_x}, {args.start_y})")
    print(f"  Goal:     ({args.goal_x}, {args.goal_y})")
    print(f"  Max steps:{args.max_steps}")
    print(f"  FPS:      {args.fps}")
    print(f"  Seed:     {args.seed}")
    print(f"{'='*60}\n")

    np.random.seed(args.seed)

    policy_fn, policy_label = build_policy_fn(
        args.policy, scan_bins=args.scan_bins, fov_deg=args.fov_deg, seed=args.seed,
    )
    print(f"  Using policy: {policy_label}")

    summary = render_episode(
        world_xml=args.world,
        policy_fn=policy_fn,
        output_path=args.output,
        max_steps=args.max_steps,
        fps=args.fps,
        scan_bins=args.scan_bins,
        seed=args.seed,
        start_xy=(args.start_x, args.start_y),
        goal_xy=(args.goal_x, args.goal_y),
    )

    print(f"\n{'='*60}")
    print(f"EPISODE SUMMARY")
    print(f"{'='*60}")
    status = (
        "SUCCESS" if summary["success"]
        else "COLLISION" if summary["collision"]
        else "TIMEOUT"
    )
    print(f"  Status:       {status}")
    print(f"  Steps:        {summary['steps']}")
    print(f"  Path length:  {summary['path_length']:.2f} m")
    print(f"  Frames saved: {summary['frames']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
