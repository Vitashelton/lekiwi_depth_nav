"""
Evaluate trained SAC/MLP policy across all MuJoCo lab worlds.

Reports per-world and aggregate success/collision/timeout metrics.

Command:
    python sim_mujoco/eval/evaluate_mujoco.py --model models/sac_mujoco_lab_empty.zip --episodes 20
    python sim_mujoco/eval/evaluate_mujoco.py --model models/mlp_policy.pt --episodes 20 --render
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from sim_mujoco.envs.lekiwi_depth_scan_env import make_env, EnvConfig

# ── Default test worlds ─────────────────────────────────────────────────────

DEFAULT_WORLDS: list[str] = [
    "lab_empty.xml",
    "lab_single_obstacle.xml",
    "lab_corridor.xml",
    "lab_narrow_gap.xml",
    "lab_cluttered.xml",
]


# ── Policy loading ──────────────────────────────────────────────────────────

def load_policy(model_path: str) -> Any:
    """Load a policy from a checkpoint path.

    Supports:
      - Stable-Baselines3 SAC .zip archives
      - MLPPolicy .pt checkpoints

    Args:
        model_path: Path to the saved model file.

    Returns:
        A callable policy object.  SB3 models use ``.predict(obs, deterministic=True)``;
        MLPPolicy uses ``.predict(obs, deterministic=True)`` as well.

    Raises:
        FileNotFoundError: If the model file does not exist.
        ImportError: If SB3 is needed but not installed.
        ValueError: If the file extension is not recognised.
    """
    model_path = str(Path(model_path).resolve())
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    suffix = Path(model_path).suffix.lower()

    if suffix == ".zip":
        try:
            from stable_baselines3 import SAC
        except ImportError:
            raise ImportError(
                "stable-baselines3 is required for loading SAC models. "
                "Install with: pip install stable-baselines3"
            )
        print(f"  Loading SAC policy from {model_path}")
        return SAC.load(model_path, device="cpu")

    elif suffix == ".pt":
        from pc.mlp_policy import MLPPolicy
        print(f"  Loading MLP policy from {model_path}")
        return MLPPolicy.load(model_path)

    else:
        raise ValueError(
            f"Unsupported model format '{suffix}'. "
            "Expected .zip (SB3 SAC) or .pt (MLPPolicy)."
        )


# ── Episode runner ──────────────────────────────────────────────────────────

def run_episode(
    env: Any,
    policy: Any,
    render: bool = False,
) -> dict[str, Any]:
    """Run a single evaluation episode.

    Args:
        env: Gymnasium-compatible environment.
        policy: Callable policy. Must implement ``predict(obs, deterministic=True)``
            returning ``(action, _)`` or a plain action array.
        render: If True, call ``env.render()`` every step (first episode only).

    Returns:
        Dict with keys: success, collision, timeout, path_length,
        avg_clearance, steps, min_clearance.
    """
    obs, _ = env.reset()
    done = False
    step_times: list[float] = []
    min_scan_values: list[float] = []

    while not done:
        t0 = time.perf_counter()
        result = policy.predict(obs, deterministic=True)
        dt = time.perf_counter() - t0
        step_times.append(dt)

        # SB3 returns (action, _states); MLPPolicy returns ndarray directly
        if isinstance(result, (list, tuple)):
            action = result[0]
        else:
            action = result

        obs, _reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        min_scan_values.append(float(info.get("min_scan", 0.0)))

        if render:
            env.render()

    return {
        "success": bool(info.get("success", False)),
        "collision": bool(info.get("collision", False)),
        "timeout": bool(info.get("timeout", False)),
        "out_of_bounds": bool(info.get("out_of_bounds", False)),
        "path_length": float(info.get("path_length", 0.0)),
        "avg_clearance": float(np.mean(min_scan_values)) if min_scan_values else 0.0,
        "min_clearance": float(np.min(min_scan_values)) if min_scan_values else 0.0,
        "steps": int(info.get("step", 0)),
        "inference_ms": float(np.mean(step_times) * 1000) if step_times else 0.0,
    }


# ── World evaluator ─────────────────────────────────────────────────────────

def evaluate_world(
    world_xml: str,
    policy: Any,
    episodes: int,
    render_first: bool = False,
    max_steps: int = 600,
) -> dict[str, Any]:
    """Evaluate the policy on a single world.

    Args:
        world_xml: World file basename (e.g. "lab_empty.xml").
        policy: Loaded policy object.
        episodes: Number of evaluation episodes.
        render_first: If True, render the first episode.
        max_steps: Max steps per episode.

    Returns:
        Aggregated metrics dict.
    """
    results: dict[str, Any] = {
        "success": 0,
        "collision": 0,
        "timeout": 0,
        "out_of_bounds": 0,
        "path_lengths": [],
        "clearances": [],
        "inference_ms": [],
    }

    for ep in range(episodes):
        env = make_env(
            world_xml=world_xml,
            max_steps=max_steps,
            render_mode="human" if (render_first and ep == 0) else "rgb_array",
        )

        ep_result = run_episode(
            env, policy,
            render=(render_first and ep == 0),
        )

        env.close()

        if ep_result["success"]:
            results["success"] += 1
        elif ep_result["collision"]:
            results["collision"] += 1
        elif ep_result["out_of_bounds"]:
            results["out_of_bounds"] += 1
        elif ep_result["timeout"]:
            results["timeout"] += 1

        results["path_lengths"].append(ep_result["path_length"])
        results["clearances"].append(ep_result["avg_clearance"])
        results["inference_ms"].append(ep_result["inference_ms"])

        status = "SUCCESS" if ep_result["success"] else (
            "COLLISION" if ep_result["collision"] else (
                "OOB" if ep_result["out_of_bounds"] else "TIMEOUT"
            )
        )
        print(f"    Ep {ep+1:3d}/{episodes}: {status:<10s} "
              f"path={ep_result['path_length']:.2f}m "
              f"clr={ep_result['avg_clearance']:.3f}m")

    total = episodes
    return {
        "world": world_xml,
        "success_pct": results["success"] / total * 100,
        "collision_pct": results["collision"] / total * 100,
        "timeout_pct": results["timeout"] / total * 100,
        "oob_pct": results["out_of_bounds"] / total * 100,
        "avg_path_length": float(np.mean(results["path_lengths"])),
        "avg_clearance": float(np.mean(results["clearances"])),
        "avg_inference_ms": float(np.mean(results["inference_ms"])),
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained policy on all MuJoCo lab worlds."
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to policy checkpoint (.zip for SB3 SAC, .pt for MLPPolicy).",
    )
    parser.add_argument(
        "--episodes", type=int, default=20,
        help="Number of evaluation episodes per world (default: 20).",
    )
    parser.add_argument(
        "--worlds", type=str, nargs="*", default=None,
        help="Specific world XML files to evaluate (default: all 5 lab worlds).",
    )
    parser.add_argument(
        "--max-steps", type=int, default=600,
        help="Maximum episode steps (default: 600).",
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Render the first episode of each world.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for reproducibility (default: 0).",
    )
    args = parser.parse_args()

    worlds = args.worlds or DEFAULT_WORLDS
    policy = load_policy(args.model)

    print(f"\n{'='*72}")
    print(f"EVALUATING POLICY: {Path(args.model).name}")
    print(f"{'='*72}")
    print(f"  Worlds:   {len(worlds)}")
    print(f"  Episodes: {args.episodes} per world")
    print(f"  Seed:     {args.seed}")
    print(f"  Render:   {args.render}")
    print(f"{'='*72}\n")

    # Set global seed
    np.random.seed(args.seed)

    all_results: list[dict[str, Any]] = []
    for wi, world_xml in enumerate(worlds):
        print(f"[{wi+1}/{len(worlds)}] World: {world_xml}")
        result = evaluate_world(
            world_xml=world_xml,
            policy=policy,
            episodes=args.episodes,
            render_first=args.render,
            max_steps=args.max_steps,
        )
        all_results.append(result)
        print()

    # ── Summary table ──
    header = (
        f"{'World':<26s} {'Success':>8s} {'Collision':>10s} {'Timeout':>8s} "
        f"{'OOB':>6s} {'Path(m)':>9s} {'Clr(m)':>8s} {'Inf(ms)':>8s}"
    )
    sep = "=" * len(header)
    print(sep)
    print("PER-WORLD RESULTS")
    print(sep)
    print(header)
    print("-" * len(header))
    for r in all_results:
        print(
            f"{r['world']:<26s} "
            f"{r['success_pct']:7.1f}% "
            f"{r['collision_pct']:9.1f}% "
            f"{r['timeout_pct']:7.1f}% "
            f"{r['oob_pct']:5.1f}% "
            f"{r['avg_path_length']:8.2f} "
            f"{r['avg_clearance']:7.3f} "
            f"{r['avg_inference_ms']:7.2f}"
        )

    # ── Aggregate row ──
    agg_success = np.mean([r["success_pct"] for r in all_results])
    agg_collision = np.mean([r["collision_pct"] for r in all_results])
    agg_timeout = np.mean([r["timeout_pct"] for r in all_results])
    agg_oob = np.mean([r["oob_pct"] for r in all_results])
    agg_path = np.mean([r["avg_path_length"] for r in all_results])
    agg_clr = np.mean([r["avg_clearance"] for r in all_results])
    agg_inf = np.mean([r["avg_inference_ms"] for r in all_results])

    print("-" * len(header))
    print(
        f"{'AGGREGATE':<26s} "
        f"{agg_success:7.1f}% "
        f"{agg_collision:9.1f}% "
        f"{agg_timeout:7.1f}% "
        f"{agg_oob:5.1f}% "
        f"{agg_path:8.2f} "
        f"{agg_clr:7.3f} "
        f"{agg_inf:7.2f}"
    )
    print(sep)
    print(f"\n  (averaged over {len(worlds)} worlds, {args.episodes} episodes each)")


if __name__ == "__main__":
    main()
