"""
Compare raw vs residual-corrected actions on all MuJoCo lab worlds.

Tests 4 action modes:
  1. Raw policy (no correction)
  2. Raw + rule safety shield
  3. Raw + learned residual correction
  4. Raw + residual + emergency shield (rule override on danger)

Command:
    python sim_mujoco/eval/evaluate_residual.py \
        --raw-policy rule \
        --residual-model models/residual_correction.pt \
        --episodes 10
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from sim_mujoco.envs.lekiwi_depth_scan_env import make_env, EnvConfig
from pc.rule_policy import RulePolicy
from pc.dwa_policy import DWAPlanner, DWAConfig
from pc.residual_correction import ResidualCorrectionNet
from pc.geometric_risk import compute_directional_risk

# ── Default test worlds ─────────────────────────────────────────────────────

DEFAULT_WORLDS: list[str] = [
    "lab_empty.xml",
    "lab_single_obstacle.xml",
    "lab_corridor.xml",
    "lab_narrow_gap.xml",
    "lab_cluttered.xml",
]

# ── Policy factory functions ────────────────────────────────────────────────

def _make_raw_policy(policy_name: str) -> Callable:
    """Create a raw (uncorrected) policy function.

    Args:
        policy_name: "rule" or "dwa".

    Returns:
        Callable(scan_m, goal_heading) -> (vx, vy, omega).
    """
    if policy_name == "rule":
        rule = RulePolicy()
        return lambda scan_m, goal_heading: rule(scan_m, goal_heading)
    elif policy_name == "dwa":
        dwa_cfg = DWAConfig()
        dwa = DWAPlanner(config=dwa_cfg)
        return lambda scan_m, goal_heading: dwa(scan_m, goal_heading)
    else:
        raise ValueError(f"Unknown raw policy: {policy_name}. Use 'rule' or 'dwa'.")


def _make_mode_functions(
    raw_fn: Callable,
    rule_policy: RulePolicy,
    residual_model: Optional[ResidualCorrectionNet],
) -> dict[str, Callable]:
    """Build the four action-mode functions.

    Each function signature: (scan_m, goal_heading, current_velocity, env) -> action.
    """
    modes: dict[str, Callable] = {}

    # Mode 1: Raw policy (no correction)
    def mode_raw(scan_m, goal_heading, _vel, _env):
        return np.asarray(raw_fn(scan_m, goal_heading), dtype=np.float32)

    modes["raw"] = mode_raw

    # Mode 2: Raw + rule safety shield
    def mode_rule_shield(scan_m, goal_heading, _vel, _env):
        raw_action = np.asarray(raw_fn(scan_m, goal_heading), dtype=np.float32)
        # Override with rule if danger is imminent
        danger = np.min(scan_m) < 0.2
        if danger:
            return np.asarray(rule_policy(scan_m, goal_heading), dtype=np.float32)
        return raw_action

    modes["rule_shield"] = mode_rule_shield

    # Mode 3: Raw + learned residual correction
    if residual_model is not None:
        def mode_residual(scan_m, goal_heading, vel, _env):
            raw_action = np.asarray(raw_fn(scan_m, goal_heading), dtype=np.float32)
            current_vel = vel[:2] if vel is not None else None
            return residual_model.correct_action(
                scan_m, raw_action, current_velocity=current_vel, goal_heading=goal_heading,
            )

        modes["residual"] = mode_residual

        # Mode 4: Raw + residual + emergency shield
        def mode_residual_shield(scan_m, goal_heading, vel, _env):
            raw_action = np.asarray(raw_fn(scan_m, goal_heading), dtype=np.float32)
            current_vel = vel[:2] if vel is not None else None
            corrected = residual_model.correct_action(
                scan_m, raw_action, current_velocity=current_vel, goal_heading=goal_heading,
            )
            # Emergency override: if any direction is extremely close,
            # use rule policy instead
            if np.min(scan_m) < 0.12:
                return np.asarray(rule_policy(scan_m, goal_heading), dtype=np.float32)
            return corrected

        modes["residual_shield"] = mode_residual_shield

    return modes


# ── Episode runner ──────────────────────────────────────────────────────────

def run_episode(
    env: Any,
    mode_fn: Callable,
    max_steps: int = 600,
    render: bool = False,
) -> dict[str, Any]:
    """Run a single episode with the given action mode.

    Args:
        env: Gymnasium-compatible environment.
        mode_fn: Action function with signature
            ``(scan_m, goal_heading, current_velocity, env) -> action``.
        max_steps: Maximum steps before truncation.
        render: Whether to call env.render() each step.

    Returns:
        Dict of metric values.
    """
    obs, _ = env.reset()
    done = False
    prev_action: Optional[np.ndarray] = None
    min_clearances: list[float] = []
    spin_steps: list[int] = []
    smoothness_vals: list[float] = []
    inference_times: list[float] = []
    current_velocity = np.zeros(2, dtype=np.float32)

    while not done:
        # Extract scan and goal heading from observation
        # Obs layout: [scan_norm(64), goal_vec_norm(2), goal_cos/sin(2), vel(3), dist(1)]
        scan_bins = env.unwrapped.cfg.scan_bins
        scan_norm = obs[:scan_bins]
        scan_m = scan_norm * env.unwrapped.cfg.scan_max_range

        goal_cos = float(obs[scan_bins + 2])
        goal_sin = float(obs[scan_bins + 3])
        goal_heading = np.arctan2(goal_sin, goal_cos)

        # Timed inference
        t0 = time.perf_counter()
        action = mode_fn(scan_m, goal_heading, current_velocity, env)
        dt = time.perf_counter() - t0
        inference_times.append(dt * 1000)  # ms

        action = np.asarray(action, dtype=np.float32)

        # Smoothness: L2 diff between consecutive actions
        if prev_action is not None:
            smoothness_vals.append(float(np.sum((action - prev_action) ** 2)))
        prev_action = action.copy()

        obs, _reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

        min_clearances.append(float(info.get("min_scan", 0.0)))
        spin_steps.append(int(info.get("spin_count", 0)))
        current_velocity = np.array([
            float(info.get("lin_vel_x", 0.0)),
            float(info.get("lin_vel_y", 0.0)),
        ], dtype=np.float32)

        if render:
            env.render()

    return {
        "success": bool(info.get("success", False)),
        "collision": bool(info.get("collision", False)),
        "timeout": bool(info.get("timeout", False)),
        "path_length": float(info.get("path_length", 0.0)),
        "min_clearance": float(np.min(min_clearances)) if min_clearances else 0.0,
        "avg_clearance": float(np.mean(min_clearances)) if min_clearances else 0.0,
        "spin_count": int(np.max(spin_steps)) if spin_steps else 0,
        "action_smoothness": float(np.mean(smoothness_vals)) if smoothness_vals else 0.0,
        "inference_latency_ms": float(np.mean(inference_times)) if inference_times else 0.0,
        "steps": int(info.get("step", 0)),
    }


# ── World evaluation ────────────────────────────────────────────────────────

def evaluate_world(
    world_xml: str,
    mode_fns: dict[str, Callable],
    episodes: int,
    render_first: bool = False,
    max_steps: int = 600,
) -> dict[str, dict[str, Any]]:
    """Evaluate all action modes on a single world.

    Args:
        world_xml: World file basename.
        mode_fns: Dict mapping mode name -> action function.
        episodes: Episodes per mode per world.
        render_first: Render first episode of each mode.
        max_steps: Max steps per episode.

    Returns:
        Dict mapping mode name -> aggregated metrics dict.
    """
    mode_results: dict[str, dict[str, Any]] = {}

    for mode_name, mode_fn in mode_fns.items():
        results: dict[str, Any] = {
            "success": 0,
            "collision": 0,
            "timeout": 0,
            "path_lengths": [],
            "min_clearances": [],
            "spin_counts": [],
            "smoothness_vals": [],
            "inference_latencies": [],
        }

        print(f"    Mode: {mode_name:<20s} ", end="", flush=True)

        for ep in range(episodes):
            env = make_env(
                world_xml=world_xml,
                max_steps=max_steps,
                render_mode="human" if (render_first and ep == 0) else "rgb_array",
            )

            ep_result = run_episode(
                env, mode_fn,
                max_steps=max_steps,
                render=(render_first and ep == 0),
            )
            env.close()

            if ep_result["success"]:
                results["success"] += 1
            elif ep_result["collision"]:
                results["collision"] += 1
            elif ep_result["timeout"]:
                results["timeout"] += 1

            results["path_lengths"].append(ep_result["path_length"])
            results["min_clearances"].append(ep_result["min_clearance"])
            results["spin_counts"].append(ep_result["spin_count"])
            results["smoothness_vals"].append(ep_result["action_smoothness"])
            results["inference_latencies"].append(ep_result["inference_latency_ms"])

            status = "." if ep_result["success"] else (
                "C" if ep_result["collision"] else "T"
            )
            print(status, end="", flush=True)

        print()

        total = episodes
        mode_results[mode_name] = {
            "success_pct": results["success"] / total * 100,
            "collision_pct": results["collision"] / total * 100,
            "timeout_pct": results["timeout"] / total * 100,
            "avg_path_length": float(np.mean(results["path_lengths"])),
            "avg_min_clearance": float(np.mean(results["min_clearances"])),
            "avg_spin_count": float(np.mean(results["spin_counts"])),
            "avg_smoothness": float(np.mean(results["smoothness_vals"])),
            "avg_inference_latency_ms": float(np.mean(results["inference_latencies"])),
        }

    return mode_results


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare raw vs residual-corrected actions on MuJoCo lab worlds."
    )
    parser.add_argument(
        "--raw-policy", type=str, default="rule",
        choices=["rule", "dwa"],
        help="Raw (uncorrected) policy to evaluate (default: rule).",
    )
    parser.add_argument(
        "--residual-model", type=str, default=None,
        help="Path to ResidualCorrectionNet checkpoint (.pt).",
    )
    parser.add_argument(
        "--episodes", type=int, default=10,
        help="Number of evaluation episodes per mode per world (default: 10).",
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
        help="Render the first episode of each mode per world.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    args = parser.parse_args()

    worlds = args.worlds or DEFAULT_WORLDS

    # Load raw policy
    raw_fn = _make_raw_policy(args.raw_policy)
    rule_policy = RulePolicy()

    # Load residual model if provided
    residual_model: Optional[ResidualCorrectionNet] = None
    if args.residual_model:
        residual_path = Path(args.residual_model)
        if not residual_path.exists():
            raise FileNotFoundError(f"Residual model not found: {args.residual_model}")
        residual_model = ResidualCorrectionNet.load(str(residual_path))
        print(f"Loaded residual model from {args.residual_model}")
    else:
        print("No residual model provided; modes 3-4 will be skipped.")

    # Build mode functions
    mode_fns = _make_mode_functions(raw_fn, rule_policy, residual_model)
    mode_names = list(mode_fns.keys())

    print(f"\n{'='*80}")
    print(f"RESIDUAL CORRECTION EVALUATION")
    print(f"{'='*80}")
    print(f"  Raw policy:   {args.raw_policy}")
    print(f"  Residual:     {args.residual_model or 'N/A'}")
    print(f"  Worlds:       {len(worlds)}")
    print(f"  Episodes:     {args.episodes} per mode per world")
    print(f"  Modes:        {', '.join(mode_names)}")
    print(f"  Seed:         {args.seed}")
    print(f"  Render:       {args.render}")
    print(f"{'='*80}\n")

    np.random.seed(args.seed)

    # Per-mode aggregator across all worlds
    agg: dict[str, dict[str, list[float]]] = {
        mn: {
            "success_pct": [], "collision_pct": [], "timeout_pct": [],
            "avg_path_length": [], "avg_min_clearance": [],
            "avg_spin_count": [], "avg_smoothness": [],
            "avg_inference_latency_ms": [],
        }
        for mn in mode_names
    }

    for wi, world_xml in enumerate(worlds):
        print(f"[{wi+1}/{len(worlds)}] World: {world_xml}")
        world_result = evaluate_world(
            world_xml=world_xml,
            mode_fns=mode_fns,
            episodes=args.episodes,
            render_first=args.render,
            max_steps=args.max_steps,
        )

        for mn in mode_names:
            wr = world_result[mn]
            for key in agg[mn]:
                agg[mn][key].append(wr[key])

        # Per-world quick summary
        for mn in mode_names:
            wr = world_result[mn]
            print(f"      {mn:<20s}  S:{wr['success_pct']:5.1f}%  "
                  f"C:{wr['collision_pct']:5.1f}%  T:{wr['timeout_pct']:5.1f}%  "
                  f"path={wr['avg_path_length']:.2f}m  clr={wr['avg_min_clearance']:.3f}m")
        print()

    # ── Aggregate comparison table ──
    header = (
        f"\n{'Mode':<22s} {'Success':>8s} {'Collision':>10s} {'Timeout':>8s} "
        f"{'Path(m)':>9s} {'Clr(m)':>8s} {'Spin':>6s} {'Smooth':>8s} {'Lat(ms)':>8s}"
    )
    sep = "=" * len(header)
    print(sep)
    print("AGGREGATE COMPARISON (mean over all worlds)")
    print(sep)
    print(header)
    print("-" * len(header))

    best_success = 0.0
    best_mode = ""

    for mn in mode_names:
        a = agg[mn]
        sp = float(np.mean(a["success_pct"]))
        cp = float(np.mean(a["collision_pct"]))
        tp = float(np.mean(a["timeout_pct"]))
        pl = float(np.mean(a["avg_path_length"]))
        cl = float(np.mean(a["avg_min_clearance"]))
        sc = float(np.mean(a["avg_spin_count"]))
        sm = float(np.mean(a["avg_smoothness"]))
        lt = float(np.mean(a["avg_inference_latency_ms"]))

        if sp > best_success:
            best_success = sp
            best_mode = mn

        print(
            f"{mn:<22s} "
            f"{sp:7.1f}% "
            f"{cp:9.1f}% "
            f"{tp:7.1f}% "
            f"{pl:8.2f} "
            f"{cl:7.3f} "
            f"{sc:5.1f} "
            f"{sm:7.4f} "
            f"{lt:7.2f}"
        )

    print(sep)
    print(f"  Best mode: {best_mode} ({best_success:.1f}% success)")
    print(f"  Metrics are averaged over {len(worlds)} worlds, "
          f"{args.episodes} episodes per mode per world.")
    print()


if __name__ == "__main__":
    main()
