"""
Evaluate and compare action correction methods on STRUCTURED test maps.

Uses 5 built-in maps (corridor, L-turn, doorway, cluttered room, obstacle
field) and runs ALL methods on the EXACT same map/start/goal.

Visualization (enabled by default):
  - Bar chart: success/collision/timeout per method per map
  - Radar chart: normalized safety profile
  - Trajectory overlay: robot paths on the same map
  - Risk-over-time curves

Usage:
    # Generate test maps first (one-time)
    python sim/structured_maps.py --output maps/test_suite/

    # Evaluate all methods on structured maps
    python tools/evaluate_correction.py --maps-dir maps/test_suite/

    # With trained residual model + viz
    python tools/evaluate_correction.py --maps-dir maps/test_suite/ --residual-model models/residual_correction.pt

    # Text-only
    python tools/evaluate_correction.py --maps-dir maps/test_suite/ --no-plot
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.simple_2d_env import Simple2DNavEnv
from pc.rule_policy import RulePolicy
from pc.geometric_risk import (
    compute_directional_risk,
    compute_clearance_cost,
    compute_action_projection,
)
from sim.structured_maps import BUILTIN_MAPS, generate_all

HAS_MPL = False
try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    pass

STYLE = {
    "candidate":  {"color": "#E53935", "marker": "o", "label": "Candidate (raw)"},
    "rule_shield": {"color": "#43A047", "marker": "s", "label": "Rule Shield"},
    "projection":  {"color": "#1E88E5", "marker": "^", "label": "Projection"},
    "residual":    {"color": "#8E24AA", "marker": "D", "label": "Residual Corr."},
}


# ── Mock candidate policy ─────────────────────────────────────────────────

class MockCandidatePolicy:
    """Imperfect policy that needs correction."""

    def __init__(self, max_v: float = 0.3, max_w: float = 90.0, seed: int = 0) -> None:
        self.max_v = max_v
        self.max_w = max_w
        self.rng = np.random.RandomState(seed)

    def __call__(self, scan_m: np.ndarray, goal_heading: float) -> tuple[float, float, float]:
        vx = float(self.max_v * 0.5 * math.cos(goal_heading)
                    + self.rng.normal(0, 0.08))
        vy = float(self.max_v * 0.5 * math.sin(goal_heading)
                    + self.rng.normal(0, 0.08))
        omega = float(30.0 * math.sin(goal_heading) + self.rng.normal(0, 20.0))
        return (max(-self.max_v, min(self.max_v, vx)),
                max(-self.max_v, min(self.max_v, vy)),
                max(-self.max_w, min(self.max_w, omega)))


# ── Apply map state to environment ────────────────────────────────────────

def apply_map_to_env(env: Simple2DNavEnv, map_data: dict) -> np.ndarray:
    """Override environment state with pre-defined map and return initial obs."""
    env._obstacles = [(o["x"], o["y"], o["r"]) for o in map_data["obstacles"]]
    sx, sy = map_data["start"]
    gx, gy = map_data["goal"]
    env._robot_x = float(sx)
    env._robot_y = float(sy)
    env._robot_theta = 0.0
    env._robot_vx = 0.0
    env._robot_vy = 0.0
    env._robot_omega = 0.0
    env._goal_x = float(gx)
    env._goal_y = float(gy)
    env._step_count = 0
    env._prev_action = None
    return env._get_obs()


# ── Run one episode with trace ────────────────────────────────────────────

def run_episode_with_trace(
    env: Simple2DNavEnv,
    policy_fn: Callable,
    max_steps: int = 800,
    record_trace: bool = False,
) -> dict:
    """Run episode, return per-step metrics."""
    obs = env._get_obs()
    done = False

    risks: list[float] = []
    clearances: list[float] = []
    deviations: list[float] = []
    smoothness_vals: list[float] = []
    prev_action: Optional[np.ndarray] = None
    trace: dict = {}

    if record_trace:
        trace = {"x": [], "y": [], "scan_m": [], "action": [], "risk": []}

    while not done:
        scan_norm = obs[:env.num_scan_bins]
        scan_m = scan_norm * env.scan_range
        goal_heading_norm = obs[-1]
        goal_heading = goal_heading_norm * math.pi

        action = np.asarray(policy_fn(scan_m, goal_heading), dtype=np.float32)

        # Metrics
        ideal_vx = 0.2 * math.cos(goal_heading)
        ideal_vy = 0.2 * math.sin(goal_heading)
        deviations.append(float(math.hypot(action[0] - ideal_vx, action[1] - ideal_vy)))

        risk = compute_directional_risk(scan_m, action)
        risks.append(float(risk))

        clr_cost = compute_clearance_cost(scan_m, action)
        clearances.append(float(1.0 - clr_cost))

        if prev_action is not None:
            smoothness_vals.append(float(np.sum((action - prev_action) ** 2)))
        prev_action = action

        if record_trace:
            trace["x"].append(float(env._robot_x))
            trace["y"].append(float(env._robot_y))
            trace["scan_m"].append(scan_m.copy())
            trace["action"].append(action.copy())
            trace["risk"].append(float(risk))

        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    reached = bool(info.get("reached_goal", False))
    collided = bool(info.get("collision", False))

    result = {
        "success": reached and not collided,
        "collision": collided,
        "timeout": not reached and not collided,
        "steps": int(info.get("step", 0)),
        "mean_risk": float(np.mean(risks)) if risks else 0.0,
        "mean_clearance": float(np.mean(clearances)) if clearances else 0.0,
        "mean_deviation": float(np.mean(deviations)) if deviations else 0.0,
        "mean_smoothness": float(np.mean(smoothness_vals)) if smoothness_vals else 0.0,
        "min_scan": float(info.get("min_scan", 0.0)),
        "dist_to_goal": float(info.get("dist_to_goal", 0.0)),
    }
    if record_trace:
        result["trace"] = {k: (np.array(v) if k != "scan_m" else v) for k, v in trace.items()}
    return result


# ── Main evaluation over structured maps ──────────────────────────────────

def evaluate_on_maps(
    maps: list[dict],
    candidate_policy: Callable,
    rule_policy: RulePolicy,
    rng: np.random.RandomState,
    residual_model: Any = None,
    scan_bins: int = 64,
    max_steps: int = 800,
) -> tuple[dict, dict]:
    """Run ALL methods on ALL maps. Returns (summary, trace_results)."""
    method_names = ["candidate", "rule_shield", "projection"]
    if residual_model is not None:
        method_names.append("residual")

    all_results: dict[str, list[dict]] = {n: [] for n in method_names}
    # Save trace of first map for visualization
    trace_results: dict[str, dict] = {}

    for mi, map_data in enumerate(maps):
        map_name = map_data.get("name", f"map_{mi:03d}")
        record_trace = (mi == 0)  # record full trace for first map only

        for mname in method_names:
            env = Simple2DNavEnv(
                num_scan_bins=scan_bins,
                num_obstacles=0,  # obstacles injected from map
                scan_fov_deg=90.0,
                scan_range=5.0,
                scan_noise_std=0.02,
                robot_radius=0.15,
                max_vx=0.3,
                max_vy=0.3,
                max_omega=90.0,
                dt=0.05,
                goal_tolerance=0.3,
                max_episode_steps=max_steps,
                seed=0,
            )
            # Inject map state
            apply_map_to_env(env, map_data)

            # Build policy function
            if mname == "candidate":
                fn = candidate_policy
            elif mname == "rule_shield":
                fn = rule_policy
            elif mname == "projection":
                def fn(sm, gh, _rng=rng, _cp=candidate_policy):
                    ca = np.asarray(_cp(sm, gh), dtype=np.float32)
                    return compute_action_projection(ca, sm, lambda_risk=2.0, num_samples=200, rng=_rng)
            elif mname == "residual" and residual_model is not None:
                def fn(sm, gh, _rm=residual_model, _cp=candidate_policy):
                    ca = np.asarray(_cp(sm, gh), dtype=np.float32)
                    return _rm.correct_action(sm, ca, goal_heading=gh)
            else:
                continue

            res = run_episode_with_trace(env, fn, max_steps=max_steps, record_trace=record_trace)
            res["map_name"] = map_name
            all_results[mname].append(res)

            if record_trace:
                trace_results[mname] = {
                    **res,
                    "map_data": map_data,
                }

            env.close()

        # Print per-map summary
        print(f"  {map_name:<22s} ", end="")
        for mname in method_names:
            r = all_results[mname][-1]
            status = "✓" if r["success"] else ("✗" if r["collision"] else "○")
            print(f" {STYLE[mname]['label'][:6]:>6s}:{status}", end="")
        print()

    # Aggregate summary
    summary = {}
    for mname in method_names:
        results = all_results[mname]
        n = len(results)
        summary[mname] = {
            "success_rate": sum(r["success"] for r in results) / n * 100,
            "collision_rate": sum(r["collision"] for r in results) / n * 100,
            "timeout_rate": sum(r["timeout"] for r in results) / n * 100,
            "mean_risk": np.mean([r["mean_risk"] for r in results]),
            "mean_clearance": np.mean([r["mean_clearance"] for r in results]),
            "mean_deviation": np.mean([r["mean_deviation"] for r in results]),
            "mean_smoothness": np.mean([r["mean_smoothness"] for r in results]),
            "mean_min_scan": np.mean([r["min_scan"] for r in results]),
            "per_map": results,
        }
    return summary, trace_results


# ── Visualization ─────────────────────────────────────────────────────────

def plot_results(summary: dict, trace_results: dict) -> None:
    """Dashboard with bar chart, radar, trajectory overlay, risk curves."""
    if not HAS_MPL:
        print("\n[!] matplotlib not available — skipping visualization.")
        return

    methods = list(summary.keys())
    colors = [STYLE[m]["color"] for m in methods]
    labels = [STYLE[m]["label"] for m in methods]

    fig = plt.figure("Correction Method Evaluation", figsize=(18, 12))

    # ── (1) Bar chart: per-map success/collision/timeout ──
    ax1 = fig.add_subplot(2, 3, 1)
    n_maps = len(summary[methods[0]]["per_map"])
    map_names = [summary[methods[0]]["per_map"][i]["map_name"][:12] for i in range(n_maps)]
    x = np.arange(n_maps)
    w = 0.2
    for i, m in enumerate(methods):
        vals = [r["success"] * 100 / 1 for r in summary[m]["per_map"]]
        ax1.bar(x + i * w, vals, w, label=labels[i], color=colors[i], alpha=0.85)
    ax1.set_xticks(x + w * (len(methods) - 1) / 2)
    ax1.set_xticklabels(map_names, fontsize=7, rotation=30, ha="right")
    ax1.set_ylabel("Success Rate (%)")
    ax1.set_title("Per-Map Success Rate", fontweight="bold")
    ax1.legend(fontsize=7)
    ax1.grid(axis="y", alpha=0.3)

    # ── (2) Radar chart ──
    ax2 = fig.add_subplot(2, 3, 2, projection="polar")
    radar_labels = ["Safety\n(1-risk)", "Clearance", "Success", "Low Dev.", "Smooth", "Min Scan"]
    angles = np.linspace(0, 2 * np.pi, len(radar_labels), endpoint=False).tolist()
    angles += angles[:1]
    for i, m in enumerate(methods):
        s = summary[m]
        values = [
            1.0 - s["mean_risk"],
            s["mean_clearance"],
            s["success_rate"] / 100.0,
            1.0 - min(s["mean_deviation"] / 0.3, 1.0),
            1.0 - min(s["mean_smoothness"] / 500.0, 1.0),
            min(s["mean_min_scan"] / 5.0, 1.0),
        ]
        values += values[:1]
        ax2.fill(angles, values, alpha=0.12, color=colors[i])
        ax2.plot(angles, values, "o-", lw=1.5, label=labels[i], color=colors[i], markersize=3)
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(radar_labels, fontsize=7)
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Safety Profile (higher = better)", fontweight="bold", pad=20)
    ax2.legend(loc="lower right", fontsize=6, bbox_to_anchor=(1.35, -0.1))

    # ── (3) Trajectory overlay ──
    ax3 = fig.add_subplot(2, 3, (3, 4))
    ax3.set_aspect("equal")
    ax3.set_title("Trajectory Comparison (first map)", fontweight="bold")

    if trace_results:
        first = next(iter(trace_results.values()))
        map_data = first.get("map_data", {})
        ms = map_data.get("map_size", 10.0)

        # Draw obstacles
        for obs in map_data.get("obstacles", []):
            circ = plt.Circle((obs["x"], obs["y"]), obs["r"],
                              fc="#FF8A80", ec="#B71C1C", lw=0.5, alpha=0.7, zorder=3)
            ax3.add_patch(circ)

        # Draw start and goal
        sx, sy = map_data.get("start", [0, 0])
        gx, gy = map_data.get("goal", [0, 0])
        ax3.plot(sx, sy, "ko", markersize=10, label="Start", zorder=9)
        ax3.plot(gx, gy, "r*", markersize=16, markeredgewidth=1.0, label="Goal", zorder=9)

        # Trajectories
        for m in methods:
            if m in trace_results and "trace" in trace_results[m]:
                t = trace_results[m]["trace"]
                xs, ys = t.get("x", []), t.get("y", [])
                if len(xs) > 0:
                    ax3.plot(xs, ys, color=STYLE[m]["color"], lw=1.5, alpha=0.85,
                             label=STYLE[m]["label"], marker=STYLE[m]["marker"],
                             markersize=3, markevery=max(1, len(xs) // 8))

        ax3.set_xlim(-0.5, ms + 0.5)
        ax3.set_ylim(-0.5, ms + 0.5)
        ax3.legend(fontsize=7, loc="upper right")
        ax3.set_xlabel("X (m)")
        ax3.set_ylabel("Y (m)")
        ax3.grid(True, alpha=0.2)

    # ── (4) Per-metric bar comparison ──
    ax4 = fig.add_subplot(2, 3, 5)
    mkeys = ["mean_risk", "mean_clearance", "mean_deviation", "mean_min_scan"]
    mlabels = ["Risk (↓)", "Clearance (↑)", "Deviation (↓)", "Min Scan (↑)"]
    x = np.arange(len(mlabels))
    w = 0.2
    for i, m in enumerate(methods):
        s = summary[m]
        vals = [s["mean_risk"], s["mean_clearance"], s["mean_deviation"], s["mean_min_scan"]]
        ax4.bar(x + i * w, vals, w, label=labels[i], color=colors[i], alpha=0.85)
    ax4.set_xticks(x + w * (len(methods) - 1) / 2)
    ax4.set_xticklabels(mlabels, fontsize=8)
    ax4.set_title("Aggregate Metrics", fontweight="bold")
    ax4.legend(fontsize=7)
    ax4.grid(axis="y", alpha=0.3)

    # ── (5) Risk curves ──
    ax5 = fig.add_subplot(2, 3, 6)
    for m in methods:
        if m in trace_results and "trace" in trace_results[m]:
            t = trace_results[m]["trace"]
            risk_arr = np.asarray(t.get("risk", []))
            if len(risk_arr) > 0:
                ax5.plot(np.arange(len(risk_arr)), risk_arr,
                         color=STYLE[m]["color"], lw=1.0, alpha=0.85,
                         label=STYLE[m]["label"])
    ax5.set_xlabel("Step")
    ax5.set_ylabel("Collision Risk")
    ax5.set_title("Risk Over Time (first map)", fontweight="bold")
    ax5.legend(fontsize=7)
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate correction methods on structured maps")
    parser.add_argument("--maps-dir", default=None,
                        help="Directory of pre-generated map JSON files.")
    parser.add_argument("--residual-model", default=None,
                        help="Path to ResidualCorrectionNet checkpoint.")
    parser.add_argument("--scan-bins", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip visualization, print tables only.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)

    # ── Load maps ──
    if args.maps_dir:
        map_dir = Path(args.maps_dir)
        if not map_dir.exists():
            print(f"Map directory not found: {args.maps_dir}")
            print("Generate maps first: python sim/structured_maps.py --output maps/test_suite/")
            sys.exit(1)
        map_files = sorted(map_dir.glob("*.json"))
        maps = []
        for mf in map_files:
            with open(mf) as f:
                maps.append(json.load(f))
        print(f"Loaded {len(maps)} maps from {args.maps_dir}")
    else:
        print("No --maps-dir specified. Generating built-in structured maps...")
        maps = generate_all(map_size=10.0)
        print(f"Using {len(maps)} built-in maps: {[m['name'] for m in maps]}")
        print("  (Save them with: python sim/structured_maps.py --output maps/test_suite/)")

    if not maps:
        print("No maps available.")
        sys.exit(1)

    candidate_policy = MockCandidatePolicy(seed=args.seed)
    rule_policy = RulePolicy(scan_bins=args.scan_bins, fov_deg=90.0)

    residual_model = None
    if args.residual_model:
        from pc.residual_correction import ResidualCorrectionNet
        residual_model = ResidualCorrectionNet.load(args.residual_model)
        print(f"Loaded residual model from {args.residual_model}")

    print(f"\nEvaluating on {len(maps)} structured maps ...\n")
    summary, trace_results = evaluate_on_maps(
        maps=maps,
        candidate_policy=candidate_policy,
        rule_policy=rule_policy,
        rng=rng,
        residual_model=residual_model,
        scan_bins=args.scan_bins,
        max_steps=args.max_steps,
    )

    # ── Print table ──
    header = (f"\n{'Method':<22s} {'Success':>8s} {'Collision':>10s} {'Timeout':>8s} "
              f"{'Risk':>7s} {'Clear':>7s} {'Dev':>7s}")
    sep = "=" * len(header)
    print(sep)
    print(header)
    print("-" * len(header))
    for m in ["candidate", "rule_shield", "projection", "residual"]:
        if m not in summary:
            continue
        s = summary[m]
        print(f"{STYLE[m]['label']:<22s} "
              f"{s['success_rate']:7.1f}% "
              f"{s['collision_rate']:9.1f}% "
              f"{s['timeout_rate']:7.1f}% "
              f"{s['mean_risk']:6.3f} "
              f"{s['mean_clearance']:6.3f} "
              f"{s['mean_deviation']:6.3f}")
    print(sep)
    print("\n  Risk = directional collision risk (lower is safer)")
    print("  Clear = clearance score (higher is safer)")
    print("  Dev = action deviation from goal-directed baseline")

    if not args.no_plot and HAS_MPL:
        plot_results(summary, trace_results)
    elif args.no_plot:
        print("\n  (--no-plot: visualization skipped)")
    else:
        print("\n  [!] matplotlib not available.")


if __name__ == "__main__":
    main()
