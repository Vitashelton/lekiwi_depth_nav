"""
Evaluate and compare action correction methods with visualization.

Compares four action sources:
  1. candidate_action     (raw LeRobot / mock policy)
  2. rule_shield           (rule-based safe action)
  3. projection            (geometric action projection)
  4. residual_correction   (learned residual correction)

Metrics:
  collision_risk | action_deviation | smoothness | min_clearance | success_rate

Visualization (enabled by default):
  - Bar chart: success/collision/timeout rates per method
  - Radar chart: normalized safety metrics per method
  - Single-episode trajectory comparison on the SAME map
  - Risk-over-time curves for each method

Usage:
    # Compare candidate, rule, projection (no residual model needed)
    python tools/evaluate_correction.py --episodes 50

    # With trained residual model
    python tools/evaluate_correction.py --episodes 50 --residual-model models/residual_correction.pt

    # Text-only (no GUI)
    python tools/evaluate_correction.py --episodes 30 --no-plot
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.simple_2d_env import Simple2DNavEnv
from pc.rule_policy import RulePolicy
from pc.geometric_risk import (
    compute_directional_risk,
    compute_clearance_cost,
    compute_action_projection,
)

HAS_MPL = False
try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle as MplCircle
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
    """Imperfect LeRobot-like policy that needs correction."""

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
        vx = max(-self.max_v, min(self.max_v, vx))
        vy = max(-self.max_v, min(self.max_v, vy))
        omega = max(-self.max_w, min(self.max_w, omega))
        return vx, vy, omega


# ── Episode runner with full trace collection ─────────────────────────────

def run_episode_with_trace(
    env: Simple2DNavEnv,
    policy_fn,
    max_steps: int = 600,
    record_trace: bool = False,
) -> dict:
    """Run one episode, optionally recording full trajectory trace."""
    obs, _ = env.reset()
    done = False

    collision_risks: list[float] = []
    clearances: list[float] = []
    deviations: list[float] = []
    smoothness_vals: list[float] = []
    prev_action: Optional[np.ndarray] = None

    trace: dict = {}
    if record_trace:
        trace = {
            "x": [], "y": [], "theta": [], "scan_m": [],
            "action": [], "risk": [],
        }

    while not done:
        scan_norm = obs[:env.num_scan_bins]
        scan_m = scan_norm * env.scan_range
        goal_heading_norm = obs[-1]
        goal_heading = goal_heading_norm * math.pi

        action = np.asarray(policy_fn(scan_m, goal_heading), dtype=np.float32)

        ideal_vx = 0.2 * math.cos(goal_heading)
        ideal_vy = 0.2 * math.sin(goal_heading)
        deviation = math.hypot(action[0] - ideal_vx, action[1] - ideal_vy)
        deviations.append(float(deviation))

        risk = compute_directional_risk(scan_m, action)
        collision_risks.append(float(risk))

        clearance_cost = compute_clearance_cost(scan_m, action)
        clearances.append(float(1.0 - clearance_cost))

        if prev_action is not None:
            smooth = float(np.sum((action - prev_action) ** 2))
            smoothness_vals.append(smooth)
        prev_action = action

        if record_trace:
            trace["x"].append(float(env._robot_x))
            trace["y"].append(float(env._robot_y))
            trace["theta"].append(float(env._robot_theta))
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
        "mean_risk": float(np.mean(collision_risks)) if collision_risks else 0.0,
        "mean_clearance": float(np.mean(clearances)) if clearances else 0.0,
        "mean_deviation": float(np.mean(deviations)) if deviations else 0.0,
        "mean_smoothness": float(np.mean(smoothness_vals)) if smoothness_vals else 0.0,
        "min_scan": float(info.get("min_scan", 0.0)),
        "dist_to_goal": float(info.get("dist_to_goal", 0.0)),
    }
    if record_trace:
        result["trace"] = {k: np.array(v) if isinstance(v, list) else v for k, v in trace.items()}
    return result


# ── Evaluation ────────────────────────────────────────────────────────────

def evaluate_all(
    env_factory,
    candidate_policy,
    rule_policy: RulePolicy,
    rng: np.random.RandomState,
    residual_model=None,
    num_episodes: int = 50,
    seed: int = 42,
    trace_episode_seed: int = 0,
) -> dict:
    """Run all methods across episodes. One episode is saved with full trace."""
    method_names = ["candidate", "rule_shield", "projection"]
    if residual_model is not None:
        method_names.append("residual")

    all_results: dict[str, list[dict]] = {n: [] for n in method_names}
    trace_results: dict[str, dict] = {}

    for ep in range(num_episodes):
        ep_seed = seed + ep
        record = (ep == 0)

        for mname in method_names:
            env = env_factory(ep_seed)
            env.reset(seed=ep_seed)

            if mname == "candidate":
                fn = candidate_policy
            elif mname == "rule_shield":
                fn = rule_policy
            elif mname == "projection":
                def fn(sm, gh, _rng=rng):
                    ca = np.asarray(candidate_policy(sm, gh), dtype=np.float32)
                    return compute_action_projection(ca, sm, lambda_risk=2.0, num_samples=200, rng=_rng)
            elif mname == "residual" and residual_model is not None:
                def fn(sm, gh, _rm=residual_model):
                    ca = np.asarray(candidate_policy(sm, gh), dtype=np.float32)
                    return _rm.correct_action(sm, ca, goal_heading=gh)
            else:
                continue

            res = run_episode_with_trace(env, fn, max_steps=600, record_trace=record)
            all_results[mname].append(res)
            if record:
                trace_results[mname] = res
            env.close()

        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep + 1}/{num_episodes}")

    summary = {}
    for name, results in all_results.items():
        n = len(results)
        summary[name] = {
            "success_rate": sum(r["success"] for r in results) / n * 100,
            "collision_rate": sum(r["collision"] for r in results) / n * 100,
            "timeout_rate": sum(r["timeout"] for r in results) / n * 100,
            "mean_risk": np.mean([r["mean_risk"] for r in results]),
            "mean_clearance": np.mean([r["mean_clearance"] for r in results]),
            "mean_deviation": np.mean([r["mean_deviation"] for r in results]),
            "mean_smoothness": np.mean([r["mean_smoothness"] for r in results]),
            "mean_min_scan": np.mean([r["min_scan"] for r in results]),
        }
    return summary, trace_results


# ── Visualization ─────────────────────────────────────────────────────────

def plot_results(summary: dict, trace_results: dict, map_size: float = 10.0) -> None:
    """Generate a 2×2 dashboard: bar chart, radar, trajectory, risk curves."""
    if not HAS_MPL:
        print("\n[!] matplotlib not available — skipping visualization.")
        print("    Install: pip install matplotlib")
        return

    methods = list(summary.keys())
    colors = [STYLE[m]["color"] for m in methods]
    labels = [STYLE[m]["label"] for m in methods]

    fig = plt.figure("Correction Method Evaluation", figsize=(16, 12))

    # ── (1) Bar chart: Success / Collision / Timeout ──
    ax1 = fig.add_subplot(2, 3, 1)
    x = np.arange(len(methods))
    w = 0.25
    for i, (key, lbl) in enumerate([("success_rate", "Success"), ("collision_rate", "Collision"), ("timeout_rate", "Timeout")]):
        vals = [summary[m][key] for m in methods]
        bars = ax1.bar(x + i * w, vals, w, label=lbl, alpha=0.85,
                       color=["#43A047", "#E53935", "#FFA726"][i])
    ax1.set_xticks(x + w)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("Rate (%)")
    ax1.set_title("Success / Collision / Timeout", fontweight="bold")
    ax1.legend(fontsize=7)
    ax1.grid(axis="y", alpha=0.3)

    # ── (2) Radar chart: safety metrics ──
    ax2 = fig.add_subplot(2, 3, 2, projection="polar")
    radar_labels = ["Safety\n(1-risk)", "Clearance", "Success\nRate", "Low Dev.\n(1-dev)", "Smooth\n(1-norm)", "Min Scan"]
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
        ax2.fill(angles, values, alpha=0.15, color=colors[i])
        ax2.plot(angles, values, "o-", linewidth=1.5, label=STYLE[m]["label"], color=colors[i], markersize=3)
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(radar_labels, fontsize=7)
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Safety Profile (higher = better)", fontweight="bold", pad=20)
    ax2.legend(loc="lower right", fontsize=6, bbox_to_anchor=(1.3, -0.1))

    # ── (3) Trajectory comparison (single episode, same map) ──
    ax3 = fig.add_subplot(2, 3, (3, 4))
    ax3.set_xlim(-0.5, map_size + 0.5)
    ax3.set_ylim(-0.5, map_size + 0.5)
    ax3.set_aspect("equal")
    ax3.set_title("Trajectory Comparison (same map, 1 episode)", fontweight="bold")

    if trace_results:
        first = next(iter(trace_results.values()))
        tr = first.get("trace", {})
        # Draw map border
        ax3.plot([0, map_size, map_size, 0, 0], [0, 0, map_size, map_size, 0], "k-", lw=1.0, alpha=0.4)
        # Draw obstacles from first trace scan — just approximate
        # Mark start
        if "x" in tr and len(tr["x"]) > 0:
            for m in methods:
                if m in trace_results and "trace" in trace_results[m]:
                    t = trace_results[m]["trace"]
                    if len(t.get("x", [])) > 0:
                        ax3.plot(t["x"], t["y"], color=STYLE[m]["color"], linewidth=1.2,
                                 alpha=0.8, label=STYLE[m]["label"], marker=STYLE[m]["marker"],
                                 markersize=3, markevery=max(1, len(t["x"]) // 8))
            # Start and goal
            ax3.plot(t["x"][0], t["y"][0], "ko", markersize=8, label="Start")
        # Goal marker (where the robot ends up if success)
        for m in methods:
            if m in trace_results and trace_results[m]["success"]:
                t2 = trace_results[m]["trace"]
                if len(t2.get("x", [])) > 0:
                    ax3.plot(t2["x"][-1], t2["y"][-1], "*", color=STYLE[m]["color"], markersize=10)
        ax3.legend(fontsize=7, loc="upper right")
        ax3.set_xlabel("X (m)")
        ax3.set_ylabel("Y (m)")
        ax3.grid(True, alpha=0.2)

    # ── (4) Metric bar comparison ──
    ax4 = fig.add_subplot(2, 3, 5)
    metric_keys = ["mean_risk", "mean_clearance", "mean_deviation", "mean_min_scan"]
    metric_labels = ["Risk (↓)", "Clearance (↑)", "Deviation (↓)", "Min Scan (↑)"]
    x = np.arange(len(metric_labels))
    w = 0.2
    for i, m in enumerate(methods):
        s = summary[m]
        vals = [s["mean_risk"], s["mean_clearance"], s["mean_deviation"], s["mean_min_scan"]]
        ax4.bar(x + i * w, vals, w, label=STYLE[m]["label"], color=colors[i], alpha=0.85)
    ax4.set_xticks(x + w * (len(methods) - 1) / 2)
    ax4.set_xticklabels(metric_labels, fontsize=8)
    ax4.set_title("Per-Metric Comparison", fontweight="bold")
    ax4.legend(fontsize=7)
    ax4.grid(axis="y", alpha=0.3)

    # ── (5) Risk-over-time curves ──
    ax5 = fig.add_subplot(2, 3, 6)
    for m in methods:
        if m in trace_results and "trace" in trace_results[m]:
            risk_arr = trace_results[m]["trace"].get("risk", np.array([]))
            if len(risk_arr) > 0:
                steps_arr = np.arange(len(risk_arr))
                ax5.plot(steps_arr, risk_arr, color=STYLE[m]["color"], linewidth=1.0,
                         alpha=0.8, label=STYLE[m]["label"])
    ax5.set_xlabel("Step")
    ax5.set_ylabel("Collision Risk")
    ax5.set_title("Risk Over Time (1 episode)", fontweight="bold")
    ax5.legend(fontsize=7)
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(bottom=-0.02, top=1.02)

    plt.tight_layout()
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate action correction methods")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Number of evaluation episodes.")
    parser.add_argument("--scan-bins", type=int, default=64)
    parser.add_argument("--obstacles", type=int, default=8)
    parser.add_argument("--residual-model", default=None,
                        help="Path to trained ResidualCorrectionNet checkpoint.")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip visualization, print table only.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)

    def env_factory(s: int) -> Simple2DNavEnv:
        return Simple2DNavEnv(
            num_scan_bins=args.scan_bins,
            num_obstacles=args.obstacles,
            scan_fov_deg=90.0,
            scan_range=5.0,
            scan_noise_std=0.02,
            robot_radius=0.15,
            max_vx=0.3,
            max_vy=0.3,
            max_omega=90.0,
            dt=0.05,
            goal_tolerance=0.3,
            max_episode_steps=600,
            seed=s,
        )

    candidate_policy = MockCandidatePolicy(seed=args.seed)
    rule_policy = RulePolicy(scan_bins=args.scan_bins, fov_deg=90.0)

    residual_model = None
    if args.residual_model:
        from pc.residual_correction import ResidualCorrectionNet
        residual_model = ResidualCorrectionNet.load(args.residual_model)
        print(f"Loaded residual model from {args.residual_model}")

    print(f"Evaluating {args.episodes} episodes ...\n")
    summary, trace_results = evaluate_all(
        env_factory=env_factory,
        candidate_policy=candidate_policy,
        rule_policy=rule_policy,
        rng=rng,
        residual_model=residual_model,
        num_episodes=args.episodes,
        seed=args.seed,
    )

    # Print table
    header = (f"{'Method':<22s} {'Success':>8s} {'Collision':>10s} {'Timeout':>8s} "
              f"{'Risk':>7s} {'Clear':>7s} {'Dev':>7s} {'Smooth':>7s} {'min_scan':>8s}")
    sep = "=" * len(header)
    print("\n" + sep)
    print(header)
    print("-" * len(header))
    for name, s in summary.items():
        print(
            f"{STYLE[name]['label']:<22s} "
            f"{s['success_rate']:7.1f}% "
            f"{s['collision_rate']:9.1f}% "
            f"{s['timeout_rate']:7.1f}% "
            f"{s['mean_risk']:6.3f} "
            f"{s['mean_clearance']:6.3f} "
            f"{s['mean_deviation']:6.3f} "
            f"{s['mean_smoothness']:6.3f} "
            f"{s['mean_min_scan']:7.3f}m"
        )
    print(sep)
    print("\n  Risk = directional collision risk (lower is safer)")
    print("  Clear = clearance score (higher is safer)")
    print("  Dev = action deviation from goal-directed baseline")
    print("  Smooth = inter-step action smoothness (lower is smoother)")

    if not args.no_plot and HAS_MPL:
        plot_results(summary, trace_results)
    elif args.no_plot:
        print("\n  (--no-plot: visualization skipped)")
    else:
        print("\n  [!] matplotlib not available. Install: pip install matplotlib")


if __name__ == "__main__":
    main()
