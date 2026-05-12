"""
Quantify sim-to-real gap by comparing MuJoCo scan distributions with real scan logs.

Uses Wasserstein distance between MuJoCo pseudo-LiDAR scans and real log scans.
Generates overlay plots of scan distributions.

Command:
    python sim_mujoco/eval/sim2real_scan_gap.py \
        --log logs/real_scans.npz \
        --world lab_cluttered.xml \
        --episodes 50 --steps 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from sim_mujoco.envs.lekiwi_depth_scan_env import make_env, EnvConfig

try:
    from scipy.stats import wasserstein_distance
    HAS_WASSERSTEIN = True
except ImportError:
    HAS_WASSERSTEIN = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ── Wasserstein computation ─────────────────────────────────────────────────

def compute_wasserstein_per_bin(
    sim_scans: np.ndarray,       # (N_sim, bins)
    real_scans: np.ndarray,      # (N_real, bins)
    scan_max_range: float = 5.0,
) -> dict:
    """Compute Wasserstein distance per scan bin between sim and real.

    Args:
        sim_scans: Simulated metric scan array, shape (N_sim, bins).
        real_scans: Real-world metric scan array, shape (N_real, bins).
        scan_max_range: Maximum scan range for normalisation.

    Returns:
        Dict with per_bin distances, mean, std, normalized_mean.
    """
    bins = sim_scans.shape[1]
    per_bin: list[float] = []

    for k in range(bins):
        sim_col = sim_scans[:, k]
        real_col = real_scans[:, k]

        # Remove invalid / zero readings
        sim_valid = sim_col[np.isfinite(sim_col) & (sim_col > 0.001)]
        real_valid = real_col[np.isfinite(real_col) & (real_col > 0.001)]

        if len(sim_valid) < 2 or len(real_valid) < 2:
            per_bin.append(np.nan)
        else:
            if HAS_WASSERSTEIN:
                d = wasserstein_distance(sim_valid, real_valid)
            else:
                # Fallback: mean absolute difference
                d = float(np.abs(np.mean(sim_valid) - np.mean(real_valid)))
            per_bin.append(d)

    per_bin_arr = np.array(per_bin)
    valid = per_bin_arr[~np.isnan(per_bin_arr)]

    return {
        "per_bin": per_bin_arr,
        "mean": float(np.mean(valid)) if len(valid) > 0 else float("nan"),
        "std": float(np.std(valid)) if len(valid) > 0 else float("nan"),
        "min": float(np.min(valid)) if len(valid) > 0 else float("nan"),
        "max": float(np.max(valid)) if len(valid) > 0 else float("nan"),
        "normalized_mean": float(np.mean(valid) / scan_max_range) if len(valid) > 0 else float("nan"),
    }


# ── MuJoCo scan collection ──────────────────────────────────────────────────

def collect_mujoco_scans(
    world_xml: str,
    episodes: int = 50,
    steps_per_episode: int = 200,
    scan_bins: int = 64,
    seed: int = 0,
) -> np.ndarray:
    """Collect LiDAR scans from MuJoCo simulation by running random exploration.

    Args:
        world_xml: World XML filename (e.g. "lab_cluttered.xml").
        episodes: Number of episodes to run.
        steps_per_episode: Steps per episode.
        scan_bins: Number of scan bins.
        seed: Random seed.

    Returns:
        Array of metric scans (N_total, scan_bins).
    """
    all_scans: list[np.ndarray] = []
    rng = np.random.RandomState(seed)

    for ep in range(episodes):
        env = make_env(world_xml=world_xml, max_steps=steps_per_episode)
        obs, _ = env.reset(seed=int(rng.randint(0, 2**31 - 1)))

        for _step in range(steps_per_episode):
            scan_norm = obs[:scan_bins]
            scan_m = scan_norm * env.unwrapped.cfg.scan_max_range
            all_scans.append(scan_m.copy())

            # Random action for diverse scan coverage
            action = np.array([
                rng.uniform(-0.3, 0.3),
                rng.uniform(-0.3, 0.3),
                rng.uniform(-90.0, 90.0),
            ], dtype=np.float32)

            obs, _reward, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                break

        env.close()

        if (ep + 1) % max(1, episodes // 5) == 0:
            print(f"    Collected {ep+1}/{episodes} episodes...")

    return np.stack(all_scans, axis=0).astype(np.float32)


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_comparison(
    sim_scans: np.ndarray,
    real_scans: np.ndarray,
    per_bin_w1: np.ndarray,
    mean_w1: float,
    world_name: str,
    log_name: str,
    scan_bins: int = 64,
    fov_deg: float = 90.0,
    output_path: Optional[str] = None,
) -> None:
    """Plot sim-vs-real scan distributions and per-bin Wasserstein distance.

    Args:
        sim_scans: Simulated scans (N_sim, bins).
        real_scans: Real-world scans (N_real, bins).
        per_bin_w1: Per-bin Wasserstein distances.
        mean_w1: Mean Wasserstein distance.
        world_name: Name of the MuJoCo world.
        log_name: Name/path of the real scan log.
        scan_bins: Number of scan bins.
        fov_deg: Scan field of view (degrees).
        output_path: If provided, save figure to this path.
    """
    if not HAS_MPL:
        print("  [Warning] matplotlib not available; skipping plot.")
        return

    angles_deg = np.linspace(-fov_deg / 2, fov_deg / 2, scan_bins)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Sim-to-Real Scan Distribution Gap\n"
        f"World: {world_name}  |  Real log: {log_name}  |  "
        f"Mean W1 = {mean_w1:.4f} m",
        fontsize=13, fontweight="bold",
    )

    # ── (1) Mean scan overlay ──
    ax1 = axes[0, 0]
    sim_mean = np.mean(sim_scans, axis=0)
    sim_std = np.std(sim_scans, axis=0)
    real_mean = np.mean(real_scans, axis=0)
    real_std = np.std(real_scans, axis=0)

    ax1.fill_between(angles_deg, sim_mean - sim_std, sim_mean + sim_std,
                     alpha=0.25, color="#2196F3", label="Sim (+/-1 std)")
    ax1.plot(angles_deg, sim_mean, color="#1565C0", lw=2.0, label="Sim mean")
    ax1.fill_between(angles_deg, real_mean - real_std, real_mean + real_std,
                     alpha=0.25, color="#FF9800", label="Real (+/-1 std)")
    ax1.plot(angles_deg, real_mean, color="#E65100", lw=2.0, label="Real mean")
    ax1.set_xlabel("Angle (deg)")
    ax1.set_ylabel("Distance (m)")
    ax1.set_title("Mean Scan Profile (with std)", fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── (2) Per-bin Wasserstein distance ──
    ax2 = axes[0, 1]
    colors = ["#4CAF50" if w < mean_w1 else "#F44336" for w in per_bin_w1]
    bar_width = 90.0 / scan_bins
    ax2.bar(angles_deg, per_bin_w1, width=bar_width, color=colors, edgecolor="none")
    ax2.axhline(y=mean_w1, color="#9C27B0", linestyle="--", lw=1.5,
                label=f"Mean W1 = {mean_w1:.4f} m")
    ax2.set_xlabel("Angle (deg)")
    ax2.set_ylabel("Wasserstein Distance (m)")
    ax2.set_title("Per-Bin Wasserstein Distance", fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── (3) Distribution histogram (pooled) ──
    ax3 = axes[1, 0]
    sim_flat = sim_scans.ravel()
    real_flat = real_scans.ravel()
    sim_flat = sim_flat[np.isfinite(sim_flat) & (sim_flat > 0)]
    real_flat = real_flat[np.isfinite(real_flat) & (real_flat > 0)]

    bins_hist = np.linspace(0, 5.0, 80)
    ax3.hist(sim_flat, bins=bins_hist, alpha=0.55, color="#2196F3",
             label=f"Sim (n={len(sim_flat)})")
    ax3.hist(real_flat, bins=bins_hist, alpha=0.55, color="#FF9800",
             label=f"Real (n={len(real_flat)})")
    ax3.set_xlabel("Distance (m)")
    ax3.set_ylabel("Count")
    ax3.set_title("Pooled Scan Distribution", fontweight="bold")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ── (4) Gap heatmap-style scatter ──
    ax4 = axes[1, 1]
    # Show random subsets for visual clarity
    n_show = min(3000, sim_scans.shape[0], real_scans.shape[0])
    si = np.random.choice(sim_scans.shape[0], n_show, replace=False)
    ri = np.random.choice(real_scans.shape[0], n_show, replace=False)

    ax4.scatter(np.tile(angles_deg, n_show),
                sim_scans[si].ravel()[: n_show * scan_bins],
                s=1, alpha=0.15, color="#2196F3", label="Sim")
    ax4.scatter(np.tile(angles_deg, n_show),
                real_scans[ri].ravel()[: n_show * scan_bins],
                s=1, alpha=0.15, color="#FF9800", label="Real")
    ax4.set_xlabel("Angle (deg)")
    ax4.set_ylabel("Distance (m)")
    ax4.set_title("Scan Cloud Overlay (sampled)", fontweight="bold")
    ax4.legend(fontsize=8, markerscale=8)
    ax4.grid(True, alpha=0.3)

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
        description="Quantify sim-to-real scan distribution gap via Wasserstein distance."
    )
    parser.add_argument(
        "--log", type=str, required=True,
        help="Path to real scan log .npz file (from tools/record_scan_log.py).",
    )
    parser.add_argument(
        "--world", type=str, default="lab_cluttered.xml",
        help="MuJoCo world XML to simulate (default: lab_cluttered.xml).",
    )
    parser.add_argument(
        "--episodes", type=int, default=50,
        help="Number of simulation episodes for scan collection (default: 50).",
    )
    parser.add_argument(
        "--steps", type=int, default=200,
        help="Max steps per episode (default: 200).",
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
        "--output-plot", type=str, default=None,
        help="Path to save the comparison plot (e.g. logs/sim2real_gap.png).",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip plotting entirely.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for reproducibility (default: 0).",
    )
    args = parser.parse_args()

    # ── Load real scan log ──
    log_path = Path(args.log)
    if not log_path.exists():
        raise FileNotFoundError(f"Real scan log not found: {args.log}")

    real_data = np.load(str(log_path))
    # Support both "scans_m" and "scans_norm" keys
    if "scans_m" in real_data:
        real_scans = real_data["scans_m"]
    elif "scans_norm" in real_data:
        real_scans = real_data["scans_norm"] * 5.0  # assume max_range=5m
    else:
        raise KeyError(
            f"Real scan log must contain 'scans_m' or 'scans_norm'. "
            f"Found keys: {list(real_data.keys())}"
        )

    # Truncate to requested bin count
    if real_scans.shape[1] > args.scan_bins:
        real_scans = real_scans[:, :args.scan_bins]
    elif real_scans.shape[1] < args.scan_bins:
        args.scan_bins = real_scans.shape[1]
        print(f"  Adjusted scan_bins to {args.scan_bins} to match real data.")

    print(f"\n{'='*60}")
    print(f"SIM-TO-REAL SCAN DISTRIBUTION GAP")
    print(f"{'='*60}")
    print(f"  Real log:      {args.log}")
    print(f"  Real frames:   {real_scans.shape[0]}")
    print(f"  MuJoCo world:  {args.world}")
    print(f"  Episodes:      {args.episodes}")
    print(f"  Steps/ep:      {args.steps}")
    print(f"  Scan bins:     {args.scan_bins}")
    print(f"  FOV:           {args.fov_deg} deg")
    print(f"  Seed:          {args.seed}")
    if not HAS_WASSERSTEIN:
        print(f"  [Warning] scipy not installed; using mean-diff fallback.")
    print(f"{'='*60}\n")

    np.random.seed(args.seed)

    # ── Collect MuJoCo scans ──
    print("Collecting MuJoCo scans...")
    sim_scans = collect_mujoco_scans(
        world_xml=args.world,
        episodes=args.episodes,
        steps_per_episode=args.steps,
        scan_bins=args.scan_bins,
        seed=args.seed,
    )
    print(f"  Collected {sim_scans.shape[0]} simulated scan frames.\n")

    # ── Compute Wasserstein distance ──
    print("Computing per-bin Wasserstein distances...")
    result = compute_wasserstein_per_bin(
        sim_scans, real_scans, scan_max_range=5.0,
    )

    # ── Print results ──
    print(f"\n{'='*60}")
    print(f"WASSERSTEIN DISTANCE (Sim-to-Real)")
    print(f"{'='*60}")
    print(f"  Sim frames:      {sim_scans.shape[0]}")
    print(f"  Real frames:     {real_scans.shape[0]}")
    print(f"  Bins:            {args.scan_bins}")
    print(f"  Mean W1:         {result['mean']:.4f} m")
    print(f"  Std W1:          {result['std']:.4f} m")
    print(f"  Min W1:          {result['min']:.4f} m")
    print(f"  Max W1:          {result['max']:.4f} m")
    print(f"  Normalized W1:   {result['normalized_mean']:.4f} (div by 5m)")
    gap_label = (
        "NEAR-ZERO" if result['normalized_mean'] < 0.05
        else "NOTABLE" if result['normalized_mean'] < 0.15
        else "SIGNIFICANT"
    )
    print(f"  Sim-to-Real Gap: {gap_label}")
    print(f"{'='*60}\n")

    # ── Plot ──
    if not args.no_plot:
        log_name_short = Path(args.log).stem
        plot_comparison(
            sim_scans=sim_scans,
            real_scans=real_scans,
            per_bin_w1=result["per_bin"],
            mean_w1=result["mean"],
            world_name=args.world,
            log_name=log_name_short,
            scan_bins=args.scan_bins,
            fov_deg=args.fov_deg,
            output_path=args.output_plot,
        )


if __name__ == "__main__":
    main()
