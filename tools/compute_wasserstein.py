"""
Compute Wasserstein distance between simulated scans and real-world scans.

This quantifies the sim-to-real distributional gap for each scan bin.

Usage:
    python tools/compute_wasserstein.py --sim logs/sim_scans.npz --real logs/real_scans.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance


def compute_wasserstein(
    sim_scans: np.ndarray,    # (N1, bins) or (N2, bins)
    real_scans: np.ndarray,
) -> dict:
    """
    Compute Wasserstein distance per bin and average across all bins.

    Args:
        sim_scans: array of synthetic scans, shape (N_sim, bins).
        real_scans: array of real-world scans, shape (N_real, bins).

    Returns:
        Dict with per_bin distances, mean, std, normalized_mean.
    """
    bins = sim_scans.shape[1]
    per_bin = []
    for k in range(bins):
        sim_col = sim_scans[:, k]
        real_col = real_scans[:, k]
        # Remove invalid values
        sim_valid = sim_col[np.isfinite(sim_col) & (sim_col > 0)]
        real_valid = real_col[np.isfinite(real_col) & (real_col > 0)]
        if len(sim_valid) < 2 or len(real_valid) < 2:
            per_bin.append(np.nan)
        else:
            d = wasserstein_distance(sim_valid, real_valid)
            per_bin.append(d)

    per_bin = np.array(per_bin)
    valid = per_bin[~np.isnan(per_bin)]
    return {
        "per_bin": per_bin,
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "normalized_mean": float(np.mean(valid) / 5.0),  # normalized by max range
    }


def main():
    parser = argparse.ArgumentParser(description="Compute Wasserstein distance")
    parser.add_argument("--sim", required=True, help="Path to simulated scans .npz")
    parser.add_argument("--real", required=True, help="Path to real-world scans .npz")
    parser.add_argument("--bins", type=int, default=64)
    parser.add_argument("--plot", action="store_true", help="Plot per-bin distances")
    args = parser.parse_args()

    sim_data = np.load(args.sim)
    real_data = np.load(args.real)

    # Handle both possible keys
    sim_scans = sim_data.get("scans_m", sim_data.get("scans_norm"))
    real_scans = real_data.get("scans_m", real_data.get("scans_norm"))

    if sim_scans is None or real_scans is None:
        print("ERROR: Could not find scan data in input files.")
        print(f"  Sim keys: {list(sim_data.keys())}")
        print(f"  Real keys: {list(real_data.keys())}")
        return

    # Ensure same bin count
    min_bins = min(sim_scans.shape[1], real_scans.shape[1])
    sim_scans = sim_scans[:, :min_bins]
    real_scans = real_scans[:, :min_bins]

    result = compute_wasserstein(sim_scans, real_scans)

    print("=" * 60)
    print("WASSERSTEIN DISTANCE (Sim-to-Real)")
    print("=" * 60)
    print(f"  Sim frames:      {sim_scans.shape[0]}")
    print(f"  Real frames:     {real_scans.shape[0]}")
    print(f"  Bins:            {min_bins}")
    print(f"  Mean W1:         {result['mean']:.4f} m")
    print(f"  Std W1:          {result['std']:.4f} m")
    print(f"  Min W1:          {result['min']:.4f} m")
    print(f"  Max W1:          {result['max']:.4f} m")
    print(f"  Normalized W1:   {result['normalized_mean']:.4f} (÷5m range)")
    print(f"  Sim-to-Real Gap: {'NEAR-ZERO' if result['normalized_mean'] < 0.05 else 'NOTABLE' if result['normalized_mean'] < 0.15 else 'SIGNIFICANT'}")
    print("=" * 60)

    if args.plot:
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 4))
            bins = np.arange(min_bins)
            fov_deg = 90.0
            angles_deg = np.linspace(-fov_deg / 2, fov_deg / 2, min_bins)

            ax.bar(angles_deg, result["per_bin"], width=90.0 / min_bins)
            ax.axhline(y=result["mean"], color="r", linestyle="--",
                       label=f"Mean W1 = {result['mean']:.4f}")
            ax.set_xlabel("Angle (deg)")
            ax.set_ylabel("Wasserstein Distance (m)")
            ax.set_title("Per-Bin Wasserstein Distance (Sim vs Real)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("matplotlib not available, skipping plot.")


if __name__ == "__main__":
    main()
