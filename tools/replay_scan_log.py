"""
Replay a recorded scan log through a policy for offline evaluation.

Usage:
    python tools/replay_scan_log.py --input logs/scan_log.npz --config config/ --policy rule
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pc.rule_policy import RulePolicy
from pc.dwa_policy import DWAPlanner, DWAConfig


def main():
    parser = argparse.ArgumentParser(description="Replay scan log through policy")
    parser.add_argument("--input", required=True, help="Path to scan log .npz")
    parser.add_argument("--config", default="config")
    parser.add_argument("--policy", default="rule", choices=["rule", "dwa"])
    parser.add_argument("--output", default=None, help="Path to save action log")
    args = parser.parse_args()

    # Load scan log
    data = np.load(args.input)
    scans_m = data["scans_m"]
    print(f"Loaded {len(scans_m)} scans from {args.input}")

    # Policy
    if args.policy == "rule":
        policy = RulePolicy(
            safe_distance=0.3, danger_distance=0.2,
            forward_speed=0.2, turn_gain=30.0,
            scan_bins=scans_m.shape[1], fov_deg=90.0,
            max_linear_vel=0.3, max_angular_vel=90.0,
        )
    else:
        policy = DWAPlanner(
            config=DWAConfig(
                max_linear_vel=0.3, max_angular_vel=90.0,
                linear_accel=0.5, angular_accel=180.0,
                dt=0.1, predict_steps=15,
                heading_weight=0.3, clearance_weight=1.0,
                velocity_weight=0.1, num_samples=50,
            ),
            scan_bins=scans_m.shape[1],
        )

    # Replay
    actions = []
    for i, scan in enumerate(scans_m):
        vx, vy, omega = policy(scan, goal_heading=0.0)
        actions.append([vx, vy, omega])
    actions = np.array(actions)

    # Stats
    print(f"\nAction Statistics ({args.policy} policy):")
    print(f"  vx:      mean={actions[:,0].mean():.3f}, std={actions[:,0].std():.3f} m/s")
    print(f"  vy:      mean={actions[:,1].mean():.3f}, std={actions[:,1].std():.3f} m/s")
    print(f"  omega:   mean={actions[:,2].mean():.1f}, std={actions[:,2].std():.1f} deg/s")
    sign_changes = np.sum(
        (actions[1:, 2] * actions[:-1, 2]) < 0
    ) / (len(actions) - 1) * 100
    print(f"  Omega sign changes: {sign_changes:.1f}% of steps")

    if args.output:
        np.savez(args.output, actions=actions)
        print(f"Saved actions to {args.output}")


if __name__ == "__main__":
    main()
