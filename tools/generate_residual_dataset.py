"""
Generate a residual correction dataset from simulation.

For each frame:
  1. Get candidate_action from a noisy / imperfect mock policy.
  2. Get safer_action from DWA (deterministic, smooth teacher).
  3. residual_label = safer_action - candidate_action.
  4. Save (X, Y) pairs as NPZ.

DWA is used as the teacher because it produces smooth, deterministic, and
geometrically safe actions — unlike random-sampling projection which is noisy.

Usage:
    python tools/generate_residual_dataset.py --sim --episodes 200 --output datasets/residual_dataset.npz
    python tools/generate_residual_dataset.py --sim --episodes 500 --obstacles 12 --output datasets/residual_dataset.npz
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pc.dwa_policy import DWAPlanner, DWAConfig


class MockCandidatePolicy:
    """Imperfect policy simulating an un-tuned LeRobot policy that needs correction."""

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


def generate_from_sim(
    num_episodes: int = 200,
    max_steps_per_episode: int = 600,
    num_obstacles: int = 10,
    scan_bins: int = 64,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Run simulation episodes, using DWA as the teacher policy."""
    from sim.simple_2d_env import Simple2DNavEnv

    env = Simple2DNavEnv(
        num_scan_bins=scan_bins,
        num_obstacles=num_obstacles,
        scan_fov_deg=90.0,
        scan_range=5.0,
        scan_noise_std=0.02,
        robot_radius=0.15,
        max_vx=0.3,
        max_vy=0.3,
        max_omega=90.0,
        dt=0.05,
        goal_tolerance=0.3,
        max_episode_steps=max_steps_per_episode,
        seed=seed,
    )

    candidate_policy = MockCandidatePolicy(seed=seed + 1)
    # DWA teacher — deterministic, smooth, safe
    teacher = DWAPlanner(
        config=DWAConfig(
            max_linear_vel=0.3,
            max_angular_vel=90.0,
            linear_accel=0.5,
            angular_accel=180.0,
            dt=0.1,
            predict_steps=15,
            heading_weight=0.3,
            clearance_weight=1.0,
            velocity_weight=0.1,
            num_samples=80,
        ),
        scan_bins=scan_bins,
        fov_deg=90.0,
    )

    X_list: list[np.ndarray] = []
    Y_list: list[np.ndarray] = []
    total_frames = 0

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        prev_vx, prev_vy = 0.0, 0.0

        while not done:
            scan_norm = obs[:scan_bins]
            scan_m = scan_norm * env.scan_range
            goal_heading_norm = obs[-1]
            goal_heading = goal_heading_norm * math.pi

            # Candidate: noisy / imperfect
            cvx, cvy, comega = candidate_policy(scan_m, goal_heading)
            candidate = np.array([cvx, cvy, comega], dtype=np.float32)

            # Teacher: DWA (deterministic, safe)
            tvx, tvy, tomega = teacher(scan_m, goal_heading)
            safer = np.array([tvx, tvy, tomega], dtype=np.float32)

            residual = safer - candidate

            current_vel = np.array([prev_vx, prev_vy], dtype=np.float32)
            inp = np.concatenate([
                scan_m, candidate, current_vel, np.array([goal_heading], dtype=np.float32)
            ])
            X_list.append(inp)
            Y_list.append(residual)
            total_frames += 1

            # Step with safer action (so we stay on a safe trajectory)
            obs, _, terminated, truncated, _info = env.step(safer)
            done = terminated or truncated
            prev_vx, prev_vy = float(safer[0]), float(safer[1])

        if (ep + 1) % 20 == 0:
            print(f"  Episode {ep + 1}/{num_episodes}, frames: {total_frames}")

    env.close()
    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)
    return X, Y


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate residual correction dataset")
    parser.add_argument("--sim", action="store_true",
                        help="Generate from simulation environment.")
    parser.add_argument("--output", default="datasets/residual_dataset.npz")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--obstacles", type=int, default=10,
                        help="More obstacles = more challenging dataset.")
    parser.add_argument("--scan-bins", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Generating dataset ({args.episodes} episodes, {args.obstacles} obstacles, DWA teacher)...")
    X, Y = generate_from_sim(
        num_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        num_obstacles=args.obstacles,
        scan_bins=args.scan_bins,
        seed=args.seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, X=X, Y=Y)
    print(f"\nSaved: {output_path}")
    print(f"  X: {X.shape}   Y: {Y.shape}")
    print(f"  Residual mean (vx,vy,omega): {Y.mean(axis=0)}")
    print(f"  Residual std  (vx,vy,omega): {Y.std(axis=0)}")
    print(f"  Residual max  |vx,vy|: {np.abs(Y[:, :2]).max(axis=0)}")
    print(f"  Residual max  |omega|: {np.abs(Y[:, 2]).max():.1f}")


if __name__ == "__main__":
    main()
