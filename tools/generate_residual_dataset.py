"""
Generate a residual correction dataset from scan logs and candidate actions.

For each frame, we:
  1. Read scan_m and a candidate_action (LeRobot policy or mock).
  2. Compute a safer_action via geometric action projection or DWA.
  3. residual_label = safer_action - candidate_action.
  4. Save as NPZ: (X, Y) where X = [scan_m, candidate_action, current_vel, goal_heading]
     and Y = residual_label.

Usage:
    # From simulation with mock candidate policy
    python tools/generate_residual_dataset.py --sim --episodes 200 --output datasets/residual_dataset.npz

    # From recorded scan log + a LeRobot-style policy
    python tools/generate_residual_dataset.py --input logs/scan_log.npz --output datasets/residual_dataset.npz

    # With custom projection parameters
    python tools/generate_residual_dataset.py --sim --lambda-risk 3.0 --num-samples 300
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pc.rule_policy import RulePolicy
from pc.dwa_policy import DWAPlanner, DWAConfig
from pc.geometric_risk import compute_action_projection


# ── Mock candidate policy (stand-in for LeRobot policy) ───────────────────

class MockCandidatePolicy:
    """Generates plausible but imperfect actions, simulating an untuned LeRobot policy."""

    def __init__(
        self,
        max_v: float = 0.3,
        max_w: float = 90.0,
        noise_v_std: float = 0.08,
        noise_w_std: float = 20.0,
        seed: int = 0,
    ) -> None:
        self.max_v = max_v
        self.max_w = max_w
        self.noise_v_std = noise_v_std
        self.noise_w_std = noise_w_std
        self.rng = np.random.RandomState(seed)

    def __call__(
        self, scan_m: np.ndarray, goal_heading: float
    ) -> tuple[float, float, float]:
        # Base: move toward goal, roughly
        vx_base = self.max_v * 0.5 * math.cos(goal_heading)
        vy_base = self.max_v * 0.5 * math.sin(goal_heading)

        # Add noise — this is what makes it need correction
        vx = float(vx_base + self.rng.normal(0, self.noise_v_std))
        vy = float(vy_base + self.rng.normal(0, self.noise_v_std))
        omega = float(30.0 * math.sin(goal_heading) + self.rng.normal(0, self.noise_w_std))

        vx = max(-self.max_v, min(self.max_v, vx))
        vy = max(-self.max_v, min(self.max_v, vy))
        omega = max(-self.max_w, min(self.max_w, omega))
        return vx, vy, omega


# ── Dataset generator ─────────────────────────────────────────────────────

def generate_from_sim(
    num_episodes: int = 200,
    max_steps_per_episode: int = 600,
    scan_bins: int = 64,
    projection_lambda: float = 2.0,
    projection_samples: int = 200,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Run simulation episodes and collect (X, Y) pairs."""
    from sim.simple_2d_env import Simple2DNavEnv

    env = Simple2DNavEnv(
        num_scan_bins=scan_bins,
        num_obstacles=8,
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
    rng = np.random.RandomState(seed + 2)

    X_list: list[np.ndarray] = []
    Y_list: list[np.ndarray] = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        prev_vx, prev_vy = 0.0, 0.0

        while not done:
            scan_norm = obs[:scan_bins]
            scan_m = scan_norm * env.scan_range
            goal_heading_norm = obs[-1]
            goal_heading = goal_heading_norm * math.pi

            # 1. Get candidate action from mock policy
            cvx, cvy, comega = candidate_policy(scan_m, goal_heading)
            candidate = np.array([cvx, cvy, comega], dtype=np.float32)

            # 2. Compute safer action via projection
            safer = compute_action_projection(
                candidate_action=candidate,
                scan_m=scan_m,
                fov_deg=90.0,
                max_range=env.scan_range,
                lambda_risk=projection_lambda,
                num_samples=projection_samples,
                rng=rng,
            )

            # 3. residual = safer - candidate
            residual = safer - candidate

            # 4. Build input vector
            current_vel = np.array([prev_vx, prev_vy], dtype=np.float32)
            inp = np.concatenate(
                [scan_m, candidate, current_vel, np.array([goal_heading], dtype=np.float32)]
            )
            X_list.append(inp)
            Y_list.append(residual)

            # Step environment with the safer action
            step_action = safer.copy()
            step_action[2] = np.clip(step_action[2], -90.0, 90.0)
            obs, _, terminated, truncated, _info = env.step(step_action)
            done = terminated or truncated
            prev_vx, prev_vy = float(step_action[0]), float(step_action[1])

        if (ep + 1) % 20 == 0:
            print(f"  Episode {ep + 1}/{num_episodes}, samples so far: {len(X_list)}")

    env.close()
    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)
    return X, Y


def generate_from_log(
    log_path: str,
    projection_lambda: float = 2.0,
    projection_samples: int = 200,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate dataset from a recorded scan log with synthetic candidate actions."""
    data = np.load(log_path)
    scans_m = data["scans_m"]  # (T, N)
    num_frames, scan_bins = scans_m.shape
    rng = np.random.RandomState(seed)

    candidate_policy = MockCandidatePolicy(seed=seed + 1)
    X_list: list[np.ndarray] = []
    Y_list: list[np.ndarray] = []
    prev_vx, prev_vy = 0.0, 0.0

    for t in range(num_frames):
        scan_m = scans_m[t].astype(np.float32)
        goal_heading = 0.0  # logs don't include goal heading; assume forward

        cvx, cvy, comega = candidate_policy(scan_m, goal_heading)
        candidate = np.array([cvx, cvy, comega], dtype=np.float32)

        safer = compute_action_projection(
            candidate_action=candidate,
            scan_m=scan_m,
            fov_deg=90.0,
            max_range=5.0,
            lambda_risk=projection_lambda,
            num_samples=projection_samples,
            rng=rng,
        )
        residual = safer - candidate

        current_vel = np.array([prev_vx, prev_vy], dtype=np.float32)
        inp = np.concatenate(
            [scan_m, candidate, current_vel, np.array([goal_heading], dtype=np.float32)]
        )
        X_list.append(inp)
        Y_list.append(residual)

        prev_vx, prev_vy = float(safer[0]), float(safer[1])

        if (t + 1) % 200 == 0:
            print(f"  Frame {t + 1}/{num_frames}")

    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)
    return X, Y


# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate residual correction dataset"
    )
    parser.add_argument("--sim", action="store_true",
                        help="Generate from simulation environment.")
    parser.add_argument("--input", default=None,
                        help="Path to recorded scan .npz log (alternative to --sim).")
    parser.add_argument("--output", default="datasets/residual_dataset.npz",
                        help="Output .npz file path.")
    parser.add_argument("--episodes", type=int, default=200,
                        help="Number of simulation episodes (with --sim).")
    parser.add_argument("--max-steps", type=int, default=600,
                        help="Max steps per episode.")
    parser.add_argument("--scan-bins", type=int, default=64)
    parser.add_argument("--lambda-risk", type=float, default=2.0,
                        help="Risk penalty weight for projection.")
    parser.add_argument("--num-samples", type=int, default=200,
                        help="Samples for projection search.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.sim:
        print(f"Generating from simulation ({args.episodes} episodes)...")
        X, Y = generate_from_sim(
            num_episodes=args.episodes,
            max_steps_per_episode=args.max_steps,
            scan_bins=args.scan_bins,
            projection_lambda=args.lambda_risk,
            projection_samples=args.num_samples,
            seed=args.seed,
        )
    elif args.input:
        print(f"Generating from log: {args.input}")
        X, Y = generate_from_log(
            log_path=args.input,
            projection_lambda=args.lambda_risk,
            projection_samples=args.num_samples,
            seed=args.seed,
        )
    else:
        print("Specify --sim or --input. Falling back to --sim with 20 episodes.")
        X, Y = generate_from_sim(
            num_episodes=20,
            max_steps_per_episode=args.max_steps,
            scan_bins=args.scan_bins,
            projection_lambda=args.lambda_risk,
            projection_samples=args.num_samples,
            seed=args.seed,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, X=X, Y=Y)
    print(f"\nSaved dataset: {output_path}")
    print(f"  X shape: {X.shape}")
    print(f"  Y shape: {Y.shape}")
    print(f"  Mean residual (vx,vy,omega): {Y.mean(axis=0)}")
    print(f"  Std residual  (vx,vy,omega): {Y.std(axis=0)}")


if __name__ == "__main__":
    main()
