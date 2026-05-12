"""
Evaluate trained policies in simulation.

Usage:
    # Evaluate a trained SB3 model
    python sim/evaluate_policy.py --config config/ --model models/mlp_sac_policy.pt

    # Compare different scan dimensions
    python sim/evaluate_policy.py --config config/ --scan-bins 32 --rule-policy
    python sim/evaluate_policy.py --config config/ --scan-bins 64 --rule-policy
    python sim/evaluate_policy.py --config config/ --scan-bins 128 --rule-policy
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sim.simple_2d_env import Simple2DNavEnv
from pc.rule_policy import RulePolicy
from pc.dwa_policy import DWAPlanner, DWAConfig
from pc.mlp_policy import MLPPolicy


def evaluate(
    env: Simple2DNavEnv,
    policy,
    num_episodes: int = 100,
    max_steps: int = 1200,
    use_scan_m: bool = True,
) -> dict:
    """
    Evaluate a policy over multiple episodes.

    Args:
        env: the environment (will be reset for each episode).
        policy: callable(scan, goal_heading) → (vx, vy, omega).
        num_episodes: number of evaluation episodes.
        max_steps: max steps per episode.
        use_scan_m: whether policy expects metric scan (True) or normalized (False).

    Returns:
        Dict with success_rate, collision_rate, timeout_rate, avg_time, etc.
    """
    successes = 0
    collisions = 0
    timeouts = 0
    total_steps = 0
    total_time = 0.0
    total_reward = 0.0
    oscillation_count = 0

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0
        prev_omega = 0.0
        sign_changes = 0

        while not done:
            # Extract scan and goal heading from observation
            scan_norm = obs[:env.num_scan_bins]
            scan_m = scan_norm * env.scan_range
            goal_heading_norm = obs[-1]
            goal_heading = goal_heading_norm * math.pi

            # Run policy
            if use_scan_m:
                vx, vy, omega = policy(scan_m, goal_heading)
            else:
                vx, vy, omega = policy(scan_norm, goal_heading)

            # Convert omega from deg/s to the action format expected by env
            # (env expects omega in deg/s in action[2])
            action = np.array([vx, vy, omega], dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_steps += 1

            # Track oscillation (sign changes in omega)
            if prev_omega * omega < 0 and abs(omega) > 1.0:
                sign_changes += 1
            prev_omega = omega

        if info.get("reached_goal"):
            successes += 1
        elif info.get("collision"):
            collisions += 1
        else:
            timeouts += 1

        total_steps += ep_steps
        total_time += ep_steps * env.dt
        total_reward += ep_reward
        oscillation_count += sign_changes

    n = num_episodes
    return {
        "num_episodes": n,
        "success_rate": successes / n * 100,
        "collision_rate": collisions / n * 100,
        "timeout_rate": timeouts / n * 100,
        "avg_steps": total_steps / n,
        "avg_time_s": total_time / n,
        "avg_reward": total_reward / n,
        "avg_oscillations": oscillation_count / n,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate navigation policy")
    parser.add_argument("--config", default="config")
    parser.add_argument("--model", default=None, help="Path to MLP model checkpoint")
    parser.add_argument("--scan-bins", type=int, default=64, choices=[32, 64, 128])
    parser.add_argument("--obstacles", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--rule-policy", action="store_true")
    parser.add_argument("--dwa-policy", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    # Create environment
    env = Simple2DNavEnv(
        map_size=10.0,
        num_obstacles=args.obstacles,
        num_scan_bins=args.scan_bins,
        scan_fov_deg=90.0,
        scan_range=5.0,
        scan_noise_std=0.02,
        robot_radius=0.15,
        max_vx=0.3,
        max_vy=0.3,
        max_omega=90.0,
        dt=0.05,
        goal_tolerance=0.3,
        max_episode_steps=1200,
        seed=args.seed,
    )

    # Load policy
    if args.rule_policy:
        policy = RulePolicy(
            safe_distance=0.3,
            danger_distance=0.2,
            forward_speed=0.2,
            turn_gain=30.0,
            scan_bins=args.scan_bins,
            fov_deg=90.0,
            max_linear_vel=0.3,
            max_angular_vel=90.0,
        )
        print(f"[Eval] Using RulePolicy, scan_bins={args.scan_bins}")
    elif args.dwa_policy:
        policy = DWAPlanner(
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
                num_samples=50,
            ),
            scan_bins=args.scan_bins,
            fov_deg=90.0,
        )
        print(f"[Eval] Using DWAPolicy, scan_bins={args.scan_bins}")
    elif args.model:
        mlp = MLPPolicy.load(args.model)
        policy = lambda scan_m, goal: mlp.predict(
            np.concatenate([
                np.clip(scan_m / 5.0, 0, 1),
                [0.0, 0.0, goal / math.pi],
            ]).astype(np.float32)
        )
        print(f"[Eval] Using MLPPolicy from {args.model}")
    else:
        # Default: trained MLP
        mlp_path = project_root / "models" / "mlp_sac_policy.pt"
        if mlp_path.exists():
            mlp = MLPPolicy.load(str(mlp_path))
            policy = lambda scan_m, goal: mlp.predict(
                np.concatenate([
                    np.clip(scan_m / 5.0, 0, 1),
                    [0.0, 0.0, goal / math.pi],
                ]).astype(np.float32)
            )
            print(f"[Eval] Using MLPPolicy from {mlp_path}")
        else:
            print("[Eval] No model found, falling back to RulePolicy")
            policy = RulePolicy(
                safe_distance=0.3, danger_distance=0.2,
                forward_speed=0.2, turn_gain=30.0,
                scan_bins=args.scan_bins, fov_deg=90.0,
                max_linear_vel=0.3, max_angular_vel=90.0,
            )

    # Evaluate
    print(f"[Eval] Running {args.episodes} episodes with {args.obstacles} obstacles...")
    start = time.perf_counter()
    results = evaluate(env, policy, num_episodes=args.episodes)
    elapsed = time.perf_counter() - start

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Episodes:        {results['num_episodes']}")
    print(f"  Success Rate:    {results['success_rate']:.1f}%")
    print(f"  Collision Rate:  {results['collision_rate']:.1f}%")
    print(f"  Timeout Rate:    {results['timeout_rate']:.1f}%")
    print(f"  Avg Steps:       {results['avg_steps']:.1f}")
    print(f"  Avg Time (s):    {results['avg_time_s']:.2f}")
    print(f"  Avg Reward:      {results['avg_reward']:.2f}")
    print(f"  Avg Oscillations:{results['avg_oscillations']:.1f}")
    print(f"  Eval Time (s):   {elapsed:.2f}")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
