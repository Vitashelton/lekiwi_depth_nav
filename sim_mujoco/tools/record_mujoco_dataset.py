"""
Record scan/action/pose data from MuJoCo episodes for offline analysis.

Outputs .npz files compatible with tools/plot_scan.py and dashboard/app.py,
as well as with the residual correction training pipeline.

Usage:
    python sim_mujoco/tools/record_mujoco_dataset.py --world lab_cluttered.xml --episodes 5 --output logs/mujoco_episodes.npz
    python sim_mujoco/tools/record_mujoco_dataset.py --world lab_empty.xml --policy dwa --episodes 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sim_mujoco.envs.lekiwi_depth_scan_env import make_env, EnvConfig
from pc.rule_policy import RulePolicy
from pc.dwa_policy import DWAPlanner, DWAConfig


def record_episodes(
    world_xml: str,
    num_episodes: int = 5,
    max_steps: int = 600,
    scan_bins: int = 64,
    policy_name: str = "rule",
    seed: int = 0,
) -> dict:
    """Run episodes and collect all sensor data.

    Returns:
        Dict with keys: scans_m, scans_norm, actions, positions, goals, timestamps, episode_info.
    """
    env = make_env(world_xml=world_xml, scan_bins=scan_bins, max_steps=max_steps)

    if policy_name == "dwa":
        policy = DWAPlanner(config=DWAConfig(), scan_bins=scan_bins, fov_deg=90.0)
    else:
        policy = RulePolicy(scan_bins=scan_bins, fov_deg=90.0)

    scans_m_list = []
    scans_norm_list = []
    actions_list = []
    positions_list = []
    goals_list = []
    timestamps_list = []
    episode_boundaries = [0]

    import time
    successes = 0
    collisions = 0
    timeouts = 0

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_scan_m = []
        ep_scan_norm = []
        ep_action = []
        ep_pos = []

        while not done:
            scan_norm = obs[:scan_bins]
            scan_m = scan_norm * env.scan_range
            goal_heading = np.arctan2(obs[scan_bins + 3], obs[scan_bins + 2])

            if policy_name == "dwa":
                vx, vy, omega = policy(scan_m, goal_heading)
            else:
                vx, vy, omega = policy(scan_m, goal_heading)
            action = np.array([vx, vy, omega], dtype=np.float32)

            ep_scan_norm.append(scan_norm.copy())
            ep_scan_m.append(scan_m.copy())
            ep_action.append(action.copy())
            ep_pos.append(env._mj.robot_pos[:2].copy())

            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if done:
                if info.get("success"):
                    successes += 1
                elif info.get("collision"):
                    collisions += 1
                else:
                    timeouts += 1

        scans_norm_list.append(np.array(ep_scan_norm, dtype=np.float32))
        scans_m_list.append(np.array(ep_scan_m, dtype=np.float32))
        actions_list.append(np.array(ep_action, dtype=np.float32))
        positions_list.append(np.array(ep_pos, dtype=np.float32))
        goals_list.append(env._mj.goal_pos[:2].copy())
        timestamps_list.append(np.arange(len(ep_scan_m), dtype=np.float64) * env.cfg.dt)
        episode_boundaries.append(episode_boundaries[-1] + len(ep_scan_m))

        print(f"  Episode {ep + 1}/{num_episodes}: {len(ep_scan_m)} steps, "
              f"status={'✓' if info.get('success') else '✗' if info.get('collision') else '○'}")

    env.close()

    return {
        "scans_m": np.concatenate(scans_m_list, axis=0),
        "scans_norm": np.concatenate(scans_norm_list, axis=0),
        "actions": np.concatenate(actions_list, axis=0),
        "positions": np.concatenate(positions_list, axis=0),
        "goals": np.array(goals_list, dtype=np.float32),
        "timestamps": np.concatenate(timestamps_list),
        "episode_boundaries": np.array(episode_boundaries, dtype=np.int32),
        "episode_info": {
            "success": successes,
            "collision": collisions,
            "timeout": timeouts,
            "num_episodes": num_episodes,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Record MuJoCo episode data")
    parser.add_argument("--world", default="lab_empty.xml")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--scan-bins", type=int, default=64)
    parser.add_argument("--policy", default="rule", choices=["rule", "dwa", "random"])
    parser.add_argument("--output", default="logs/mujoco_episodes.npz")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"Recording {args.episodes} episodes from {args.world} (policy={args.policy})...")
    data = record_episodes(
        world_xml=args.world,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        scan_bins=args.scan_bins,
        policy_name=args.policy,
        seed=args.seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **data)
    print(f"\nSaved: {output_path}")
    ei = data["episode_info"]
    print(f"  Episodes: {ei['num_episodes']} | success={ei['success']} collision={ei['collision']} timeout={ei['timeout']}")
    print(f"  Total frames: {len(data['scans_m'])}")


if __name__ == "__main__":
    main()
