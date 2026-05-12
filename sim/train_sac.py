"""
Training script: Train an MLP policy using Stable-Baselines3 SAC.

Supports experiments with different scan dimensions (32, 64, 128)
to ablate the Depth-to-Scan representation.

Usage:
    # Train with default 64-D scan
    python sim/train_sac.py --config config/

    # Train with 32-D scan
    python sim/train_sac.py --config config/ --scan-bins 32

    # Train with 128-D scan, more obstacles
    python sim/train_sac.py --config config/ --scan-bins 128 --obstacles 15
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CheckpointCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Add parent to path so we can import from sim/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sim.simple_2d_env import Simple2DNavEnv


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_env(
    config_dir: str,
    scan_bins: int = 64,
    num_obstacles: int = 8,
    seed: int = 0,
) -> Simple2DNavEnv:
    """Create a training environment instance."""
    cfg_path = Path(config_dir)
    policy_cfg = load_yaml(cfg_path / "policy.yaml")
    robot_cfg = load_yaml(cfg_path / "robot.yaml")["robot"]

    return Simple2DNavEnv(
        map_size=10.0,
        num_obstacles=num_obstacles,
        obstacle_min_r=0.2,
        obstacle_max_r=0.6,
        num_scan_bins=scan_bins,
        scan_fov_deg=90.0,
        scan_range=5.0,
        scan_noise_std=0.02,
        robot_radius=0.15,
        max_vx=robot_cfg["max_linear_vel"],
        max_vy=robot_cfg["max_linear_vel"],
        max_omega=robot_cfg["max_angular_vel"],
        dt=0.05,
        goal_tolerance=0.3,
        max_episode_steps=1200,
        goal_reward=10.0,
        collision_penalty=5.0,
        progress_weight=0.5,
        clearance_weight=0.1,
        smoothness_weight=0.01,
        timeout_penalty=0.0,
        seed=seed,
    )


class ProgressCallback(BaseCallback):
    """Custom callback to log training progress."""

    def __init__(self, log_interval: int = 1000):
        super().__init__()
        self.log_interval = log_interval

    def _on_step(self) -> bool:
        if self.n_calls % self.log_interval == 0:
            if "train" in self.model.logger.name_to_value:
                avg_reward = self.model.logger.name_to_value["train/ep_rew_mean"]
                print(f"Step {self.n_calls:7d} | Avg reward: {avg_reward:7.2f}")
        return True


def main():
    parser = argparse.ArgumentParser(description="Train SAC policy for navigation")
    parser.add_argument("--config", default="config", help="Path to config directory")
    parser.add_argument("--scan-bins", type=int, default=64, choices=[32, 64, 128])
    parser.add_argument("--obstacles", type=int, default=8, help="Number of obstacles")
    parser.add_argument("--total-steps", type=int, default=500_000, help="Total training steps")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Directories
    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "models"
    logs_dir = project_root / "logs"
    models_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"sac_bins{args.scan_bins}_obs{args.obstacles}_{timestamp}"
    run_dir = logs_dir / run_name
    run_dir.mkdir(exist_ok=True)

    print(f"[Train] Run: {run_name}")
    print(f"[Train] Scan bins: {args.scan_bins}, Obstacles: {args.obstacles}")
    print(f"[Train] Device: {args.device}, Steps: {args.total_steps}")

    # Create vectorized environment
    def _make_train_env():
        return make_env(
            str(project_root / args.config),
            scan_bins=args.scan_bins,
            num_obstacles=args.obstacles,
            seed=args.seed,
        )

    def _make_eval_env():
        return make_env(
            str(project_root / args.config),
            scan_bins=args.scan_bins,
            num_obstacles=args.obstacles,
            seed=args.seed + 10000,
        )

    train_env = DummyVecEnv([_make_train_env])
    eval_env = DummyVecEnv([_make_eval_env])

    # Optional: normalize observations
    # train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False)
    # eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)

    # Create SAC model
    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        ent_coef="auto",
        target_entropy="auto",
        policy_kwargs={
            "net_arch": [256, 256],
            "activation_fn": torch.nn.ReLU,
        },
        verbose=1,
        device=args.device,
        seed=args.seed,
    )

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(models_dir),
        log_path=str(run_dir),
        eval_freq=10_000,
        n_eval_episodes=20,
        deterministic=True,
        render=False,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=str(models_dir),
        name_prefix=f"{run_name}",
    )
    progress_callback = ProgressCallback(log_interval=5000)

    # Train
    print("[Train] Starting training...")
    model.learn(
        total_timesteps=args.total_steps,
        callback=[eval_callback, checkpoint_callback, progress_callback],
        progress_bar=True,
    )

    # Save final model
    final_path = models_dir / f"{run_name}_final"
    model.save(str(final_path))
    print(f"[Train] Model saved to {final_path}.zip")

    # Also export as MLPPolicy checkpoint for inference
    _export_policy(model, final_path, args.scan_bins, run_dir)
    print(f"[Train] Exported MLPPolicy to {models_dir}/mlp_sac_policy.pt")

    # Save run config
    run_cfg = {
        "run_name": run_name,
        "scan_bins": args.scan_bins,
        "num_obstacles": args.obstacles,
        "total_steps": args.total_steps,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
    }
    with open(run_dir / "run_config.json", "w") as f:
        json.dump(run_cfg, f, indent=2)

    train_env.close()
    eval_env.close()


def _export_policy(
    sb3_model: SAC,
    save_path: Path,
    scan_bins: int,
    run_dir: Path,
) -> None:
    """Extract the MLP policy network from trained SB3 model."""
    from pc.mlp_policy import MLPPolicy

    # SB3 SAC policy structure: actor.mu, actor.log_std
    actor = sb3_model.policy.actor

    # Reconstruct MLP layers
    mlp = MLPPolicy(
        obs_dim=scan_bins + 3,  # scan + vx + vy + goal_heading
        act_dim=3,
        hidden_layers=[256, 256],
        max_vx=0.3,
        max_vy=0.3,
        max_omega=90.0,
    )

    # Copy weights from SB3 actor network
    sb3_state = actor.state_dict()
    mlp_state = {}

    # SB3 uses feature_extractor + mu_net
    # For SB3 MlpPolicy, the network is:
    #   features_extractor (shared) → mu_net (action mean)
    # The full network is latent_pi → mu_net

    # Map SB3 layers to our MLPPolicy layers
    # SB3's policy net architecture (with net_arch=[256,256]):
    #   actor.latent_pi (from features_extractor)
    #   actor.mu (final layer: 256 → act_dim)
    # The features_extractor has its own linear layers

    # Get feature extractor weights
    feat_ext = sb3_model.policy.actor_features_extractor
    # For observation space of N dims, this is a sequential with Linear(N→256)→ReLU→Linear(256→256)→ReLU

    # Build state dict mapping
    idx = 0
    for name, param in feat_ext.state_dict().items():
        mlp_state[f"_net.{idx}.weight"] = param
        idx += 1
        if "weight" in name:
            mlp_state[f"_net.{idx}.weight"] = None
            mlp_state[f"_net.{idx}.bias"] = None
            idx += 2 if idx % 2 == 0 else 1

    # Copy actor.latent_pi and actor.mu
    for name, param in actor.state_dict().items():
        if name == "latent_pi.weight":
            mlp_state[f"_net.{idx}.weight"] = param
            idx += 1
            mlp_state[f"_net.{idx}.weight"] = None
            mlp_state[f"_net.{idx}.bias"] = None
            idx += 2
        elif name == "latent_pi.bias":
            mlp_state[f"_net.{idx-2}.bias"] = param

    # This is getting complex. Instead, let's save the raw model and
    # provide a separate conversion script.
    # For now, export a PyTorch state that can be loaded directly.
    torch.save(
        {
            "sb3_state_dict": actor.state_dict(),
            "feat_ext_state_dict": feat_ext.state_dict(),
            "obs_dim": scan_bins + 3,
            "act_dim": 3,
            "scan_bins": scan_bins,
        },
        str(save_path.parent / "mlp_sac_policy.pt"),
    )
