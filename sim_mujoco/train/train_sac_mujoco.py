#!/usr/bin/env python3
"""
Train SAC policy on MuJoCo LeKiwi depth navigation environment.

Uses Stable-Baselines3 SAC with vectorized environments, VecNormalize,
and supports checkpoint resumption and tensorboard logging.

Usage:
    python sim_mujoco/train/train_sac_mujoco.py --world lab_empty.xml --timesteps 500000
    python sim_mujoco/train/train_sac_mujoco.py --resume models/sac_mujoco_lab_empty_200000_steps.zip
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

# Ensure project root is on path for sim_mujoco / pc imports.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sim_mujoco.envs.lekiwi_depth_scan_env import make_env, EnvConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_sac")


# ---------------------------------------------------------------------------
# Default training hyper-parameters
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict = {
    "sac": {
        "learning_rate": 3e-4,
        "buffer_size": 200_000,
        "batch_size": 256,
        "tau": 0.005,
        "gamma": 0.99,
        "ent_coef": "auto",
        "train_freq": 1,
        "gradient_steps": 1,
        "policy_kwargs": {
            "net_arch": [256, 256],
            "activation_fn": "relu",
        },
    },
    "env": {
        "world_xml": "lab_empty.xml",
        "scan_bins": 64,
        "max_episode_steps": 600,
        "n_envs": 4,
        "apply_dr": False,
    },
    "training": {
        "total_timesteps": 500_000,
        "save_freq": 50_000,
        "eval_episodes": 10,
        "log_dir": "tensorboard/",
        "model_dir": "models/",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(config_path: Optional[str]) -> dict:
    """Load YAML config, or return defaults if path is None/missing."""
    if config_path is None:
        logger.info("No config file provided; using built-in defaults.")
        return DEFAULT_CONFIG.copy()

    path = Path(config_path)
    if not path.exists():
        logger.warning(
            "Config file '%s' not found. Using built-in defaults.", config_path
        )
        return DEFAULT_CONFIG.copy()

    with open(path) as fh:
        user = yaml.safe_load(fh) or {}

    # Deep-merge user overrides into defaults.
    merged = DEFAULT_CONFIG.copy()
    _deep_update(merged, user)
    logger.info("Loaded config from %s", config_path)
    return merged


def _deep_update(base: dict, override: dict) -> dict:
    """Recursively update *base* with *override* values in-place."""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def build_model_dir(model_dir: str, world_name: str) -> Path:
    """Ensure the model output directory exists and return its Path."""
    p = Path(model_dir).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def world_stem(world_xml: str) -> str:
    """Return world filename without .xml extension."""
    return Path(world_xml).stem


# ---------------------------------------------------------------------------
# Main training entry-point
# ---------------------------------------------------------------------------

def train(parsed_args: argparse.Namespace) -> None:
    """Orchestrate SAC training end-to-end."""

    # -- 1. Load configuration -------------------------------------------------
    cfg = load_config(parsed_args.config)
    env_cfg = cfg["env"]
    sac_cfg = cfg["sac"]
    train_cfg = cfg["training"]

    # CLI overrides
    world_xml = parsed_args.world or env_cfg["world_xml"]
    total_timesteps = parsed_args.timesteps or train_cfg["total_timesteps"]
    resume_path: Optional[str] = parsed_args.resume

    model_dir = build_model_dir(train_cfg["model_dir"], world_xml)
    log_dir = Path(train_cfg["log_dir"]).resolve()
    stem = world_stem(world_xml)

    logger.info("World: %s", world_xml)
    logger.info("Total timesteps: %d", total_timesteps)
    logger.info("Model dir: %s", model_dir)
    logger.info("Log dir: %s", log_dir)

    # -- 2. Create vectorized environment -------------------------------------
    # Import stable-baselines3 lazily so the script can still be imported for
    # type-checking even when sb3 is not installed.
    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
    except ImportError as exc:
        raise ImportError(
            "stable_baselines3 is required. Install with: pip install stable-baselines3"
        ) from exc

    n_envs = env_cfg.get("n_envs", 1)

    def _env_factory() -> callable:
        return make_env(
            world_xml=world_xml,
            scan_bins=env_cfg.get("scan_bins", 64),
            max_steps=env_cfg.get("max_episode_steps", 600),
            render_mode="rgb_array",
            apply_dr=env_cfg.get("apply_dr", False),
        )

    logger.info("Creating %d vectorized environment(s)...", n_envs)

    if n_envs > 1:
        env = make_vec_env(
            _env_factory,
            n_envs=n_envs,
            seed=42,
            vec_env_cls=DummyVecEnv,
        )
    else:
        env = DummyVecEnv([_env_factory])

    # Wrap with VecNormalize for stable training.
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=sac_cfg["gamma"],
    )
    logger.info("Environment observation space: %s", env.observation_space)
    logger.info("Environment action space: %s", env.action_space)

    # -- 3. Create or resume model --------------------------------------------
    if resume_path is not None:
        resume_full = resume_path
        if not Path(resume_full).exists():
            # Try resolving relative to model_dir.
            resume_full = str(model_dir / resume_path)
        if not Path(resume_full).exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        logger.info("Resuming SAC from checkpoint: %s", resume_full)
        model = SAC.load(
            resume_full,
            env=env,
            tensorboard_log=str(log_dir),
            device="auto",
            print_system_info=True,
        )
    else:
        # Convert string activation name to PyTorch class if present.
        policy_kwargs = dict(sac_cfg.get("policy_kwargs", {}))
        if "activation_fn" in policy_kwargs:
            import torch.nn as nn
            fn_name = policy_kwargs["activation_fn"]
            policy_kwargs["activation_fn"] = getattr(nn, fn_name.title(), nn.ReLU)

        logger.info("Creating new SAC model…")
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=sac_cfg["learning_rate"],
            buffer_size=sac_cfg["buffer_size"],
            batch_size=sac_cfg["batch_size"],
            tau=sac_cfg["tau"],
            gamma=sac_cfg["gamma"],
            ent_coef=sac_cfg["ent_coef"],
            train_freq=sac_cfg["train_freq"],
            gradient_steps=sac_cfg["gradient_steps"],
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(log_dir),
            device="auto",
            verbose=1,
        )

    # -- 4. Training loop with periodic checkpointing -------------------------
    save_freq = train_cfg["save_freq"]
    remaining = total_timesteps
    start_time = time.time()

    # Build a descriptive prefix for saved models.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_prefix = f"sac_mujoco_{stem}_{timestamp}"

    logger.info(
        "Starting training… %d total timesteps, saving every %d steps.",
        total_timesteps,
        save_freq,
    )

    while remaining > 0:
        step = min(save_freq, remaining)
        model.learn(
            total_timesteps=step,
            reset_num_timesteps=False,
            tb_log_name=model_prefix,
        )
        remaining -= step
        steps_done = total_timesteps - remaining

        # Save checkpoint
        ckpt_name = f"{model_prefix}_{steps_done}_steps"
        ckpt_path = model_dir / ckpt_name
        model.save(str(ckpt_path))
        # Also save VecNormalize statistics alongside.
        norm_path = model_dir / f"{ckpt_name}_vecnormalize.pkl"
        env.save(str(norm_path))
        logger.info(
            "Checkpoint saved: %s (steps=%d/%d)",
            ckpt_path,
            steps_done,
            total_timesteps,
        )

    elapsed = time.time() - start_time
    logger.info("Training completed in %.1f seconds (%.1f minutes).", elapsed, elapsed / 60)

    # -- 5. Save final model --------------------------------------------------
    final_path = model_dir / f"{model_prefix}_final"
    model.save(str(final_path))
    final_norm = model_dir / f"{model_prefix}_final_vecnormalize.pkl"
    env.save(str(final_norm))
    logger.info("Final model saved to: %s", final_path)

    # -- 6. Quick evaluation --------------------------------------------------
    eval_episodes = train_cfg.get("eval_episodes", 0)
    if eval_episodes > 0:
        logger.info("Running evaluation for %d episodes…", eval_episodes)
        env_eval = make_env(
            world_xml=world_xml,
            scan_bins=env_cfg.get("scan_bins", 64),
            max_steps=env_cfg.get("max_episode_steps", 600),
            render_mode="rgb_array",
            apply_dr=False,
        )

        rewards: list[float] = []
        successes: list[bool] = []
        collisions: list[bool] = []

        for ep in range(eval_episodes):
            obs, _ = env_eval.reset()
            ep_reward = 0.0
            done = False

            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env_eval.step(action)
                ep_reward += float(reward)
                done = terminated or truncated

            rewards.append(ep_reward)
            collisions.append(bool(info.get("collision", False)))
            successes.append(bool(info.get("success", False)))

        env_eval.close()

        logger.info(
            "Evaluation (n=%d): mean_reward=%.2f, success_rate=%.2f%%, collision_rate=%.2f%%",
            eval_episodes,
            np.mean(rewards),
            100.0 * np.mean(successes),
            100.0 * np.mean(collisions),
        )

    env.close()
    logger.info("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SAC policy on MuJoCo LeKiwi depth navigation environment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python sim_mujoco/train/train_sac_mujoco.py --world lab_empty.xml --timesteps 500000
  python sim_mujoco/train/train_sac_mujoco.py --config sim_mujoco/configs/train_sac.yaml
  python sim_mujoco/train/train_sac_mujoco.py --resume sac_mujoco_lab_empty_200000_steps.zip
        """,
    )
    parser.add_argument(
        "--world",
        type=str,
        default=None,
        help="World XML filename under sim_mujoco/worlds/ (default: lab_empty.xml).",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Total training timesteps (default: 500000).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (default: sim_mujoco/configs/train_sac.yaml).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume training from a checkpoint .zip file.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    train(args)
