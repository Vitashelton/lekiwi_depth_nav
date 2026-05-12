#!/usr/bin/env python3
"""
Train residual safety correction model using MuJoCo rollout data.

The model learns to predict a corrective delta_action that makes raw
actions safer. DWA is used as a safety teacher to produce ground-truth
corrections. The trained model is compatible with pc/policy_server.py
(mode="residual_correction").

Usage:
    python sim_mujoco/train/train_residual_safety.py --episodes 200 --raw-policy rule
    python sim_mujoco/train/train_residual_safety.py --episodes 500 --raw-policy dwa --world lab_empty.xml
    python sim_mujoco/train/train_residual_safety.py --raw-policy models/sac_mujoco_final.zip --episodes 300
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Ensure project root is on path for sim_mujoco / pc imports.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sim_mujoco.envs.lekiwi_depth_scan_env import make_env, EnvConfig
from pc.residual_correction import ResidualCorrectionNet
from pc.dwa_policy import DWAPlanner, DWAConfig
from pc.rule_policy import RulePolicy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_residual")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default DWA teacher configuration (conservative settings for safety).
TEACHER_DWA_CONFIG = DWAConfig(
    max_linear_vel=0.3,
    max_angular_vel=90.0,
    linear_accel=0.5,
    angular_accel=180.0,
    dt=0.1,
    predict_steps=15,
    heading_weight=0.3,
    clearance_weight=1.5,  # Higher clearance weight for safety teacher.
    velocity_weight=0.05,
    obstacle_cost_gain=1.0,
    num_samples=80,  # More samples for better teacher quality.
)

# Default Rule policy configuration (simple but fast).
DEFAULT_RULE_SAFE_DIST = 0.3
DEFAULT_RULE_DANGER_DIST = 0.2
DEFAULT_RULE_FORWARD_SPEED = 0.2
DEFAULT_RULE_TURN_GAIN = 30.0

# Training hyper-parameters.
DEFAULT_BATCH_SIZE = 128
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_NUM_EPOCHS = 100
DEFAULT_VAL_SPLIT = 0.1
DEFAULT_HIDDEN_DIMS = (128, 64)
DEFAULT_MAX_RESIDUAL_V = 0.15  # m/s
DEFAULT_MAX_RESIDUAL_OMEGA = 30.0  # deg/s

# Observation components sizes.
SCAN_BINS = 64
ACT_DIM = 3
VEL_DIM = 2
GOAL_DIM = 1


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

class RawPolicyWrapper:
    """Wraps the raw policy source (rule / dwa / SAC model) into a unified
    callable that returns (vx, vy, omega)."""

    def __init__(
        self,
        source: str,
        scan_bins: int = SCAN_BINS,
        fov_deg: float = 90.0,
        env=None,  # provided for SAC obs normalization when needed
    ) -> None:
        self._source = source
        self._scan_bins = scan_bins
        self._fov_deg = fov_deg
        self._env = env
        self._sac_model = None
        self._rule = None
        self._dwa = None

        source_lower = source.lower()

        if source_lower == "rule":
            self._rule = RulePolicy(
                safe_distance=DEFAULT_RULE_SAFE_DIST,
                danger_distance=DEFAULT_RULE_DANGER_DIST,
                forward_speed=DEFAULT_RULE_FORWARD_SPEED,
                turn_gain=DEFAULT_RULE_TURN_GAIN,
                scan_bins=scan_bins,
                fov_deg=fov_deg,
                max_linear_vel=0.3,
                max_angular_vel=90.0,
            )
            self._mode = "rule"
        elif source_lower == "dwa":
            self._dwa = DWAPlanner(
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
                scan_bins=scan_bins,
                fov_deg=fov_deg,
            )
            self._mode = "dwa"
        else:
            # Assume it is a path to a SAC checkpoint.
            self._mode = "sac"
            self._sac_path = source
            self._load_sac_model(source)

        logger.info("Raw policy: %s", self._mode)

    def _load_sac_model(self, path: str) -> None:
        """Lazy-import SB3 and load SAC policy."""
        try:
            from stable_baselines3 import SAC
        except ImportError as exc:
            raise ImportError(
                "stable_baselines3 is required for SAC-based raw policy. "
                "Install with: pip install stable-baselines3"
            ) from exc

        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"SAC model not found: {path}")

        self._sac_model = SAC.load(str(model_path), device="cpu")
        logger.info("Loaded SAC model from %s", path)

    def __call__(
        self, scan_m: np.ndarray, goal_heading: float
    ) -> np.ndarray:
        """
        Returns:
            (3,) np.ndarray [vx, vy, omega].
        """
        if self._mode == "rule":
            vx, vy, omega = self._rule(scan_m, goal_heading)
            return np.array([vx, vy, omega], dtype=np.float32)
        elif self._mode == "dwa":
            vx, vy, omega = self._dwa(scan_m, goal_heading)
            return np.array([vx, vy, omega], dtype=np.float32)
        elif self._mode == "sac":
            # Build observation as expected by the SAC policy (72-D, normalized).
            obs = self._build_sac_obs(scan_m, goal_heading)
            action, _states = self._sac_model.predict(obs, deterministic=True)
            return np.asarray(action, dtype=np.float32)
        else:
            raise RuntimeError(f"Unknown raw policy mode: {self._mode}")

    def _build_sac_obs(
        self, scan_m: np.ndarray, goal_heading: float
    ) -> np.ndarray:
        """Build a 72-D observation compatible with the SAC policy's input.
        This matches the observation layout of LeKiwiDepthScanEnv._get_obs()."""
        import math

        scan_norm = np.clip(scan_m / 5.0, 0.0, 1.0).astype(np.float32)

        # Goal vector placeholder (teacher provides heading, not full goal pos).
        goal_vec_norm = np.array(
            [0.1 * math.cos(goal_heading), 0.1 * math.sin(goal_heading)],
            dtype=np.float32,
        )

        goal_cos = math.cos(goal_heading)
        goal_sin = math.sin(goal_heading)

        # Velocity placeholder (unknown during data collection; use zeros).
        vx_norm = 0.0
        vy_norm = 0.0
        omega_norm = 0.0

        dist_norm = 0.1  # placeholder

        return np.concatenate([
            scan_norm,
            goal_vec_norm,
            np.array([goal_cos, goal_sin], dtype=np.float32),
            np.array([vx_norm, vy_norm, omega_norm], dtype=np.float32),
            np.array([dist_norm], dtype=np.float32),
        ]).astype(np.float32)


def collect_rollout_data(
    raw_policy: RawPolicyWrapper,
    teacher_planner: DWAPlanner,
    world_xml: str,
    num_episodes: int,
    max_steps: int = 600,
    scan_bins: int = SCAN_BINS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Roll out the raw policy in the MuJoCo environment and use DWA as
    safety teacher to produce correction labels.

    For each timestep we record:
      - scan_m:  (scan_bins,) metric scan in meters
      - raw_action: (act_dim,) [vx, vy, omega] from raw policy
      - current_velocity: (vel_dim,) [vx, vy]
      - goal_heading: (1,) heading to goal relative to robot
      - safer_action: (act_dim,) [vx, vy, omega] from teacher

    Returns:
        scans, raw_actions, velocities, goal_headings, safer_actions
        Each is a (T, D) numpy array where T is total collected timesteps.
    """
    env = make_env(
        world_xml=world_xml,
        scan_bins=scan_bins,
        max_steps=max_steps,
        render_mode="rgb_array",
        apply_dr=False,
    )

    scans_list: list[np.ndarray] = []
    raws_list: list[np.ndarray] = []
    vels_list: list[np.ndarray] = []
    goals_list: list[np.ndarray] = []
    safes_list: list[np.ndarray] = []

    total_steps = 0

    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        done = False
        ep_steps = 0
        prev_vel = np.zeros(VEL_DIM, dtype=np.float32)

        while not done:
            # Extract scan_m from the full observation.
            # Observation layout (72-D):
            #   [0:64] scan_norm, [64:66] goal_vec_norm, [66:68] (cos,sin),
            #   [68:71] velocity_norm, [71] dist_norm
            scan_norm = obs[:scan_bins]
            scan_m = scan_norm * 5.0  # denormalize (max_range=5.0)

            # Goal heading from observation.
            goal_cos = float(obs[66])
            goal_sin = float(obs[67])
            import math
            goal_heading = math.atan2(goal_sin, goal_cos)

            # Current velocity (denormalized).
            vx = float(obs[68]) * 0.3
            vy = float(obs[69]) * 0.3
            curr_vel = np.array([vx, vy], dtype=np.float32)

            # Query raw policy.
            raw_action = raw_policy(scan_m, goal_heading)

            # Query safety teacher (DWA).
            safer_vx, safer_vy, safer_omega = teacher_planner(scan_m, goal_heading)
            safer_action = np.array([safer_vx, safer_vy, safer_omega], dtype=np.float32)

            # Store data point.
            scans_list.append(scan_m.astype(np.float32))
            raws_list.append(raw_action)
            vels_list.append(curr_vel)
            goals_list.append(np.array([goal_heading], dtype=np.float32))
            safes_list.append(safer_action)

            # Step the environment using the *raw* action (so we see what
            # the raw policy would actually encounter).
            obs, _reward, terminated, truncated, _info = env.step(raw_action)
            done = terminated or truncated

            prev_vel = curr_vel
            ep_steps += 1

        total_steps += ep_steps
        if ep % 10 == 0 or ep == num_episodes:
            logger.info(
                "  episode %d/%d | steps=%d | total_steps=%d",
                ep,
                num_episodes,
                ep_steps,
                total_steps,
            )

    env.close()

    scans = np.stack(scans_list, axis=0)  # (T, scan_bins)
    raws = np.stack(raws_list, axis=0)    # (T, act_dim)
    vels = np.stack(vels_list, axis=0)    # (T, vel_dim)
    goals = np.stack(goals_list, axis=0)  # (T, 1)
    safes = np.stack(safes_list, axis=0)  # (T, act_dim)

    logger.info(
        "Data collection complete: %d total timesteps from %d episodes.",
        total_steps,
        num_episodes,
    )
    logger.info("Shapes — scans:%s raws:%s vels:%s goals:%s safes:%s",
                 scans.shape, raws.shape, vels.shape, goals.shape, safes.shape)

    return scans, raws, vels, goals, safes


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def compute_statistics(
    scans: np.ndarray,
    raws: np.ndarray,
    vels: np.ndarray,
    goals: np.ndarray,
    safes: np.ndarray,
) -> dict:
    """Compute input and output normalization statistics."""
    # Input: [scan, raw_action, velocity, goal_heading]
    X = np.concatenate([scans, raws, vels, goals], axis=1)
    # Output: residual = safer - raw
    Y = safes - raws

    x_mean = X.mean(axis=0).astype(np.float32)
    x_std = X.std(axis=0).astype(np.float32) + 1e-8
    y_mean = Y.mean(axis=0).astype(np.float32)
    y_std = Y.std(axis=0).astype(np.float32) + 1e-8

    logger.info("Input  mean range: [%.4f, %.4f]", float(x_mean.min()), float(x_mean.max()))
    logger.info("Output mean: [%.4f, %.4f, %.4f]", float(y_mean[0]), float(y_mean[1]), float(y_mean[2]))
    logger.info("Output std:  [%.4f, %.4f, %.4f]", float(y_std[0]), float(y_std[1]), float(y_std[2]))

    return {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }


def train_residual_model(
    scans: np.ndarray,
    raws: np.ndarray,
    vels: np.ndarray,
    goals: np.ndarray,
    safes: np.ndarray,
    output_path: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    val_split: float = DEFAULT_VAL_SPLIT,
    hidden_dims: tuple[int, ...] = DEFAULT_HIDDEN_DIMS,
    max_residual_v: float = DEFAULT_MAX_RESIDUAL_V,
    max_residual_omega: float = DEFAULT_MAX_RESIDUAL_OMEGA,
    device: str = "cpu",
) -> ResidualCorrectionNet:
    """Train ResidualCorrectionNet on collected rollout data.

    Args:
        scans: (T, scan_bins) metric scans.
        raws: (T, act_dim) raw policy actions.
        vels: (T, vel_dim) current velocities.
        goals: (T, 1) goal headings.
        safes: (T, act_dim) teacher actions.
        output_path: where to save the trained model.
        batch_size, learning_rate, num_epochs, val_split: training params.
        hidden_dims: tuple of hidden layer sizes.
        max_residual_v, max_residual_omega: residual magnitude bounds.
        device: torch device string.

    Returns:
        Trained ResidualCorrectionNet.
    """

    # -- Build dataset ---------------------------------------------------------
    # Input features: [scan, raw_action, velocity, goal_heading]
    X = np.concatenate([scans, raws, vels, goals], axis=1).astype(np.float32)
    # Target: residual = safer - raw
    Y = safes - raws
    Y = Y.astype(np.float32)

    # Train / validation split.
    n_total = X.shape[0]
    n_val = max(1, int(n_total * val_split))
    indices = np.random.RandomState(42).permutation(n_total)
    train_idx = indices[n_val:]
    val_idx = indices[:n_val]

    X_train = torch.from_numpy(X[train_idx])
    Y_train = torch.from_numpy(Y[train_idx])
    X_val = torch.from_numpy(X[val_idx])
    Y_val = torch.from_numpy(Y[val_idx])

    logger.info(
        "Training samples: %d | Validation samples: %d",
        len(train_idx),
        len(val_idx),
    )

    train_loader = DataLoader(
        TensorDataset(X_train, Y_train),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, Y_val),
        batch_size=batch_size,
        shuffle=False,
    )

    # -- Model -----------------------------------------------------------------
    model = ResidualCorrectionNet(
        scan_dim=scans.shape[1],
        act_dim=raws.shape[1],
        vel_dim=vels.shape[1],
        goal_dim=goals.shape[1],
        hidden_dims=hidden_dims,
        activation="relu",
        max_residual_v=max_residual_v,
        max_residual_omega=max_residual_omega,
    ).to(device)

    # Set normalization stats from the full training set.
    stats = compute_statistics(scans, raws, vels, goals, safes)
    model.set_normalization(
        x_mean=stats["x_mean"],
        x_std=stats["x_std"],
        y_mean=stats["y_mean"],
        y_std=stats["y_std"],
    )

    # -- Optimizer & loss ------------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, verbose=True,
    )
    loss_fn = nn.MSELoss()

    # -- Training loop ---------------------------------------------------------
    best_val_loss = float("inf")
    best_epoch = 0
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            pred = model.forward(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)

        train_loss /= len(train_idx)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                pred = model.forward(batch_x)
                loss = loss_fn(pred, batch_y)
                val_loss += loss.item() * batch_x.size(0)
        val_loss /= len(val_idx)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            # Save best model immediately.
            model.save(output_path)

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            logger.info(
                "epoch %3d/%d | train_loss=%.6f | val_loss=%.6f | lr=%.2e | elapsed=%.0fs",
                epoch,
                num_epochs,
                train_loss,
                val_loss,
                optimizer.param_groups[0]["lr"],
                elapsed,
            )

    elapsed = time.time() - start_time
    logger.info("Training finished. Best val_loss=%.6f at epoch %d (%.1fs)",
                 best_val_loss, best_epoch, elapsed)

    # Save final model.
    model.save(output_path)
    logger.info("Model saved to %s", output_path)

    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(parsed_args: argparse.Namespace) -> None:
    """Orchestrate data collection and residual model training."""

    world_xml = parsed_args.world
    num_episodes = parsed_args.episodes
    raw_policy_src = parsed_args.raw_policy
    output_path = parsed_args.output
    device = parsed_args.device

    logger.info("=== Residual Safety Training ===")
    logger.info("World: %s", world_xml)
    logger.info("Episodes: %d", num_episodes)
    logger.info("Raw policy source: %s", raw_policy_src)
    logger.info("Output: %s", output_path)
    logger.info("Device: %s", device)

    # --- 1. Build raw policy wrapper ------------------------------------------
    raw_policy = RawPolicyWrapper(
        source=raw_policy_src,
        scan_bins=SCAN_BINS,
        fov_deg=90.0,
    )

    # --- 2. Build DWA safety teacher ------------------------------------------
    teacher = DWAPlanner(
        config=TEACHER_DWA_CONFIG,
        scan_bins=SCAN_BINS,
        fov_deg=90.0,
    )
    logger.info("DWA teacher configured (clearance_weight=%.1f, num_samples=%d).",
                 TEACHER_DWA_CONFIG.clearance_weight, TEACHER_DWA_CONFIG.num_samples)

    # --- 3. Collect rollout data ----------------------------------------------
    logger.info("Collecting rollout data (%d episodes)...", num_episodes)
    scans, raws, vels, goals, safes = collect_rollout_data(
        raw_policy=raw_policy,
        teacher_planner=teacher,
        world_xml=world_xml,
        num_episodes=num_episodes,
        max_steps=parsed_args.max_steps,
        scan_bins=SCAN_BINS,
    )

    if scans.shape[0] < 100:
        logger.warning(
            "Only %d samples collected. Training may be unstable. "
            "Consider increasing --episodes.",
            scans.shape[0],
        )

    # --- 4. Train residual correction model -----------------------------------
    logger.info("Training ResidualCorrectionNet...")
    train_residual_model(
        scans=scans,
        raws=raws,
        vels=vels,
        goals=goals,
        safes=safes,
        output_path=output_path,
        batch_size=parsed_args.batch_size,
        learning_rate=parsed_args.learning_rate,
        num_epochs=parsed_args.epochs,
        val_split=parsed_args.val_split,
        hidden_dims=DEFAULT_HIDDEN_DIMS,
        max_residual_v=parsed_args.max_residual_v,
        max_residual_omega=parsed_args.max_residual_omega,
        device=device,
    )

    # --- 5. Quick sanity check ------------------------------------------------
    logger.info("Running sanity check on trained model...")
    model = ResidualCorrectionNet.load(output_path)
    sample_scan = scans[0]
    sample_raw = raws[0]  # noqa: F841 (used in f-string below)
    sample_vel = vels[0]
    sample_goal = float(goals[0, 0])
    residual = model.predict_residual(
        scan_m=sample_scan,
        candidate_action=raws[0],
        current_velocity=sample_vel,
        goal_heading=sample_goal,
    )
    corrected = model.correct_action(
        scan_m=sample_scan,
        candidate_action=raws[0],
        current_velocity=sample_vel,
        goal_heading=sample_goal,
    )
    logger.info(
        "Sample: raw=[%.3f,%.3f,%.1f] residual=[%.3f,%.3f,%.1f] corrected=[%.3f,%.3f,%.1f]",
        float(raws[0, 0]), float(raws[0, 1]), float(raws[0, 2]),
        float(residual[0]), float(residual[1]), float(residual[2]),
        float(corrected[0]), float(corrected[1]), float(corrected[2]),
    )

    logger.info("Done. Trained model: %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train residual safety correction model using MuJoCo rollout data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python sim_mujoco/train/train_residual_safety.py --episodes 200 --raw-policy rule
  python sim_mujoco/train/train_residual_safety.py --episodes 500 --raw-policy dwa
  python sim_mujoco/train/train_residual_safety.py --raw-policy models/sac_mujoco_final.zip --episodes 300
""",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=200,
        help="Number of rollout episodes for data collection (default: 200).",
    )
    parser.add_argument(
        "--raw-policy",
        type=str,
        default="rule",
        help=(
            "Source of raw (candidate) actions: 'rule', 'dwa', "
            "or path to a SAC model .zip (default: rule)."
        ),
    )
    parser.add_argument(
        "--world",
        type=str,
        default="lab_empty.xml",
        help="World XML filename under sim_mujoco/worlds/ (default: lab_empty.xml).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=600,
        help="Maximum steps per episode (default: 600).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Path to save the trained model .pt file "
            "(default: models/residual_correction.pt)."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Training batch size (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f"Learning rate (default: {DEFAULT_LEARNING_RATE}).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_NUM_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_NUM_EPOCHS}).",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=DEFAULT_VAL_SPLIT,
        help=f"Validation split fraction (default: {DEFAULT_VAL_SPLIT}).",
    )
    parser.add_argument(
        "--max-residual-v",
        type=float,
        default=DEFAULT_MAX_RESIDUAL_V,
        help=f"Max linear residual magnitude in m/s (default: {DEFAULT_MAX_RESIDUAL_V}).",
    )
    parser.add_argument(
        "--max-residual-omega",
        type=float,
        default=DEFAULT_MAX_RESIDUAL_OMEGA,
        help=f"Max angular residual magnitude in deg/s (default: {DEFAULT_MAX_RESIDUAL_OMEGA}).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device: 'cpu', 'cuda', or 'mps' (default: cpu).",
    )

    args = parser.parse_args(argv)

    # Resolve default output path.
    if args.output is None:
        args.output = str(
            _PROJECT_ROOT / "computed" / "models" / "residual_correction.pt"
        )

    return args


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    main(args)
