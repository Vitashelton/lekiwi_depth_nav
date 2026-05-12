"""
Residual correction network for geometry-aware action safety.

A lightweight PyTorch MLP that predicts a residual Δa to adjust
a candidate LeRobot action into a safer navigation command.

Input:  [scan_m, candidate_action, current_velocity, goal_heading]
Output: Δa = [Δvx, Δvy, Δomega]

The residual magnitude is bounded by max_residual to prevent the
correction from fully overriding the original policy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class ResidualCorrectionNet(nn.Module):
    """
    MLP that predicts a bounded residual action correction.

    Architecture:
        Linear(scan_dim + 2*act_dim + vel_dim + goal_dim) → hidden → hidden → act_dim
    """

    def __init__(
        self,
        scan_dim: int = 64,
        act_dim: int = 3,
        vel_dim: int = 2,
        goal_dim: int = 1,
        hidden_dims: tuple[int, ...] = (128, 64),
        activation: str = "relu",
        max_residual_v: float = 0.15,
        max_residual_omega: float = 30.0,
    ) -> None:
        """
        Args:
            scan_dim: pseudo-LiDAR scan bins.
            act_dim: action dimension (vx, vy, omega).
            vel_dim: current velocity dimensions (vx, vy).
            goal_dim: goal heading dimension.
            hidden_dims: tuple of hidden layer sizes.
            activation: "relu" or "silu".
            max_residual_v: max residual magnitude for linear velocity (m/s).
            max_residual_omega: max residual magnitude for angular velocity (deg/s).
        """
        super().__init__()
        input_dim = scan_dim + act_dim + vel_dim + goal_dim
        self.scan_dim = scan_dim
        self.act_dim = act_dim
        self.vel_dim = vel_dim
        self.goal_dim = goal_dim
        self.max_residual_v = max_residual_v
        self.max_residual_omega = max_residual_omega

        layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU() if activation == "relu" else nn.SiLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, act_dim))
        self._net = nn.Sequential(*layers)

        # Residual scale vector: [max_v, max_v, max_omega]
        self.register_buffer(
            "residual_scale",
            torch.tensor(
                [max_residual_v, max_residual_v, max_residual_omega],
                dtype=torch.float32,
            ),
        )

        # Normalization stats (set after training)
        self.register_buffer("input_mean", torch.zeros(input_dim))
        self.register_buffer("input_std", torch.ones(input_dim))

        # Max action bounds (for clipping final action)
        self.register_buffer(
            "action_max",
            torch.tensor([0.3, 0.3, 90.0], dtype=torch.float32),
        )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) concatenated input vector.

        Returns:
            (B, act_dim) residual Δa in physical units.
        """
        # Normalize
        x_norm = (x - self.input_mean) / (self.input_std + 1e-8)
        raw = self._net(x_norm)
        # Tanh squashing → bounded in [-residual_scale, residual_scale]
        residual = torch.tanh(raw) * self.residual_scale
        return residual

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_residual(
        self,
        scan_m: np.ndarray,
        candidate_action: np.ndarray,
        current_velocity: Optional[np.ndarray] = None,
        goal_heading: float = 0.0,
    ) -> np.ndarray:
        """
        Predict a residual correction for a single observation.

        Args:
            scan_m: (scan_dim,) metric scan in meters.
            candidate_action: (act_dim,) candidate action [vx, vy, omega].
            current_velocity: (vel_dim,) current vx, vy. Defaults to zeros.
            goal_heading: goal heading relative to robot (rad).

        Returns:
            (act_dim,) residual Δa in physical units.
        """
        self.eval()

        if current_velocity is None:
            current_velocity = np.zeros(self.vel_dim, dtype=np.float32)

        scan_m = np.asarray(scan_m, dtype=np.float32)
        ca = np.asarray(candidate_action, dtype=np.float32)
        cv = np.asarray(current_velocity, dtype=np.float32)
        gh = np.array([float(goal_heading)], dtype=np.float32)

        inp = np.concatenate([scan_m, ca, cv, gh])
        inp_t = torch.from_numpy(inp).float().unsqueeze(0)
        residual: torch.Tensor = self.forward(inp_t)
        return residual.squeeze(0).cpu().numpy()

    # ------------------------------------------------------------------
    def correct_action(
        self,
        scan_m: np.ndarray,
        candidate_action: np.ndarray,
        current_velocity: Optional[np.ndarray] = None,
        goal_heading: float = 0.0,
        max_linear_vel: float = 0.3,
        max_angular_vel: float = 90.0,
    ) -> np.ndarray:
        """
        Apply residual correction and clip to valid range.

        Returns:
            (act_dim,) final safe action [vx, vy, omega].
        """
        residual = self.predict_residual(
            scan_m, candidate_action, current_velocity, goal_heading,
        )
        final = np.asarray(candidate_action, dtype=np.float32) + residual

        final[0] = np.clip(final[0], -max_linear_vel, max_linear_vel)
        final[1] = np.clip(final[1], -max_linear_vel, max_linear_vel)
        final[2] = np.clip(final[2], -max_angular_vel, max_angular_vel)
        return final

    # ------------------------------------------------------------------
    def set_normalization(
        self, mean: np.ndarray, std: np.ndarray
    ) -> None:
        """Set input normalization statistics."""
        self.input_mean = torch.from_numpy(mean).float()
        self.input_std = torch.from_numpy(std).float()

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "input_mean": self.input_mean,
                "input_std": self.input_std,
                "residual_scale": self.residual_scale,
                "action_max": self.action_max,
                "scan_dim": self.scan_dim,
                "act_dim": self.act_dim,
                "vel_dim": self.vel_dim,
                "goal_dim": self.goal_dim,
                "max_residual_v": self.max_residual_v,
                "max_residual_omega": self.max_residual_omega,
            },
            str(path),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ResidualCorrectionNet":
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        model = cls(
            scan_dim=ckpt["scan_dim"],
            act_dim=ckpt["act_dim"],
            vel_dim=ckpt["vel_dim"],
            goal_dim=ckpt["goal_dim"],
            max_residual_v=ckpt["max_residual_v"],
            max_residual_omega=ckpt["max_residual_omega"],
        )
        model.load_state_dict(ckpt["state_dict"])
        model.input_mean = ckpt["input_mean"]
        model.input_std = ckpt["input_std"]
        model.residual_scale = ckpt["residual_scale"]
        model.action_max = ckpt["action_max"]
        return model
