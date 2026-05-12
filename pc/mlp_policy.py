"""
PyTorch MLP policy network for navigation.

Takes scan + goal heading + current velocity, outputs (vx, vy, omega).
omega is in deg/s following lerobot convention.

Can be trained with SAC (stable-baselines3) and loaded for inference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class MLPPolicy(nn.Module):
    """
    Multi-layer perceptron policy network for omnidirectional navigation.

    Action space: (vx [m/s], vy [m/s], omega [deg/s]).
    """

    def __init__(
        self,
        obs_dim: int = 67,
        act_dim: int = 3,
        hidden_layers: list[int] | None = None,
        activation: str = "relu",
        max_vx: float = 0.3,
        max_vy: float = 0.3,
        max_omega: float = 90.0,
    ):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [256, 256]

        layers = []
        in_dim = obs_dim
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU() if activation == "relu" else nn.SiLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, act_dim * 2))  # mean + log_std
        self._net = nn.Sequential(*layers)

        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))

        # Action scaling to physical ranges
        self.register_buffer(
            "action_scale", torch.tensor([max_vx, max_vy, max_omega])
        )
        self._act_dim = act_dim
        self._obs_dim = obs_dim

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self._net(obs)
        mean, log_std = out.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -5, 2)
        return mean, log_std

    def sample(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        if deterministic:
            action_raw = mean
        else:
            dist = torch.distributions.Normal(mean, std)
            action_raw = dist.rsample()
        action = torch.tanh(action_raw) * self.action_scale.to(action_raw.device)
        return action, mean

    @torch.no_grad()
    def predict(self, obs_np: np.ndarray, deterministic: bool = True) -> np.ndarray:
        self.eval()
        was_1d = obs_np.ndim == 1
        if was_1d:
            obs_np = obs_np[None, :]
        obs_tensor = torch.from_numpy(obs_np).float()
        obs_tensor = (obs_tensor - self.obs_mean) / (self.obs_std + 1e-8)
        action, _ = self.sample(obs_tensor, deterministic=deterministic)
        action_np = action.cpu().numpy()
        if was_1d:
            action_np = action_np[0]
        return action_np

    def set_normalization(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.obs_mean = torch.from_numpy(mean).float()
        self.obs_std = torch.from_numpy(std).float()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "obs_mean": self.obs_mean,
                "obs_std": self.obs_std,
                "action_scale": self.action_scale,
                "obs_dim": self._obs_dim,
                "act_dim": self._act_dim,
            },
            str(path),
        )

    @classmethod
    def load(cls, path: str | Path) -> "MLPPolicy":
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        model = cls(
            obs_dim=checkpoint["obs_dim"],
            act_dim=checkpoint["act_dim"],
            max_vx=checkpoint["action_scale"][0].item(),
            max_vy=checkpoint["action_scale"][1].item(),
            max_omega=checkpoint["action_scale"][2].item(),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.obs_mean = checkpoint["obs_mean"]
        model.obs_std = checkpoint["obs_std"]
        model.action_scale = checkpoint["action_scale"]
        return model
