"""
Train the residual correction network.

Loads a dataset of (X, Y) pairs where X = [scan, candidate_action, velocity, goal_heading]
and Y = safer_action - candidate_action, then trains an MLP with MSE loss
plus an L2 penalty on the residual norm to encourage small corrections.

Usage:
    # First generate a dataset:
    python tools/generate_residual_dataset.py --sim --episodes 200 --output datasets/residual_dataset.npz

    # Then train:
    python train/train_residual_correction.py --dataset datasets/residual_dataset.npz --epochs 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pc.residual_correction import ResidualCorrectionNet


def load_dataset(
    path: str, val_split: float = 0.1, seed: int = 42
) -> tuple[DataLoader, DataLoader, np.ndarray, np.ndarray]:
    """Load NPZ dataset and return train/val dataloaders and normalization stats.

    Returns:
        train_loader, val_loader, x_mean, x_std
    """
    data = np.load(path)
    X = data["X"].astype(np.float32)
    Y = data["Y"].astype(np.float32)

    print(f"Loaded dataset: X={X.shape}, Y={Y.shape}")

    # Shuffle and split
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))
    split = int(len(X) * (1 - val_split))
    train_idx = idx[:split]
    val_idx = idx[split:]

    # Compute normalization stats from training split
    x_mean = X[train_idx].mean(axis=0)
    x_std = X[train_idx].std(axis=0)
    x_std = np.where(x_std < 1e-8, 1.0, x_std)

    # Per-dimension Y normalization — critical: vx,vy ~ ±0.3, omega ~ ±90
    y_mean = Y[train_idx].mean(axis=0)
    y_std = Y[train_idx].std(axis=0)
    y_std = np.where(y_std < 1e-8, 1.0, y_std)
    print(f"  y_mean: {y_mean}, y_std: {y_std}")

    # Build datasets
    X_train = (X[train_idx] - x_mean) / x_std
    Y_train = (Y[train_idx] - y_mean) / y_std
    X_val = (X[val_idx] - x_mean) / x_std
    Y_val = (Y[val_idx] - y_mean) / y_std

    train_ds = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(Y_train)
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val), torch.from_numpy(Y_val)
    )

    return train_ds, val_ds, x_mean, x_std, y_mean, y_std


def train(
    model: ResidualCorrectionNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    residual_norm_weight: float = 0.01,
    device: str = "cpu",
    checkpoint_dir: str = "models",
) -> None:
    """Train the residual correction network."""
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    mse_loss = nn.MSELoss()

    best_val_loss = float("inf")
    checkpoint_path = Path(checkpoint_dir) / "residual_correction.pt"

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_loss = 0.0
        train_mse = 0.0
        train_penalty = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            pred = model.forward(batch_x)  # (B, 3)

            mse = mse_loss(pred, batch_y)
            # L2 penalty on residual magnitude — encourages small corrections
            norm_penalty = pred.pow(2).mean()
            loss = mse + residual_norm_weight * norm_penalty

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_mse += mse.item()
            train_penalty += norm_penalty.item()

        n_batches = len(train_loader)
        train_loss /= n_batches
        train_mse /= n_batches
        train_penalty /= n_batches

        # --- Val ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                pred = model.forward(batch_x)
                val_loss += mse_loss(pred, batch_y).item()
        val_loss /= len(val_loader)

        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch + 1:3d}/{epochs} | "
                f"train_loss={train_loss:.6f} (mse={train_mse:.6f} pen={train_penalty:.6f}) | "
                f"val_loss={val_loss:.6f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(str(checkpoint_path))

    print(f"\nBest val loss: {best_val_loss:.6f}")
    print(f"Model saved to {checkpoint_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train residual correction network"
    )
    parser.add_argument("--dataset", default="datasets/residual_dataset.npz",
                        help="Path to residual dataset .npz.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--residual-norm-weight", type=float, default=0.01,
                        help="L2 penalty weight on residual norm.")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--max-residual-v", type=float, default=0.15)
    parser.add_argument("--max-residual-omega", type=float, default=30.0)
    parser.add_argument("--checkpoint-dir", default="models")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Device: {args.device}")
    print(f"Loading {args.dataset} ...")

    train_ds, val_ds, x_mean, x_std, y_mean, y_std = load_dataset(
        args.dataset, val_split=args.val_split, seed=args.seed,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = ResidualCorrectionNet(
        hidden_dims=tuple(args.hidden_dims),
        max_residual_v=args.max_residual_v,
        max_residual_omega=args.max_residual_omega,
    )
    model.set_normalization(x_mean, x_std, y_mean, y_std)

    print(f"Model: {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Training {args.epochs} epochs, batch_size={args.batch_size} ...\n")

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        residual_norm_weight=args.residual_norm_weight,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
