"""
Geometric risk functions for omnidirectional robot navigation.

Computes continuous, differentiable risk scores from pseudo-LiDAR scans
and candidate velocity actions. Used to evaluate action safety and to
generate safer alternatives via sampling-based projection.

All functions operate on numpy arrays and avoid hard if-else branching
where possible, using smooth approximations instead.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


def _sigmoid(x: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Smooth sigmoid: 0→0, ∞→1. Numerically stable."""
    x = np.asarray(x, dtype=np.float32)
    sx = scale * x
    # Clamp for numerical stability
    sx = np.clip(sx, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-sx))


def _softmin(x: np.ndarray, temperature: float = 0.1) -> np.ndarray:
    """Differentiable soft minimum approximation."""
    x = np.asarray(x, dtype=np.float32)
    x_shift = x - x.min()
    w = np.exp(-x_shift / (temperature + 1e-8))
    return np.sum(x * w) / (np.sum(w) + 1e-8)


# ═══════════════════════════════════════════════════════════════════════════
# Directional Risk
# ═══════════════════════════════════════════════════════════════════════════

def compute_directional_risk(
    scan_m: np.ndarray,
    action: np.ndarray,
    fov_deg: float = 90.0,
    max_range: float = 5.0,
    safe_distance: float = 0.3,
    danger_distance: float = 0.15,
    temperature: float = 0.1,
) -> float:
    """
    Compute a continuous risk score in [0, 1] by projecting the action
    direction onto the scan's angular bins.

    Risk is high when the robot moves toward close obstacles.

    Args:
        scan_m: (N,) metric scan distances in meters.
        action: (3,) [vx, vy, omega]; vx,vy define the motion direction.
        fov_deg: scan field of view in degrees.
        max_range: max scan range (m).
        safe_distance: below this, risk begins to rise.
        danger_distance: below this, risk saturates at 1.
        temperature: soft attention sharpness over angular bins.

    Returns:
        Scalar risk ∈ [0, 1].
    """
    N = len(scan_m)
    half_fov = np.deg2rad(fov_deg / 2.0)
    bin_angles = np.linspace(-half_fov, half_fov, N)

    # Motion direction angle in robot frame
    vx, vy = float(action[0]), float(action[1])
    motion_angle = np.arctan2(vy, vx) if abs(vx) + abs(vy) > 1e-6 else 0.0

    # Angular difference to each bin (wrapped)
    angle_diff = np.abs(bin_angles - motion_angle)
    angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)

    # Soft attention: weight bins close to the motion direction
    weights = np.exp(-angle_diff / (temperature + 1e-8))
    weights /= weights.sum() + 1e-8

    # Per-bin risk: 0 when far, 1 when dangerously close
    # Smooth transition between safe_distance and danger_distance
    bin_risk = _sigmoid(safe_distance - scan_m, scale=10.0)
    # Boost for danger zone
    danger_boost = _sigmoid(danger_distance - scan_m, scale=20.0)
    bin_risk = np.clip(bin_risk + 0.5 * danger_boost, 0.0, 1.0)

    # Weighted sum over angular bins
    risk = float(np.dot(weights, bin_risk))
    return np.clip(risk, 0.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# Clearance Cost
# ═══════════════════════════════════════════════════════════════════════════

def compute_clearance_cost(
    scan_m: np.ndarray,
    action: np.ndarray,
    fov_deg: float = 90.0,
    max_range: float = 5.0,
    min_clearance: float = 0.15,
    dt: float = 0.5,
    num_lookahead_pts: int = 8,
) -> float:
    """
    Simulate forward motion for `dt` seconds and compute the minimum
    clearance along the predicted trajectory.

    A low clearance → high cost.

    Args:
        scan_m: (N,) metric scan in meters.
        action: (3,) [vx, vy, omega].
        fov_deg: scan FOV in degrees.
        max_range: maximum scan range.
        min_clearance: clearance below which cost goes to 1.
        dt: lookahead duration in seconds.
        num_lookahead_pts: points to sample along trajectory.

    Returns:
        Scalar cost ∈ [0, 1], where 1 = imminent collision.
    """
    N = len(scan_m)
    half_fov = np.deg2rad(fov_deg / 2.0)
    bin_angles = np.linspace(-half_fov, half_fov, N)

    vx, vy, omega_deg = float(action[0]), float(action[1]), float(action[2])
    omega = np.deg2rad(omega_deg)

    clearances: list[float] = []
    for k in range(1, num_lookahead_pts + 1):
        t = dt * k / num_lookahead_pts
        # Predicted displacement in robot frame (approximate)
        if abs(omega) > 1e-4:
            dx = (vx * np.sin(omega * t) - vy * (1 - np.cos(omega * t))) / omega
            dy = (vx * (1 - np.cos(omega * t)) + vy * np.sin(omega * t)) / omega
        else:
            dx = vx * t
            dy = vy * t

        dist = np.sqrt(dx**2 + dy**2)
        traj_angle = np.arctan2(dy, dx) if dist > 0.01 else 0.0

        # Find closest bin to this lookahead direction
        angle_diff = np.abs(bin_angles - traj_angle)
        angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
        nearest_bin = int(np.argmin(angle_diff))

        # Estimate clearance at this point
        scan_at_bin = scan_m[nearest_bin]
        clearance = scan_at_bin - dist
        clearances.append(max(0.0, clearance))

    min_c = float(np.min(clearances)) if clearances else max_range

    # Smooth cost
    cost = _sigmoid(min_clearance - min_c, scale=8.0)
    return float(np.clip(cost, 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════════════════
# Action Projection (sampling-based safety projection)
# ═══════════════════════════════════════════════════════════════════════════

def compute_action_projection(
    candidate_action: np.ndarray,
    scan_m: np.ndarray,
    fov_deg: float = 90.0,
    max_range: float = 5.0,
    max_linear_vel: float = 0.3,
    max_angular_vel: float = 90.0,
    lambda_risk: float = 2.0,
    num_samples: int = 200,
    search_radius_v: float = 0.15,
    search_radius_omega: float = 45.0,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """
    Find a safer action near the candidate by sampling and minimizing:
        cost = ||a - a_candidate||^2 + lambda_risk * directional_risk(scan, a)

    Args:
        candidate_action: (3,) original action [vx, vy, omega].
        scan_m: (N,) metric scan.
        fov_deg: scan FOV.
        max_range: scan max range.
        max_linear_vel: velocity bound.
        max_angular_vel: angular velocity bound (deg/s).
        lambda_risk: risk penalty weight (higher = safer, more deviation).
        num_samples: number of random samples around candidate.
        search_radius_v: max vx/vy perturbation.
        search_radius_omega: max omega perturbation (deg/s).
        rng: optional random state for reproducibility.

    Returns:
        (3,) projected action [vx, vy, omega].
    """
    if rng is None:
        rng = np.random.RandomState()

    ca = np.asarray(candidate_action, dtype=np.float32)

    # Generate samples: candidate + uniform noise
    samples = np.tile(ca, (num_samples, 1))
    noise_v = rng.uniform(-search_radius_v, search_radius_v, size=(num_samples, 2))
    noise_w = rng.uniform(-search_radius_omega, search_radius_omega, size=(num_samples, 1))
    noise = np.concatenate([noise_v, noise_w], axis=1)
    samples = samples + noise

    # Clip to physical limits
    samples[:, 0] = np.clip(samples[:, 0], -max_linear_vel, max_linear_vel)
    samples[:, 1] = np.clip(samples[:, 1], -max_linear_vel, max_linear_vel)
    samples[:, 2] = np.clip(samples[:, 2], -max_angular_vel, max_angular_vel)

    # Evaluate cost for each sample
    best_cost = float("inf")
    best_action = ca.copy()

    for i in range(num_samples):
        a_i = samples[i]
        deviation = np.sum((a_i - ca) ** 2)
        risk = compute_directional_risk(
            scan_m, a_i, fov_deg=fov_deg, max_range=max_range
        )
        cost = deviation + lambda_risk * risk

        if cost < best_cost:
            best_cost = cost
            best_action = a_i.copy()

    return best_action
