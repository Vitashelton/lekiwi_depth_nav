"""
Structured test maps simulating real indoor environments: corridors, rooms,
doorways, and lab setups — instead of random circles.

Each map is a layout of rectangular walls and cylindrical obstacles that
form meaningful spatial constraints the robot must navigate around.

Usage:
    python sim/structured_maps.py --output maps/test_suite/
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np


def _rect_to_cylinders(
    cx: float, cy: float, w: float, h: float, spacing: float = 0.12
) -> list[tuple[float, float, float]]:
    """Approximate a rectangular wall with a row of small cylinders."""
    obs = []
    r = spacing / 2.0
    # Top and bottom edges
    for x in np.arange(cx - w / 2, cx + w / 2 + spacing, spacing):
        obs.append((x, cy - h / 2 - r, r))
        obs.append((x, cy + h / 2 + r, r))
    # Left and right edges
    for y in np.arange(cy - h / 2, cy + h / 2 + spacing, spacing):
        obs.append((cx - w / 2 - r, y, r))
        obs.append((cx + w / 2 + r, y, r))
    return obs


def _wall_row(
    x1: float, y1: float, x2: float, y2: float, spacing: float = 0.12
) -> list[tuple[float, float, float]]:
    """A line of small cylinders forming a wall segment."""
    obs = []
    r = spacing / 2.0
    dist = math.hypot(x2 - x1, y2 - y1)
    n_pts = max(2, int(dist / spacing))
    for t in np.linspace(0, 1, n_pts):
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        obs.append((x, y, r))
    return obs


# ── Map builders ──────────────────────────────────────────────────────────

def corridor_straight(map_size: float = 10.0) -> dict:
    """A straight 2m-wide corridor with obstacles along the walls."""
    half = map_size / 2
    corr_half = 1.2  # corridor half-width

    obstacles: list[tuple[float, float, float]] = []

    # Top wall
    obstacles += _wall_row(0, half + corr_half, map_size, half + corr_half, 0.12)
    # Bottom wall
    obstacles += _wall_row(0, half - corr_half, map_size, half - corr_half, 0.12)

    # Some clutter along walls (boxes, equipment)
    rng = np.random.RandomState(42)
    for _ in range(6):
        ox = rng.uniform(1.0, map_size - 1.0)
        side = 1 if rng.rand() > 0.5 else -1
        oy = half + side * (corr_half + 0.25)
        obstacles.append((ox, oy, rng.uniform(0.15, 0.25)))

    # Start at left, goal at right
    start = (1.0, half)
    goal = (map_size - 1.0, half)

    return {
        "map_size": map_size,
        "start": [float(start[0]), float(start[1])],
        "goal": [float(goal[0]), float(goal[1])],
        "obstacles": [{"x": float(x), "y": float(y), "r": float(r)} for x, y, r in obstacles],
        "name": "corridor_straight",
    }


def corridor_l_turn(map_size: float = 10.0) -> dict:
    """An L-shaped corridor: go right then up."""
    corr_half = 1.2

    obstacles: list[tuple[float, float, float]] = []

    # Horizontal corridor section: top wall, bottom wall
    obstacles += _wall_row(0, 3.0 + corr_half, 6.0, 3.0 + corr_half, 0.12)
    obstacles += _wall_row(0, 3.0 - corr_half, 6.0, 3.0 - corr_half, 0.12)

    # Vertical corridor section: left wall, right wall
    obstacles += _wall_row(6.0 - corr_half, 3.0, 6.0 - corr_half, map_size, 0.12)
    obstacles += _wall_row(6.0 + corr_half, 3.0, 6.0 + corr_half, map_size, 0.12)

    # Inner corner block
    obstacles += _rect_to_cylinders(6.0, 3.0, corr_half * 2, corr_half * 2, 0.12)

    # Start bottom-left, goal top-right
    start = (1.0, 3.0)
    goal = (6.0, map_size - 1.0)

    return {
        "map_size": map_size,
        "start": [float(start[0]), float(start[1])],
        "goal": [float(goal[0]), float(goal[1])],
        "obstacles": [{"x": float(x), "y": float(y), "r": float(r)} for x, y, r in obstacles],
        "name": "corridor_l_turn",
    }


def doorway_challenge(map_size: float = 10.0) -> dict:
    """A room with a narrow doorway — the robot must find and go through it."""
    half = map_size / 2
    door_width = 0.8
    door_x = 6.0

    obstacles: list[tuple[float, float, float]] = []

    # Dividing wall with a door opening
    obstacles += _wall_row(door_x, 0, door_x, half - door_width / 2, 0.10)
    obstacles += _wall_row(door_x, half + door_width / 2, door_x, map_size, 0.10)

    # Side walls
    obstacles += _wall_row(0, 0, 0, map_size, 0.12)
    obstacles += _wall_row(map_size, 0, map_size, map_size, 0.12)
    obstacles += _wall_row(0, 0, map_size, 0, 0.12)
    obstacles += _wall_row(0, map_size, map_size, map_size, 0.12)

    # Obstacles near doorway to make it trickier
    rng = np.random.RandomState(99)
    for _ in range(3):
        ox = door_x + rng.uniform(-1.5, 1.5)
        oy = half + rng.uniform(-2.0, 2.0)
        if abs(ox - door_x) < 0.6 and abs(oy - half) < door_width / 2:
            continue  # don't block the door
        obstacles.append((ox, oy, rng.uniform(0.15, 0.3)))

    start = (1.5, half)
    goal = (map_size - 1.5, half)

    return {
        "map_size": map_size,
        "start": [float(start[0]), float(start[1])],
        "goal": [float(goal[0]), float(goal[1])],
        "obstacles": [{"x": float(x), "y": float(y), "r": float(r)} for x, y, r in obstacles],
        "name": "doorway_challenge",
    }


def cluttered_room(map_size: float = 10.0) -> dict:
    """A room with scattered furniture-equivalent obstacles and walls."""
    obstacles: list[tuple[float, float, float]] = []
    rng = np.random.RandomState(77)

    # Outer walls
    obstacles += _wall_row(0, 0, map_size, 0, 0.12)
    obstacles += _wall_row(0, map_size, map_size, map_size, 0.12)
    obstacles += _wall_row(0, 0, 0, map_size, 0.12)
    obstacles += _wall_row(map_size, 0, map_size, map_size, 0.12)

    # Furniture clusters — rectangular groups (simulate desks, shelves)
    furniture = [
        (2.5, 2.5, 1.2, 0.6, 0.0),   # desk
        (7.0, 3.0, 1.5, 0.5, 0.3),   # shelf
        (4.0, 7.5, 1.0, 0.8, 0.5),   # cabinet
        (8.0, 7.0, 0.8, 0.8, 0.0),   # pillar
        (2.0, 6.0, 0.6, 1.5, 0.2),   # bench
    ]
    for fx, fy, fw, fh, angle in furniture:
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        # Sample points along the furniture edges
        for t in np.linspace(-0.5, 0.5, max(3, int(max(fw, fh) / 0.15))):
            # Along width
            px = fx + t * fw * cos_a
            py = fy + t * fw * sin_a
            obstacles.append((px, py, 0.10))
            # Along height
            px2 = fx + t * fh * (-sin_a)
            py2 = fy + t * fh * cos_a
            obstacles.append((px2, py2, 0.10))

    # Scattered small obstacles
    for _ in range(8):
        ox = rng.uniform(0.8, map_size - 0.8)
        oy = rng.uniform(0.8, map_size - 0.8)
        # Don't place near start or goal
        if math.hypot(ox - 1.2, oy - 1.2) < 1.5 or math.hypot(ox - 8.8, oy - 8.8) < 1.5:
            continue
        obstacles.append((ox, oy, rng.uniform(0.12, 0.25)))

    start = (1.2, 1.2)
    goal = (map_size - 1.2, map_size - 1.2)

    return {
        "map_size": map_size,
        "start": [float(start[0]), float(start[1])],
        "goal": [float(goal[0]), float(goal[1])],
        "obstacles": [{"x": float(x), "y": float(y), "r": float(r)} for x, y, r in obstacles],
        "name": "cluttered_room",
    }


def obstacle_field(map_size: float = 10.0) -> dict:
    """Dense structured obstacle field between start and goal — no clear path."""
    obstacles: list[tuple[float, float, float]] = []
    rng = np.random.RandomState(55)

    # Outer walls
    obstacles += _wall_row(0, 0, map_size, 0, 0.12)
    obstacles += _wall_row(map_size, 0, map_size, map_size, 0.12)
    obstacles += _wall_row(0, 0, 0, map_size, 0.12)
    obstacles += _wall_row(0, map_size, map_size, map_size, 0.12)

    # Grid of obstacles — guaranteed to block the direct path
    start_x, start_y = 1.0, 1.0
    goal_x, goal_y = map_size - 1.0, map_size - 1.0

    for gx in np.arange(2.0, map_size - 1.5, 2.0):
        for gy in np.arange(2.0, map_size - 1.5, 2.0):
            # Offset alternating rows for staggered pattern
            offset = 0.0 if int(gy / 2.0) % 2 == 0 else 1.0
            ox = gx + offset
            oy = gy
            if ox < 0.5 or ox > map_size - 0.5:
                continue
            # Random gap sometimes (like a doorway)
            if rng.rand() < 0.15:
                continue
            obstacles.append((ox, oy, rng.uniform(0.18, 0.35)))

    return {
        "map_size": map_size,
        "start": [start_x, start_y],
        "goal": [goal_x, goal_y],
        "obstacles": [{"x": float(x), "y": float(y), "r": float(r)} for x, y, r in obstacles],
        "name": "obstacle_field",
    }


# ── Registry ──────────────────────────────────────────────────────────────

BUILTIN_MAPS = {
    "corridor_straight": corridor_straight,
    "corridor_l_turn": corridor_l_turn,
    "doorway_challenge": doorway_challenge,
    "cluttered_room": cluttered_room,
    "obstacle_field": obstacle_field,
}


def generate_all(map_size: float = 10.0) -> list[dict]:
    return [builder(map_size) for builder in BUILTIN_MAPS.values()]


def save_maps(maps: list[dict], output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i, m in enumerate(maps):
        name = m.get("name", f"map_{i:03d}")
        path = out / f"{name}.json"
        with open(path, "w") as f:
            json.dump(m, f, indent=2)
        print(f"  {path}  ({len(m['obstacles'])} obstacles)")


def load_map(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate structured test maps")
    parser.add_argument("--output", default="maps/test_suite")
    parser.add_argument("--map-size", type=float, default=10.0)
    args = parser.parse_args()

    maps = generate_all(args.map_size)
    save_maps(maps, args.output)
    print(f"\nGenerated {len(maps)} structured maps in {args.output}/")
    print(f"Maps: {', '.join(BUILTIN_MAPS.keys())}")


if __name__ == "__main__":
    main()
