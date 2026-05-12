"""
Generate random obstacle maps for navigation testing.

Usage:
    python sim/generate_random_maps.py --num-maps 10 --output maps/
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def generate_map(
    map_size: float = 10.0,
    num_obstacles: int = 8,
    min_r: float = 0.2,
    max_r: float = 0.6,
    num_targets: int = 5,
    seed: int = 0,
) -> dict:
    """Generate a random obstacle map with target positions."""
    rng = np.random.RandomState(seed)
    margin = 0.5

    obstacles = []
    for _ in range(num_obstacles):
        for _ in range(50):
            x = rng.uniform(margin, map_size - margin)
            y = rng.uniform(margin, map_size - margin)
            r = rng.uniform(min_r, max_r)
            ok = True
            for ox, oy, orad in obstacles:
                if math.hypot(x - ox, y - oy) < r + orad + 0.3:
                    ok = False
                    break
            if ok:
                obstacles.append((x, y, r))
                break

    # Start position
    start_x, start_y = rng.uniform(margin, map_size - margin, 2)
    # Ensure start not in obstacle
    for _ in range(100):
        ok = True
        for ox, oy, orad in obstacles:
            if math.hypot(start_x - ox, start_y - oy) < orad + 0.5:
                ok = False
                break
        if ok:
            break
        start_x, start_y = rng.uniform(margin, map_size - margin, 2)

    # Targets
    targets = []
    for _ in range(num_targets):
        for _ in range(100):
            tx = rng.uniform(margin, map_size - margin)
            ty = rng.uniform(margin, map_size - margin)
            ok = True
            for ox, oy, orad in obstacles:
                if math.hypot(tx - ox, ty - oy) < orad + 0.4:
                    ok = False
                    break
            if math.hypot(tx - start_x, ty - start_y) < 1.0:
                ok = False
            if ok:
                targets.append((tx, ty))
                break

    return {
        "map_size": map_size,
        "start": [float(start_x), float(start_y)],
        "targets": [[float(tx), float(ty)] for tx, ty in targets],
        "obstacles": [
            {"x": float(ox), "y": float(oy), "r": float(or_)}
            for ox, oy, or_ in obstacles
        ],
        "seed": seed,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate random navigation maps")
    parser.add_argument("--num-maps", type=int, default=10)
    parser.add_argument("--output", default="maps")
    parser.add_argument("--map-size", type=float, default=10.0)
    parser.add_argument("--obstacles", type=int, default=8)
    parser.add_argument("--targets", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.num_maps):
        seed = args.seed + i
        map_data = generate_map(
            map_size=args.map_size,
            num_obstacles=args.obstacles,
            num_targets=args.targets,
            seed=seed,
        )
        fpath = output_dir / f"map_{i:03d}.json"
        with open(fpath, "w") as f:
            json.dump(map_data, f, indent=2)
        print(f"  Saved {fpath}")

    print(f"Generated {args.num_maps} maps in {output_dir}/")


if __name__ == "__main__":
    main()
