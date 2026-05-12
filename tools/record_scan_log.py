"""
Record pseudo-LiDAR scan data from a running Pi stack to a log file.

Can save in NPZ (numpy archive) or CSV format for later analysis.

Usage:
    # Record from live ZMQ stream (PC connected to Pi)
    python tools/record_scan_log.py --config config/ --output logs/scan_log_001.npz --duration 60

    # Record from simulation
    python tools/record_scan_log.py --config config/ --sim --output logs/sim_scans.npz --duration 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def record_from_zmq(config_dir: str, output: str, duration: float) -> None:
    """Record scan data from live ZMQ stream."""
    import zmq
    import yaml

    with open(Path(config_dir) / "network.yaml") as f:
        net_cfg = yaml.safe_load(f)["network"]

    transport = net_cfg.get("transport", "tcp")
    pi_ip = net_cfg.get("pi_known_ip", "127.0.0.1")
    scan_addr = f"{transport}://{pi_ip}:{net_cfg['pi']['scan_pub_port']}"

    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect(scan_addr)
    sub.setsockopt(zmq.SUBSCRIBE, net_cfg.get("scan_topic", "scan").encode())
    sub.setsockopt(zmq.RCVHWM, 10)

    print(f"Recording from {scan_addr} for {duration}s...")
    scans_norm = []
    scans_m = []
    timestamps = []

    t0 = time.time()
    while time.time() - t0 < duration:
        try:
            if sub.poll(timeout=100) != 0:
                parts = sub.recv_multipart(zmq.NOBLOCK)
                msg = json.loads(parts[1].decode())
                scans_norm.append(msg["scan"])
                scans_m.append(msg["scan_m"])
                timestamps.append(msg["timestamp"])
        except zmq.ZMQError:
            pass

    sub.close()
    ctx.term()

    _save(output, scans_norm, scans_m, timestamps)


def record_from_sim(output: str, duration: float, scan_bins: int = 64) -> None:
    """Record scan data from simulation environment."""
    from sim.simple_2d_env import Simple2DNavEnv

    env = Simple2DNavEnv(num_scan_bins=scan_bins)
    obs, _ = env.reset()

    scans_norm = []
    scans_m = []
    timestamps = []

    t0 = time.time()
    step = 0
    while time.time() - t0 < duration:
        # Random action for exploration
        action = np.array([0.1, 0.0, np.random.uniform(-30, 30)], dtype=np.float32)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

        scans_norm.append(obs[:scan_bins].tolist())
        scans_m.append((np.array(obs[:scan_bins]) * 5.0).tolist())
        timestamps.append(time.time())
        step += 1

    _save(output, scans_norm, scans_m, timestamps)


def _save(output: str, scans_norm, scans_m, timestamps) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.suffix == ".npz":
        np.savez(
            str(output),
            scans_norm=np.array(scans_norm, dtype=np.float32),
            scans_m=np.array(scans_m, dtype=np.float32),
            timestamps=np.array(timestamps, dtype=np.float64),
        )
    elif output.suffix == ".csv":
        # Save as CSV (only metric scans for simplicity)
        data = np.array(scans_m)
        header = ",".join([f"bin_{i}" for i in range(data.shape[1])]) + ",timestamp"
        np.savetxt(output, np.column_stack([data, timestamps]), delimiter=",", header=header, comments="")
    else:
        raise ValueError(f"Unsupported format: {output.suffix} (use .npz or .csv)")

    print(f"Saved {len(timestamps)} scans to {output}")


def main():
    parser = argparse.ArgumentParser(description="Record scan logs")
    parser.add_argument("--config", default="config")
    parser.add_argument("--output", default="logs/scan_log.npz")
    parser.add_argument("--duration", type=float, default=30.0, help="Recording duration (s)")
    parser.add_argument("--sim", action="store_true", help="Record from simulation")
    parser.add_argument("--scan-bins", type=int, default=64)
    args = parser.parse_args()

    if args.sim:
        record_from_sim(args.output, args.duration, args.scan_bins)
    else:
        record_from_zmq(args.config, args.output, args.duration)


if __name__ == "__main__":
    main()
