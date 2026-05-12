"""
Latency benchmark: Measure each component's processing time.

Tests:
  1. Depth-to-Scan processing time (Raspberry Pi)
  2. ZMQ round-trip latency (Pi → PC → Pi)
  3. Policy inference time (PC)

Usage:
    python tools/latency_test.py --config config/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from raspberry_pi.camera_node import MockCamera, CameraConfig
from raspberry_pi.depth_to_scan import DepthToScan, ScanConfig
from pc.rule_policy import RulePolicy
from pc.dwa_policy import DWAPlanner, DWAConfig
from pc.mlp_policy import MLPPolicy


def test_depth_to_scan(num_runs: int = 1000) -> dict:
    """Test Depth-to-Scan processing latency."""
    cam = MockCamera(CameraConfig())
    d2s = DepthToScan(ScanConfig())

    depth = cam.get_depth_frame()
    times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        d2s(depth)
        times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    return {
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
        "p99_ms": np.percentile(times, 99),
    }


def test_policy_inference(policy_type: str, num_runs: int = 500) -> dict:
    """Test policy inference latency."""
    scan_m = np.random.uniform(0.15, 5.0, 64).astype(np.float32)
    scan_m[30:35] = 0.3  # simulate obstacle
    goal = 0.0

    if policy_type == "rule":
        policy = RulePolicy()
    elif policy_type == "dwa":
        policy = DWAPlanner(DWAConfig())
    elif policy_type == "mlp":
        policy = MLPPolicy(obs_dim=67, act_dim=3)
        policy.eval()
        # Build obs for MLP
        scan_norm = np.clip(scan_m / 5.0, 0, 1)
        obs = np.concatenate([scan_norm, [0.0, 0.0, 0.0]]).astype(np.float32)
        times = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            policy.predict(obs)
            times.append((time.perf_counter() - t0) * 1000)
        times = np.array(times)
        return {
            "mean_ms": np.mean(times),
            "std_ms": np.std(times),
            "min_ms": np.min(times),
            "max_ms": np.max(times),
            "p99_ms": np.percentile(times, 99),
        }
    else:
        raise ValueError(f"Unknown policy: {policy_type}")

    times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        policy(scan_m, goal)
        times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    return {
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
        "p99_ms": np.percentile(times, 99),
    }


def test_zmq_roundtrip(num_runs: int = 200) -> dict:
    """Test ZMQ pub-sub round-trip latency (simulated local)."""
    import zmq

    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.bind("tcp://127.0.0.1:15555")
    sub = ctx.socket(zmq.SUB)
    sub.connect("tcp://127.0.0.1:15555")
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    time.sleep(0.2)  # warm-up

    times = []
    for i in range(num_runs):
        t0 = time.perf_counter()
        pub.send_multipart([b"test", json.dumps({"seq": i, "ts": t0}).encode()])
        sub.recv_multipart()
        times.append((time.perf_counter() - t0) * 1000)

    pub.close()
    sub.close()
    ctx.term()

    times = np.array(times)
    return {
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
        "p99_ms": np.percentile(times, 99),
    }


def main():
    parser = argparse.ArgumentParser(description="Latency benchmark")
    parser.add_argument("--config", default="config")
    parser.add_argument("--runs", type=int, default=1000)
    args = parser.parse_args()

    print("=" * 60)
    print("LATENCY BENCHMARK")
    print("=" * 60)

    print("\n[1] Depth-to-Scan Processing (MockCamera, 1000 runs)")
    result = test_depth_to_scan(args.runs)
    print(f"    Mean: {result['mean_ms']:.2f} ms")
    print(f"    Std:  {result['std_ms']:.2f} ms")
    print(f"    P99:  {result['p99_ms']:.2f} ms")

    print("\n[2] Policy Inference (500 runs each)")
    for ptype in ["rule", "dwa", "mlp"]:
        result = test_policy_inference(ptype, 500)
        print(f"    {ptype:6s}: mean={result['mean_ms']:.2f} ms, p99={result['p99_ms']:.2f} ms")

    print("\n[3] ZMQ Round-trip (local, 200 runs)")
    result = test_zmq_roundtrip(200)
    print(f"    Mean: {result['mean_ms']:.2f} ms")
    print(f"    P99:  {result['p99_ms']:.2f} ms")

    print("\n[4] Theoretical Control Loop Budget (20 Hz)")
    budget = 50.0  # ms
    d2s_lat = test_depth_to_scan(100)["mean_ms"]
    zmq_lat = result["mean_ms"]
    rule_lat = test_policy_inference("rule", 100)["mean_ms"]
    mlp_lat = test_policy_inference("mlp", 100)["mean_ms"]
    total_rule = d2s_lat + zmq_lat + rule_lat
    total_mlp = d2s_lat + zmq_lat + mlp_lat
    print(f"    Budget:         {budget:.1f} ms")
    print(f"    Depth-to-Scan:  {d2s_lat:.1f} ms")
    print(f"    ZMQ:            {zmq_lat:.1f} ms")
    print(f"    Rule Policy:    {rule_lat:.1f} ms")
    print(f"    MLP Policy:     {mlp_lat:.1f} ms")
    print(f"    Total (Rule):   {total_rule:.1f} ms {'OK' if total_rule < budget else 'EXCEEDED'}")
    print(f"    Total (MLP):    {total_mlp:.1f} ms {'OK' if total_mlp < budget else 'EXCEEDED'}")


if __name__ == "__main__":
    main()
