"""
Real-time pseudo-LiDAR scan viewer via ZeroMQ subscription.

Subscribes to scan JSON messages published by the Raspberry Pi stack
and renders three live views: linear scan, polar scan, and min-range history.

Works without hardware: supply --mock to generate synthetic scans locally.

Usage:
    # Connect to a real Pi stack (configured in config/network.yaml)
    python tools/live_scan_viewer.py --config config/

    # Connect directly to a known endpoint
    python tools/live_scan_viewer.py --endpoint tcp://192.168.1.100:5555

    # Mock mode (no hardware / no ZMQ)
    python tools/live_scan_viewer.py --mock
"""

from __future__ import annotations

import argparse
import json
import sys
import struct
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ── Mock scan generator (for --mock mode) ──────────────────────────────────

class MockScanSource:
    """Produces synthetic scan messages that mimic the Pi publisher."""

    def __init__(self, num_bins: int = 64, max_range: float = 5.0) -> None:
        self.num_bins = num_bins
        self.max_range = max_range
        self._seq: int = 0
        self._t0: float = time.time()
        self._rng = np.random.RandomState(42)

    def recv(self, timeout_ms: int = 100) -> Optional[dict]:
        """Return a single mock scan dict (blocking), or None on timeout."""
        time.sleep(0.02)
        now = time.time() - self._t0
        # Generate a plausible scan: clear ahead, closer on sides
        half = self.num_bins // 2
        angles = np.linspace(-np.pi / 4, np.pi / 4, self.num_bins)
        # Base: distant values
        scan = np.full(self.num_bins, self.max_range, dtype=np.float32)
        # Nearer values at the edges (simulate walls)
        for i in range(self.num_bins):
            # Central bins are clear; edge bins see "walls"
            edge_factor = abs(i - half) / half  # 0 at center, 1 at edges
            noisy = self.max_range * (0.3 + 0.5 * edge_factor + 0.05 * np.sin(now * 3.0 + i * 0.1))
            scan[i] = float(np.clip(noisy + self._rng.normal(0, 0.05), 0.1, self.max_range))

        self._seq += 1
        return {
            "scan": (scan / self.max_range).tolist(),
            "scan_m": scan.tolist(),
            "timestamp": time.time(),
            "seq": self._seq,
        }


# ── ZMQ scan source ───────────────────────────────────────────────────────

class ZmqScanSource:
    """Subscribes to scan messages from a ZeroMQ publisher."""

    def __init__(self, endpoint: str, topic: str = "scan") -> None:
        import zmq
        self._ctx = zmq.Context()
        self._sub: zmq.Socket = self._ctx.socket(zmq.SUB)
        self._sub.connect(endpoint)
        self._sub.setsockopt(zmq.SUBSCRIBE, topic.encode())
        self._sub.setsockopt(zmq.RCVHWM, 5)
        self._topic = topic
        self._poller = zmq.Poller()
        self._poller.register(self._sub, zmq.POLLIN)

    def recv(self, timeout_ms: int = 100) -> Optional[dict]:
        socks = dict(self._poller.poll(timeout_ms))
        if self._sub in socks and socks[self._sub] == zmq.POLLIN:
            parts = self._sub.recv_multipart(zmq.NOBLOCK)
            if len(parts) >= 2:
                return json.loads(parts[1].decode())
        return None

    def close(self) -> None:
        self._sub.close()
        self._ctx.term()


# ── Live Viewer ───────────────────────────────────────────────────────────

class LiveScanViewer:
    """Real-time matplotlib dashboard for pseudo-LiDAR scans."""

    def __init__(
        self,
        num_bins: int = 64,
        fov_deg: float = 90.0,
        max_range: float = 5.0,
        history_sec: float = 30.0,
        fps_target: float = 20.0,
    ) -> None:
        if not HAS_MPL:
            raise RuntimeError("matplotlib is required. Install with: pip install matplotlib")

        self.num_bins = num_bins
        self.max_range = max_range
        self.fps_target = fps_target
        self._fov_half_rad = np.deg2rad(fov_deg / 2.0)
        self._angles_rad = np.linspace(-self._fov_half_rad, self._fov_half_rad, num_bins)
        self._angles_deg = np.rad2deg(self._angles_rad)

        # History ring buffer for min-distance-over-time plot
        buf_len = int(history_sec * fps_target)
        self._time_history: deque[float] = deque(maxlen=buf_len)
        self._min_dist_history: deque[float] = deque(maxlen=buf_len)

        # Stats
        self._last_seq: int = -1
        self._msg_count: int = 0
        self._t_start: float = time.time()
        self._t_last: float = self._t_start
        self._fps_smooth: float = 0.0

        # Build figure
        self.fig = plt.figure("LeKiwi Live Scan Viewer", figsize=(14, 8))
        gs = self.fig.add_gridspec(2, 3, height_ratios=[2, 1], hspace=0.35, wspace=0.35)

        # (a) Linear scan
        self.ax_linear: plt.Axes = self.fig.add_subplot(gs[0, 0])
        self.ax_linear.set_title("Linear Scan", fontweight="bold")
        self.ax_linear.set_xlabel("Angle (deg)")
        self.ax_linear.set_ylabel("Range (m)")
        self.ax_linear.set_ylim(0, max_range * 1.1)
        self.ax_linear.grid(True, alpha=0.3)
        self.ax_linear.axhline(y=max_range, color="gray", ls="--", alpha=0.4)
        (self._linear_line,) = self.ax_linear.plot(
            self._angles_deg, np.full(num_bins, max_range), "b-", linewidth=1.5
        )
        self._linear_fill = self.ax_linear.fill_between(
            self._angles_deg, 0, np.full(num_bins, max_range), alpha=0.2, color="blue"
        )

        # (b) Polar scan
        self.ax_polar: plt.Axes = self.fig.add_subplot(gs[0, 1], projection="polar")
        self.ax_polar.set_title("Polar Scan", fontweight="bold")
        self.ax_polar.set_theta_zero_location("N")
        self.ax_polar.set_theta_direction(-1)
        self.ax_polar.set_thetamin(-45)
        self.ax_polar.set_thetamax(45)
        self.ax_polar.set_ylim(0, max_range * 1.1)
        (self._polar_line,) = self.ax_polar.plot(
            self._angles_rad, np.full(num_bins, max_range), "b-", linewidth=1.5
        )
        self._polar_fill = self.ax_polar.fill_between(
            self._angles_rad, 0, np.full(num_bins, max_range), alpha=0.2, color="blue"
        )

        # (c) Stats panel (text only)
        self.ax_stats: plt.Axes = self.fig.add_subplot(gs[0, 2])
        self.ax_stats.axis("off")
        self.ax_stats.set_title("Telemetry", fontweight="bold")
        self._stats_text = self.ax_stats.text(
            0.05, 0.95, "", transform=self.ax_stats.transAxes,
            fontsize=12, fontfamily="monospace", verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5", edgecolor="#BDBDBD"),
        )

        # (d) Min distance over time
        self.ax_history: plt.Axes = self.fig.add_subplot(gs[1, :])
        self.ax_history.set_title("Min Range Over Time", fontweight="bold")
        self.ax_history.set_xlabel("Time (s)")
        self.ax_history.set_ylabel("Min Range (m)")
        self.ax_history.set_ylim(0, max_range * 1.1)
        self.ax_history.grid(True, alpha=0.3)
        self.ax_history.axhline(y=0.3, color="orange", ls="--", alpha=0.5, label="safe dist")
        self.ax_history.axhline(y=0.15, color="red", ls="--", alpha=0.5, label="danger")
        self.ax_history.legend(loc="upper right", fontsize=7)
        (self._history_line,) = self.ax_history.plot([], [], "r-", linewidth=1.2)

        plt.ion()
        self.fig.show()

    # ------------------------------------------------------------------
    def update(self, msg: dict) -> None:
        """Ingest one scan message and refresh the display.

        Args:
            msg: Dict with keys scan (list), scan_m (list), timestamp (float), seq (int).
        """
        scan_m = np.asarray(msg["scan_m"], dtype=np.float32)
        ts = msg.get("timestamp", time.time())
        seq = msg.get("seq", 0)

        now = time.time()

        # --- FPS smooth ---
        if self._msg_count > 0:
            alpha = 0.05
            dt = now - self._t_last if (now - self._t_last) > 0 else 0.05
            self._fps_smooth = (1 - alpha) * self._fps_smooth + alpha * (1.0 / dt)
        self._t_last = now
        self._msg_count += 1

        # Drop detection
        dropped = ""
        if self._last_seq >= 0 and seq > self._last_seq + 1:
            dropped = f"  DROPPED: {seq - self._last_seq - 1}"
        self._last_seq = seq

        # --- (a) Linear scan ---
        self._linear_line.set_ydata(scan_m)
        # Update fill_between — remove old collection, create new
        for coll in self._linear_fill.collections:
            coll.remove()
        self._linear_fill = self.ax_linear.fill_between(
            self._angles_deg, 0, scan_m, alpha=0.2, color="blue"
        )

        # --- (b) Polar scan ---
        self._polar_line.set_ydata(scan_m)
        for coll in self._polar_fill.collections:
            coll.remove()
        self._polar_fill = self.ax_polar.fill_between(
            self._angles_rad, 0, scan_m, alpha=0.2, color="blue"
        )

        # --- (c) Stats ---
        elapsed = now - self._t_start
        min_range = float(np.min(scan_m))
        network_delay = now - ts if ts else 0
        self._stats_text.set_text(
            f"FPS:       {self._fps_smooth:5.1f}\n"
            f"Seq:       {seq:5d}{dropped}\n"
            f"Elapsed:   {elapsed:6.1f} s\n"
            f"Min range: {min_range:5.2f} m\n"
            f"Net delay: {network_delay * 1000:5.1f} ms\n"
            f"Messages:  {self._msg_count:5d}"
        )

        # --- (d) History ---
        self._time_history.append(elapsed)
        self._min_dist_history.append(min_range)
        self._history_line.set_data(
            list(self._time_history), list(self._min_dist_history)
        )
        if self._time_history:
            self.ax_history.set_xlim(
                max(0, self._time_history[-1] - 30), max(30, self._time_history[-1] + 2)
            )

        self.fig.canvas.flush_events()
        plt.pause(0.001)

    # ------------------------------------------------------------------
    def close(self) -> None:
        plt.ioff()
        plt.close(self.fig)


# ═══════════════════════════════════════════════════════════════════════════

def load_endpoint_from_config(config_dir: str) -> tuple[str, str]:
    """Read network.yaml and return (endpoint, topic)."""
    try:
        import yaml
    except ImportError:
        print("pyyaml is required. Install with: pip install pyyaml")
        sys.exit(1)

    cfg_path = Path(config_dir) / "network.yaml"
    if not cfg_path.exists():
        print(f"Config file not found: {cfg_path}")
        sys.exit(1)

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)["network"]

    transport = cfg.get("transport", "tcp")
    pi_ip = cfg.get("pi_known_ip", "127.0.0.1")
    port = cfg["pi"]["scan_pub_port"]
    endpoint = f"{transport}://{pi_ip}:{port}"
    topic = cfg.get("scan_topic", "scan")
    return endpoint, topic


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live pseudo-LiDAR scan viewer via ZeroMQ"
    )
    parser.add_argument("--config", default=None,
                        help="Path to config directory (reads network.yaml).")
    parser.add_argument("--endpoint", default=None,
                        help="ZMQ endpoint directly (e.g. tcp://192.168.1.100:5555).")
    parser.add_argument("--topic", default="scan",
                        help="ZMQ topic to subscribe to.")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock scan generator instead of ZMQ.")
    parser.add_argument("--scan-bins", type=int, default=64,
                        help="Number of scan bins for the mock generator.")
    parser.add_argument("--max-range", type=float, default=5.0,
                        help="Maximum scan range in meters.")
    parser.add_argument("--history-sec", type=float, default=30.0,
                        help="Seconds of history to show in min-range plot.")
    args = parser.parse_args()

    if not HAS_MPL:
        print("matplotlib is required. Install with: pip install matplotlib")
        sys.exit(1)

    # Determine source
    if args.mock:
        source = MockScanSource(num_bins=args.scan_bins, max_range=args.max_range)
        print(f"[mock] Using synthetic scan source, {args.scan_bins} bins")
    elif args.endpoint:
        source = ZmqScanSource(args.endpoint, args.topic)
        print(f"[zmq] Subscribed to {args.endpoint}  topic={args.topic}")
    elif args.config:
        endpoint, topic = load_endpoint_from_config(args.config)
        source = ZmqScanSource(endpoint, topic)
        print(f"[zmq] Using config: {endpoint}  topic={topic}")
    else:
        print("No source specified. Use --config, --endpoint, or --mock.")
        print("Falling back to --mock ...")
        source = MockScanSource(num_bins=args.scan_bins, max_range=args.max_range)

    viewer = LiveScanViewer(
        num_bins=args.scan_bins,
        max_range=args.max_range,
        history_sec=args.history_sec,
    )

    print("Live viewer running. Close the figure window to stop.\n")
    try:
        while plt.fignum_exists(viewer.fig.number):
            msg = source.recv(timeout_ms=100)
            if msg is not None:
                # Handle truncated message
                if "scan_m" not in msg and "scan" in msg:
                    scan_norm = np.asarray(msg["scan"])
                    msg["scan_m"] = (scan_norm * args.max_range).tolist()
                viewer.update(msg)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if hasattr(source, "close"):
            source.close()
        viewer.close()


if __name__ == "__main__":
    main()
