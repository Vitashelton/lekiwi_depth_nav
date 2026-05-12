"""
Launch the PC-side policy server.

Usage:
    python run_policy_server.py --config config/

    The policy server receives pseudo-LiDAR scans from the Raspberry Pi,
    runs the configured policy (rule / DWA / MLP), and sends velocity
    commands back.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys

from policy_server import PolicyServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main():
    parser = argparse.ArgumentParser(description="PC policy server")
    parser.add_argument("--config", default="config", help="Path to config directory")
    parser.add_argument("--freq", type=float, default=20.0, help="Policy loop frequency (Hz)")
    args = parser.parse_args()

    server = PolicyServer(args.config)

    def _sig_handler(signum, frame):
        print("\nShutting down...")
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    try:
        server.run(freq_hz=args.freq)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
