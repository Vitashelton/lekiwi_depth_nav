"""
Launch the PC-side policy server.

Usage:
    # Raw LeRobot policy (default)
    python run_policy_server.py --config config/ --mode lerobot_raw

    # Rule-based safety shield
    python run_policy_server.py --config config/ --mode rule_shield

    # Learned residual correction
    python run_policy_server.py --config config/ --mode residual_correction --residual-model models/residual_correction.pt
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


def main() -> None:
    parser = argparse.ArgumentParser(description="PC policy server")
    parser.add_argument("--config", default="config", help="Path to config directory")
    parser.add_argument("--freq", type=float, default=20.0, help="Policy loop frequency (Hz)")
    parser.add_argument(
        "--mode", default="lerobot_raw",
        choices=["lerobot_raw", "rule_shield", "residual_correction"],
        help="Operation mode.",
    )
    parser.add_argument(
        "--residual-model", default=None,
        help="Path to ResidualCorrectionNet checkpoint (required for residual_correction mode).",
    )
    args = parser.parse_args()

    server = PolicyServer(
        args.config,
        mode=args.mode,
        residual_model_path=args.residual_model,
    )

    def _sig_handler(signum, frame) -> None:
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
