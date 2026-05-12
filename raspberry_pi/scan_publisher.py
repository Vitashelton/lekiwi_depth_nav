"""
ZMQ publisher: send pseudo-LiDAR scan from Raspberry Pi to PC.
"""

from __future__ import annotations

import time
import json
import struct
import numpy as np

import zmq


class ScanPublisher:
    """Publish scan data via ZeroMQ PUB socket."""

    def __init__(self, address: str, topic: str = "scan"):
        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.PUB)
        self._socket.setsockopt(zmq.SNDHWM, 1)
        self._socket.setsockopt(zmq.LINGER, 100)
        self._socket.bind(address)
        self._topic = topic.encode() if isinstance(topic, str) else topic
        self._address = address
        self._seq: int = 0

    def publish(self, scan_norm: np.ndarray, scan_m: np.ndarray) -> None:
        """
        Publish scan data as JSON message with binary payload.

        Message format (multi-part ZMQ):
          Part 0: topic bytes
          Part 1: JSON metadata
          Part 2: (optional) binary float32 array for efficiency
        """
        ts = time.time()
        msg = {
            "scan": scan_norm.tolist(),
            "scan_m": scan_m.tolist(),
            "timestamp": ts,
            "seq": self._seq,
        }
        self._seq += 1
        self._socket.send_multipart(
            [self._topic, json.dumps(msg).encode()]
        )

    def close(self) -> None:
        self._socket.close()
        self._ctx.term()

    @property
    def address(self) -> str:
        return self._address
