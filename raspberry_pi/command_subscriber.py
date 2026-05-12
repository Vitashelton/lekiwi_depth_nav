"""
ZMQ subscriber: receive velocity commands from PC on Raspberry Pi.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Optional

import zmq


@dataclass
class VelocityCommand:
    """Velocity command for omnidirectional robot."""
    vx: float       # m/s (forward)
    vy: float       # m/s (lateral right)
    omega: float    # rad/s (counterclockwise positive)
    timestamp: float


class CommandSubscriber:
    """Subscribe to velocity commands from PC via ZeroMQ SUB socket."""

    def __init__(self, address: str, topic: str = "cmd_vel"):
        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.SUB)
        self._socket.setsockopt(zmq.RCVHWM, 1)
        self._socket.setsockopt(zmq.LINGER, 100)
        self._socket.connect(address)
        self._socket.setsockopt(zmq.SUBSCRIBE, topic.encode())
        self._address = address
        self._topic = topic
        self._last_cmd: Optional[VelocityCommand] = None
        self._last_recv_time: float = 0.0

    def recv(self, timeout_ms: int = 10) -> Optional[VelocityCommand]:
        """
        Non-blocking receive with timeout. Returns None if no message.
        """
        try:
            if self._socket.poll(timeout=timeout_ms) == 0:
                return None
            parts = self._socket.recv_multipart(zmq.NOBLOCK)
            if len(parts) >= 2:
                msg = json.loads(parts[1].decode())
                cmd = VelocityCommand(
                    vx=msg["vx"],
                    vy=msg["vy"],
                    omega=msg["omega"],
                    timestamp=msg.get("timestamp", time.time()),
                )
                self._last_cmd = cmd
                self._last_recv_time = time.time()
                return cmd
        except zmq.ZMQError:
            pass
        return None

    def get_last_cmd(self) -> Optional[VelocityCommand]:
        return self._last_cmd

    def get_time_since_last_cmd(self) -> float:
        return time.time() - self._last_recv_time

    def close(self) -> None:
        self._socket.close()
        self._ctx.term()
