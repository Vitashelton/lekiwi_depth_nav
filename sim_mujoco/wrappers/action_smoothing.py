"""
Action smoothing wrapper — prevents jerky motion and spin artifacts.

Wraps a Gymnasium env and applies exponential moving average (EMA)
to the action before passing it through.
"""

from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np


class ActionSmoothingWrapper(gym.Wrapper):
    """Applies exponential moving average to actions across steps.

    Reduces jitter and prevents the robot from oscillating / spinning.
    """

    def __init__(
        self,
        env: gym.Env,
        smoothing_alpha: float = 0.4,
        history_size: int = 3,
    ) -> None:
        super().__init__(env)
        self._alpha = smoothing_alpha
        self._history: deque = deque(maxlen=history_size)
        self._smoothed: np.ndarray | None = None

    def step(self, action: np.ndarray) -> tuple:
        action = np.asarray(action, dtype=np.float32)

        if self._smoothed is None:
            self._smoothed = action.copy()
        else:
            self._smoothed = (
                self._alpha * action + (1 - self._alpha) * self._smoothed
            )

        self._history.append(self._smoothed.copy())

        # Also apply median filter over history for extra stability
        if len(self._history) >= 2:
            stacked = np.stack(list(self._history))
            median_action = np.median(stacked, axis=0)
            final_action = 0.7 * self._smoothed + 0.3 * median_action
        else:
            final_action = self._smoothed

        return self.env.step(final_action)

    def reset(self, **kwargs):
        self._smoothed = None
        self._history.clear()
        return self.env.reset(**kwargs)
