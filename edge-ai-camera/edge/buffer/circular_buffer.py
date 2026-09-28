"""Bounded in-memory frame history for pre-motion event footage."""

from collections import deque
from typing import Deque

import numpy as np


class CircularFrameBuffer:
    def __init__(self, max_frames: int) -> None:
        if max_frames < 1:
            raise ValueError("max_frames must be at least 1")
        self._frames: Deque[np.ndarray] = deque(maxlen=max_frames)

    def append(self, frame: np.ndarray) -> None:
        # Copy so callers can safely reuse or modify their capture buffer.
        self._frames.append(frame.copy())

    def snapshot(self) -> list[np.ndarray]:
        return [frame.copy() for frame in self._frames]

    def __len__(self) -> int:
        return len(self._frames)

