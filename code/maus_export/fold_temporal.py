"""Causal majority masks without future frames or stale occlusion history."""

from collections import deque

import numpy as np

MASK_WINDOW = 5
MIN_MASK_SAMPLES = 3
MASK_VALUE = 255


class MaskWarming(ValueError):
    """There are not yet enough post-occlusion masks for a causal vote."""


class TemporalMask:
    """Average binary foreground masks over the latest five valid frames."""

    def __init__(self) -> None:
        self.history: deque[np.ndarray] = deque(maxlen=MASK_WINDOW)

    def reset(self) -> None:
        """Discard stale foreground after an occlusion or failed measurement."""
        self.history.clear()

    def update(self, mask: np.ndarray) -> np.ndarray | None:
        """Return a strict-majority mask, or unknown during warm-up.

        At 10 fps the five-frame window spans 0.4 seconds and introduces
        about 0.2 seconds of transition delay. No timestamps are shifted.
        """
        if self.history and mask.shape != self.history[0].shape:
            self.reset()
        self.history.append((mask != 0).astype(np.uint8))
        if len(self.history) < MIN_MASK_SAMPLES:
            return None
        votes = np.sum(self.history, axis=0)
        return (votes * 2 > len(self.history)).astype(np.uint8) * MASK_VALUE
