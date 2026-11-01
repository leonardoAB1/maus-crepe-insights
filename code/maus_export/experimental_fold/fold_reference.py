"""Causal full-crepe area reference learned from vision, never log labels."""

from collections import deque
from statistics import median

REFERENCE_WINDOW = 9
MIN_REFERENCE_SAMPLES = 3


class VisualFlatReference:
    """Use the median of recent visually flat frames, after classification."""

    def __init__(self) -> None:
        self.area: float | None = None
        self.samples: deque[float] = deque(maxlen=REFERENCE_WINDOW)

    def observe(self, prediction: str | None, area: float | None) -> None:
        """Update for subsequent frames; no expected state or verdict input."""
        if prediction != "flat" or area is None or area <= 0:
            self.samples.clear()
            return
        self.samples.append(area)
        if len(self.samples) >= MIN_REFERENCE_SAMPLES:
            candidate = float(median(self.samples))
            self.area = max(candidate, self.area or candidate)
