"""Conservative live foreground adapter with explicit occlusion abstention."""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from .fold_mask import CrepeSegmenter
from .fold_plate import PlateEllipse

MIN_DOUGH_B = 136
MAX_DOUGH_A = 145
MIN_LIGHTNESS = 90
MIN_SEED_FRACTION = 0.35
BOUNDARY_MARGIN = 0.08
MIN_EXTERNAL_PIXELS = 30
REFERENCE_FRAMES = 5
MIN_REFERENCE_FRAMES = 3
MAX_EMPTY_FRACTION = 0.01


class ForegroundOccluded(ValueError):
    """The image cannot support a reliable food-area measurement."""


class LiveForeground:
    """Keep warm dough components; abstain on bright rim-crossing objects.

    Neutral metal and highlights are not food seeds. Components may include
    pale dough when sufficient warm pixels support them. White dough and
    coloured tools remain known limitations, not solved recognition problems.
    The adapter can be replaced without changing geometry or check voting.
    """

    def __init__(self, shape: tuple[int, int], plate: PlateEllipse) -> None:
        self.segmenter = CrepeSegmenter(shape, plate)
        self.plate = plate
        self.references: deque[np.ndarray] = deque(maxlen=REFERENCE_FRAMES)
        self.reference_ready = False
        self.rim = max(plate.width, plate.height) * BOUNDARY_MARGIN
        self.inner = (
            cv2.distanceTransform(
                self.segmenter.inside.astype(np.uint8), cv2.DIST_L2, 3
            )
            > self.rim
        )

    def segment(
        self, frame: np.ndarray, *, learn_empty: bool = False
    ) -> np.ndarray:
        """Measure only visible foreground and learn pre-dispense references."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        bright = (lab[:, :, 0] > MIN_LIGHTNESS).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(bright, 8)
        inside = self.segmenter.inside
        for label in range(1, count):
            x, y, width, height, _ = stats[label]
            touches_image = (
                x == 0
                or y == 0
                or x + width == frame.shape[1]
                or y + height == frame.shape[0]
            )
            if not touches_image:
                continue
            pixels = labels == label
            if np.count_nonzero(
                pixels & ~inside
            ) >= MIN_EXTERNAL_PIXELS and np.any(pixels & self.inner):
                raise ForegroundOccluded("Bright object enters from image edge")
        warm = (
            (lab[:, :, 2] >= MIN_DOUGH_B)
            & (lab[:, :, 1] <= MAX_DOUGH_A)
            & (lab[:, :, 0] > MIN_LIGHTNESS)
        )
        if learn_empty and not self.reference_ready:
            fraction = np.count_nonzero(warm & inside) / inside.sum()
            if fraction < MAX_EMPTY_FRACTION:
                self.references.append(frame.copy())
            else:
                self.references.clear()
            if len(self.references) >= MIN_REFERENCE_FRAMES:
                reference = np.median(self.references, axis=0).astype(np.uint8)
                self.segmenter = CrepeSegmenter(
                    frame.shape[:2], self.plate, reference
                )
                self.reference_ready = True
        mask = self.segmenter.segment(frame)
        count, labels = cv2.connectedComponents(mask)
        result = np.zeros_like(mask)
        for label in range(1, count):
            pixels = labels == label
            if warm[pixels].mean() >= MIN_SEED_FRACTION:
                result[pixels] = mask[pixels]
        return result
