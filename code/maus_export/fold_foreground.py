"""Conservative live foreground adapter with explicit occlusion abstention."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

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
MAX_ARM_WIDTH_FRACTION = 0.50
REFERENCE_FRAMES = 15
MIN_REFERENCE_FRAMES = 3
REFERENCE_UPDATE_INTERVAL = 5
REFERENCE_SIZE = 225
HIGHLIGHT_PERCENTILE = 90
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
        self.reference_count = 0
        self.used_samples: list[np.ndarray] = []
        self.used_sample_count = 0
        self.reference_image: np.ndarray | None = None
        self.reference_upper_l: np.ndarray | None = None
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
            narrow = min(width, height) < (
                min(self.plate.width, self.plate.height)
                * MAX_ARM_WIDTH_FRACTION
            )
            if not touches_image or not narrow:
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
        if learn_empty:
            fraction = np.count_nonzero(warm & inside) / inside.sum()
            if fraction < MAX_EMPTY_FRACTION:
                self.references.append(
                    cv2.resize(frame, (REFERENCE_SIZE, REFERENCE_SIZE))
                )
                self.reference_count += 1
            elif not self.reference_ready:
                self.references.clear()
            if (
                fraction < MAX_EMPTY_FRACTION
                and len(self.references) >= MIN_REFERENCE_FRAMES
                and (
                    not self.reference_ready
                    or self.reference_count % REFERENCE_UPDATE_INTERVAL == 0
                )
            ):
                self.used_samples = list(self.references)
                self.used_sample_count = self.reference_count
                size = (frame.shape[1], frame.shape[0])
                self.reference_image = cv2.resize(
                    np.median(self.references, axis=0).astype(np.uint8), size
                )
                lights = [
                    cv2.cvtColor(sample, cv2.COLOR_BGR2LAB)[:, :, 0]
                    for sample in self.references
                ]
                self.reference_upper_l = cv2.resize(
                    np.percentile(lights, HIGHLIGHT_PERCENTILE, axis=0).astype(
                        np.uint8
                    ),
                    size,
                )
                self.segmenter = CrepeSegmenter(
                    frame.shape[:2],
                    self.plate,
                    self.reference_image,
                    reference_upper_l=self.reference_upper_l,
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

    def save_reference(self, directory: Path) -> None:
        """Write a private reproducible reference set, not a public fixture."""
        directory.mkdir(parents=True, exist_ok=True)
        images = {
            f"sample-{index:02d}.png": sample
            for index, sample in enumerate(self.used_samples)
        }
        if self.reference_image is not None:
            images["median.png"] = self.reference_image
        if self.reference_upper_l is not None:
            images["highlight-l.png"] = self.reference_upper_l
        for name, image in images.items():
            if not cv2.imwrite(str(directory / name), image):
                raise OSError("Cannot write private background reference")
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "selection": "pre-dispense low-warmth unoccluded frames",
                    "observed_samples": self.reference_count,
                    "used_sample_count": self.used_sample_count,
                    "retained_samples": len(self.used_samples),
                    "ready": self.reference_ready,
                    "highlight_percentile": HIGHLIGHT_PERCENTILE,
                    "sample_size_px": REFERENCE_SIZE,
                },
                indent=2,
            )
        )
