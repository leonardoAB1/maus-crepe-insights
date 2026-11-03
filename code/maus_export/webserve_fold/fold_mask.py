"""Crepe segmentation with optional empty-plate exposure matching."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .fold_plate import KERNEL_SIZE, PlateEllipse, ellipse_mask

GREY_FLOOR = 90
DARK_L_LIMIT = 80
CHANGE_THRESHOLD = 25
MIN_CHANGED_FRACTION = 0.50
GLARE_L_LIMIT = 180
GLARE_B_LIMIT = 134
MIN_COMPONENT_FRACTION = 0.05
MIN_PLATE_FRACTION = 0.03
MAX_RUN_DISTANCE = 3


@dataclass(frozen=True)
class EmptyReference:
    """A candidate empty frame; identifier distinguishes frames within a run."""

    frame_id: str
    run_id: int
    robot: str
    width: int
    check_ok: bool


def select_reference(
    candidates: list[EmptyReference],
    frame_id: str,
    run_id: int,
    robot: str,
    width: int,
) -> EmptyReference | None:
    """Prefer a passed same-run check, then nearby compatible run IDs."""
    compatible = [
        c
        for c in candidates
        if c.frame_id != frame_id
        and c.robot == robot
        and c.width == width
        and abs(c.run_id - run_id) <= MAX_RUN_DISTANCE
    ]
    if not compatible:
        return None
    return min(
        compatible,
        key=lambda c: (
            c.run_id != run_id,
            not c.check_ok if c.run_id == run_id else False,
            abs(c.run_id - run_id),
            not c.check_ok,
            c.run_id,
            c.frame_id,
        ),
    )


class CrepeSegmenter:
    """Cache fixed plate and reference data for successive camera frames."""

    def __init__(
        self,
        shape: tuple[int, int],
        plate: PlateEllipse,
        reference: np.ndarray | None = None,
    ) -> None:
        self.shape = shape
        self.inside = ellipse_mask(shape, plate) != 0
        self.plate_area = int(np.count_nonzero(self.inside))
        if not self.plate_area:
            raise ValueError("Plate mask is empty")
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (KERNEL_SIZE, KERNEL_SIZE)
        )
        self.ref_l = None
        if reference is not None:
            if reference.shape != (*shape, 3):
                raise ValueError("Reference dimensions must match the frame")
            self.ref_l = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)[:, :, 0]

    def segment(self, frame: np.ndarray) -> np.ndarray:
        """Return a mask for a frame in the cached camera coordinates."""
        if frame.shape != (*self.shape, 3):
            raise ValueError("Frame dimensions must match the plate")
        return _segment(frame, self)


def segment_crepe(
    frame: np.ndarray, plate: PlateEllipse, reference: np.ndarray | None = None
) -> np.ndarray:
    """Return a 0/255 crepe mask using bright, changed components.

    Reference and frame must already share dimensions and camera coordinates.
    If no dark pixels can support exposure matching, fail rather than assume
    an unmeasured exposure shift.
    """
    return CrepeSegmenter(frame.shape[:2], plate, reference).segment(frame)


def _segment(frame: np.ndarray, context: CrepeSegmenter) -> np.ndarray:
    inside = context.inside
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    threshold, _ = cv2.threshold(
        grey[inside].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    bright = ((grey > max(threshold, GREY_FLOOR)) & inside).astype(np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, context.kernel)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, context.kernel)
    bright &= inside.astype(np.uint8)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    changed = None
    if context.ref_l is not None:
        ref_l = context.ref_l
        light = lab[:, :, 0]
        dark = inside & (light < DARK_L_LIMIT) & (ref_l < DARK_L_LIMIT)
        if not dark.any():
            raise ValueError("No dark pixels for exposure matching")
        difference = light.astype(np.int16) - ref_l.astype(np.int16)
        shift = float(np.median(difference[dark]))
        changed = difference - shift > CHANGE_THRESHOLD
    count, labels, stats, _ = cv2.connectedComponentsWithStats(bright, 8)
    keep: list[tuple[int, int]] = []
    for label in range(1, count):
        pixels = labels == label
        if changed is not None:
            if float(changed[pixels].mean()) < MIN_CHANGED_FRACTION:
                continue
        elif (
            float(lab[:, :, 0][pixels].mean()) < GLARE_L_LIMIT
            and float(lab[:, :, 2][pixels].mean()) < GLARE_B_LIMIT
        ):
            continue
        keep.append((label, int(stats[label, cv2.CC_STAT_AREA])))
    result = np.zeros(frame.shape[:2], dtype=np.uint8)
    if keep:
        largest = max(area for _, area in keep)
        for label, area in keep:
            if area >= MIN_COMPONENT_FRACTION * largest:
                result[labels == label] = 255
    if np.count_nonzero(result) < MIN_PLATE_FRACTION * context.plate_area:
        result.fill(0)
    return result
