"""Crêpe mask extraction with optional empty-plate references."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .fold_plate import PlateEllipse, ellipse_mask

KERNEL_SIZE = 7
MIN_BRIGHT_FLOOR = 90
DARK_REFERENCE_FLOOR = 80
CHANGE_L_FLOOR = 25
MIN_CHANGED_SHARE = 0.50
GLARE_MEAN_L = 180
GLARE_MEAN_B = 134
SMALL_COMPONENT_SHARE = 0.05
MIN_PLATE_AREA_SHARE = 0.03
MAX_PIXEL = 255


@dataclass(frozen=True)
class MaskResult:
    """Crepe mask and component summary."""

    mask: np.ndarray
    area_px: int
    plate_area_px: int
    component_count: int
    has_reference: bool

    @property
    def area_plate_share(self) -> float:
        """Mask area divided by plate area."""
        if self.plate_area_px <= 0:
            return 0.0
        return self.area_px / self.plate_area_px


def exposure_shift(frame: np.ndarray, reference: np.ndarray) -> float:
    """Median Lab-L shift over dark pixels in both images."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)
    dark = (lab[..., 0] < DARK_REFERENCE_FLOOR) & (
        ref_lab[..., 0] < DARK_REFERENCE_FLOOR
    )
    if not np.any(dark):
        return 0.0
    delta = lab[..., 0].astype(np.float32) - ref_lab[..., 0].astype(np.float32)
    return float(np.median(delta[dark]))


def changed_pixels(frame: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Pixels whose exposure-corrected lightness changed."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)
    shift = exposure_shift(frame, reference)
    delta = lab[..., 0].astype(np.float32) - ref_lab[..., 0].astype(np.float32)
    return (delta - shift > CHANGE_L_FLOOR).astype(np.uint8) * MAX_PIXEL


def _bright_candidates(frame: np.ndarray, plate: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pixels = gray[plate > 0]
    if len(pixels) == 0:
        return np.zeros(gray.shape, dtype=np.uint8)
    threshold, _ = cv2.threshold(
        pixels, 0, MAX_PIXEL, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    floor = max(float(threshold), MIN_BRIGHT_FLOOR)
    bright = np.where((gray > floor) & (plate > 0), MAX_PIXEL, 0).astype(
        np.uint8
    )
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (KERNEL_SIZE, KERNEL_SIZE),
    )
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)


def _component_keep_mask(
    candidates: np.ndarray,
    frame: np.ndarray,
    reference: np.ndarray | None,
) -> tuple[np.ndarray, int]:
    components, labels, stats, _ = cv2.connectedComponentsWithStats(candidates)
    if components <= 1:
        return np.zeros(candidates.shape, dtype=np.uint8), 0
    largest = int(stats[1:, cv2.CC_STAT_AREA].max())
    changed = (
        changed_pixels(frame, reference) if reference is not None else None
    )
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    kept = np.zeros(candidates.shape, dtype=np.uint8)
    count = 0
    for label in range(1, components):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < SMALL_COMPONENT_SHARE * largest:
            continue
        pixels = labels == label
        if changed is not None:
            changed_share = float(np.mean(changed[pixels] > 0))
            if changed_share < MIN_CHANGED_SHARE:
                continue
        else:
            mean_l = float(np.mean(lab[..., 0][pixels]))
            mean_b = float(np.mean(lab[..., 2][pixels]))
            if mean_l < GLARE_MEAN_L and mean_b < GLARE_MEAN_B:
                continue
        kept[pixels] = MAX_PIXEL
        count += 1
    return kept, count


def crepe_mask(
    frame: np.ndarray,
    plate_ellipse: PlateEllipse,
    reference: np.ndarray | None = None,
) -> MaskResult:
    """Segment the crepe inside the plate ellipse."""
    plate = ellipse_mask(frame.shape[:2], plate_ellipse)
    candidates = _bright_candidates(frame, plate)
    mask, count = _component_keep_mask(candidates, frame, reference)
    mask = cv2.bitwise_and(mask, plate)
    area = int(np.count_nonzero(mask))
    plate_area = int(np.count_nonzero(plate))
    if plate_area <= 0 or area < MIN_PLATE_AREA_SHARE * plate_area:
        mask[:] = 0
        area = 0
        count = 0
    return MaskResult(mask, area, plate_area, count, reference is not None)
