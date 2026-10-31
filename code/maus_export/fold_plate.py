"""Hotplate ellipse detection for corrected fold-segmentation frames."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

BLUR_SIZE = 7
HOUGH_DP = 1.5
HOUGH_PARAM1 = 100
HOUGH_PARAM2 = 40
MIN_RADIUS_HEIGHT_SHARE = 0.28
MAX_RADIUS_HEIGHT_SHARE = 0.60
GROW_SHARE = 1.05
DARK_VALUE = 70
KERNEL_SIZE = 7
MAX_CENTER_SHIFT_RADIUS = 0.30
MIN_AXIS_DIAMETER_SHARE = 0.60
MAX_AXIS_DIAMETER_SHARE = 1.15
AXIS_SHRINK = 0.94
MIN_ELLIPSE_POINTS = 5


@dataclass(frozen=True)
class PlateEllipse:
    """OpenCV-style ellipse in image coordinates."""

    cx: float
    cy: float
    major: float
    minor: float
    angle: float
    source: str

    @property
    def diameter(self) -> float:
        """Mean fitted ellipse diameter."""
        return float((self.major + self.minor) / 2.0)


def ellipse_mask(shape: tuple[int, int], ellipse: PlateEllipse) -> np.ndarray:
    """Return a uint8 mask for a plate ellipse."""
    mask = np.zeros(shape, dtype=np.uint8)
    axes = (max(1, round(ellipse.major / 2)), max(1, round(ellipse.minor / 2)))
    cv2.ellipse(
        mask,
        (round(ellipse.cx), round(ellipse.cy)),
        axes,
        float(ellipse.angle),
        0,
        360,
        255,
        -1,
    )
    return mask


def _fallback_circle(width: int, height: int) -> tuple[float, float, float]:
    radius = min(width, height) * 0.44
    return width / 2.0, height / 2.0, radius


def _hough_circle(gray: np.ndarray) -> tuple[float, float, float]:
    height, width = gray.shape[:2]
    blurred = cv2.medianBlur(gray, BLUR_SIZE)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP,
        minDist=height / 2,
        param1=HOUGH_PARAM1,
        param2=HOUGH_PARAM2,
        minRadius=round(height * MIN_RADIUS_HEIGHT_SHARE),
        maxRadius=round(height * MAX_RADIUS_HEIGHT_SHARE),
    )
    if circles is None:
        return _fallback_circle(width, height)
    circle = circles[0][np.argmax(circles[0, :, 2])]
    return float(circle[0]), float(circle[1]), float(circle[2])


def detect_plate_ellipse(frame: np.ndarray) -> PlateEllipse:
    """Detect the hotplate as an ellipse, falling back to Hough circle."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cx, cy, radius = _hough_circle(gray)
    diameter = radius * 2.0
    circle = PlateEllipse(
        cx,
        cy,
        diameter * AXIS_SHRINK,
        diameter * AXIS_SHRINK,
        0.0,
        "circle",
    )
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(
        mask,
        (round(cx), round(cy)),
        round(radius * GROW_SHARE),
        255,
        -1,
    )
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dark = np.where((hsv[..., 2] < DARK_VALUE) & (mask > 0), 255, 0).astype(
        np.uint8
    )
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (KERNEL_SIZE, KERNEL_SIZE),
    )
    opened = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)
    components, labels, stats, _ = cv2.connectedComponentsWithStats(opened)
    if components <= 1:
        return circle
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    points = np.column_stack(np.where(labels == largest)).astype(np.int32)
    if len(points) < MIN_ELLIPSE_POINTS:
        return circle
    contour = cv2.convexHull(points[:, ::-1])
    if len(contour) < MIN_ELLIPSE_POINTS:
        return circle
    (ex, ey), (axis_a, axis_b), angle = cv2.fitEllipse(contour)
    shift = float(np.hypot(ex - cx, ey - cy))
    if shift > MAX_CENTER_SHIFT_RADIUS * radius:
        return circle
    major = max(float(axis_a), float(axis_b))
    minor = min(float(axis_a), float(axis_b))
    min_axis = MIN_AXIS_DIAMETER_SHARE * diameter
    max_axis = MAX_AXIS_DIAMETER_SHARE * diameter
    if not (min_axis <= minor <= max_axis and min_axis <= major <= max_axis):
        return circle
    return PlateEllipse(
        ex,
        ey,
        major * AXIS_SHRINK,
        minor * AXIS_SHRINK,
        angle,
        "dark",
    )


def median_ellipse(ellipses: list[PlateEllipse]) -> PlateEllipse:
    """Return the per-coordinate median plate ellipse for one run."""
    if not ellipses:
        raise ValueError("Need at least one ellipse")
    values = np.asarray(
        [[e.cx, e.cy, e.major, e.minor, e.angle] for e in ellipses],
        dtype=np.float64,
    )
    median = np.median(values, axis=0)
    return PlateEllipse(
        float(median[0]),
        float(median[1]),
        float(median[2]),
        float(median[3]),
        float(median[4]),
        "run_median",
    )
