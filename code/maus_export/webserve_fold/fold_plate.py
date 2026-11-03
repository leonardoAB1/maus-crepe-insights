"""Estimate the heating plate in a corrected camera image."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

BLUR_SIZE = 7
HOUGH_DP = 1.5
HOUGH_CANNY = 100
HOUGH_VOTES = 40
MIN_RADIUS = 0.28
MAX_RADIUS = 0.60
CIRCLE_GROWTH = 1.05
DARK_VALUE = 70
KERNEL_SIZE = 7
MAX_CENTRE_SHIFT = 0.30
MIN_AXIS_RATIO = 0.60
MAX_AXIS_RATIO = 1.15
AXIS_SHRINK = 0.94
ANGLE_PERIOD = 180.0


@dataclass(frozen=True)
class PlateEllipse:
    """OpenCV ellipse with full axis lengths and image-space angle."""

    x: float
    y: float
    width: float
    height: float
    angle: float


def ellipse_mask(shape: tuple[int, int], ellipse: PlateEllipse) -> np.ndarray:
    """Rasterize an ellipse as a uint8 mask with values 0 and 255."""
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.ellipse(
        mask,
        (
            (ellipse.x, ellipse.y),
            (ellipse.width, ellipse.height),
            ellipse.angle,
        ),
        255,
        -1,
    )
    return mask


def detect_plate(frame: np.ndarray) -> PlateEllipse | None:
    """Detect a plate ellipse, falling back to the detected Hough circle."""
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grey = cv2.medianBlur(grey, BLUR_SIZE)
    height = frame.shape[0]
    circles = cv2.HoughCircles(
        grey,
        cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP,
        minDist=height,
        param1=HOUGH_CANNY,
        param2=HOUGH_VOTES,
        minRadius=round(height * MIN_RADIUS),
        maxRadius=round(height * MAX_RADIUS),
    )
    if circles is None:
        return None
    x, y, radius = (float(v) for v in circles[0, 0])
    diameter = 2 * radius
    chosen = PlateEllipse(x, y, diameter, diameter, 0.0)
    circle = PlateEllipse(
        x, y, diameter * CIRCLE_GROWTH, diameter * CIRCLE_GROWTH, 0.0
    )
    dark = cv2.inRange(
        cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2], 0, DARK_VALUE - 1
    )
    dark = cv2.bitwise_and(dark, ellipse_mask(frame.shape[:2], circle))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (KERNEL_SIZE, KERNEL_SIZE)
    )
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(
        dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        hull = cv2.convexHull(max(contours, key=cv2.contourArea))
        if len(hull) >= 5:
            (cx, cy), (width, axis_height), angle = cv2.fitEllipse(hull)
            shift = math.hypot(cx - x, cy - y)
            if shift < MAX_CENTRE_SHIFT * radius and all(
                MIN_AXIS_RATIO * diameter <= axis <= MAX_AXIS_RATIO * diameter
                for axis in (width, axis_height)
            ):
                chosen = PlateEllipse(cx, cy, width, axis_height, angle)
    return PlateEllipse(
        chosen.x,
        chosen.y,
        chosen.width * AXIS_SHRINK,
        chosen.height * AXIS_SHRINK,
        chosen.angle,
    )


def median_plate(ellipses: list[PlateEllipse]) -> PlateEllipse:
    """Combine run ellipses, treating angles as periodic over 180 degrees."""
    if not ellipses:
        raise ValueError("At least one plate ellipse is required")
    angles = [ellipse.angle % ANGLE_PERIOD for ellipse in ellipses]

    def angular_cost(angle: float) -> float:
        return sum(
            min(abs(angle - other), ANGLE_PERIOD - abs(angle - other))
            for other in angles
        )

    angle = min(angles, key=angular_cost)
    return PlateEllipse(
        float(np.median([e.x for e in ellipses])),
        float(np.median([e.y for e in ellipses])),
        float(np.median([e.width for e in ellipses])),
        float(np.median([e.height for e in ellipses])),
        angle,
    )
