"""Draw interpretable fold-analysis annotations on private video frames."""

from __future__ import annotations

import math

import cv2
import numpy as np

from .fold_geometry import CrepeGeometry
from .fold_plate import PlateEllipse

MASK_COLOR = (80, 210, 95)
PLATE_COLOR = (235, 200, 70)
EDGE_COLOR = (30, 210, 255)
CENTRE_COLOR = (255, 255, 255)
CORNER_COLOR = (240, 90, 240)
TEXT_COLOR = (245, 245, 245)
PANEL_COLOR = (15, 20, 25)
MASK_OPACITY = 0.35
PANEL_OPACITY = 0.70
LINE_WIDTH = 2
EDGE_WIDTH = 3
MARKER_RADIUS = 6
FONT_SCALE = 0.55
TEXT_X = 15
TEXT_Y = 25
TEXT_STEP = 22
PANEL_WIDTH = 500
PANEL_MARGIN = 12


def _point(point: tuple[float, float]) -> tuple[int, int]:
    return round(point[0]), round(point[1])


def annotate_frame(
    frame: np.ndarray,
    plate: PlateEllipse | None,
    mask: np.ndarray | None,
    geometry: CrepeGeometry | None,
    state: str,
    prediction: str | None,
    status: str,
) -> np.ndarray:
    """Draw the current mask and measurements without modifying the input.

    On failed inference, pass no mask or geometry so stale detections are not
    shown. All positions and dimensions are in corrected-image pixels.
    """
    output = frame.copy()
    if mask is not None:
        if mask.shape != frame.shape[:2]:
            raise ValueError("Mask dimensions must match the frame")
        overlay = np.full_like(frame, MASK_COLOR)
        blended = cv2.addWeighted(
            frame, 1 - MASK_OPACITY, overlay, MASK_OPACITY, 0
        )
        cv2.copyTo(blended, mask, output)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(output, contours, -1, MASK_COLOR, LINE_WIDTH)
    if plate is not None:
        cv2.ellipse(
            output,
            ((plate.x, plate.y), (plate.width, plate.height), plate.angle),
            PLATE_COLOR,
            LINE_WIDTH,
            cv2.LINE_AA,
        )
    if geometry is not None:
        for edge in geometry.edges:
            cv2.line(
                output,
                _point(edge.start),
                _point(edge.end),
                EDGE_COLOR,
                EDGE_WIDTH,
                cv2.LINE_AA,
            )
        if geometry.centre is not None:
            cv2.drawMarker(
                output,
                _point(geometry.centre),
                CENTRE_COLOR,
                cv2.MARKER_CROSS,
                MARKER_RADIUS * 2,
                LINE_WIDTH,
            )
        if geometry.corner is not None:
            x, y = geometry.corner
            height, width = frame.shape[:2]
            if (
                math.isfinite(x)
                and math.isfinite(y)
                and 0 <= x < width
                and 0 <= y < height
            ):
                cv2.circle(
                    output,
                    _point(geometry.corner),
                    MARKER_RADIUS,
                    CORNER_COLOR,
                    LINE_WIDTH,
                    cv2.LINE_AA,
                )
    lines = [f"State: {state} | Raw: {prediction or '-'} | {status}"]
    if geometry is not None and geometry.centre is not None:
        lines.append(
            f"Area: {geometry.plate_fraction:.1%} plate | "
            f"Equivalent diameter: {geometry.diameter:.0f} px"
        )
        lines.append(
            f"Offset: {geometry.offset_fraction:.1%} plate diameter | "
            f"Holes: {geometry.hole_fraction:.1%}"
        )
        if geometry.edges:
            lines.append(
                f"Fold edges: {len(geometry.edges)} | "
                f"Main angle: {geometry.edges[0].angle:.0f} deg"
            )
        if geometry.corner_angle is not None:
            lines.append(f"Corner angle: {geometry.corner_angle:.0f} deg")
    panel_height = min(frame.shape[0], TEXT_Y + len(lines) * TEXT_STEP)
    panel_width = min(frame.shape[1], PANEL_WIDTH)
    panel = output[:panel_height, :panel_width]
    shade = np.full_like(panel, PANEL_COLOR)
    cv2.addWeighted(panel, 1 - PANEL_OPACITY, shade, PANEL_OPACITY, 0, panel)
    for index, text in enumerate(lines):
        y = TEXT_Y + index * TEXT_STEP
        if y >= frame.shape[0] - PANEL_MARGIN:
            break
        cv2.putText(
            output,
            text,
            (TEXT_X, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            TEXT_COLOR,
            LINE_WIDTH,
            cv2.LINE_AA,
        )
    return output
