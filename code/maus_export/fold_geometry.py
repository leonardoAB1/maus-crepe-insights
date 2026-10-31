"""Geometry features for fold-state masks."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .fold_plate import PlateEllipse, ellipse_mask

APPROX_EPSILON_SHARE = 0.015
MIN_EDGE_PLATE_SHARE = 0.18
MAX_RMS_EDGE_SHARE = 0.025
MIN_RIM_DISTANCE_SHARE = 0.03
DEGREES_HALF_TURN = 180.0
MIN_CONTOUR_AREA = 1.0
CIRCULARITY_SCALE = 4.0 * np.pi
EQUIVALENT_DIAMETER_SCALE = 4.0 / np.pi


@dataclass(frozen=True)
class StraightEdge:
    """One straight fold-like contour edge."""

    x1: float
    y1: float
    x2: float
    y2: float
    angle_deg: float
    length_px: float
    rms_px: float
    median_rim_distance_px: float


@dataclass(frozen=True)
class GeometryFeatures:
    """Feature set used by state rules and metrics."""

    area_px: int
    area_plate_share: float
    center_x: float | None
    center_y: float | None
    offset_plate_share: float | None
    equivalent_diameter_px: float
    diameter_plate_share: float
    solidity: float
    circularity: float
    hole_fraction: float
    straight_edges: tuple[StraightEdge, ...]
    corner_x: float | None
    corner_y: float | None
    corner_angle_deg: float | None


def fill_external(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Fill holes by drawing the largest external contour."""
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        return np.zeros(mask.shape, dtype=np.uint8), None
    contour = max(contours, key=cv2.contourArea)
    filled = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(filled, [contour], -1, 255, -1)
    return filled, contour


def _edge_points(
    contour: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    points = contour[:, 0, :]
    distances = _point_segment_distance(
        points.astype(np.float64),
        first,
        second,
    )
    limit = max(
        2.0,
        float(np.linalg.norm(second - first)) * MAX_RMS_EDGE_SHARE * 3.0,
    )
    return points[distances <= limit]


def _point_segment_distance(
    points: np.ndarray, first: np.ndarray, second: np.ndarray
) -> np.ndarray:
    chord = second - first
    length_sq = float(chord @ chord)
    if length_sq <= 0:
        return np.linalg.norm(points - first, axis=1)
    t = np.clip(((points - first) @ chord) / length_sq, 0.0, 1.0)
    projection = first + t[:, None] * chord
    return np.linalg.norm(points - projection, axis=1)


def _rim_distance(plate_mask: np.ndarray) -> np.ndarray:
    inside = cv2.distanceTransform(plate_mask, cv2.DIST_L2, 5)
    outside = cv2.distanceTransform(255 - plate_mask, cv2.DIST_L2, 5)
    return np.minimum(inside, outside)


def straight_edges(
    contour: np.ndarray, plate: PlateEllipse, shape: tuple[int, int]
) -> tuple[StraightEdge, ...]:
    """Find straight edges that are far enough from the hotplate rim."""
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, APPROX_EPSILON_SHARE * perimeter, True)
    if len(approx) < 2:
        return ()
    plate_diameter = plate.diameter
    rim = _rim_distance(ellipse_mask(shape, plate))
    found: list[StraightEdge] = []
    vertices = approx[:, 0, :].astype(np.float64)
    for index, first in enumerate(vertices):
        second = vertices[(index + 1) % len(vertices)]
        chord = second - first
        length = float(np.linalg.norm(chord))
        if length < MIN_EDGE_PLATE_SHARE * plate_diameter:
            continue
        points = _edge_points(contour, first, second)
        if len(points) < 2:
            continue
        distances = _point_segment_distance(
            points.astype(np.float64),
            first,
            second,
        )
        rms = float(np.sqrt(np.mean(distances**2)))
        if rms > MAX_RMS_EDGE_SHARE * length:
            continue
        samples = rim[
            points[:, 1].clip(0, shape[0] - 1),
            points[:, 0].clip(0, shape[1] - 1),
        ]
        median_rim = float(np.median(samples))
        if median_rim < MIN_RIM_DISTANCE_SHARE * plate_diameter:
            continue
        angle = float(
            np.degrees(np.arctan2(chord[1], chord[0])) % DEGREES_HALF_TURN
        )
        found.append(
            StraightEdge(
                float(first[0]),
                float(first[1]),
                float(second[0]),
                float(second[1]),
                angle,
                length,
                rms,
                median_rim,
            )
        )
    return tuple(sorted(found, key=lambda edge: edge.length_px, reverse=True))


def _intersection(
    first: StraightEdge,
    second: StraightEdge,
) -> tuple[float, float] | None:
    p = np.array([first.x1, first.y1], dtype=np.float64)
    r = np.array(
        [first.x2 - first.x1, first.y2 - first.y1],
        dtype=np.float64,
    )
    q = np.array([second.x1, second.y1], dtype=np.float64)
    s = np.array(
        [second.x2 - second.x1, second.y2 - second.y1],
        dtype=np.float64,
    )
    cross = float(r[0] * s[1] - r[1] * s[0])
    if abs(cross) < np.finfo(np.float64).eps:
        return None
    t = float(((q - p)[0] * s[1] - (q - p)[1] * s[0]) / cross)
    point = p + t * r
    return float(point[0]), float(point[1])


def mask_geometry(mask: np.ndarray, plate: PlateEllipse) -> GeometryFeatures:
    """Compute geometry features for one crepe mask."""
    filled, contour = fill_external(mask)
    plate_area = max(1, int(np.count_nonzero(ellipse_mask(mask.shape, plate))))
    area = int(np.count_nonzero(filled))
    original_area = int(np.count_nonzero(mask))
    holes = max(0, area - original_area) / max(1, area)
    if contour is None or area <= MIN_CONTOUR_AREA:
        return GeometryFeatures(
            0,
            0.0,
            None,
            None,
            None,
            0.0,
            0.0,
            0.0,
            0.0,
            holes,
            (),
            None,
            None,
            None,
        )
    moments = cv2.moments(filled)
    center_x = float(moments["m10"] / moments["m00"])
    center_y = float(moments["m01"] / moments["m00"])
    offset = float(
        np.hypot(center_x - plate.cx, center_y - plate.cy) / plate.diameter
    )
    equivalent = float(np.sqrt(EQUIVALENT_DIAMETER_SCALE * area))
    perimeter = float(cv2.arcLength(contour, True))
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    contour_area = float(cv2.contourArea(contour))
    solidity = contour_area / hull_area if hull_area > 0 else 0.0
    circularity = (
        CIRCULARITY_SCALE * contour_area / (perimeter**2)
        if perimeter > 0
        else 0.0
    )
    edges = straight_edges(contour, plate, mask.shape)
    corner_x = corner_y = corner_angle = None
    if len(edges) >= 2:
        intersection = _intersection(edges[0], edges[1])
        if intersection is not None:
            corner_x, corner_y = intersection
            diff = abs(edges[0].angle_deg - edges[1].angle_deg)
            corner_angle = float(min(diff, DEGREES_HALF_TURN - diff))
    return GeometryFeatures(
        area,
        area / plate_area,
        center_x,
        center_y,
        offset,
        equivalent,
        equivalent / plate.diameter,
        solidity,
        circularity,
        holes,
        edges,
        corner_x,
        corner_y,
        corner_angle,
    )
