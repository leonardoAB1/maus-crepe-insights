"""Measure a segmented crepe and locate straight fold edges."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .fold_plate import PlateEllipse, ellipse_mask

POLYGON_EPSILON = 0.015
MIN_EDGE_LENGTH = 0.18
MAX_EDGE_RMS = 0.025
MIN_RIM_DISTANCE = 0.03
STRAIGHT_ANGLE = 180.0
DISTANCE_MASK_SIZE = 5
PARALLEL_DETERMINANT_EPSILON = 1e-6


@dataclass(frozen=True)
class FoldEdge:
    """A straight contour chord, with image-space orientation in degrees."""

    start: tuple[float, float]
    end: tuple[float, float]
    length: float
    angle: float


@dataclass(frozen=True)
class CrepeGeometry:
    """Filled-mask metrics and detected straight edges."""

    area: float
    plate_fraction: float
    centre: tuple[float, float] | None
    diameter: float
    diameter_fraction: float
    offset_fraction: float | None
    hole_fraction: float
    solidity: float | None
    circularity: float | None
    edges: tuple[FoldEdge, ...]
    corner: tuple[float, float] | None
    corner_angle: float | None
    rect_aspect: float | None
    rect_angle: float | None


@dataclass(frozen=True)
class GeometryContext:
    """Cached plate raster and rim distance for a fixed camera view."""

    plate: PlateEllipse
    plate_mask: np.ndarray
    rim_distance: np.ndarray
    plate_area: float


def prepare_geometry(
    shape: tuple[int, int], plate: PlateEllipse
) -> GeometryContext:
    """Precompute plate data for repeated geometry measurements."""
    plate_mask = ellipse_mask(shape, plate)
    plate_area = float(np.count_nonzero(plate_mask))
    if plate_area == 0:
        raise ValueError("Plate mask is empty")
    distance = cv2.distanceTransform(
        plate_mask, cv2.DIST_L2, DISTANCE_MASK_SIZE
    )
    return GeometryContext(plate, plate_mask, distance, plate_area)


def _edges(
    contour: np.ndarray,
    plate_mask: np.ndarray,
    plate_diameter: float,
    rim_distance: np.ndarray | None = None,
) -> tuple[FoldEdge, ...]:
    perimeter = cv2.arcLength(contour, True)
    polygon = cv2.approxPolyDP(contour, POLYGON_EPSILON * perimeter, True)
    points = contour[:, 0, :]
    vertices = polygon[:, 0, :]
    distance = (
        rim_distance
        if rim_distance is not None
        else cv2.distanceTransform(plate_mask, cv2.DIST_L2, DISTANCE_MASK_SIZE)
    )
    indices = [
        int(np.flatnonzero(np.all(points == p, axis=1))[0]) for p in vertices
    ]
    edges = []
    for i, index in enumerate(indices):
        end_index = indices[(i + 1) % len(indices)]
        if end_index <= index:
            segment = np.concatenate((points[index:], points[: end_index + 1]))
        else:
            segment = points[index : end_index + 1]
        start = vertices[i].astype(float)
        end = vertices[(i + 1) % len(vertices)].astype(float)
        chord = end - start
        length = float(np.linalg.norm(chord))
        if length < MIN_EDGE_LENGTH * plate_diameter:
            continue
        delta = segment.astype(float) - start
        errors = (delta[:, 0] * chord[1] - delta[:, 1] * chord[0]) / length
        rms = float(np.sqrt(np.mean(errors**2)))
        rim = float(np.median(distance[segment[:, 1], segment[:, 0]]))
        if (
            rms <= MAX_EDGE_RMS * length
            and rim >= MIN_RIM_DISTANCE * plate_diameter
        ):
            edges.append(
                FoldEdge(
                    tuple(start),
                    tuple(end),
                    length,
                    math.degrees(math.atan2(chord[1], chord[0]))
                    % STRAIGHT_ANGLE,
                )
            )
    return tuple(sorted(edges, key=lambda edge: -edge.length))


def _corner(
    edges: tuple[FoldEdge, ...],
) -> tuple[tuple[float, float] | None, float | None]:
    if len(edges) < 2:
        return None, None
    first, second = edges[:2]
    p, q = np.array(first.start), np.array(second.start)
    a = np.array(first.end) - p
    b = np.array(second.end) - q
    matrix = np.column_stack((a, -b))
    if abs(np.linalg.det(matrix)) < PARALLEL_DETERMINANT_EPSILON:
        return None, None
    corner = p + np.linalg.solve(matrix, q - p)[0] * a
    vectors = []
    for edge in (first, second):
        endpoints = [np.array(edge.start), np.array(edge.end)]
        endpoint = max(
            endpoints, key=lambda point: np.linalg.norm(point - corner)
        )
        vector = endpoint - corner
        if np.linalg.norm(vector) == 0:
            return None, None
        vectors.append(vector / np.linalg.norm(vector))
    angle = math.degrees(math.acos(float(np.clip(np.dot(*vectors), -1, 1))))
    return (float(corner[0]), float(corner[1])), angle


def measure_geometry(
    mask: np.ndarray,
    plate: PlateEllipse,
    *,
    context: GeometryContext | None = None,
) -> CrepeGeometry:
    """Measure filled components, holes, shape, position and fold edges."""
    if context is not None and (
        context.plate != plate or context.plate_mask.shape != mask.shape
    ):
        raise ValueError("Geometry context does not match mask and plate")
    plate_mask = (
        context.plate_mask if context else ellipse_mask(mask.shape, plate)
    )
    plate_area = (
        context.plate_area if context else float(np.count_nonzero(plate_mask))
    )
    if plate_area == 0:
        raise ValueError("Plate mask is empty")
    binary = ((mask != 0) & (plate_mask != 0)).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    plate_diameter = math.sqrt(plate.width * plate.height)
    if not contours:
        return CrepeGeometry(
            0, 0, None, 0, 0, None, 0, None, None, (), None, None, None, None
        )
    filled = np.zeros_like(binary)
    cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
    filled &= plate_mask
    area = float(np.count_nonzero(filled))
    moments = cv2.moments(filled, binaryImage=True)
    centre = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
    diameter = math.sqrt(4 * area / math.pi)
    offset = (
        math.hypot(centre[0] - plate.x, centre[1] - plate.y) / plate_diameter
    )
    hole_fraction = (area - np.count_nonzero(binary)) / area
    all_points = np.concatenate(contours)
    hull_area = cv2.contourArea(cv2.convexHull(all_points))
    contour_area = sum(cv2.contourArea(contour) for contour in contours)
    perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
    solidity = contour_area / hull_area if hull_area else None
    circularity = (
        4 * math.pi * contour_area / perimeter**2 if perimeter else None
    )
    edges = _edges(
        max(contours, key=cv2.contourArea),
        plate_mask,
        plate_diameter,
        context.rim_distance if context else None,
    )
    corner, corner_angle = _corner(edges)
    rect_aspect, rect_angle = _minimum_rect(filled)
    return CrepeGeometry(
        area,
        area / plate_area,
        centre,
        diameter,
        diameter / plate_diameter,
        offset,
        hole_fraction,
        solidity,
        circularity,
        edges,
        corner,
        corner_angle,
        rect_aspect,
        rect_angle,
    )


def _minimum_rect(filled: np.ndarray) -> tuple[float | None, float | None]:
    """Aspect ratio and long-axis angle of the mask minimum rectangle.

    A half fold is a half disc (about 2:1) while a quarter fold and a
    full crepe are close to square. The angle is the long axis in image
    degrees from 0 to 180, so it tracks fold rotation on rectified
    frames even when no straight edge is detected.
    """
    contours, _ = cv2.findContours(
        filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None, None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 1.0:
        return None, None
    (_, _), (width, height), _ = cv2.minAreaRect(largest)
    short = min(width, height)
    long = max(width, height)
    if short <= 0:
        return None, None
    box = cv2.boxPoints(cv2.minAreaRect(largest))
    best = 0.0
    best_length = -1.0
    for i in range(4):
        delta = box[(i + 1) % 4] - box[i]
        length = float(np.linalg.norm(delta))
        if length > best_length:
            best_length = length
            best = math.degrees(math.atan2(delta[1], delta[0])) % STRAIGHT_ANGLE
    return long / short, best
