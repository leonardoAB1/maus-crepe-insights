"""Synthetic geometric measurements of crepe masks."""

import cv2
import numpy as np
import pytest

from maus_export.experimental_fold.fold_geometry import (
    measure_geometry,
    prepare_geometry,
)
from maus_export.experimental_fold.fold_plate import PlateEllipse


def test_cached_geometry_matches_uncached() -> None:
    mask = np.zeros((240, 240), np.uint8)
    cv2.rectangle(mask, (80, 80), (160, 160), 255, -1)
    plate = PlateEllipse(120, 120, 220, 220, 0)
    context = prepare_geometry(mask.shape, plate)
    assert measure_geometry(mask, plate, context=context) == measure_geometry(
        mask, plate
    )
    with pytest.raises(ValueError, match="context"):
        measure_geometry(mask[:100], plate, context=context)
    with pytest.raises(ValueError, match="context"):
        measure_geometry(
            mask, PlateEllipse(100, 100, 220, 220, 0), context=context
        )


def test_circle_centre_diameter_and_holes() -> None:
    mask = np.zeros((240, 240), np.uint8)
    cv2.circle(mask, (120, 120), 60, 255, -1)
    cv2.circle(mask, (120, 120), 10, 0, -1)
    geometry = measure_geometry(mask, PlateEllipse(120, 120, 220, 220, 0))
    assert geometry.centre == pytest.approx((120, 120))
    assert geometry.offset_fraction == pytest.approx(0)
    assert geometry.diameter == pytest.approx(120, abs=1)
    assert geometry.hole_fraction == pytest.approx(1 / 36, abs=0.002)
    assert geometry.solidity > 0.98
    assert geometry.circularity > 0.85


def test_rectangle_fold_edges_and_corner() -> None:
    mask = np.zeros((240, 240), np.uint8)
    cv2.rectangle(mask, (80, 80), (160, 160), 255, -1)
    geometry = measure_geometry(mask, PlateEllipse(120, 120, 220, 220, 0))
    assert len(geometry.edges) == 4
    assert geometry.corner is not None
    assert geometry.corner_angle == pytest.approx(90)
    assert all(edge.angle in (0, 90) for edge in geometry.edges)


def test_empty_geometry() -> None:
    geometry = measure_geometry(
        np.zeros((200, 200), np.uint8), PlateEllipse(100, 100, 150, 150, 0)
    )
    assert geometry.area == 0
    assert geometry.centre is None
    assert geometry.edges == ()
    assert geometry.corner is None
    assert geometry.rect_aspect is None
    assert geometry.rect_angle is None


def test_minimum_rect_aspect_and_angle() -> None:
    plate = PlateEllipse(120, 120, 220, 220, 0)
    square = np.zeros((240, 240), np.uint8)
    cv2.rectangle(square, (90, 90), (150, 150), 255, -1)
    geometry = measure_geometry(square, plate)
    assert geometry.rect_aspect == pytest.approx(1, abs=0.05)
    assert geometry.rect_angle in (0, 90)
    bar = np.zeros((240, 240), np.uint8)
    cv2.rectangle(bar, (70, 115), (170, 125), 255, -1)
    geometry = measure_geometry(bar, plate)
    assert geometry.rect_aspect == pytest.approx(10, rel=0.1)
    assert geometry.rect_angle == pytest.approx(0, abs=1)
