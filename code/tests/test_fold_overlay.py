"""Synthetic visual overlays and failure handling."""

from dataclasses import replace

import cv2
import numpy as np
import pytest

from maus_export.fold_geometry import measure_geometry
from maus_export.fold_overlay import annotate_frame
from maus_export.fold_plate import PlateEllipse


def test_mask_and_geometry_are_drawn_without_changing_input() -> None:
    image = np.full((600, 600, 3), 50, np.uint8)
    original = image.copy()
    mask = np.zeros((600, 600), np.uint8)
    cv2.rectangle(mask, (250, 300), (450, 450), 255, -1)
    plate = PlateEllipse(300, 300, 550, 550, 0)
    geometry = measure_geometry(mask, plate)
    output = annotate_frame(
        image,
        plate,
        mask,
        geometry,
        "half",
        "ok",
        check="fold1",
        expected="half",
    )
    assert np.array_equal(image, original)
    assert output[350, 300, 1] > original[350, 300, 1]
    assert np.array_equal(output[580, 580], original[580, 580])
    assert len(geometry.edges) > 0
    assert not np.array_equal(output[300, 300], original[300, 300])
    assert not np.array_equal(output[375, 350], original[375, 350])


def test_plate_estimate_is_not_drawn() -> None:
    image = np.full((600, 600, 3), 50, np.uint8)
    plate = PlateEllipse(300, 300, 550, 550, 0)
    with_plate = annotate_frame(image, plate, None, None, None, "ok")
    without_plate = annotate_frame(image, None, None, None, None, "ok")
    assert np.array_equal(with_plate, without_plate)


def test_purple_corner_circle_is_not_drawn() -> None:
    image = np.full((600, 600, 3), 50, np.uint8)
    mask = np.zeros((600, 600), np.uint8)
    cv2.rectangle(mask, (250, 300), (450, 450), 255, -1)
    plate = PlateEllipse(300, 300, 550, 550, 0)
    geometry = measure_geometry(mask, plate)
    with_corner = annotate_frame(
        image, plate, mask, replace(geometry, corner=(550, 550)), None, "ok"
    )
    without_corner = annotate_frame(
        image, plate, mask, replace(geometry, corner=None), None, "ok"
    )
    assert np.array_equal(with_corner, without_corner)


def test_crepe_contour_remains_drawn() -> None:
    image = np.full((600, 600, 3), 50, np.uint8)
    mask = np.zeros((600, 600), np.uint8)
    cv2.rectangle(mask, (250, 300), (450, 450), 255, -1)
    output = annotate_frame(image, None, mask, None, None, "ok")
    assert tuple(output[450, 350]) == (80, 210, 95)
    assert output[350, 350, 1] > image[350, 350, 1]


def test_failed_detection_does_not_draw_stale_mask() -> None:
    image = np.full((300, 600, 3), 50, np.uint8)
    output = annotate_frame(
        image, None, None, None, None, "segmentation_failed"
    )
    assert np.array_equal(output[250, 550], image[250, 550])
    with pytest.raises(ValueError, match="dimensions"):
        annotate_frame(
            image, None, np.zeros((10, 10), np.uint8), None, None, "ok"
        )
