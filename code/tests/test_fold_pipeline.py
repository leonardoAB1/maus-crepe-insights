"""Synthetic integration of the image primitives and state rule."""

import cv2
import numpy as np
import pytest

from maus_export.fold_geometry import measure_geometry
from maus_export.fold_mask import segment_crepe
from maus_export.fold_plate import detect_plate
from maus_export.fold_rectify import rectify_video_frame
from maus_export.fold_state import classify_state


def test_synthetic_quarter_state(monkeypatch: pytest.MonkeyPatch) -> None:
    calibration = {
        "schema_version": 1,
        "robot_corrections": {
            "robot": {
                "source_size_px": [200, 200],
                "target_size_px": [200, 200],
                "matrix_mean": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            }
        },
    }
    monkeypatch.setattr(
        cv2,
        "HoughCircles",
        lambda *args, **kwargs: np.array([[[100, 100, 90]]], np.float32),
    )
    empty = np.full((200, 200, 3), 25, np.uint8)
    frame = empty.copy()
    cv2.rectangle(frame, (65, 65), (135, 125), (230, 230, 230), -1)
    frame, _ = rectify_video_frame(frame, "robot", calibration)
    plate = detect_plate(empty)
    assert plate is not None
    mask = segment_crepe(frame, plate, empty)
    geometry = measure_geometry(mask, plate)
    plate_area = np.count_nonzero(
        cv2.ellipse(
            np.zeros((200, 200), np.uint8),
            ((plate.x, plate.y), (plate.width, plate.height), plate.angle),
            255,
            -1,
        )
    )
    assert geometry.centre == pytest.approx((100, 95), abs=1)
    assert classify_state(
        geometry.area, plate_area, len(geometry.edges), 18000
    ) == ("quarter")
