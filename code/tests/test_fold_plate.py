"""Synthetic plate detection and fallback checks."""

import cv2
import numpy as np
import pytest

from maus_export.webserve_fold.fold_plate import (
    PlateEllipse,
    detect_plate,
    ellipse_mask,
    median_plate,
)


def test_fallback_to_circle_and_shrink(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cv2,
        "HoughCircles",
        lambda *args, **kwargs: np.array([[[100, 100, 80]]], np.float32),
    )
    frame = np.full((200, 200, 3), 200, np.uint8)
    plate = detect_plate(frame)
    assert plate is not None
    assert (plate.x, plate.y) == (100, 100)
    assert plate.width == pytest.approx(150.4)
    assert plate.height == pytest.approx(150.4)
    mask = ellipse_mask(frame.shape[:2], plate)
    assert mask[100, 100] == 255
    assert mask[0, 0] == 0


def test_detect_dark_ellipse(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cv2,
        "HoughCircles",
        lambda *args, **kwargs: np.array([[[100, 100, 80]]], np.float32),
    )
    frame = np.full((200, 200, 3), 200, np.uint8)
    cv2.ellipse(frame, (102, 99), (65, 75), 10, 0, 360, (20, 20, 20), -1)
    plate = detect_plate(frame)
    assert plate is not None
    assert plate.x == pytest.approx(102, abs=2)
    assert plate.y == pytest.approx(99, abs=2)
    assert plate.width == pytest.approx(130 * 0.94, abs=4)
    assert plate.height == pytest.approx(150 * 0.94, abs=4)


def test_missing_circle_and_periodic_angle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cv2, "HoughCircles", lambda *args, **kwargs: None)
    assert detect_plate(np.zeros((200, 200, 3), np.uint8)) is None
    plate = median_plate(
        [
            PlateEllipse(0, 0, 4, 8, 179),
            PlateEllipse(1, 1, 5, 9, 1),
            PlateEllipse(2, 2, 6, 10, 0),
        ]
    )
    assert plate == PlateEllipse(1, 1, 5, 9, 0)
    with pytest.raises(ValueError):
        median_plate([])
