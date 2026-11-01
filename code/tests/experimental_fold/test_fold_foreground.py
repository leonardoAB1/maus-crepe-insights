"""Synthetic warm food, neutral reflections and boundary arm tests."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from maus_export.experimental_fold.fold_foreground import (
    ForegroundOccluded,
    LiveForeground,
)
from maus_export.experimental_fold.fold_plate import PlateEllipse


def detector() -> LiveForeground:
    return LiveForeground((200, 200), PlateEllipse(100, 100, 180, 180, 0))


def empty() -> np.ndarray:
    return np.full((200, 200, 3), 25, np.uint8)


def test_bright_neutral_reflection_is_not_dough() -> None:
    frame = empty()
    cv2.circle(frame, (100, 100), 40, (250, 250, 250), -1)
    assert not detector().segment(frame).any()


def test_warm_dough_survives_and_static_glare_does_not() -> None:
    foreground = detector()
    reference = empty()
    cv2.circle(reference, (155, 100), 12, (230, 230, 230), -1)
    for _ in range(3):
        foreground.segment(reference, learn_empty=True)
    frame = reference.copy()
    cv2.circle(frame, (85, 100), 35, (120, 185, 220), -1)
    mask = foreground.segment(frame)
    assert foreground.reference_ready
    assert mask[100, 85] == 255
    assert mask[100, 155] == 0


@pytest.mark.parametrize("color", [(220, 220, 220), (120, 185, 220)])
def test_arm_entering_plate_is_unknown_not_food(color: tuple) -> None:
    frame = empty()
    cv2.circle(frame, (100, 100), 35, (120, 185, 220), -1)
    cv2.rectangle(frame, (85, 0), (115, 105), color, -1)
    with pytest.raises(ForegroundOccluded):
        detector().segment(frame)


def test_broad_bright_background_is_not_an_arm() -> None:
    frame = np.full((200, 200, 3), 220, np.uint8)
    cv2.circle(frame, (100, 100), 75, (25, 25, 25), -1)
    assert not detector().segment(frame).any()


def test_private_reference_dataset_tracks_used_samples(tmp_path: Path) -> None:
    foreground = detector()
    for value in range(20, 37):
        foreground.segment(
            np.full((200, 200, 3), value, np.uint8), learn_empty=True
        )
    foreground.save_reference(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["observed_samples"] == 17
    assert manifest["used_sample_count"] == 15
    assert manifest["retained_samples"] == 15
    assert len(list(tmp_path.glob("sample-*.png"))) == 15
    assert (tmp_path / "median.png").is_file()
    assert (tmp_path / "highlight-l.png").is_file()
    before = foreground.reference_count
    foreground.segment(empty())
    assert foreground.reference_count == before


def test_no_reference_learning_after_dispensing_or_from_dough() -> None:
    foreground = detector()
    for _ in range(5):
        foreground.segment(empty())
    assert not foreground.reference_ready
    frame = empty()
    cv2.circle(frame, (100, 100), 35, (120, 185, 220), -1)
    for _ in range(5):
        foreground.segment(frame, learn_empty=True)
    assert not foreground.reference_ready
