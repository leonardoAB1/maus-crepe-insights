"""Synthetic warm food, neutral reflections and boundary arm tests."""

import cv2
import numpy as np
import pytest

from maus_export.fold_foreground import ForegroundOccluded, LiveForeground
from maus_export.fold_plate import PlateEllipse


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
