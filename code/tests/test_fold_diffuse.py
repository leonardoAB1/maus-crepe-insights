"""Synthetic diffuse food versus connected, warm and moving reflections."""

import cv2
import numpy as np
import pytest

from maus_export.fold_diffuse import diffuse_mask


def lab(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)


def test_connected_highlight_cannot_be_kept_by_food_component() -> None:
    frame = np.full((200, 200, 3), 25, np.uint8)
    cv2.circle(frame, (80, 100), 40, (120, 185, 220), -1)
    cv2.rectangle(frame, (110, 90), (180, 110), (255, 255, 255), -1)
    mask = diffuse_mask(lab(frame), np.ones((200, 200), bool))
    assert mask[100, 80] == 255
    assert mask[100, 150] == 0


def test_static_warm_rim_and_exposure_shift_are_not_food() -> None:
    reference = np.full((200, 200, 3), 25, np.uint8)
    cv2.rectangle(reference, (10, 60), (60, 150), (90, 140, 170), -1)
    frame = cv2.add(reference, np.full_like(reference, 10))
    assert not diffuse_mask(
        lab(frame), np.ones((200, 200), bool), lab(reference)
    ).any()


def test_colour_change_recovers_food_over_a_known_glare_patch() -> None:
    reference = np.full((200, 200, 3), 25, np.uint8)
    cv2.circle(reference, (100, 100), 40, (245, 245, 245), -1)
    frame = reference.copy()
    cv2.circle(frame, (100, 100), 40, (120, 185, 220), -1)
    ref = lab(reference)
    mask = diffuse_mask(
        lab(frame), np.ones((200, 200), bool), ref, ref[:, :, 0]
    )
    assert mask[100, 100] == 255


def test_new_neutral_flash_is_rejected_without_a_reference() -> None:
    frame = np.full((100, 100, 3), 25, np.uint8)
    cv2.circle(frame, (50, 50), 35, (255, 255, 255), -1)
    assert not diffuse_mask(lab(frame), np.ones((100, 100), bool)).any()


def test_unmeasurable_exposure_is_unknown_not_empty() -> None:
    frame = np.full((100, 100, 3), 255, np.uint8)
    with pytest.raises(ValueError, match="exposure"):
        diffuse_mask(lab(frame), np.ones((100, 100), bool), lab(frame))
