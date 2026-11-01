"""Synthetic empty-reference and component-filtering tests."""

import cv2
import numpy as np
import pytest

from maus_export.experimental_fold.fold_mask import (
    CrepeSegmenter,
    EmptyReference,
    segment_crepe,
    select_reference,
)
from maus_export.experimental_fold.fold_plate import PlateEllipse


def test_cached_segmentation_matches_uncached() -> None:
    plate = PlateEllipse(100, 100, 180, 180, 0)
    reference = np.full((200, 200, 3), 25, np.uint8)
    frame = reference.copy()
    cv2.circle(frame, (100, 100), 40, (230, 230, 230), -1)
    for ref in (None, reference):
        cached = CrepeSegmenter(frame.shape[:2], plate, ref)
        assert np.array_equal(
            cached.segment(frame), segment_crepe(frame, plate, ref)
        )
        assert np.array_equal(
            cached.segment(frame), segment_crepe(frame, plate, ref)
        )
        with pytest.raises(ValueError, match="dimensions"):
            cached.segment(frame[:100])


def test_reference_preserves_changed_component_and_removes_glare() -> None:
    plate = PlateEllipse(100, 100, 180, 180, 0)
    reference = np.full((200, 200, 3), 25, np.uint8)
    cv2.circle(reference, (60, 100), 10, (200, 200, 200), -1)
    cv2.circle(reference, (150, 100), 12, (200, 200, 200), -1)
    frame = reference.copy()
    cv2.circle(frame, (70, 100), 32, (230, 230, 230), -1)
    result = segment_crepe(frame, plate, reference)
    assert result[100, 70] == 255
    assert result[100, 60] == 255
    assert result[100, 150] == 0
    assert result[0, 0] == 0


def test_no_reference_rejects_grey_glare_and_tiny_components() -> None:
    plate = PlateEllipse(100, 100, 180, 180, 0)
    frame = np.full((200, 200, 3), 25, np.uint8)
    cv2.circle(frame, (100, 100), 40, (230, 230, 230), -1)
    cv2.circle(frame, (165, 100), 6, (230, 230, 230), -1)
    mask = segment_crepe(frame, plate)
    assert mask[100, 100] == 255
    assert mask[100, 165] == 0
    frame.fill(25)
    cv2.circle(frame, (100, 100), 30, (110, 110, 110), -1)
    assert not segment_crepe(frame, plate).any()


def test_rejects_reference_that_cannot_be_exposure_matched() -> None:
    frame = np.full((100, 100, 3), 255, np.uint8)
    with pytest.raises(ValueError, match="dark pixels"):
        segment_crepe(frame, PlateEllipse(50, 50, 80, 80, 0), frame)
    with pytest.raises(ValueError, match="dimensions"):
        segment_crepe(frame, PlateEllipse(50, 50, 80, 80, 0), frame[:50])


def test_highlight_envelope_rejects_variable_specular_patch() -> None:
    plate = PlateEllipse(100, 100, 180, 180, 0)
    reference = np.full((200, 200, 3), 25, np.uint8)
    frame = reference.copy()
    cv2.circle(frame, (100, 100), 35, (240, 240, 240), -1)
    upper = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[:, :, 0]
    assert CrepeSegmenter((200, 200), plate, reference).segment(frame).any()
    protected = CrepeSegmenter(
        (200, 200), plate, reference, reference_upper_l=upper
    )
    assert not protected.segment(frame).any()
    dough = frame.copy()
    cv2.circle(dough, (55, 100), 45, (120, 185, 220), -1)
    assert protected.segment(dough)[100, 50] == 255
    with pytest.raises(ValueError, match="envelope"):
        CrepeSegmenter((200, 200), plate, reference_upper_l=upper)


def test_select_reference_priorities_and_compatibility() -> None:
    candidates = [
        EmptyReference("self", 10, "robot", 200, True),
        EmptyReference("near", 11, "robot", 200, True),
        EmptyReference("same-fail", 10, "robot", 200, False),
        EmptyReference("same-pass", 10, "robot", 200, True),
        EmptyReference("other", 10, "other", 200, True),
        EmptyReference("size", 10, "robot", 100, True),
        EmptyReference("far", 20, "robot", 200, True),
    ]
    assert select_reference(candidates, "self", 10, "robot", 200).frame_id == (
        "same-pass"
    )
    assert select_reference(
        candidates[:2], "self", 10, "robot", 200
    ).frame_id == ("near")
    assert select_reference(candidates[-1:], "self", 10, "robot", 200) is None
