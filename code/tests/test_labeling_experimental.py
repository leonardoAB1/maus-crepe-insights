"""Experimental transfer tests use synthetic images and explicit opt-in."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from test_labeling_workflow import load_workflow

from maus_export.labeling_experimental import (
    experimental_alignment,
    register_video_frame,
)

PHOTO_WIDTH = 640
PHOTO_HEIGHT = 480
VIDEO_SCALE = 1.5


def test_similarity_registration_measures_resize_from_texture() -> None:
    rng = np.random.default_rng(42)
    texture = rng.integers(20, 230, (PHOTO_HEIGHT, PHOTO_WIDTH), np.uint8)
    texture = cv2.GaussianBlur(texture, (3, 3), 0)
    photo = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)
    video = cv2.resize(photo, None, fx=VIDEO_SCALE, fy=VIDEO_SCALE)
    result = register_video_frame(video, photo)
    matrix = np.asarray(result["matrix"])
    expected = np.eye(2) / VIDEO_SCALE
    assert np.allclose(matrix[:2, :2], expected, atol=0.001)
    assert result["validation"]["median_error_px"] < 1
    assert result["validation"]["held_out_coverage"] > 0.1


def test_textureless_registration_is_rejected() -> None:
    blank = np.zeros((PHOTO_HEIGHT, PHOTO_WIDTH, 3), np.uint8)
    with pytest.raises(ValueError, match="texture"):
        register_video_frame(blank, blank)


def test_transfer_requires_two_temporal_registrations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = np.full((80, 80, 3), 100, np.uint8)
    reference = tmp_path / "01_spread_overlay.jpg"
    photo = tmp_path / "01_spread_frame.jpg"
    for path in (reference, photo):
        cv2.imwrite(str(path), image)
    transform = {
        "schema_version": 1,
        "source_size": [80, 80],
        "target_size": [80, 80],
        "matrix": np.eye(3).tolist(),
        "validation": {"synthetic": True},
    }
    monkeypatch.setattr(
        "maus_export.labeling_experimental.estimate_alignment_for_robot",
        lambda *args: transform,
    )
    monkeypatch.setattr(
        "maus_export.labeling_experimental.register_video_frame",
        lambda *args: transform,
    )
    with pytest.raises(ValueError, match="consistent"):
        experimental_alignment([(0, photo)], [reference])
    with pytest.raises(ValueError, match="consistent"):
        experimental_alignment([(0, photo), (0, photo)], [reference])
    result = experimental_alignment([(0, photo), (10, photo)], [reference])
    assert result["video_validation_frames"] == [0, 10]
    assert result["kind"] == "experimental_photo_video_transfer"
    assert "visual review" in result["warning"]
    assert len(result["video_to_photo"]) == 2


def test_experimental_contract_requires_explicit_marker(
    tmp_path: Path,
) -> None:
    # This rejects before any geometry or artifact validation is attempted.
    manifest = {
        "preprocessing": {
            "version": "upstream-rectified-experimental-v1",
            "coordinate_space": "rectified_upstream",
            "source_manifest_sha256": "0" * 64,
        },
        "runs": [],
    }
    with pytest.raises(ValueError, match="must be explicit"):
        load_workflow()._validate_rectified(tmp_path, manifest)
