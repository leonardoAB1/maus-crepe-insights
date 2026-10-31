"""Video rectification using a synthetic camera calibration."""

import numpy as np
import pytest

from maus_export.fold_rectify import rectify_video_frame


def calibration() -> dict:
    return {
        "schema_version": 1,
        "robot_corrections": {
            "robot": {
                "source_size_px": [100, 50],
                "target_size_px": [100, 50],
                "matrix_mean": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            }
        },
    }


def test_rectify_resizes_and_uses_transform() -> None:
    frame = np.full((100, 200, 3), 123, np.uint8)
    warped, valid = rectify_video_frame(frame, "robot", calibration())
    assert warped.shape == (50, 100, 3)
    assert np.all(warped == 123)
    assert np.all(valid == 255)


def test_reject_missing_correction_and_mismatched_aspect() -> None:
    frame = np.zeros((100, 100, 3), np.uint8)
    with pytest.raises(ValueError, match="No camera correction"):
        rectify_video_frame(frame, "unknown", calibration())
    with pytest.raises(ValueError, match="aspect ratio"):
        rectify_video_frame(frame, "robot", calibration())
