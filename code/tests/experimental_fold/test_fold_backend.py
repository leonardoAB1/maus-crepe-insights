"""Raw inference, binary perspective alignment and backend isolation."""

from pathlib import Path

import numpy as np
import pytest

from maus_export.crepe_student import CrepeStudent
from maus_export.experimental_fold.fold_backend import (
    OnnxForeground,
    analysis_root,
    rectify_student_mask,
)
from maus_export.experimental_fold.fold_plate import PlateEllipse


def calibration() -> dict:
    return {
        "schema_version": 1,
        "robot_corrections": {
            "synthetic": {
                "source_size_px": [100, 100],
                "target_size_px": [100, 100],
                "matrix_mean": [[1, 0, 10], [0, 1, 5], [0, 0, 1]],
            }
        },
    }


def test_mask_resized_before_warp_with_binary_interpolation() -> None:
    mask = np.zeros((10, 10), bool)
    mask[3:5, 2:4] = True
    result = rectify_student_mask(mask, (200, 200), "synthetic", calibration())
    expected = np.zeros((100, 100), np.uint8)
    expected[35:55, 30:50] = 255
    assert np.array_equal(result, expected)
    assert set(np.unique(result)) == {0, 255}


def test_onnx_receives_raw_pixels_and_returns_corrected_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = np.full((200, 200, 3), 37, np.uint8)

    class Student:
        def mask(self, frame: np.ndarray) -> np.ndarray:
            assert frame is raw
            mask = np.zeros((10, 10), bool)
            mask[3:5, 2:4] = True
            return mask

    monkeypatch.setattr(CrepeStudent, "load", lambda: Student())
    backend = OnnxForeground((100, 100), PlateEllipse(50, 50, 90, 90, 0))
    result = backend.segment_raw(raw, "synthetic", calibration())
    assert result[40, 35] == 255
    assert result[30, 20] == 0
    assert not result[~backend.segmenter.inside].any()


def test_backend_paths_are_distinct_and_allowlisted(tmp_path: Path) -> None:
    assert analysis_root(tmp_path).name == "live"
    assert analysis_root(tmp_path, "onnx").name == "live-onnx"
    with pytest.raises(ValueError, match="Unknown"):
        analysis_root(tmp_path, "../../raw")


def test_mask_correction_rejects_wrong_camera() -> None:
    with pytest.raises(ValueError, match="No camera"):
        rectify_student_mask(
            np.zeros((10, 10), bool), (200, 200), "other", calibration()
        )
