"""Segmentation backend selection and private output isolation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .fold_mask import CrepeSegmenter
from .fold_plate import PlateEllipse
from .fold_rectify import rectify_video_frame

MODELS = ("classical", "onnx")
MASK_VALUE = 255


def analysis_root(data_dir: Path, model: str = "classical") -> Path:
    """Keep existing classical outputs separate from raw-frame ONNX results."""
    if model not in MODELS:
        raise ValueError("Unknown segmentation model")
    name = "live" if model == "classical" else "live-onnx"
    return data_dir / "fold_segmentation" / name


def rectify_student_mask(
    mask: np.ndarray,
    raw_shape: tuple[int, int],
    robot: str,
    calibration: dict[str, Any],
) -> np.ndarray:
    """Map the student's low-resolution raw mask to corrected camera pixels."""
    height, width = raw_shape
    native = cv2.resize(
        mask.astype(np.uint8) * MASK_VALUE,
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    corrected, valid = rectify_video_frame(
        native, robot, calibration, is_mask=True
    )
    corrected[valid == 0] = 0
    return corrected


class OnnxForeground:
    """Infer on raw frames, then align masks with classical geometry."""

    def __init__(self, shape: tuple[int, int], plate: PlateEllipse) -> None:
        from ..crepe_student import CrepeStudent

        self.student = CrepeStudent.load()
        self.segmenter = CrepeSegmenter(shape, plate)

    def segment_raw(
        self, raw: np.ndarray, robot: str, calibration: dict[str, Any]
    ) -> np.ndarray:
        """Never feed rectified frames to a model trained on raw cameras."""
        mask = rectify_student_mask(
            self.student.mask(raw), raw.shape[:2], robot, calibration
        )
        mask[~self.segmenter.inside] = 0
        return mask

    def save_reference(self, directory: Path) -> None:
        """Record that this backend does not learn an empty-plate reference."""
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "model": "onnx",
                    "input_space": "raw",
                    "empty_reference": False,
                }
            )
        )
