"""Apply a repository camera correction to video frames."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ..rectification import rectify

ASPECT_RATIO_TOLERANCE = 0.01
CALIBRATION_VERSION = 1


def rectify_video_frame(
    frame: np.ndarray, robot: str, calibration: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Resize to the calibrated source size, then apply the robot transform.

    Return both the warped image and its supported-pixel mask. A transform for
    another robot or substantially different aspect ratio must not be guessed.
    """
    if calibration.get("schema_version") != CALIBRATION_VERSION:
        raise ValueError("Unsupported camera calibration schema")
    correction = calibration.get("robot_corrections", {}).get(robot)
    if correction is None:
        raise ValueError("No camera correction for this robot")
    source_size = tuple(correction["source_size_px"])
    target_size = correction["target_size_px"]
    width, height = source_size
    frame_height, frame_width = frame.shape[:2]
    if min(width, height, frame_width, frame_height) <= 0:
        raise ValueError("Frame and calibration dimensions must be positive")
    ratio = (frame_width / frame_height) / (width / height)
    if abs(ratio - 1) > ASPECT_RATIO_TOLERANCE:
        raise ValueError("Frame aspect ratio does not match camera calibration")
    if (frame_width, frame_height) != source_size:
        frame = cv2.resize(frame, source_size, interpolation=cv2.INTER_AREA)
    transform = {
        "schema_version": CALIBRATION_VERSION,
        "source_size": source_size,
        "target_size": target_size,
        "matrix": correction["matrix_mean"],
    }
    return rectify(frame, transform)
