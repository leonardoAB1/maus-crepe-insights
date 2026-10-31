"""One preprocessing contract for frame models: training, export, caching
and on-robot inference all go through `preprocess`.

Input is a raw camera check frame (900x509 BGR, as read by cv2), never the
rectified overlay. The plate is cropped with a fixed box that covers the
plate on every robot seen so far, then resized and ImageNet-normalized.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

# (x, y, w, h) on the raw 900x509 frame; covers the plate on all robots
PLATE_CROP = (190, 0, 570, 509)
INPUT_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)  # RGB
IMAGENET_STD = (0.229, 0.224, 0.225)
PIXEL_MAX = 255.0


def load_frame(path: Path | str) -> np.ndarray:
    """BGR uint8 image; raises if the file is missing or not an image."""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"cannot read image {path}")
    return img


def crop_plate(
    bgr: np.ndarray, box: tuple[int, int, int, int] = PLATE_CROP
) -> np.ndarray:
    """The plate region, clipped to the image bounds."""
    x, y, w, h = box
    height, width = bgr.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + w), min(height, y + h)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"crop {box} is outside a {width}x{height} image")
    return bgr[y0:y1, x0:x1]


def preprocess(
    bgr: np.ndarray, box: tuple[int, int, int, int] = PLATE_CROP
) -> np.ndarray:
    """float32 NCHW tensor of shape (1, 3, INPUT_SIZE, INPUT_SIZE)."""
    crop = crop_plate(bgr, box)
    small = cv2.resize(
        crop, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA
    )
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb /= PIXEL_MAX
    rgb -= np.asarray(IMAGENET_MEAN, dtype=np.float32)
    rgb /= np.asarray(IMAGENET_STD, dtype=np.float32)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])


def preprocess_config() -> dict[str, Any]:
    """The contract as stored in model cards."""
    return {
        "source": "raw camera frame (BGR), never the overlay",
        "crop_xywh": list(PLATE_CROP),
        "resize": [INPUT_SIZE, INPUT_SIZE],
        "interpolation": "cv2.INTER_AREA",
        "color": "RGB",
        "scale": f"1/{PIXEL_MAX:g}",
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "layout": "NCHW",
        "function": "maus_export.frame_preprocess.preprocess",
    }
