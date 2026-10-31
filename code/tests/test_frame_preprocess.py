from __future__ import annotations

import cv2
import numpy as np
import pytest

from maus_export.frame_preprocess import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_SIZE,
    PLATE_CROP,
    crop_plate,
    load_frame,
    preprocess,
    preprocess_config,
)

RAW_H, RAW_W = 509, 900
GREY = 128


def raw(value: int = GREY) -> np.ndarray:
    return np.full((RAW_H, RAW_W, 3), value, dtype=np.uint8)


def test_preprocess_shape_and_dtype() -> None:
    x = preprocess(raw())
    assert x.shape == (1, 3, INPUT_SIZE, INPUT_SIZE)
    assert x.dtype == np.float32
    assert x.flags["C_CONTIGUOUS"]


def test_preprocess_normalizes_per_channel() -> None:
    x = preprocess(raw())
    want = (GREY / 255 - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    np.testing.assert_allclose(x[0].mean(axis=(1, 2)), want, atol=1e-5)


def test_preprocess_converts_bgr_to_rgb() -> None:
    img = raw(0)
    img[..., 2] = 255  # pure red in BGR
    x = preprocess(img)[0]
    assert x[0].mean() > x[1].mean() and x[0].mean() > x[2].mean()


def test_crop_uses_plate_box() -> None:
    x, y, w, h = PLATE_CROP
    assert crop_plate(raw()).shape == (h, w, 3)
    img = raw(0)
    img[y : y + h, x : x + w] = 255
    assert crop_plate(img).min() == 255


def test_crop_clips_to_small_image() -> None:
    small = np.zeros((100, 300, 3), dtype=np.uint8)
    assert crop_plate(small).shape == (100, 300 - PLATE_CROP[0], 3)
    assert preprocess(small).shape == (1, 3, INPUT_SIZE, INPUT_SIZE)


def test_crop_outside_image_raises() -> None:
    with pytest.raises(ValueError):
        crop_plate(np.zeros((50, 100, 3), dtype=np.uint8))


def test_load_frame(tmp_path) -> None:
    path = tmp_path / "f.jpg"
    cv2.imwrite(str(path), raw())
    assert load_frame(path).shape == (RAW_H, RAW_W, 3)
    (tmp_path / "bad.jpg").write_bytes(b"not an image")
    with pytest.raises(ValueError):
        load_frame(tmp_path / "bad.jpg")
    with pytest.raises(ValueError):
        load_frame(tmp_path / "missing.jpg")


def test_config_matches_constants() -> None:
    cfg = preprocess_config()
    assert cfg["crop_xywh"] == list(PLATE_CROP)
    assert cfg["resize"] == [INPUT_SIZE, INPUT_SIZE]
