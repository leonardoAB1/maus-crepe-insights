"""Human-label evaluation: RLE decoding, label loading and metrics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from maus_export import human_eval as he
from maus_export.crepe_student import CrepeStudent

HEIGHT, WIDTH = 12, 16
MAX_RUN = 2**8  # longest run one block encodes with 8-bit length fields
WORD = 8
RUN_BITS = 8


def encode_rle(alpha: np.ndarray) -> list[int]:
    """Label Studio brush RLE of an RGBA image with this alpha channel.

    Only repeat blocks, all with the same 8-bit run length field: enough
    to exercise the decoder's bit layout.
    """
    rgba = np.zeros((*alpha.shape, he.RGBA_CHANNELS), dtype=np.uint8)
    rgba[..., -1] = alpha
    flat = rgba.ravel()
    bits = [f"{flat.size:032b}", f"{WORD - 1:05b}"]
    bits += [f"{RUN_BITS - 1:04b}"] * he.SIZE_COUNT
    index = 0
    while index < flat.size:
        value = flat[index]
        end = index + 1
        while end < flat.size and flat[end] == value and end - index < MAX_RUN:
            end += 1
        bits.append("1" + "00" + f"{end - index - 1:08b}" + f"{value:08b}")
        index = end
    stream = "".join(bits)
    stream += "0" * (-len(stream) % he.BITS_PER_BYTE)
    return [
        int(stream[i : i + he.BITS_PER_BYTE], 2)
        for i in range(0, len(stream), he.BITS_PER_BYTE)
    ]


def square(top: int, left: int, size: int) -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[top : top + size, left : left + size] = True
    return mask


def brush(mask: np.ndarray, label: str) -> dict:
    return {
        "type": "brushlabels",
        "original_width": WIDTH,
        "original_height": HEIGHT,
        "value": {
            "format": "rle",
            "rle": encode_rle(mask.astype(np.uint8) * 255),
            "brushlabels": [label],
        },
    }


def test_decode_round_trip() -> None:
    mask = square(2, 3, 5)
    rle = encode_rle(mask.astype(np.uint8) * 255)
    assert np.array_equal(he.decode_rle(rle, HEIGHT, WIDTH), mask)


def test_decode_literal_block() -> None:
    # header: 4 pixels of RGBA, 8-bit words, 2-bit length fields
    total = 2 * 2 * he.RGBA_CHANNELS
    bits = f"{total:032b}" + f"{WORD - 1:05b}" + "0001" * he.SIZE_COUNT
    alpha = [0, 9, 0, 9]
    for value in alpha:
        bits += (
            "0" + "00" + "11" + "".join(f"{v:08b}" for v in (0, 0, 0, value))
        )
    bits += "0" * (-len(bits) % 8)
    rle = [int(bits[i : i + 8], 2) for i in range(0, len(bits), 8)]
    decoded = he.decode_rle(rle, 2, 2)
    assert decoded.tolist() == [[False, True], [False, True]]


def test_decode_rejects_wrong_size() -> None:
    rle = encode_rle(np.zeros((HEIGHT, WIDTH), dtype=np.uint8))
    with pytest.raises(ValueError):
        he.decode_rle(rle, HEIGHT + 1, WIDTH)


def test_brush_masks_union_per_class() -> None:
    result = [
        {"type": "choices", "value": {"choices": ["crepe_present"]}},
        brush(square(0, 0, 3), he.MATERIAL),
        brush(square(5, 5, 3), he.MATERIAL),
        brush(square(1, 1, 1), he.HOLE),
    ]
    masks = he.brush_masks(result, HEIGHT, WIDTH)
    assert masks[he.MATERIAL].sum() == 18
    assert masks[he.HOLE].sum() == 1


def write_labels(root: Path, set_name: str, kind: str, frames: list) -> None:
    folder = root / set_name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{kind}-v1.json").write_text(json.dumps({"frames": frames}))


def test_load_phase_merges_label_and_phase_columns(tmp_path: Path) -> None:
    write_labels(tmp_path, "set-1", "phase", [{"run_id": 1, "label": "circle"}])
    write_labels(
        tmp_path, "set-2", "phase", [{"run_id": 2, "phase": "halffold"}]
    )
    table = he.load_phase(tmp_path).sort_values("run_id")
    assert table["phase"].tolist() == ["circle", "halffold"]
    assert table["set"].tolist() == ["set-1", "set-2"]


def test_iou_handles_empty_masks() -> None:
    empty = np.zeros((2, 2), dtype=bool)
    assert np.isnan(he.iou(empty, empty))
    assert he.iou(square(0, 0, 4), square(0, 0, 2)) == pytest.approx(0.25)


def test_annotator_iou_on_shared_frames() -> None:
    def record(set_name: str, mask: np.ndarray) -> dict:
        return {
            "set": set_name,
            "run_id": 1,
            "decoded_frame_index": 5,
            "state": he.CREPE_PRESENT,
            "image_size": [WIDTH, HEIGHT],
            "result": [brush(mask, he.MATERIAL)],
        }

    masks = pd.DataFrame(
        [record("set-1", square(0, 0, 4)), record("set-2", square(0, 0, 2))]
    )
    assert he.annotator_iou(masks).tolist() == [pytest.approx(0.25)]


def test_phase_metrics_uses_the_last_frame_before_the_label() -> None:
    phase = pd.DataFrame(
        {
            "run_id": [1, 1],
            "phase": ["circle", "halffold"],
            "decoded_pts": [10, 30],
            "start_pts": [0, 0],
            "time_base": ["1/10", "1/10"],
        }
    )
    timeline = pd.DataFrame(
        {
            "second": [0.5, 1.5, 2.5, 3.5],
            "state": ["flat", "flat", "flat", "half"],
            "raw_state": ["flat", "flat", "half", "half"],
        }
    )
    table = he.phase_metrics(phase, {1: timeline})
    assert table["state"].tolist() == ["flat", "flat"]
    assert table["raw_state"].tolist() == ["flat", "half"]
    assert table["human_state"].tolist() == ["flat", "half"]


class FixedSession:
    def __init__(self, mask: np.ndarray) -> None:
        self.mask = mask

    def run(self, output_names, feeds):
        return [self.mask[None, None].astype(np.uint8)]


def test_mask_metrics_and_summary(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    path = he.frame_file(frames_dir, 1, 0)
    path.parent.mkdir(parents=True)
    cv2.imwrite(str(path), np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8))
    material = square(0, 0, 8)
    holes = square(2, 2, 2)
    masks = pd.DataFrame(
        [
            {
                "set": "set-1",
                "run_id": 1,
                "decoded_frame_index": 0,
                "state": he.CREPE_PRESENT,
                "validation": he.VALID,
                "result": [
                    brush(material & ~holes, he.MATERIAL),
                    brush(holes, he.HOLE),
                ],
            }
        ]
    )
    student = CrepeStudent(FixedSession(np.ones((216, 384), dtype=bool)))
    table = he.mask_metrics(masks, frames_dir, student)
    row = table.iloc[0]
    assert row["hole_as_crepe"] == 1.0
    assert row["iou"] == pytest.approx((64 - 4) / (HEIGHT * WIDTH))
    result = he.summary(table, pd.DataFrame(), pd.Series([0.9]))
    assert set(result["metric"]) >= {
        "mask iou, crepe frames",
        "mask iou, human vs human",
    }


def test_decode_label_frames_checks_hashes(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    path = he.frame_file(frames_dir, 3, 7)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"png bytes")
    good = hashlib.sha256(b"png bytes").hexdigest()
    keys = pd.DataFrame(
        {
            "run_id": [3, 3],
            "decoded_frame_index": [7, 7],
            "image_sha256": [good, "other"],
        }
    )
    table = he.decode_label_frames(tmp_path, keys, frames_dir)
    assert table["decoded"].all()
    assert table["sha_match"].tolist() == [True, False]
