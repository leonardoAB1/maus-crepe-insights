"""Hand mask decoding and record filtering, on made-up annotations only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from maus_export import human_masks as hm

WIDTH, HEIGHT = 8, 4


def encode(values: list[int]) -> list[int]:
    """Smallest valid brush RLE: every byte in its own literal block."""
    bits = format(len(values), "032b") + format(7, "05b")
    bits += format(0, "04b") * 4
    for value in values:
        bits += "0" + "00" + "0" + format(value, "08b")
    bits += "0" * (-len(bits) % 8)
    return [int(bits[i : i + 8], 2) for i in range(0, len(bits), 8)]


def brush(mask: np.ndarray, label: str) -> dict:
    flat = np.zeros((HEIGHT, WIDTH, 4), np.uint8)
    flat[..., 3] = mask * 255
    return {
        "type": "brushlabels",
        "original_width": WIDTH,
        "original_height": HEIGHT,
        "value": {
            "rle": encode(flat.ravel().tolist()),
            "brushlabels": [label],
        },
    }


def record(state: str, regions: list, **extra: object) -> dict:
    return {
        "image_size": f"[{WIDTH}, {HEIGHT}]",
        "state": state,
        "result": regions,
        "validation": "ok",
        "image_sha256": "a",
    } | extra


def test_decode_literal_and_repeated_blocks() -> None:
    assert hm.decode_rle(encode([1, 2, 255])).tolist() == [1, 2, 255]
    # Repeat flag 1, run selector 0, run length 3 (one bit stores 2 extra is
    # not enough, so the first run size is set to two bits), value 9
    bits = format(3, "032b") + format(7, "05b") + format(1, "04b")
    bits += format(0, "04b") * 3 + "1" + "00" + "10" + format(9, "08b")
    bits += "0" * (-len(bits) % 8)
    rle = [int(bits[i : i + 8], 2) for i in range(0, len(bits), 8)]
    assert hm.decode_rle(rle).tolist() == [9, 9, 9]


def test_record_masks_separates_material_and_holes() -> None:
    material = np.zeros((HEIGHT, WIDTH), bool)
    material[1:3, 2:6] = True
    hole = np.zeros((HEIGHT, WIDTH), bool)
    hole[0, 0] = True
    found, holes = hm.record_masks(
        record(
            hm.PRESENT,
            [brush(material, hm.MATERIAL), brush(hole, hm.HOLE)],
        )
    )
    assert np.array_equal(found, material)
    assert np.array_equal(holes, hole)


def test_record_masks_refuses_a_contradicting_state() -> None:
    empty = np.zeros((HEIGHT, WIDTH), bool)
    with pytest.raises(ValueError, match="needs a material"):
        hm.record_masks(record(hm.PRESENT, []))
    with pytest.raises(ValueError, match="no mask"):
        hm.record_masks(record(hm.ABSENT, [brush(~empty, hm.MATERIAL)]))


def test_wrong_brush_size_is_refused() -> None:
    region = brush(np.zeros((HEIGHT, WIDTH), bool), hm.MATERIAL)
    region["original_width"] = WIDTH + 1
    with pytest.raises(ValueError, match="dimensions"):
        hm.decode_region(region, (WIDTH, HEIGHT))


def test_usable_records_keep_ok_reviewed_frames_only() -> None:
    frames = [
        record(hm.ABSENT, [], image_sha256="a"),
        record(hm.ABSENT, [], image_sha256="b", validation="needs_review"),
        record(None, [], image_sha256="c"),
    ]
    kept = hm.usable_records([json.dumps({"frames": frames})])
    assert [r["image_sha256"] for r in kept] == ["a"]


def test_usable_records_refuse_the_same_image_twice() -> None:
    text = json.dumps({"frames": [record(hm.ABSENT, [])]})
    with pytest.raises(ValueError, match="two snapshots"):
        hm.usable_records([text, text])


def test_identity_index_finds_a_frame_by_run_frame_and_timestamp(
    tmp_path: Path,
) -> None:
    (tmp_path / "run-000").mkdir()
    (tmp_path / "run-000" / "000000.png").write_bytes(b"x")
    manifest = {
        "runs": [
            {
                "run_id": 7,
                "frames": [
                    {
                        "image": "run-000/000000.png",
                        "decoded_frame_index": 3,
                        "decoded_pts": 90,
                    },
                    {
                        "image": "run-000/missing.png",
                        "decoded_frame_index": 4,
                        "decoded_pts": 120,
                    },
                ],
            }
        ]
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    found = hm.identity_index([tmp_path])
    assert found == {(7, 3, 90): tmp_path / "run-000" / "000000.png"}
    wanted = {"run_id": "7", "decoded_frame_index": "3", "decoded_pts": "90"}
    assert hm.frame_identity(wanted) == (7, 3, 90)
