from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from maus_export.frame_embeddings import (
    OUT_NAME,
    build_frame_embeddings,
    scan_frames,
)

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

RAW_SHAPE = (509, 900, 3)
DIM = 3  # tiny test model: per-channel mean


def write_frame(media: Path, run_id: int, name: str, value: int) -> None:
    folder = media / str(run_id)
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / name), np.full(RAW_SHAPE, value, np.uint8))


def write_run(data_dir: Path, run_id: int, phases: list[str]) -> None:
    runs = data_dir / "raw" / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    checks = [{"phase": p} for p in phases]
    (runs / f"{run_id}.json").write_text(json.dumps({"visionChecks": checks}))


def tiny_model(path: Path) -> None:
    """(N,3,H,W) -> (N,3) channel means, standing in for the CNN."""
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [
            helper.make_node("GlobalAveragePool", ["x"], ["p"]),
            helper.make_node("Flatten", ["p"], ["y"], axis=1),
        ],
        "tiny",
        [
            helper.make_tensor_value_info(
                "x", TensorProto.FLOAT, [None, 3, 224, 224]
            )
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, DIM])],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8
    )
    path.write_bytes(model.SerializeToString())


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    media = tmp_path / "raw" / "media"
    write_frame(media, 7, "00_spread_frame.jpg", 10)
    write_frame(media, 7, "00_spread_overlay.jpg", 10)  # never embedded
    write_frame(media, 7, "01_push_off_frame.jpg", 20)
    write_frame(media, 7, "02_push_off_frame.jpg", 30)
    write_frame(media, 8, "00_spread_frame.jpg", 40)
    # run 9: the first push_off frame is missing locally
    write_frame(media, 9, "02_push_off_frame.jpg", 50)
    write_run(tmp_path, 9, ["spread", "push_off", "push_off"])
    (media / "notes").mkdir()
    return tmp_path


def test_scan_frames_keys(data_dir: Path) -> None:
    frames = scan_frames(data_dir)
    assert list(frames.columns) == [
        "run_id",
        "check",
        "attempt",
        "check_index",
        "frame_path",
    ]
    rows = frames[["run_id", "check", "attempt", "check_index"]]
    assert rows.values.tolist() == [
        [7, "spread", 1, 0],
        [7, "push_off", 1, 1],
        [7, "push_off", 2, 2],
        [8, "spread", 1, 0],
        [9, "push_off", 2, 2],  # counts the missing frame from the JSON
    ]
    assert frames["frame_path"][0] == "raw/media/7/00_spread_frame.jpg"
    assert frames["attempt"].dtype == np.int32
    assert frames["run_id"].dtype == np.int64


def test_scan_frames_empty(tmp_path: Path) -> None:
    assert scan_frames(tmp_path).empty


def test_build_frame_embeddings(data_dir: Path) -> None:
    model = data_dir / "model.onnx"
    tiny_model(model)
    (data_dir / "raw" / "media" / "8" / "01_fold1_frame.jpg").write_bytes(
        b"broken"
    )
    out = build_frame_embeddings(data_dir, model, batch_size=2)
    assert out == data_dir / OUT_NAME
    table = pd.read_parquet(out)
    assert len(table) == 5  # broken frame skipped
    emb = [f"e{j}" for j in range(DIM)]
    assert all(table[c].dtype == np.float32 for c in emb)
    assert not table.duplicated(["run_id", "check", "attempt"]).any()
    assert table["model_sha256"].nunique() == 1
    # brighter grey frames give larger channel means, in row order
    assert table["e0"].is_monotonic_increasing
