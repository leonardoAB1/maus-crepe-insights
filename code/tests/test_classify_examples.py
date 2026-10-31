from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("onnxruntime")

from sklearn.linear_model import LogisticRegression  # noqa: E402

from maus_classify.check_verdict import CHECKS, FAIL, PASS  # noqa: E402
from maus_classify.examples import (  # noqa: E402
    EMBEDDINGS,
    FRAMES,
    check_verdict_example,
    embedder_example,
)
from maus_classify.harness import export_onnx  # noqa: E402

DIM = 4
N_RUNS = 12
RAW_SHAPE = (509, 900, 3)
ROBOTS = ("robot-a", "robot-b")
MAX_BYTES = 400_000


def write_data(data_dir: Path) -> None:
    rng = np.random.default_rng(0)
    frames, emb = [], []
    for run in range(N_RUNS):
        for idx, check in enumerate(CHECKS):
            ok = (run + idx) % 3 != 0
            rel = f"raw/media/{run}/{idx:02d}_{check}_frame.jpg"
            path = data_dir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), np.full(RAW_SHAPE, 90, np.uint8))
            frames.append(
                {
                    "run_id": run,
                    "check": check,
                    "attempt": 1,
                    "ok": ok,
                    "robot": ROBOTS[run % len(ROBOTS)],
                    "split": "test" if run % 2 else "train",
                    "frame_path": rel,
                }
            )
            vec = rng.normal(0, 1, DIM) + (0 if ok else 2)
            emb.append(
                {"run_id": run, "check": check, "attempt": 1}
                | {f"e{j}": np.float32(v) for j, v in enumerate(vec)}
            )
    (data_dir / "training").mkdir()
    pd.DataFrame(frames).to_parquet(data_dir / FRAMES)
    pd.DataFrame(emb).to_parquet(data_dir / EMBEDDINGS)


def write_heads(data_dir: Path, model_dir: Path) -> None:
    emb = pd.read_parquet(data_dir / EMBEDDINGS)
    frames = pd.read_parquet(data_dir / FRAMES)
    df = emb.merge(frames, on=["run_id", "check", "attempt"])
    for check in CHECKS:
        rows = df[df["check"] == check]
        X = rows[[f"e{j}" for j in range(DIM)]].to_numpy(np.float32)
        y = np.where(rows["ok"], PASS, FAIL)
        model = LogisticRegression().fit(X, y)
        export_onnx(model, DIM, model_dir / check / "model.onnx")
        card = {"outputs": [{}, {"classes": [str(c) for c in model.classes_]}]}
        (model_dir / check / "model.json").write_text(json.dumps(card))


def test_check_verdict_example(tmp_path: Path) -> None:
    write_data(tmp_path)
    models = tmp_path / "models"
    write_heads(tmp_path, models)
    out = check_verdict_example(tmp_path, models, tmp_path / "cv.jpg")
    assert cv2.imread(str(out)) is not None
    assert out.stat().st_size < MAX_BYTES


def test_embedder_example(tmp_path: Path) -> None:
    write_data(tmp_path)
    out = embedder_example(tmp_path, tmp_path / "emb.jpg")
    assert cv2.imread(str(out)) is not None
    assert out.stat().st_size < MAX_BYTES
