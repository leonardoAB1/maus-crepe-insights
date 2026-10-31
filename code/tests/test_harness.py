"""End to end harness test on synthetic features (ml environment)."""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")  # harness runs in the ml environment
pytest.importorskip("skl2onnx")
pytest.importorskip("onnxruntime")

from sklearn.linear_model import LogisticRegression  # noqa: E402

from maus_classify.harness import (  # noqa: E402
    HarnessError,
    train_and_evaluate,
)

N_RUNS = 60
FRAMES_PER_RUN = 3
N_TRAIN_RUNS = 45
CLASSES = np.array(["flat", "half", "quarter"])


def synthetic():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(1, N_RUNS + 1), FRAMES_PER_RUN)
    y = CLASSES[rng.integers(0, len(CLASSES), len(groups))]
    centres = {c: i * 3.0 for i, c in enumerate(CLASSES)}
    X = np.stack([[centres[c], -centres[c]] for c in y]) + rng.normal(
        size=(len(y), 2)
    )
    split = np.where(groups <= N_TRAIN_RUNS, "train", "test")
    ts = np.datetime64("2026-10-13T10:00") + groups.astype("timedelta64[h]")
    return X, y, split, groups, ts


def test_train_score_export_end_to_end(tmp_path: Path) -> None:
    X, y, split, groups, ts = synthetic()
    card = train_and_evaluate(
        LogisticRegression(max_iter=500),
        X,
        y,
        split,
        groups,
        name="dummy",
        feature_names=["a", "b"],
        out_dir=tmp_path,
        timestamps=ts,
    )
    assert (tmp_path / "model.onnx").is_file()
    saved = json.loads((tmp_path / "model.json").read_text(encoding="utf-8"))
    assert saved["name"] == "dummy"
    assert saved["inputs"][0]["features"] == ["a", "b"]
    assert saved["outputs"][1]["classes"] == list(CLASSES)
    assert saved["data"]["train"]["n_runs"] == N_TRAIN_RUNS
    assert saved["data"]["test"]["run_id_min"] == N_TRAIN_RUNS + 1
    assert saved["onnx"]["parity"] == "ok"

    m = card["metrics"]["test"]
    assert m["n"] == (N_RUNS - N_TRAIN_RUNS) * FRAMES_PER_RUN
    assert m["balanced_accuracy"] > 0.9
    assert 0 <= m["f1_macro"] <= 1
    assert set(m["per_class"]) == set(CLASSES)
    assert {"precision", "recall"} <= set(m["per_class"]["half"])
    cm = np.array(m["confusion_matrix"]["rows_true_cols_pred"])
    assert cm.shape == (3, 3) and cm.sum() == m["n"]
    cal = m["calibration"]
    assert 0 <= cal["brier_mean"] <= 1
    assert "ece" in cal["per_class"]["flat"]


def test_run_spanning_both_splits_is_rejected(tmp_path: Path) -> None:
    X, y, split, groups, _ = synthetic()
    split[0] = "test"  # run 1 now has rows on both sides
    with pytest.raises(HarnessError):
        train_and_evaluate(
            LogisticRegression(),
            X,
            y,
            split,
            groups,
            name="bad",
            feature_names=["a", "b"],
            out_dir=tmp_path,
        )
