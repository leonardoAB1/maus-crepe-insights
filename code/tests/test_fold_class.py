from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("onnxruntime")

from maus_classify.fold_class import (  # noqa: E402
    CLASSES,
    EMBEDDINGS,
    FRAMES,
    FoldClassError,
    choose_c,
    confidence_summary,
    embedding_columns,
    load_frames,
    train_fold_class,
)

DIM = 6
N_RUNS = 30
TEST_RUNS = 6
SPREAD = 0.3
LABEL_OF = {"spread": "flat", "fold1": "half", "fold2": "quarter"}
# Each check gets a distinct mean so the synthetic classes separate;
# push_off looks like fold2 (quarter), plate_clean like nothing seen
CENTRE = {
    "spread": 0,
    "fold1": 1,
    "fold2": 2,
    "push_off": 2,
    "plate_clean": 3,
}


def write_data(data_dir: Path) -> None:
    rng = np.random.default_rng(0)
    frames, emb = [], []
    start = pd.Timestamp("2026-10-01", tz="UTC")
    for run in range(N_RUNS):
        split = "test" if run >= N_RUNS - TEST_RUNS else "train"
        for idx, check in enumerate(CENTRE):
            ok = not (check == "fold1" and run % 7 == 0)  # a few failures
            frames.append(
                {
                    "run_id": run,
                    "check": check,
                    "check_index": idx,
                    "attempt": 1,
                    "ts_utc": start + pd.Timedelta(hours=run),
                    "ok": ok,
                    "fold_class": LABEL_OF.get(check) if ok else None,
                    "split": split,
                }
            )
            vec = rng.normal(0, SPREAD, DIM)
            vec[CENTRE[check]] += 1
            emb.append(
                {"run_id": run, "check": check, "attempt": 1}
                | {f"e{j}": np.float32(v) for j, v in enumerate(vec)}
            )
    (data_dir / "training").mkdir(parents=True)
    pd.DataFrame(frames).to_parquet(data_dir / FRAMES)
    pd.DataFrame(emb).to_parquet(data_dir / EMBEDDINGS)


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    write_data(tmp_path)
    return tmp_path


def test_embedding_columns_ignore_other_e_names() -> None:
    df = pd.DataFrame(columns=["e10", "check", "e2", "model_sha256", "e0"])
    assert embedding_columns(df) == ["e0", "e2", "e10"]


def test_load_frames_joins_on_key(data_dir: Path) -> None:
    df = load_frames(data_dir)
    assert len(df) == N_RUNS * len(CENTRE)
    assert embedding_columns(df) == [f"e{j}" for j in range(DIM)]


def test_missing_inputs_raise(tmp_path: Path) -> None:
    with pytest.raises(FoldClassError):
        load_frames(tmp_path)


def test_choose_c_prefers_regularization_on_ties() -> None:
    rng = np.random.default_rng(1)
    y = np.repeat(np.array(CLASSES), 20)
    X = rng.normal(0, 1, (len(y), DIM))
    X[np.arange(len(y)), np.repeat([0, 1, 2], 20)] += 3
    groups = np.arange(len(y)) // 2
    c, scores = choose_c(X, y, groups)
    best = min(scores.values())
    assert scores[str(c)] <= best * 1.05 + 1e-12
    assert all(float(k) >= c or v > best * 1.05 for k, v in scores.items())


def test_confidence_summary() -> None:
    y = np.array(["flat", "half"])
    proba = np.array([[0.9, 0.1, 0.0], [0.6, 0.4, 0.0]])
    out = confidence_summary(y, proba, CLASSES)
    assert out["accuracy"] == 0.5
    assert out["mean_confidence"] == pytest.approx(0.75)
    assert out["per_true_class_accuracy"] == {"flat": 1.0, "half": 0.0}


def test_train_fold_class_card(data_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "model"
    card = train_fold_class(data_dir, out)
    assert (out / "model.onnx").is_file()
    assert json.loads((out / "model.json").read_text()) == card

    assert card["outputs"][1]["classes"] == CLASSES
    test = card["metrics"]["test"]
    assert set(test["per_class"]) == set(CLASSES)
    assert set(test["calibration"]["per_class"]) == set(CLASSES)
    assert test["confidence"]["accuracy"] > 0.9
    # failed frames carry no label and are left out
    n_failed_train = sum(1 for r in range(N_RUNS - TEST_RUNS) if r % 7 == 0)
    assert (
        card["data"]["train"]["n_rows"]
        == 3 * (N_RUNS - TEST_RUNS) - n_failed_train
    )

    sanity = card["sanity_checks"]
    assert sanity["push_off_passed"]["predicted_share"]["quarter"] > 0.9
    assert sanity["plate_clean_out_of_scope"]["n"] == TEST_RUNS
    assert "triangle" in card["caveats"]["triangle_fold"].lower()
    assert "#87" in card["caveats"]["triangle_fold"]
    assert "label_leakage" in card["caveats"]
