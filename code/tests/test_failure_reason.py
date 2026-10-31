"""Failure-reason heads on synthetic failed frames and embeddings."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")  # heads run in the ml environment
pytest.importorskip("onnxruntime")

from maus_classify import failure_reason as fr  # noqa: E402

N_DIM = 8
PERM_SMALL = 49


def test_reason_codes_merge_and_fallbacks() -> None:
    assert fr.reason_codes(
        "spread", ["Large hole in the centre", "large hole off-center"], None
    ) == ["hole"]
    assert fr.reason_codes(
        "fold2", ["not a quarter fold", "wrong fold shape"], None
    ) == ["shape"]
    # No errors: message is used; unknown text maps to other
    assert fr.reason_codes("spread", [], "No crepe in the plate") == [
        "no_crepe"
    ]
    assert fr.reason_codes("push_off", ["something new"], None) == [fr.OTHER]


def synthetic(n_runs: int = 60, seed: int = 0, signal: bool = True):
    """Failed spread frames: too_small shifts embedding dim 0 when
    `signal`, holes otherwise; two robots, last 20% of runs in test."""
    rng = np.random.default_rng(seed)
    frames, emb = [], []
    for run in range(n_runs):
        small = run % 2 == 0
        codes = ["crepe too small"] if small else ["large hole off-center"]
        robot = "r-a" if run % 3 else "r-b"
        split = "test" if run >= n_runs * 0.8 else "train"
        frames.append(
            {
                "run_id": run,
                "check": "spread",
                "attempt": 1,
                "ok": False,
                "errors": codes,
                "message": "",
                "split": split,
                "robot": robot,
                "is_last_attempt": True,
                "ts_utc": pd.Timestamp("2026-10-13T10:00Z")
                + pd.Timedelta(hours=run),
                "frame_path": f"raw/media/{run}/00_spread_frame.jpg",
            }
        )
        vec = rng.normal(size=N_DIM)
        if signal and small:
            vec[0] += 3.0
        emb.append(
            {"run_id": run, "check": "spread", "attempt": 1}
            | {f"e{i}": v for i, v in enumerate(vec)}
        )
    return pd.DataFrame(frames), pd.DataFrame(emb)


def test_support_and_min_support() -> None:
    frames, emb = synthetic()
    rows = fr.failed_rows(frames, emb, "spread")
    sup = fr.support(rows, "spread")
    assert sup["too_small"]["train"] + sup["too_small"]["test"] == 30
    assert fr.modelled_classes(rows, "spread") == ["hole", "too_small"]
    few = rows[rows["run_id"] < 12]
    assert fr.modelled_classes(few, "spread") == []


def test_threshold_sets_and_metrics() -> None:
    probas = {"a": np.array([0.9, 0.2, 0.6]), "b": np.array([0.8, 0.1, 0.4])}
    assert fr.predict_sets(probas, 0.5) == [["a", "b"], [fr.OTHER], ["a"]]
    truth = [["a", "b"], [fr.OTHER], ["b"]]
    t, scores = fr.choose_threshold(truth, probas)
    assert str(t) in scores
    m = fr.set_metrics(truth, fr.predict_sets(probas, t), ["a", "b"])
    assert m["per_class"]["a"]["support"] == 1
    assert m["per_class"][fr.OTHER]["support"] == 1


def test_permutation_test_signal_and_null() -> None:
    for signal, expected in ((True, True), (False, False)):
        frames, emb = synthetic(signal=signal)
        rows = fr.failed_rows(frames, emb, "spread")
        X = rows[[f"e{i}" for i in range(N_DIM)]].to_numpy(float)
        y = fr.binary_labels(rows, "too_small")
        res = fr.permutation_test(
            X,
            y,
            rows["run_id"].to_numpy(),
            rows["robot"].to_numpy(),
            "too_small",
            n_perm=PERM_SMALL,
        )
        assert res["beats_null"] is expected


def test_train_check_end_to_end(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(fr, "N_PERM", PERM_SMALL)
    frames, emb = synthetic()
    card = fr.train_check(frames, emb, "spread", tmp_path)
    assert card["modelled"] == ["hole", "too_small"]
    assert (tmp_path / "too_small" / "model.onnx").is_file()
    head = json.loads((tmp_path / "too_small" / "model.json").read_text())
    assert head["reason"]["majority_baseline"] is not None
    assert set(head["reason"]["train_cv"]["per_robot"]) == {"r-a", "r-b"}
    assert card["abstain"]["threshold"] in fr.THRESHOLDS
    per = card["holdout"]["per_class"]
    assert {"too_small", "hole", fr.OTHER} <= set(per)
    assert card["heads"]["too_small"]["verdict"] == "deployable"


def test_check_without_support_reports_not_enough_data(tmp_path) -> None:
    frames, emb = synthetic(n_runs=10)
    card = fr.train_check(frames, emb, "spread", tmp_path)
    assert card["modelled"] == []
    assert "not enough data" in card["verdict"]
    assert (tmp_path / "model.json").is_file()
