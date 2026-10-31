"""Per-check verdict heads on synthetic embeddings with a planted signal."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("skl2onnx")
ort = pytest.importorskip("onnxruntime")

from maus_classify.check_verdict import (  # noqa: E402
    FAIL,
    PASS,
    check_rows,
    embedding_columns,
    make_head,
    summary_table,
    train_check,
    verdict,
)

DIM = 8
N_RUNS = 120
TEST_FROM = 96  # runs at or after this index are holdout
FAIL_RATE = 0.25
SIGNAL = 3.0


def synthetic(seed: int = 0) -> tuple[pd.DataFrame, ...]:
    """Frames, embeddings and vision scores shaped like the real tables.
    Failing frames are shifted along e0; push_off fails once then passes."""
    rng = np.random.default_rng(seed)
    frames, embs, vision = [], [], []
    start = pd.Timestamp("2026-10-01", tz="UTC")
    for i in range(N_RUNS):
        run_id = 3000 + i
        split = "test" if i >= TEST_FROM else "train"
        ts = start + pd.Timedelta(hours=i)
        oks = {
            "spread": [rng.random() > FAIL_RATE],
            "push_off": [False, True],
        }
        for check, verdicts in oks.items():
            for k, ok in enumerate(verdicts, start=1):
                frames.append(
                    {
                        "run_id": run_id,
                        "check": check,
                        "attempt": k,
                        "ts_utc": ts,
                        "ok": ok,
                        "is_last_attempt": k == len(verdicts),
                        "split": split,
                        "robot": "quin" if i < N_RUNS // 2 else "pixi",
                    }
                )
                e = rng.normal(size=DIM).astype(np.float32)
                e[0] += 0.0 if ok else SIGNAL
                embs.append(
                    {"run_id": run_id, "check": check, "attempt": k}
                    | {f"e{j}": e[j] for j in range(DIM)}
                    | {"frame_path": "x.jpg", "model_sha256": "abc"}
                )
        ok = oks["spread"][0]
        vision.append(
            {
                "id": run_id,
                "vision.spread.area_rel": rng.normal(0 if ok else 2),
                "vision.fold1.iou": rng.normal(),
            }
        )
    return pd.DataFrame(frames), pd.DataFrame(embs), pd.DataFrame(vision)


def test_embedding_columns_are_numeric_order():
    df = pd.DataFrame(columns=["e10", "e2", "run_id", "e0", "frame_path"])
    assert embedding_columns(df) == ["e0", "e2", "e10"]


def test_check_rows_keeps_one_check_and_labels_fail():
    frames, embs, _ = synthetic()
    rows = check_rows(frames, embs, "spread")
    assert set(rows["check"]) == {"spread"}
    assert "frame_path" not in rows.columns.difference(frames.columns)
    assert (rows["label"] == np.where(rows["ok"], PASS, FAIL)).all()


def test_head_uses_balanced_class_weights():
    clf = make_head().steps[-1][1]
    assert clf.class_weight == "balanced"


def test_spread_head_beats_majority_and_exports(tmp_path):
    frames, embs, vision = synthetic()
    card = train_check(frames, embs, vision, "spread", tmp_path)
    test = card["metrics"]["test"]
    assert test["balanced_accuracy"] > 0.8
    assert card["baselines"]["majority"]["balanced_accuracy"] == 0.5
    assert card["verdict"].startswith("Beats")
    assert card["baselines"]["overlay_scores"]["features"] == [
        "vision.spread.area_rel"
    ]
    assert 0 <= test["agreement"] <= 1
    assert card["label"]["positive"] == FAIL
    assert card["embedder_sha256"] == ["abc"]
    assert card["data"]["test"]["robots"] == ["pixi"]
    assert "quin" in card["data"]["holdout_note"]
    assert card["weighting"]["chosen"] in ("class", "robot_class")
    recall = card["intervals"]["metrics"]["recall_fail"]
    assert recall["ci95"][0] <= recall["value"] <= recall["ci95"][1]
    assert set(card["leave_one_robot_out"]) == {"pixi", "quin"}

    saved = json.loads((tmp_path / "model.json").read_text("utf-8"))
    assert saved["verdict"] == card["verdict"]
    sess = ort.InferenceSession(str(tmp_path / "model.onnx"))
    x = np.zeros((2, DIM), np.float32)
    x[1, 0] = 2 * SIGNAL
    label, proba = sess.run(None, {sess.get_inputs()[0].name: x})
    assert list(np.asarray(label).astype(str)) == [PASS, FAIL]
    assert proba.shape == (2, 2)


def test_push_off_uses_all_attempts_and_has_no_overlay(tmp_path):
    frames, embs, vision = synthetic()
    card = train_check(frames, embs, vision, "push_off", tmp_path)
    assert card["baselines"]["overlay_scores"] is None
    assert "overlay_scores_missing" in card["baselines"]
    counts = card["counts"]
    assert counts["train"][FAIL] == counts["train"][PASS] == TEST_FROM
    assert "test_last_attempt" in card["metrics"]


def test_verdict_flags_missed_and_few_failures():
    head = {
        "balanced_accuracy": 0.7,
        "f1_macro": 0.7,
        "per_class": {FAIL: {"recall": 3 / 7, "support": 7}},
    }
    majority = {"balanced_accuracy": 0.5, "f1_macro": 0.4}
    text = verdict(head, majority)
    assert text.startswith("Beats")
    assert "catches 3 of 7" in text
    assert "Only 7 failures" in text
    worse = verdict(head | {"balanced_accuracy": 0.5}, majority)
    assert worse.startswith("Does not beat")


def test_summary_table_has_one_row_per_check(tmp_path):
    frames, embs, vision = synthetic()
    cards = {
        c: train_check(frames, embs, vision, c, tmp_path / c)
        for c in ("spread", "push_off")
    }
    table = summary_table(cards)
    assert table.count("\n") == 2 + len(cards)
    assert "| push_off |" in table
