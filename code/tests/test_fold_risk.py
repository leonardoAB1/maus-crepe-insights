"""Offline tests for the fold-risk dataset and lift report."""

import math

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")  # lift metrics run in the ml environment

from maus_classify.fold_risk import (  # noqa: E402
    FAIL,
    PASS,
    SCORE_FEATURES,
    VERDICT_FEATURE,
    fold_risk_table,
    fold_target,
    lift_report,
    robot_breakdown,
    signal_verdict,
    verdict_baseline,
    wilson,
)


def frame(run_id, check, attempt, ok, last=True, split="train", robot="r1"):
    return {
        "run_id": run_id,
        "check": check,
        "attempt": attempt,
        "ok": ok,
        "is_last_attempt": last,
        "split": split,
        "robot": robot,
        "ts_utc": pd.Timestamp("2026-10-13T10:00Z")
        + pd.Timedelta(hours=run_id),
    }


FRAMES = pd.DataFrame(
    [
        # run 1: spread retried, fold1 fails once then passes -> fail
        frame(1, "spread", 1, False, last=False),
        frame(1, "spread", 2, True),
        frame(1, "fold1", 1, False, last=False),
        frame(1, "fold1", 2, True),
        frame(1, "fold2", 1, True),
        # run 2: clean -> ok
        frame(2, "spread", 1, True),
        frame(2, "fold1", 1, True),
        frame(2, "fold2", 1, True),
        # run 3: spread failed and the run stopped -> no target
        frame(3, "spread", 1, False),
        # run 4: fold2 fails, test split
        frame(4, "spread", 1, False, split="test", robot="r2"),
        frame(4, "fold1", 1, True, split="test", robot="r2"),
        frame(4, "fold2", 1, False, split="test", robot="r2"),
        # run 5: no vision scores -> dropped
        frame(5, "spread", 1, True),
        frame(5, "fold1", 1, True),
    ]
)


def vision(ids):
    rows = []
    for i in ids:
        row = {"id": i}
        for k, f in enumerate(SCORE_FEATURES):
            row[f"vision.spread.{f}"] = float(i + k)
        rows.append(row)
    return pd.DataFrame(rows)


def test_fold_target_any_fold_failure() -> None:
    assert fold_target(FRAMES).to_dict() == {
        1: True,
        2: False,
        4: True,
        5: False,
    }


def test_table_rows_features_and_labels() -> None:
    table = fold_risk_table(FRAMES, vision([1, 2, 3, 4]))
    assert table.index.tolist() == [1, 2, 4]
    assert table["label"].tolist() == [FAIL, PASS, FAIL]
    # Verdict of the last spread attempt, not the first
    assert table[VERDICT_FEATURE].tolist() == [1.0, 1.0, 0.0]
    assert table["split"].tolist() == ["train", "train", "test"]
    assert set(SCORE_FEATURES) <= set(table.columns)


def test_table_joins_last_spread_embedding() -> None:
    emb = pd.DataFrame(
        [
            {"run_id": 1, "check": "spread", "attempt": 1, "e0": 9.0},
            {"run_id": 1, "check": "spread", "attempt": 2, "e0": 1.0},
            {"run_id": 2, "check": "spread", "attempt": 1, "e0": 2.0},
        ]
    )
    table = fold_risk_table(FRAMES, vision([1, 2, 4]), emb)
    # Run 4 has no embedding and is dropped; run 1 uses attempt 2
    assert table["e0"].to_dict() == {1: 1.0, 2: 2.0}


def test_lift_report_perfect_and_constant() -> None:
    y = np.array([1, 0, 1, 0, 0, 0, 1, 0])
    good = lift_report(y, y * 0.9 + 0.05, n_boot=200)
    assert good["roc_auc"]["value"] == 1.0
    assert good["base_rate"] == pytest.approx(3 / 8)
    assert good["ap_lift"]["value"] == pytest.approx(8 / 3)
    assert good["roc_auc"]["ci"][0] <= 1.0 <= good["roc_auc"]["ci"][1] + 1e-9
    assert signal_verdict(good).startswith("signal")

    flat = lift_report(y, np.full(len(y), 0.5), n_boot=200)
    assert flat["roc_auc"]["value"] == 0.5
    assert signal_verdict(flat).startswith("no clear signal")


def test_wilson_interval() -> None:
    lo, hi = wilson(8, 20)
    assert lo < 0.4 < hi
    assert all(math.isnan(v) for v in wilson(0, 0))


def test_verdict_baseline_uses_train_rates() -> None:
    table = fold_risk_table(FRAMES, vision([1, 2, 4]))
    # No train run had a failed spread verdict -> train overall rate
    assert verdict_baseline(table).tolist() == [0.5]


def test_robot_breakdown_hides_names() -> None:
    table = fold_risk_table(FRAMES, vision([1, 2, 4]))
    rows = robot_breakdown(table)
    assert {r["robot"] for r in rows} == {"A", "B"}
    assert all("r1" not in str(r) and "r2" not in str(r) for r in rows)
