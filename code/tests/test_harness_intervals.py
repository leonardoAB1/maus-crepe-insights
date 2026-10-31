"""Run bootstrap intervals, robot views and time-ordered CV (ml env)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("skl2onnx")
pytest.importorskip("onnxruntime")

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from maus_classify.harness import (  # noqa: E402
    HarnessError,
    ci95,
    cross_val_proba,
    holdout_intervals,
    leave_one_robot_out,
    per_robot_metrics,
    run_bootstrap,
    run_starts,
    time_group_folds,
    train_and_evaluate,
)

FRAMES_PER_RUN = 2
N_BOOT = 200
START = np.datetime64("2026-10-13T10:00")


def labelled(n_runs: int, seed: int = 0, noise: float = 0.3):
    """Runs of two frames; labels follow the run, predictions are right
    except for a `noise` share of frames."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_runs), FRAMES_PER_RUN)
    y = np.where(rng.random(n_runs) < 0.3, "fail", "pass")[groups]
    flip = rng.random(len(y)) < noise
    pred = np.where(flip, np.where(y == "fail", "pass", "fail"), y)
    return y, pred, groups


def separable(n_runs: int, seed: int = 0):
    """Features with a planted signal, three robots in time order."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_runs), FRAMES_PER_RUN)
    y = np.where(rng.random(len(groups)) < 0.4, "fail", "pass")
    X = rng.normal(size=(len(y), 3))
    X[:, 0] += np.where(y == "fail", 3.0, 0.0)
    ts = START + groups.astype("timedelta64[h]")
    robots = np.array(["quin", "otto", "pixi"])[
        np.minimum(groups * 3 // n_runs, 2)
    ]
    return X, y, groups, ts, robots


def test_ci95_ignores_nan_and_handles_empty():
    assert ci95([1.0, math.nan, 1.0]) == [1.0, 1.0]
    assert all(math.isnan(v) for v in ci95([math.nan]))


def test_bootstrap_value_is_full_sample_and_inside_interval():
    y, pred, groups = labelled(80)
    out = holdout_intervals(y, pred, groups, positive="fail", n_boot=N_BOOT)
    m = out["metrics"]
    full = float(np.mean(pred[y == "fail"] == "fail"))
    assert m["recall_fail"]["value"] == pytest.approx(full)
    for entry in m.values():
        lo, hi = entry["ci95"]
        assert lo <= entry["value"] <= hi
    assert out["n_runs"] == 80


def test_interval_narrows_with_more_runs():
    def width(n: int) -> float:
        y, pred, groups = labelled(n, seed=1)
        lo, hi = holdout_intervals(y, pred, groups, n_boot=N_BOOT)["metrics"][
            "balanced_accuracy"
        ]["ci95"]
        return hi - lo

    assert width(400) < width(40)


def test_bootstrap_resamples_whole_runs_within_strata():
    groups = np.array([1, 1, 2, 2, 3, 3])
    strata = np.array(["a", "a", "a", "a", "b", "b"])
    seen = []

    def stat(idx: np.ndarray) -> dict[str, float]:
        seen.append(idx)
        return {"n_b": float((strata[idx] == "b").sum())}

    out = run_bootstrap(stat, groups, strata=strata, n_boot=50)
    for idx in seen:
        # each run contributes both of its rows, and stratum b keeps 1 run
        assert len(idx) % FRAMES_PER_RUN == 0
        assert (strata[idx] == "b").sum() == FRAMES_PER_RUN
    assert out["n_b"]["ci95"] == [2.0, 2.0]


def test_time_folds_respect_order_and_keep_runs_whole():
    groups = np.repeat(np.arange(30), FRAMES_PER_RUN)
    ts = START + (groups // 2).astype("timedelta64[h]")  # tied pairs
    starts = run_starts(groups, ts)
    folds = time_group_folds(starts, n_folds=4)
    assert len(folds) == 4
    for fit, val in folds:
        assert not set(fit) & set(val)
        assert starts[fit].max() < starts[val].min()
        # tied start times never straddle fit and validation
        assert not set(starts[fit]) & set(starts[val])
    with pytest.raises(HarnessError):
        time_group_folds(starts.iloc[:3], n_folds=4)


def test_cross_val_proba_is_out_of_fold():
    X, y, groups, ts, _ = separable(60)
    proba, classes = cross_val_proba(
        LogisticRegression(max_iter=500), X, y, groups, ts, n_folds=4
    )
    assert classes == ["fail", "pass"]
    has = ~np.isnan(proba).any(axis=1)
    first = groups < groups[has].min()
    assert first.any() and np.isnan(proba[first]).all()
    np.testing.assert_allclose(proba[has].sum(axis=1), 1.0)
    pred = np.array(classes)[proba[has].argmax(axis=1)]
    assert np.mean(pred == y[has]) > 0.8


def test_cross_val_proba_routes_sample_weight_into_pipeline():
    X, y, groups, ts, _ = separable(60)
    head = make_pipeline(StandardScaler(), LogisticRegression())
    w = np.where(y == "fail", 5.0, 1.0)
    weighted, _ = cross_val_proba(
        head, X, y, groups, ts, sample_weight=w, n_folds=4
    )
    plain, _ = cross_val_proba(head, X, y, groups, ts, n_folds=4)
    has = ~np.isnan(plain).any(axis=1)
    # up-weighting fails raises their predicted probability on average
    assert weighted[has, 0].mean() > plain[has, 0].mean()


def test_per_robot_and_leave_one_robot_out():
    X, y, groups, ts, robots = separable(90)
    model = LogisticRegression(max_iter=500).fit(X, y)
    per = per_robot_metrics(
        y, model.predict(X), model.predict_proba(X), robots, ["fail", "pass"]
    )
    assert set(per) == {"otto", "pixi", "quin"}
    assert sum(v["n"] for v in per.values()) == len(y)

    loro = leave_one_robot_out(
        LogisticRegression(max_iter=500),
        X,
        y,
        robots,
        groups=groups,
        positive="fail",
        score=lambda yt, p, cls: {"n_fail": int((yt == "fail").sum())},
        n_boot=N_BOOT,
    )
    for r, scores in loro.items():
        assert scores["roc_auc_fail"] > 0.8
        assert scores["n_fail"] == int(((robots == r) & (y == "fail")).sum())
        assert "recall_fail" in scores["intervals"]["metrics"]


def test_leave_one_robot_out_skips_one_class_train():
    X = np.zeros((4, 2))
    y = np.array(["fail", "fail", "pass", "pass"])
    robots = np.array(["a", "a", "b", "b"])
    out = leave_one_robot_out(LogisticRegression(), X, y, robots)
    assert out["a"]["skipped"] and out["b"]["n"] == 2


def test_card_names_holdout_robots(tmp_path):
    X, y, groups, ts, robots = separable(90)
    split = np.where(robots == "pixi", "test", "train")
    card = train_and_evaluate(
        LogisticRegression(max_iter=500),
        X,
        y,
        split,
        groups,
        name="dummy",
        feature_names=["a", "b", "c"],
        out_dir=tmp_path,
        timestamps=ts,
        robots=robots,
        positive="fail",
        n_boot=N_BOOT,
    )
    assert card["data"]["test"]["robots"] == ["pixi"]
    assert card["data"]["train"]["robots"] == ["otto", "quin"]
    assert "otto, quin" in card["data"]["holdout_note"]
    assert card["data"]["split_version"]
    assert set(card["metrics"]["per_robot"]["test"]) == {"pixi"}
    assert "within robot" in card["intervals"]["method"]
    assert "recall_fail" in card["intervals"]["metrics"]


def test_card_has_intervals_without_robots(tmp_path):
    X, y, groups, ts, _ = separable(60)
    split = np.where(groups < 45, "train", "test")
    card = train_and_evaluate(
        LogisticRegression(max_iter=500),
        X,
        y,
        split,
        groups,
        name="dummy",
        feature_names=["a", "b", "c"],
        out_dir=tmp_path,
        n_boot=N_BOOT,
    )
    assert "balanced_accuracy" in card["intervals"]["metrics"]
    assert "leave_one_robot_out" not in card
    assert isinstance(run_starts(groups, ts), pd.Series)


def test_robot_confound_is_exposed(tmp_path):
    """Features only identify the robot, and fail rates differ by robot.
    A pooled score looks fine, but no robot beats majority on its own, so
    the card must not claim to generalize."""
    rng = np.random.default_rng(3)
    n_runs = 150
    groups = np.repeat(np.arange(n_runs), FRAMES_PER_RUN)
    names = np.array(["quin", "otto", "pixi"])
    robots = names[groups % 3]
    fail_rate = {"quin": 0.1, "otto": 0.5, "pixi": 0.9}
    y = np.where(
        rng.random(len(groups)) < np.vectorize(fail_rate.get)(robots),
        "fail",
        "pass",
    )
    X = np.stack(
        [(robots == r).astype(float) for r in names], axis=1
    ) + rng.normal(scale=0.05, size=(len(y), 3))
    ts = START + groups.astype("timedelta64[h]")
    split = np.where(groups % 5 == 4, "test", "train")
    card = train_and_evaluate(
        LogisticRegression(max_iter=500),
        X,
        y,
        split,
        groups,
        name="confound",
        feature_names=["a", "b", "c"],
        out_dir=tmp_path,
        timestamps=ts,
        robots=robots,
        positive="fail",
        n_boot=N_BOOT,
    )
    assert card["metrics"]["test"]["balanced_accuracy"] > 0.6  # looks fine
    gen = card["generalization"]
    assert gen["generalizes"] is False
    assert gen["not_above_majority"]
