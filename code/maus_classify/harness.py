"""Shared train and evaluate harness for every frame classifier.

It does not care where features come from: pass a feature matrix, labels,
the fixed holdout split and the run id of each row. It fits on train, scores
on test, exports to ONNX, checks onnxruntime against the fitted model and
writes `model.onnx` and `model.json` (see code/models/README.md).

Needs the ml environment: `pixi run -e ml ...`.
"""

from __future__ import annotations

import math
import subprocess
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import pandas as pd
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType
from sklearn.base import ClassifierMixin, clone
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from maus_export.doe_analyze import dump_json

from .split import TEST, TRAIN

ONNX_INPUT = "x"
ONNX_LABEL = "label"
ONNX_PROBA = "probabilities"
OPSET = 17
ATOL = 1e-4  # float32 graph vs float64 sklearn
CAL_BINS = 10
N_BOOT = 1000
CI_LEVEL = 0.95
SEED = 0
N_FOLDS = 5
MIN_CLASSES = 2


class HarnessError(ValueError):
    """Bad harness input."""


def _check_inputs(
    X: np.ndarray, y: np.ndarray, split: np.ndarray, groups: np.ndarray
) -> None:
    if X.ndim != 2:
        raise HarnessError("features must be a 2-D array")
    if not len(X) == len(y) == len(split) == len(groups):
        raise HarnessError("features, labels, split and groups differ in size")
    if not set(split) <= {TRAIN, TEST}:
        raise HarnessError(f"split values must be {TRAIN!r} or {TEST!r}")
    if not ((split == TRAIN).any() and (split == TEST).any()):
        raise HarnessError("need rows in both train and test")
    sides = pd.Series(split).groupby(pd.Series(groups)).nunique()
    if (sides > 1).any():
        raise HarnessError("a run has rows in both train and test")
    if not np.isfinite(X).all():
        raise HarnessError("features must be finite; impute before training")


def calibration(
    y_true: np.ndarray, proba: np.ndarray, classes: list[Any]
) -> dict[str, Any]:
    """One-vs-rest Brier score, expected calibration error and reliability
    bins for each class."""
    out: dict[str, Any] = {}
    edges = np.linspace(0.0, 1.0, CAL_BINS + 1)
    briers = []
    for k, cls in enumerate(classes):
        hit = (y_true == cls).astype(float)
        p = proba[:, k]
        brier = float(np.mean((p - hit) ** 2))
        briers.append(brier)
        which = np.clip(np.digitize(p, edges[1:-1]), 0, CAL_BINS - 1)
        ece = sum(
            abs(hit[which == b].mean() - p[which == b].mean())
            * (which == b).mean()
            for b in range(CAL_BINS)
            if (which == b).any()
        )
        frac, mean = (
            calibration_curve(hit, p, n_bins=CAL_BINS)
            if 0 < hit.sum() < len(hit)
            else (np.array([]), np.array([]))
        )
        out[str(cls)] = {
            "brier": brier,
            "ece": float(ece),
            "bins": {
                "mean_predicted": mean.tolist(),
                "fraction_positive": frac.tolist(),
            },
        }
    return {"per_class": out, "brier_mean": float(np.mean(briers))}


def metrics(
    y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray, classes: list
) -> dict[str, Any]:
    """Holdout metrics shared by every classifier."""
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    return {
        "n": len(y_true),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(
            f1_score(
                y_true, y_pred, labels=classes, average="macro", zero_division=0
            )
        ),
        "per_class": {
            str(c): {
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, c in enumerate(classes)
        },
        "confusion_matrix": {
            "labels": [str(c) for c in classes],
            "rows_true_cols_pred": confusion_matrix(
                y_true, y_pred, labels=classes
            ).tolist(),
        },
        "calibration": calibration(y_true, proba, classes),
    }


def ci95(values: Any) -> list[float]:
    """Central CI_LEVEL interval of the finite values, NaN when none."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return [math.nan, math.nan]
    tail = (1 - CI_LEVEL) / 2
    return [float(np.quantile(v, tail)), float(np.quantile(v, 1 - tail))]


def run_bootstrap(
    stat: Callable[[np.ndarray], dict[str, float]],
    groups: np.ndarray,
    *,
    strata: np.ndarray | None = None,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict[str, dict[str, Any]]:
    """Point value and interval of each statistic, resampling whole runs.

    `stat(idx)` scores the rows `idx` and returns named floats. `groups`
    holds the run id of each row. With `strata` (one value per row, constant
    within a run, e.g. the robot) runs are resampled within each stratum, so
    every draw keeps the robot mix.
    """
    groups = np.asarray(groups)
    runs = pd.unique(groups)
    rows_of = {g: np.flatnonzero(groups == g) for g in runs}
    if strata is None:
        pools = [runs]
    else:
        strata = np.asarray(strata)
        of_run = np.array([strata[rows_of[g][0]] for g in runs])
        pools = [runs[of_run == s] for s in pd.unique(of_run)]
    point = stat(np.arange(len(groups)))
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {k: [] for k in point}
    for _ in range(n_boot):
        picked = np.concatenate([rng.choice(p, len(p)) for p in pools])
        idx = np.concatenate([rows_of[g] for g in picked])
        for k, v in stat(idx).items():
            draws[k].append(v)
    return {
        k: {"value": float(point[k]), "ci95": ci95(draws[k])} for k in point
    }


def holdout_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    *,
    strata: np.ndarray | None = None,
    positive: str | None = None,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict[str, Any]:
    """Run-bootstrap intervals for balanced accuracy, macro F1 and, given
    a positive class, its recall. A draw missing a class scores NaN for the
    metrics that need it, and NaN draws are left out of the interval."""
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)
    labels = sorted(set(y_true) | set(y_pred))

    def stat(idx: np.ndarray) -> dict[str, float]:
        t, p = y_true[idx], y_pred[idx]
        out = {"balanced_accuracy": math.nan, "f1_macro": math.nan}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if len(set(t)) >= MIN_CLASSES:
                out["balanced_accuracy"] = balanced_accuracy_score(t, p)
                out["f1_macro"] = f1_score(
                    t, p, labels=labels, average="macro", zero_division=0
                )
            if positive is not None:
                out[f"recall_{positive}"] = (
                    recall_score(t, p, labels=[positive], average="macro")
                    if (t == positive).any()
                    else math.nan
                )
        return out

    return {
        "method": "bootstrap of runs"
        + (" within robot" if strata is not None else ""),
        "level": CI_LEVEL,
        "n_boot": n_boot,
        "n_runs": int(len(pd.unique(np.asarray(groups)))),
        "metrics": run_bootstrap(
            stat, groups, strata=strata, n_boot=n_boot, seed=seed
        ),
    }


def _quiet_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray, classes: list
) -> dict[str, Any]:
    """metrics() on subsets that may hold one class only."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return metrics(y_true, y_pred, proba, classes)


def per_robot_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray,
    robots: np.ndarray,
    classes: list,
) -> dict[str, Any]:
    """metrics() for the rows of each robot."""
    robots = np.asarray(robots).astype(str)
    return {
        r: _quiet_metrics(
            y_true[robots == r],
            y_pred[robots == r],
            proba[robots == r],
            classes,
        )
        for r in sorted(set(robots))
    }


def _fit(
    estimator: ClassifierMixin,
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> ClassifierMixin:
    """Fit a fresh clone, routing sample weights to a pipeline's last step."""
    model = clone(estimator)
    if sample_weight is None:
        return model.fit(X, y)
    if isinstance(model, Pipeline):
        key = f"{model.steps[-1][0]}__sample_weight"
        return model.fit(X, y, **{key: sample_weight})
    return model.fit(X, y, sample_weight=sample_weight)


def leave_one_robot_out(
    estimator: ClassifierMixin,
    X: np.ndarray,
    y: np.ndarray,
    robots: np.ndarray,
    *,
    groups: np.ndarray | None = None,
    positive: str | None = None,
    score: Callable[[np.ndarray, np.ndarray, list[str]], dict[str, Any]]
    | None = None,
    n_boot: int = N_BOOT,
) -> dict[str, Any]:
    """Train on every other robot, score the held-out one. A transfer check
    that ignores the time split; a robot is skipped when its train side has
    one class only.

    With `positive`, adds its ROC AUC when the held-out robot has both
    classes. `score(y_true, proba, classes)` adds extra named scores."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y).astype(str)
    robots = np.asarray(robots).astype(str)
    out: dict[str, Any] = {}
    for r in sorted(set(robots)):
        test = robots == r
        train = ~test
        if len(set(y[train])) < MIN_CLASSES:
            out[r] = {"n": int(test.sum()), "skipped": "one class in train"}
            continue
        model = _fit(estimator, X[train], y[train])
        classes = [str(c) for c in model.classes_]
        pred = model.predict(X[test]).astype(str)
        proba = model.predict_proba(X[test])
        scores = _quiet_metrics(y[test], pred, proba, classes)
        scores["n_train"] = int(train.sum())
        if positive in classes and len(set(y[test])) >= MIN_CLASSES:
            scores[f"roc_auc_{positive}"] = float(
                roc_auc_score(
                    y[test] == positive, proba[:, classes.index(positive)]
                )
            )
        if score is not None:
            scores.update(score(y[test], proba, classes))
        if groups is not None:
            scores["intervals"] = holdout_intervals(
                y[test],
                pred,
                np.asarray(groups)[test],
                positive=positive,
                n_boot=n_boot,
            )
        out[r] = scores
    return out


def run_starts(groups: np.ndarray, timestamps: np.ndarray) -> pd.Series:
    """Earliest timestamp of each run, indexed by run id."""
    ts = pd.to_datetime(pd.Series(np.asarray(timestamps)), utc=True)
    return ts.groupby(np.asarray(groups)).min()


def time_group_folds(
    starts: pd.Series, n_folds: int = N_FOLDS
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds over runs ordered by start time.

    Distinct start times are cut into `n_folds + 1` consecutive blocks, so
    runs with tied start times always share a block. Fold k fits on blocks
    0..k and validates on block k + 1. Returns (fit run ids, validation run
    ids) per fold; the first block is never validated.
    """
    times = np.sort(starts.dropna().unique())
    if len(times) < n_folds + 1:
        raise HarnessError("too few distinct run start times for the folds")
    blocks = np.array_split(times, n_folds + 1)
    block_of = {t: b for b, chunk in enumerate(blocks) for t in chunk}
    block = starts.map(block_of).to_numpy()
    ids = starts.index.to_numpy()
    return [(ids[block <= k], ids[block == k + 1]) for k in range(n_folds)]


def cross_val_proba(
    estimator: ClassifierMixin,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    timestamps: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    n_folds: int = N_FOLDS,
) -> tuple[np.ndarray, list[str]]:
    """Out-of-fold class probabilities from time-ordered, run-grouped folds.

    Pass train rows only; the holdout stays untouched for the final score.
    Rows in the first time block, or in a fold whose fit side has one class,
    get NaN. Returns (probabilities, classes) with columns in class order.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y).astype(str)
    groups = np.asarray(groups)
    classes = sorted(set(y))
    proba = np.full((len(y), len(classes)), np.nan)
    for fit_runs, val_runs in time_group_folds(
        run_starts(groups, timestamps), n_folds
    ):
        fit, val = np.isin(groups, fit_runs), np.isin(groups, val_runs)
        if len(set(y[fit])) < MIN_CLASSES or not val.any():
            continue
        w = None if sample_weight is None else np.asarray(sample_weight)[fit]
        model = _fit(estimator, X[fit], y[fit], w)
        cols = [classes.index(str(c)) for c in model.classes_]
        rows = np.flatnonzero(val)
        proba[rows] = 0.0  # classes the fold never saw get probability 0
        proba[np.ix_(rows, cols)] = model.predict_proba(X[val])
    return proba, classes


def _data_range(
    mask: np.ndarray, groups: np.ndarray, timestamps: np.ndarray | None
) -> dict[str, Any]:
    ids = groups[mask]
    out: dict[str, Any] = {
        "n_rows": int(mask.sum()),
        "n_runs": int(len(set(ids))),
        "run_id_min": int(ids.min()),
        "run_id_max": int(ids.max()),
    }
    if timestamps is not None:
        ts = pd.to_datetime(pd.Series(timestamps[mask]), utc=True).dropna()
        if not ts.empty:
            out["ts_utc_min"] = ts.min().isoformat()
            out["ts_utc_max"] = ts.max().isoformat()
    return out


def _commit() -> str | None:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        )
    except OSError:
        return None
    return res.stdout.strip() or None


def export_onnx(
    model: ClassifierMixin, n_features: int, path: Path
) -> ort.InferenceSession:
    """Write the fitted classifier as ONNX and open it in onnxruntime."""
    onx = to_onnx(
        model,
        initial_types=[(ONNX_INPUT, FloatTensorType([None, n_features]))],
        target_opset=OPSET,
        options={id(model): {"zipmap": False}},
    )
    outputs = [o.name for o in onx.graph.output]
    if outputs != [ONNX_LABEL, ONNX_PROBA]:
        raise HarnessError(f"unexpected ONNX outputs {outputs}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(onx.SerializeToString())
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def parity(
    model: ClassifierMixin, sess: ort.InferenceSession, X: np.ndarray
) -> None:
    """Fail unless onnxruntime matches the fitted model on these rows."""
    x32 = X.astype(np.float32)
    label, proba = sess.run([ONNX_LABEL, ONNX_PROBA], {ONNX_INPUT: x32})
    want = model.predict_proba(x32)
    np.testing.assert_allclose(proba, want, atol=ATOL)
    np.testing.assert_array_equal(
        np.asarray(label).astype(str), model.predict(x32).astype(str)
    )


def train_and_evaluate(
    estimator: ClassifierMixin,
    X: np.ndarray,
    y: np.ndarray,
    split: np.ndarray,
    groups: np.ndarray,
    *,
    name: str,
    feature_names: list[str],
    out_dir: Path,
    timestamps: np.ndarray | None = None,
    description: str = "",
    robots: np.ndarray | None = None,
    positive: str | None = None,
    n_boot: int = N_BOOT,
) -> dict[str, Any]:
    """Fit on train rows, score on test rows, export and check ONNX, write
    `model.onnx` and `model.json` in `out_dir`. Returns the model card.

    The card always carries run-bootstrap intervals for the holdout. With
    `robots` (one per row) it also names the robots each side covers and
    adds per-robot holdout metrics and a leave-one-robot-out check. With
    `positive` the intervals include that class's recall."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(str)
    split = np.asarray(split)
    groups = np.asarray(groups)
    _check_inputs(X, y, split, groups)
    if len(feature_names) != X.shape[1]:
        raise HarnessError("one feature name per column is required")
    train, test = split == TRAIN, split == TEST

    model = clone(estimator).fit(X[train].astype(np.float32), y[train])
    classes = [str(c) for c in model.classes_]
    x_test = X[test].astype(np.float32)
    proba = model.predict_proba(x_test)
    pred = model.predict(x_test).astype(str)

    sess = export_onnx(model, X.shape[1], out_dir / "model.onnx")
    parity(model, sess, X[test])

    card: dict[str, Any] = {
        "name": name,
        "description": description,
        "estimator": type(estimator).__name__,
        "inputs": [
            {
                "name": ONNX_INPUT,
                "dtype": "float32",
                "shape": [None, X.shape[1]],
                "features": list(feature_names),
            }
        ],
        "outputs": [
            {"name": ONNX_LABEL, "dtype": "string", "shape": [None]},
            {
                "name": ONNX_PROBA,
                "dtype": "float32",
                "shape": [None, len(classes)],
                "classes": classes,
            },
        ],
        "data": {
            "split": "chronological, run-grouped holdout (frames.split)",
            TRAIN: _data_range(train, groups, timestamps),
            TEST: _data_range(test, groups, timestamps),
        },
        "metrics": {TEST: metrics(y[test], pred, proba, classes)},
        "onnx": {"opset": OPSET, "parity_atol": ATOL, "parity": "ok"},
        "commit": _commit(),
    }
    strata = None
    if robots is not None:
        robots = np.asarray(robots).astype(str)
        if len(robots) != len(y):
            raise HarnessError("one robot per row is required")
        strata = robots[test]
        for side, mask in ((TRAIN, train), (TEST, test)):
            card["data"][side]["robots"] = sorted(set(robots[mask]))
        covered = ", ".join(card["data"][TEST]["robots"])
        card["data"]["holdout_note"] = (
            f"The holdout covers robot(s) {covered} only. Robots ran in "
            f"separate time windows, so robot also stands for era."
        )
        card["metrics"]["per_robot"] = {
            TEST: per_robot_metrics(y[test], pred, proba, strata, classes)
        }
        card["leave_one_robot_out"] = leave_one_robot_out(
            estimator,
            X,
            y,
            robots,
            groups=groups,
            positive=positive,
            n_boot=n_boot,
        )
    card["intervals"] = holdout_intervals(
        y[test],
        pred,
        groups[test],
        strata=strata,
        positive=positive,
        n_boot=n_boot,
    )
    dump_json(card, out_dir / "model.json")
    return card
