"""Shared train and evaluate harness for every frame classifier.

It does not care where features come from: pass a feature matrix, labels,
the fixed holdout split and the run id of each row. It fits on train, scores
on test, exports to ONNX, checks onnxruntime against the fitted model and
writes `model.onnx` and `model.json` (see code/models/README.md).

Needs the ml environment: `pixi run -e ml ...`.
"""

from __future__ import annotations

import subprocess
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
)

from maus_export.doe_analyze import dump_json

from .split import TEST, TRAIN

ONNX_INPUT = "x"
ONNX_LABEL = "label"
ONNX_PROBA = "probabilities"
OPSET = 17
ATOL = 1e-4  # float32 graph vs float64 sklearn
CAL_BINS = 10


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
) -> dict[str, Any]:
    """Fit on train rows, score on test rows, export and check ONNX, write
    `model.onnx` and `model.json` in `out_dir`. Returns the model card."""
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
    dump_json(card, out_dir / "model.json")
    return card
