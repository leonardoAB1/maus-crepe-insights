"""Fold state from a single frame: flat, half or quarter (challenge Task 3).

A logistic-regression head on the frozen frame embeddings (#90), trained
and exported with the shared harness. Labels are the weak `fold_class`
column of the frame table: a passed spread check is flat, a passed fold1 is
half, a passed fold2 is quarter. Failed checks have no label.

Needs the ml environment: `pixi run -e ml fold-class`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from maus_export.doe_analyze import dump_json

from .harness import ONNX_INPUT, ONNX_PROBA, train_and_evaluate
from .split import TEST, TRAIN

NAME = "fold-class"
CLASSES = ["flat", "half", "quarter"]
LABEL = "fold_class"
KEY = ["run_id", "check", "attempt"]
FRAMES = Path("training") / "frames.parquet"
EMBEDDINGS = Path("training") / "frame_embeddings.parquet"
EMBED_RE = re.compile(r"^e\d+$")
FOLD_CHECKS = ["spread", "fold1", "fold2"]
# push_off frames show the finished quarter fold still in the pan, but the
# head never trains on that check: a check-independent test of the shape
SHAPE_CHECK = ("push_off", "quarter")
EMPTY_CHECK = "plate_clean"  # empty pan, outside the three classes
C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
# Pick the strongest regularization whose CV loss is within this share of
# the best: the loss curve is flat at the top, and a smaller C is less
# overconfident
LOSS_TOL = 0.05
CV_FOLDS = 5
MAX_ITER = 5000
SEED = 0


class FoldClassError(ValueError):
    """Missing or unusable input."""


def embedding_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if EMBED_RE.match(c)]
    return sorted(cols, key=lambda c: int(c[1:]))


def load_frames(data_dir: Path) -> pd.DataFrame:
    """Frame table joined with embeddings on (run_id, check, attempt).
    Frames without an embedding (missing image) are dropped."""
    for rel in (FRAMES, EMBEDDINGS):
        if not (data_dir / rel).is_file():
            raise FoldClassError(f"missing {rel}; build it first")
    frames = pd.read_parquet(data_dir / FRAMES)
    emb = pd.read_parquet(data_dir / EMBEDDINGS)
    cols = embedding_columns(emb)
    if not cols:
        raise FoldClassError("no e<j> embedding columns")
    return frames.merge(emb[KEY + cols], on=KEY, how="inner")


def build_estimator(c: float) -> Pipeline:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, max_iter=MAX_ITER, random_state=SEED),
    )


def choose_c(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray
) -> tuple[float, dict[str, float]]:
    """Regularization by run-grouped CV log loss, on train rows only.
    Log loss rewards calibrated probabilities, not just right labels."""
    folds = GroupKFold(n_splits=min(CV_FOLDS, len(set(groups))))
    scores: dict[str, float] = {}
    for c in C_GRID:
        losses = []
        for tr, va in folds.split(X, y, groups):
            model = build_estimator(c).fit(X[tr], y[tr])
            proba = model.predict_proba(X[va])
            losses.append(log_loss(y[va], proba, labels=model.classes_))
        scores[str(c)] = float(np.mean(losses))
    floor = min(scores.values()) * (1 + LOSS_TOL)
    chosen = min(c for c in C_GRID if scores[str(c)] <= floor)
    return chosen, scores


def predict(sess: ort.InferenceSession, X: np.ndarray) -> np.ndarray:
    return sess.run([ONNX_PROBA], {ONNX_INPUT: X.astype(np.float32)})[0]


def confidence_summary(
    y: np.ndarray, proba: np.ndarray, classes: list[str]
) -> dict[str, Any]:
    """Mean top-class confidence against accuracy: the gap shows over- or
    under-confidence at a glance, overall and per predicted class."""
    pred = np.asarray(classes)[proba.argmax(axis=1)]
    conf = proba.max(axis=1)
    hit = pred == y
    per_class = {}
    for cls in classes:
        mask = pred == cls
        if mask.any():
            per_class[cls] = {
                "n_predicted": int(mask.sum()),
                "mean_confidence": float(conf[mask].mean()),
                "accuracy": float(hit[mask].mean()),
            }
    return {
        "accuracy": float(hit.mean()),
        "mean_confidence": float(conf.mean()),
        "per_true_class_accuracy": {
            cls: float(hit[y == cls].mean())
            for cls in classes
            if (y == cls).any()
        },
        "per_predicted_class": per_class,
    }


def _shares(proba: np.ndarray, classes: list[str]) -> dict[str, Any]:
    pred = np.asarray(classes)[proba.argmax(axis=1)]
    return {
        "n": len(pred),
        "predicted_share": {c: float((pred == c).mean()) for c in classes},
        "mean_confidence": float(proba.max(axis=1).mean()),
    }


def sanity_checks(
    sess: ort.InferenceSession,
    df: pd.DataFrame,
    cols: list[str],
    classes: list[str],
) -> dict[str, Any]:
    """Predictions on holdout frames the head never saw a label for."""
    test = df[df["split"] == TEST]
    out: dict[str, Any] = {}
    check, expected = SHAPE_CHECK
    shape = test[(test["check"] == check) & test["ok"]]
    if len(shape):
        res = _shares(predict(sess, shape[cols].to_numpy()), classes)
        res["expected"] = expected
        out[f"{check}_passed"] = res
    failed = test[test["check"].isin(FOLD_CHECKS) & ~test["ok"]]
    out["failed_fold_checks"] = {
        name: _shares(predict(sess, g[cols].to_numpy()), classes)
        for name, g in failed.groupby("check")
    }
    empty = test[test["check"] == EMPTY_CHECK]
    if len(empty):
        out[f"{EMPTY_CHECK}_out_of_scope"] = _shares(
            predict(sess, empty[cols].to_numpy()), classes
        )
    return out


CAVEATS = {
    "triangle_fold": (
        "Unsupported. No frame in the data shows a triangle fold, so the "
        "head can only answer flat, half or quarter. Human fold labels "
        "from the labelling workstream (#87) are the path to add it, but "
        "its visual taxonomy (crepe-visual-v2) has no triangle label yet; "
        "the follow-up (#106) depends on adding one."
    ),
    "label_leakage": (
        "Weak labels: the class equals the check that took the frame "
        "(spread, fold1, fold2). The head may learn check-specific camera "
        "timing or plate state instead of the fold shape. See "
        "sanity_checks: push_off frames show a quarter fold from a check "
        "the head never trained on. The stronger check against human "
        "labels on video frames is tracked in #104."
    ),
    "no_crepe": (
        "There is no 'no crepe' class. Empty pans (plate_clean frames) "
        "come out as quarter with near-certain confidence, so quarter "
        "partly means 'little crepe visible'. High confidence is not a "
        "presence check: only run the head on frames known to show a "
        "crepe (spread and fold checks)."
    ),
    "failed_checks": (
        "Failed spread/fold checks are excluded from training because "
        "their true state is unknown."
    ),
}


def train_fold_class(data_dir: Path, out_dir: Path) -> dict[str, Any]:
    """Fit, score and export the fold head; returns the model card."""
    df = load_frames(data_dir)
    cols = embedding_columns(df)
    labeled = df[df[LABEL].isin(CLASSES)].reset_index(drop=True)
    if labeled.empty:
        raise FoldClassError("no labeled fold frames")
    X = labeled[cols].to_numpy(np.float64)
    y = labeled[LABEL].astype(str).to_numpy()
    split = labeled["split"].to_numpy()
    groups = labeled["run_id"].to_numpy()

    train = split == TRAIN
    c, cv_scores = choose_c(X[train], y[train], groups[train])
    card = train_and_evaluate(
        build_estimator(c),
        X,
        y,
        split,
        groups,
        name=NAME,
        feature_names=cols,
        out_dir=out_dir,
        timestamps=labeled["ts_utc"].to_numpy(),
        description=(
            "Fold state of the crepe from one raw check frame: flat, half "
            "or quarter, with probabilities. Logistic regression on the "
            "frozen frame-embedder output (code/models/frame-embedder)."
        ),
    )

    classes = card["outputs"][1]["classes"]
    sess = ort.InferenceSession(
        str(out_dir / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    test = split == TEST
    card["labels"] = {
        "column": f"frames.{LABEL}",
        "rule": "spread ok -> flat, fold1 ok -> half, fold2 ok -> quarter",
        "counts": {
            side: labeled.loc[mask, LABEL].value_counts().to_dict()
            for side, mask in ((TRAIN, train), (TEST, test))
        },
    }
    card["features"] = {
        "source": "data/training/frame_embeddings.parquet",
        "embedder": "code/models/frame-embedder (preprocess first)",
        "join": KEY,
    }
    card["selection"] = {
        "param": "LogisticRegression C",
        "chosen": c,
        "cv": f"{CV_FOLDS}-fold run-grouped, train rows only",
        "rule": f"smallest C within {LOSS_TOL:.0%} of the best CV log loss",
        "cv_log_loss": cv_scores,
    }
    card["metrics"][TEST]["confidence"] = confidence_summary(
        y[test], predict(sess, X[test]), classes
    )
    card["sanity_checks"] = sanity_checks(sess, df, cols, classes)
    card["caveats"] = CAVEATS
    card["out_of_scope"] = "edge angle for plate alignment"
    dump_json(card, out_dir / "model.json")
    return card
