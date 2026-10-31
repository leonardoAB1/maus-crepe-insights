"""Pass or fail verdict per check frame, one logistic head per check.

Each head reads a frame embedding (training/frame_embeddings.parquet) and
predicts the robot's own verdict for that frame (frames.ok). The labels are
the robot's verdicts, so they are weak: the model learns to agree with the
robot, not to judge crepe quality.

Every frame attempt is a training row, because each attempt has its own
verdict (push_off fails once by design and is retried). Holdout metrics are
reported on all attempts and on last attempts only, which is what the run's
final verdict and the overlay baseline use.

Needs the ml environment: `pixi run -e ml ...`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from maus_export.doe_analyze import dump_json

from .harness import metrics, train_and_evaluate
from .split import TEST, TRAIN

CHECKS = ("spread", "fold1", "fold2", "push_off")
PASS, FAIL = "pass", "fail"
KEYS = ["run_id", "check", "attempt"]
EMBEDDING_COLUMN = re.compile(r"^e(\d+)$")
VISION_PREFIX = "vision."
VISION_ID = "id"
# L2 strength of the heads; a few hundred rows against hundreds of features
HEAD_C = 0.1
MAX_ITER = 2000
WEAK_LABELS = (
    "Labels are the robot's own check verdicts (weak labels). Metrics measure "
    "agreement with the robot, not verified crepe quality."
)
OVERLAY_NOTE = (
    "Overlay scores are measured on the robot's overlay image, which is drawn "
    "from the same detection that produced the verdict, so this baseline is "
    "close to an upper bound."
)


class CheckVerdictError(ValueError):
    """Input problem with a fixed message that is safe to print."""


def embedding_columns(df: pd.DataFrame) -> list[str]:
    """Embedding columns e0..eN in numeric order."""
    cols = [c for c in df.columns if EMBEDDING_COLUMN.match(str(c))]
    if not cols:
        raise CheckVerdictError("no embedding columns e0..eN found")
    return sorted(cols, key=lambda c: int(EMBEDDING_COLUMN.match(c)[1]))


def overlay_columns(vision: pd.DataFrame, check: str) -> list[str]:
    """vision.parquet score columns for one check (none for push_off)."""
    prefix = f"{VISION_PREFIX}{check}."
    return [c for c in vision.columns if str(c).startswith(prefix)]


def check_rows(
    frames: pd.DataFrame, embeddings: pd.DataFrame, check: str
) -> pd.DataFrame:
    """Labelled rows of one check with their embedding, sorted by key."""
    rows = frames[(frames["check"] == check) & frames["ok"].notna()]
    emb = embeddings[[*KEYS, *embedding_columns(embeddings)]]
    rows = rows.merge(emb, on=KEYS, how="inner", validate="1:1")
    rows = rows.sort_values(KEYS, kind="stable").reset_index(drop=True)
    rows["label"] = np.where(rows["ok"].astype(bool), PASS, FAIL)
    return rows


def make_head(c: float = HEAD_C) -> Pipeline:
    """Scaled logistic regression with class weights for the imbalance."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", C=c, max_iter=MAX_ITER),
    )


def _score(
    model: ClassifierMixin, X: np.ndarray, y: np.ndarray
) -> dict[str, Any]:
    """Harness metrics plus plain agreement with the robot verdict."""
    x32 = X.astype(np.float32)
    pred = model.predict(x32).astype(str)
    out = metrics(
        y, pred, model.predict_proba(x32), [str(c) for c in model.classes_]
    )
    out["agreement"] = float(np.mean(pred == y))
    return out


def _fit_score(
    estimator: ClassifierMixin,
    X: np.ndarray,
    y: np.ndarray,
    split: np.ndarray,
) -> dict[str, Any] | None:
    """Fit on train rows and score on test rows; None if a side is empty
    or train has a single class."""
    train, test = split == TRAIN, split == TEST
    if not test.any() or len(set(y[train])) < 2:
        return None
    model = clone(estimator).fit(X[train].astype(np.float32), y[train])
    return _score(model, X[test], y[test])


def overlay_baseline(
    rows: pd.DataFrame, vision: pd.DataFrame, check: str
) -> dict[str, Any] | None:
    """Logistic regression on the overlay scores, last attempts only."""
    cols = overlay_columns(vision, check)
    if not cols:
        return None
    last = rows[rows["is_last_attempt"]]
    joined = last.merge(
        vision[[VISION_ID, *cols]],
        left_on="run_id",
        right_on=VISION_ID,
        how="inner",
    )
    if joined.empty:
        return None
    X = joined[cols].to_numpy(np.float64)
    keep = ~np.isnan(X).all(axis=1)
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=MAX_ITER),
    )
    scored = _fit_score(
        model,
        X[keep],
        joined["label"].to_numpy(str)[keep],
        joined["split"].to_numpy()[keep],
    )
    if scored is not None:
        scored["features"] = cols
        scored["note"] = OVERLAY_NOTE
    return scored


def verdict(head: dict[str, Any], majority: dict[str, Any]) -> str:
    """One sentence on whether the head beats the majority baseline."""
    ba, ba0 = head["balanced_accuracy"], majority["balanced_accuracy"]
    f1, f10 = head["f1_macro"], majority["f1_macro"]
    if ba > ba0 and f1 > f10:
        return (
            f"Beats the majority baseline on the holdout: balanced accuracy "
            f"{ba:.2f} vs {ba0:.2f}, macro F1 {f1:.2f} vs {f10:.2f}."
        )
    return (
        f"Does not beat the majority baseline on the holdout: balanced "
        f"accuracy {ba:.2f} vs {ba0:.2f}, macro F1 {f1:.2f} vs {f10:.2f}. "
        f"Do not rely on this head for {FAIL!r} calls."
    )


def train_check(
    frames: pd.DataFrame,
    embeddings: pd.DataFrame,
    vision: pd.DataFrame,
    check: str,
    out_dir: Path,
) -> dict[str, Any]:
    """Train, evaluate and export the head for one check. Writes
    `model.onnx` and `model.json` in `out_dir` and returns the card."""
    rows = check_rows(frames, embeddings, check)
    if rows.empty:
        raise CheckVerdictError(f"no labelled frames with embeddings: {check}")
    features = embedding_columns(embeddings)
    X = rows[features].to_numpy(np.float64)
    y = rows["label"].to_numpy(str)
    split = rows["split"].to_numpy()
    last = rows["is_last_attempt"].to_numpy(bool)

    card = train_and_evaluate(
        make_head(),
        X,
        y,
        split,
        rows["run_id"].to_numpy(),
        name=f"check-verdict-{check}",
        feature_names=features,
        out_dir=out_dir,
        timestamps=rows["ts_utc"].to_numpy(),
        description=(
            f"Robot pass/fail verdict for the {check} check frame, from the "
            f"frame embedding. {WEAK_LABELS}"
        ),
    )
    # The harness exported the head fitted on train; refit the same way to
    # add agreement and the last-attempt view without touching its card
    train = split == TRAIN
    head = clone(make_head()).fit(X[train].astype(np.float32), y[train])
    test = split == TEST
    card["metrics"][TEST]["agreement"] = _score(head, X[test], y[test])[
        "agreement"
    ]
    if (test & last).any():
        card["metrics"]["test_last_attempt"] = _score(
            head, X[test & last], y[test & last]
        )
    majority = _fit_score(
        DummyClassifier(strategy="most_frequent"), X, y, split
    )
    card["check"] = check
    card["label"] = {
        "positive": FAIL,
        "source": "frames.ok (robot verdict per frame attempt)",
        "rows": "all attempts with a verdict and an embedding",
        "weak": WEAK_LABELS,
    }
    card["counts"] = {
        side: {
            lab: int(((split == side) & (y == lab)).sum())
            for lab in (PASS, FAIL)
        }
        for side in (TRAIN, TEST)
    }
    card["baselines"] = {
        "majority": majority,
        "overlay_scores": overlay_baseline(rows, vision, check),
    }
    if card["baselines"]["overlay_scores"] is None:
        card["baselines"]["overlay_scores_missing"] = (
            f"vision.parquet has no {check} scores for these frames"
        )
    card["verdict"] = (
        verdict(card["metrics"][TEST], majority)
        if majority is not None
        else "No majority baseline: train or test rows missing."
    )
    dump_json(card, out_dir / "model.json")
    return card


def summary_table(cards: dict[str, dict[str, Any]]) -> str:
    """Markdown table of holdout scores per check, aggregates only."""
    head = (
        "| check | test frames | fail share | balanced acc | macro F1 "
        "| agreement | majority BA | overlay BA |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )

    def ba(scores: dict[str, Any] | None) -> str:
        return f"{scores['balanced_accuracy']:.2f}" if scores else "-"

    lines = []
    for check, card in cards.items():
        m = card["metrics"][TEST]
        n = card["counts"][TEST]
        total = n[PASS] + n[FAIL]
        share = f"{n[FAIL] / total:.2f}" if total else "-"
        base = card["baselines"]
        cells = [
            check,
            str(total),
            share,
            f"{m['balanced_accuracy']:.2f}",
            f"{m['f1_macro']:.2f}",
            f"{m['agreement']:.2f}",
            ba(base["majority"]),
            ba(base["overlay_scores"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return head + "\n".join(lines) + "\n"
