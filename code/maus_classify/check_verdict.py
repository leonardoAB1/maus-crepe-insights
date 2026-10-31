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

from .harness import (
    cross_val_proba,
    fit_model,
    metrics,
    train_and_evaluate,
)
from .split import SPLIT_VERSION, TEST, TRAIN

CHECKS = ("spread", "fold1", "fold2", "push_off")
PASS, FAIL = "pass", "fail"
KEYS = ["run_id", "check", "attempt"]
EMBEDDING_COLUMN = re.compile(r"^e(\d+)$")
VISION_PREFIX = "vision."
VISION_ID = "id"
EMBEDDER_HASH = "model_sha256"
# L2 strength of the heads; a few hundred rows against hundreds of features
HEAD_C = 0.1
MAX_ITER = 2000
# Sample weighting candidates, picked per check by cross validation on the
# train part: class balance only (class_weight), or class balance within
# each robot so no single robot dominates the fit
CLASS_WEIGHTING = "class"
ROBOT_WEIGHTING = "robot_class"
WEIGHTINGS = (CLASS_WEIGHTING, ROBOT_WEIGHTING)
# Below these the verdict warns: too few holdout failures to trust the
# score, or too many robot failures missed
MIN_FAIL_SUPPORT = 10
MIN_FAIL_RECALL = 0.7
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


def make_head(
    c: float = HEAD_C, class_weight: str | None = "balanced"
) -> Pipeline:
    """Scaled logistic regression, by default with class weights for the
    imbalance. Pass `class_weight=None` when sample weights balance it."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight=class_weight, C=c, max_iter=MAX_ITER),
    )


def robot_balanced_weights(y: np.ndarray, robots: np.ndarray) -> np.ndarray:
    """Sample weights giving every (robot, class) cell the same total
    weight, with a mean weight of 1."""
    cells = pd.Series(list(zip(robots, y, strict=True)))
    counts = cells.map(cells.value_counts())
    n_cells = cells.nunique()
    return (len(cells) / (n_cells * counts)).to_numpy(float)


def weighting_setup(
    name: str, y: np.ndarray, robots: np.ndarray | None
) -> tuple[Pipeline, np.ndarray | None]:
    """Head and sample weights for a weighting candidate."""
    if name == ROBOT_WEIGHTING and robots is not None:
        return make_head(class_weight=None), robot_balanced_weights(y, robots)
    return make_head(), None


def choose_weighting(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    timestamps: np.ndarray,
    robots: np.ndarray | None,
) -> dict[str, Any]:
    """Out-of-fold balanced accuracy and fail recall of each weighting on
    train rows; picks the best balanced accuracy (class weighting on ties
    or when cross validation cannot run)."""
    if robots is None:
        return {"chosen": CLASS_WEIGHTING, "cv": {}, "note": "no robots"}
    cv: dict[str, Any] = {}
    for name in WEIGHTINGS:
        head, w = weighting_setup(name, y, robots)
        try:
            proba, classes = cross_val_proba(
                head, X, y, groups, timestamps, sample_weight=w, robots=robots
            )
        except ValueError as exc:
            return {"chosen": CLASS_WEIGHTING, "cv": {}, "note": str(exc)}
        has = ~np.isnan(proba).any(axis=1)
        if not has.any() or len(set(y[has])) < len(classes):
            continue
        pred = np.array(classes)[proba[has].argmax(axis=1)]
        m = metrics(y[has], pred, proba[has], classes)
        cv[name] = {
            "n_rows": int(has.sum()),
            "balanced_accuracy": m["balanced_accuracy"],
            f"recall_{FAIL}": m["per_class"].get(FAIL, {}).get("recall"),
        }
    best = CLASS_WEIGHTING
    if cv.get(ROBOT_WEIGHTING) and cv.get(CLASS_WEIGHTING):
        if (
            cv[ROBOT_WEIGHTING]["balanced_accuracy"]
            > cv[CLASS_WEIGHTING]["balanced_accuracy"]
        ):
            best = ROBOT_WEIGHTING
    helped = best == ROBOT_WEIGHTING
    return {
        "chosen": best,
        "cv": cv,
        "robot_weights_helped": helped,
        "selection": "out-of-fold balanced accuracy, time-ordered folds "
        "within each robot on the train part",
    }


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
    """Whether the head beats the majority baseline, and where it falls
    short on catching robot failures."""
    ba, ba0 = head["balanced_accuracy"], majority["balanced_accuracy"]
    f1, f10 = head["f1_macro"], majority["f1_macro"]
    scores = (
        f"balanced accuracy {ba:.2f} vs {ba0:.2f}, "
        f"macro F1 {f1:.2f} vs {f10:.2f}."
    )
    if not (ba > ba0 and f1 > f10):
        return (
            f"Does not beat the majority baseline on the holdout: {scores} "
            f"Do not rely on this head for {FAIL!r} calls."
        )
    text = f"Beats the majority baseline on the holdout: {scores}"
    fail = head["per_class"].get(FAIL)
    if fail is None:
        return text
    caught = round(fail["recall"] * fail["support"])
    if fail["recall"] < MIN_FAIL_RECALL:
        text += (
            f" Falls short on failures: catches {caught} of "
            f"{fail['support']} robot fails."
        )
    if fail["support"] < MIN_FAIL_SUPPORT:
        text += (
            f" Only {fail['support']} failures in the holdout, so these "
            f"scores are rough."
        )
    return text


def train_check(
    frames: pd.DataFrame,
    embeddings: pd.DataFrame,
    vision: pd.DataFrame,
    check: str,
    out_dir: Path,
) -> dict[str, Any]:
    """Train, evaluate and export the head for one check. Writes
    `model.onnx` and `model.json` in `out_dir` and returns the card."""
    if "split_version" in frames and set(frames["split_version"]) != {
        SPLIT_VERSION
    }:
        raise CheckVerdictError(
            f"frames.parquet was built with another split; expected "
            f"{SPLIT_VERSION}, rebuild it with pixi run frames"
        )
    rows = check_rows(frames, embeddings, check)
    if rows.empty:
        raise CheckVerdictError(f"no labelled frames with embeddings: {check}")
    features = embedding_columns(embeddings)
    X = rows[features].to_numpy(np.float64)
    y = rows["label"].to_numpy(str)
    split = rows["split"].to_numpy()
    last = rows["is_last_attempt"].to_numpy(bool)
    groups = rows["run_id"].to_numpy()
    timestamps = rows["ts_utc"].to_numpy()
    robots = rows["robot"].to_numpy(str) if "robot" in rows else None
    train, test = split == TRAIN, split == TEST

    weighting = choose_weighting(
        X[train],
        y[train],
        groups[train],
        timestamps[train],
        None if robots is None else robots[train],
    )
    head_spec, weights = weighting_setup(weighting["chosen"], y, robots)
    card = train_and_evaluate(
        head_spec,
        X,
        y,
        split,
        groups,
        name=f"check-verdict-{check}",
        feature_names=features,
        out_dir=out_dir,
        timestamps=timestamps,
        robots=robots,
        positive=FAIL,
        sample_weight=weights,
        description=(
            f"Robot pass/fail verdict for the {check} check frame, from the "
            f"frame embedding. {WEAK_LABELS}"
        ),
    )
    # The harness exported the head fitted on train; refit the same way to
    # add agreement and the last-attempt view without touching its card
    head = fit_model(
        head_spec,
        X[train].astype(np.float32),
        y[train],
        None if weights is None else weights[train],
    )
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
    card["weighting"] = weighting
    if EMBEDDER_HASH in embeddings.columns:
        # Heads only fit embeddings from the exact embedder they saw
        card["embedder_sha256"] = sorted(
            embeddings[EMBEDDER_HASH].dropna().astype(str).unique()
        )
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
        "| check | test frames | fail share | balanced acc [95% CI] "
        "| fail recall [95% CI] | macro F1 | agreement | majority BA "
        "| overlay BA |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )

    def ba(scores: dict[str, Any] | None) -> str:
        return f"{scores['balanced_accuracy']:.2f}" if scores else "-"

    def with_ci(card: dict[str, Any], key: str) -> str:
        entry = card.get("intervals", {}).get("metrics", {}).get(key)
        if entry is None:
            return "-"
        lo, hi = entry["ci95"]
        return f"{entry['value']:.2f} [{lo:.2f}, {hi:.2f}]"

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
            with_ci(card, "balanced_accuracy"),
            with_ci(card, f"recall_{FAIL}"),
            f"{m['f1_macro']:.2f}",
            f"{m['agreement']:.2f}",
            ba(base["majority"]),
            ba(base["overlay_scores"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return head + "\n".join(lines) + "\n"
