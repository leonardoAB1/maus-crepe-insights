"""Why did a check fail? Multi-label reason heads on failed frames.

The robot reports one or more error strings per failed check, and they
co-occur: "not a quarter fold" nearly always comes with "wrong fold shape".
A single multi-class label would have to invent a primary error, so each
reason class gets its own binary head (multi-label). Related error strings
are merged into one class by a named table (`REASON_CLASSES`). A class is
modelled only with at least `MIN_SUPPORT` train frames on both sides; the
rest are reported with their support and fall back to `other`.

At inference a check returns every modelled class whose probability reaches
the check's abstain threshold, sorted by confidence, or `other` when none
does. The threshold is picked by run-grouped CV on train rows only.

Labels are the robot's own error strings (weak labels). The holdout is the
per-robot split and small, so each head is also tested by train CV against
labels shuffled within robot, and per-robot and leave-one-robot-out scores
are reported.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from .check_verdict import FAIL, check_rows, embedding_columns, make_head
from .frames import normalize
from .harness import metrics, train_and_evaluate, write_card
from .harness import permutation_test as harness_permutation_test
from .split import TEST, TRAIN

# Robot error string -> reason class, per check. Merges are deliberate:
# both hole messages describe a hole, and the fold messages that say "not a
# half/quarter fold" and "wrong fold shape" describe the same shape failure.
REASON_CLASSES: dict[str, dict[str, str]] = {
    "spread": {
        "crepe too small": "too_small",
        "large hole in the centre": "hole",
        "large hole off-center": "hole",
        "batter left on the plate": "residue",
        "no crepe in the plate": "no_crepe",
    },
    "fold1": {
        "not a half fold": "shape",
        "wrong fold shape": "shape",
        "batter left on the plate": "residue",
        "no crepe detected": "no_crepe",
    },
    "fold2": {
        "not a quarter fold": "shape",
        "wrong fold shape": "shape",
        "batter left on the plate": "residue",
        "no crepe detected": "no_crepe",
    },
    "push_off": {
        "crepe not aligned with the plate": "misaligned",
        "serving plate not found": "no_plate",
        "no crepe detected": "no_crepe",
    },
}
CHECKS = tuple(REASON_CLASSES)
OTHER = "other"
NOT = "not_"
# Train frames needed on each side of a binary head; fewer and the head is
# not trained (5 CV folds need at least one positive per fold, plus slack)
MIN_SUPPORT = 8
CV_FOLDS = 5
N_PERM = 1000
ALPHA = 0.05
THRESHOLDS = np.round(np.arange(0.5, 0.96, 0.05), 2)
SEED = 0


def reason_codes(check: str, errors: list[str], message: str | None) -> list:
    """Sorted reason classes of one failed frame. A failed frame without
    errors uses its message; unknown strings map to `other`."""
    table = REASON_CLASSES[check]
    texts = [normalize(e) for e in errors] or (
        [normalize(message)] if message else []
    )
    return sorted({table.get(t, OTHER) for t in texts}) or [OTHER]


def failed_rows(
    frames: pd.DataFrame, embeddings: pd.DataFrame, check: str
) -> pd.DataFrame:
    """Failed frames of one check with embedding and `codes`."""
    rows = check_rows(frames, embeddings, check)
    rows = rows[rows["label"] == FAIL].reset_index(drop=True)
    rows["codes"] = [
        reason_codes(check, list(e), m)
        for e, m in zip(rows["errors"], rows["message"], strict=True)
    ]
    return rows


def support(rows: pd.DataFrame, check: str) -> dict[str, dict[str, int]]:
    """Frames per reason class and split."""
    classes = sorted(set(REASON_CLASSES[check].values()) | {OTHER})
    out = {}
    for c in classes:
        has = rows["codes"].map(lambda cs, c=c: c in cs)
        out[c] = {
            side: int((has & (rows["split"] == side)).sum())
            for side in (TRAIN, TEST)
        }
    return out


def modelled_classes(rows: pd.DataFrame, check: str) -> list[str]:
    """Classes with at least MIN_SUPPORT train frames with and without."""
    train = rows[rows["split"] == TRAIN]
    keep = []
    for c in sorted(set(REASON_CLASSES[check].values())):
        pos = int(train["codes"].map(lambda cs, c=c: c in cs).sum())
        if min(pos, len(train) - pos) >= MIN_SUPPORT:
            keep.append(c)
    return keep


def binary_labels(rows: pd.DataFrame, cls: str) -> np.ndarray:
    return np.where(
        rows["codes"].map(lambda cs: cls in cs), cls, NOT + cls
    ).astype(str)


def _folds(y: np.ndarray, groups: np.ndarray) -> StratifiedGroupKFold:
    runs = pd.Series(y).groupby(groups).first()
    k = int(min(CV_FOLDS, runs.value_counts().min()))
    return StratifiedGroupKFold(
        n_splits=max(k, 2), shuffle=True, random_state=SEED
    )


def oof_proba(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, positive: str
) -> np.ndarray:
    """Run-grouped out-of-fold probability of `positive`."""
    out = np.full(len(y), np.nan)
    for tr, va in _folds(y, groups).split(X, y, groups):
        if len(set(y[tr])) < 2:
            continue
        head = clone(make_head()).fit(X[tr].astype(np.float32), y[tr])
        col = list(head.classes_).index(positive)
        out[va] = head.predict_proba(X[va].astype(np.float32))[:, col]
    return out


def _auc(y: np.ndarray, p: np.ndarray, positive: str) -> float:
    ok = np.isfinite(p)
    hit = y[ok] == positive
    if hit.all() or not hit.any():
        return math.nan
    return float(roc_auc_score(hit, p[ok]))


def permutation_test(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    robots: np.ndarray,
    positive: str,
    *,
    n_perm: int = N_PERM,
    seed: int = SEED,
) -> dict[str, Any]:
    """Train-CV AUC against labels shuffled within robot (one-sided),
    through the shared `harness.permutation_test`."""
    res = harness_permutation_test(
        lambda labels: _auc(
            labels, oof_proba(X, labels, groups, positive), positive
        ),
        y,
        strata=robots,
        n_perm=n_perm,
        seed=seed,
    )
    return {
        "cv_auc": res["value"],
        "n_perm": res["n_perm"],
        "null_q05_q95": res["null_q05_q95"],
        "p_one_sided": res["p_value"],
        "beats_null": bool(res["p_value"] < ALPHA),
    }


def per_robot(
    y: np.ndarray, p: np.ndarray, robots: np.ndarray, positive: str
) -> dict[str, Any]:
    """Out-of-fold AUC and counts per robot (train CV)."""
    out = {}
    for r in sorted(set(robots)):
        m = robots == r
        out[str(r)] = {
            "n": int(m.sum()),
            "positives": int((y[m] == positive).sum()),
            "cv_auc": _auc(y[m], p[m], positive),
        }
    return out


def predict_sets(
    probas: dict[str, np.ndarray], threshold: float
) -> list[list[str]]:
    """Classes at or above the threshold by falling confidence, else other."""
    n = len(next(iter(probas.values())))
    out = []
    for i in range(n):
        hits = sorted(
            (c for c, p in probas.items() if p[i] >= threshold),
            key=lambda c: -probas[c][i],
        )
        out.append(hits or [OTHER])
    return out


def truth_sets(codes: pd.Series, classes: list[str]) -> list[list[str]]:
    return [sorted(set(cs) & set(classes)) or [OTHER] for cs in codes]


def set_f1(truth: list[list[str]], pred: list[list[str]]) -> float:
    """Mean per-frame F1 between true and predicted class sets."""
    scores = []
    for t, p in zip(truth, pred, strict=True):
        inter = len(set(t) & set(p))
        scores.append(2 * inter / (len(t) + len(p)))
    return float(np.mean(scores)) if scores else math.nan


def choose_threshold(
    truth: list[list[str]], probas: dict[str, np.ndarray]
) -> tuple[float, dict[str, float]]:
    """Abstain threshold with the best train-CV set F1; ties go to the
    higher (more cautious) threshold."""
    scores = {
        str(t): set_f1(truth, predict_sets(probas, float(t)))
        for t in THRESHOLDS
    }
    best = max(scores.values())
    chosen = max(float(t) for t, s in scores.items() if s >= best - 1e-12)
    return chosen, scores


def set_metrics(
    truth: list[list[str]], pred: list[list[str]], classes: list[str]
) -> dict[str, Any]:
    """Per-class precision, recall and support of set membership."""
    out: dict[str, Any] = {
        "n": len(truth),
        "set_f1": set_f1(truth, pred),
        "abstained": int(sum(p == [OTHER] for p in pred)),
    }
    per = {}
    for c in [*classes, OTHER]:
        t = np.array([c in s for s in truth])
        p = np.array([c in s for s in pred])
        tp = int((t & p).sum())
        per[c] = {
            "support": int(t.sum()),
            "predicted": int(p.sum()),
            "precision": tp / int(p.sum()) if p.any() else None,
            "recall": tp / int(t.sum()) if t.any() else None,
        }
    out["per_class"] = per
    return out


def majority_head(y_train: np.ndarray, y_test: np.ndarray) -> dict | None:
    if not len(y_test):
        return None
    dummy = DummyClassifier(strategy="most_frequent").fit(
        np.zeros((len(y_train), 1)), y_train
    )
    classes = [str(c) for c in dummy.classes_]
    pred = dummy.predict(np.zeros((len(y_test), 1))).astype(str)
    proba = dummy.predict_proba(np.zeros((len(y_test), 1)))
    return metrics(y_test, pred, proba, classes)


def train_head(
    rows: pd.DataFrame, features: list[str], check: str, cls: str, out: Path
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """One binary head: harness export and holdout, train CV, null, per
    robot. Returns the card, train OOF and test probabilities."""
    X = rows[features].to_numpy(np.float64)
    y = binary_labels(rows, cls)
    split = rows["split"].to_numpy()
    groups = rows["run_id"].to_numpy()
    robots = rows["robot"].astype(str).to_numpy()
    train, test = split == TRAIN, split == TEST

    card = train_and_evaluate(
        make_head(),
        X,
        y,
        split,
        groups,
        name=f"failure-reason-{check}-{cls}",
        feature_names=features,
        out_dir=out,
        timestamps=rows["ts_utc"].to_numpy(),
        robots=rows["robot"].fillna("unknown").to_numpy(),
        positive=cls,
        description=f"Is '{cls}' among the robot's reasons for a failed "
        f"{check} check? Frame embedding in, weak labels (robot errors).",
    )
    oof = oof_proba(X[train], y[train], groups[train], cls)
    head = clone(make_head()).fit(X[train].astype(np.float32), y[train])
    p_test = head.predict_proba(X[test].astype(np.float32))[
        :, list(head.classes_).index(cls)
    ]
    card["reason"] = {
        "check": check,
        "class": cls,
        "error_strings": sorted(
            e for e, c in REASON_CLASSES[check].items() if c == cls
        ),
        "counts": {
            side: {
                lab: int(((split == side) & (y == lab)).sum())
                for lab in (cls, NOT + cls)
            }
            for side in (TRAIN, TEST)
        },
        "majority_baseline": majority_head(y[train], y[test]),
        "train_cv": {
            "auc": _auc(y[train], oof, cls),
            "per_robot": per_robot(y[train], oof, robots[train], cls),
        },
        "null": permutation_test(
            X[train], y[train], groups[train], robots[train], cls
        ),
        "holdout_auc": _auc(y[test], p_test, cls),
    }
    write_card(card, out / "model.json")
    return card, oof, p_test


def head_verdict(card: dict[str, Any]) -> str:
    """Deployable only if train CV beats the shuffled-label null and the
    holdout beats the majority baseline on balanced accuracy."""
    r = card["reason"]
    test = card["metrics"][TEST]
    base = r["majority_baseline"]
    counts = r["counts"][TEST]
    if min(counts.values()) == 0:
        return "not deployable: holdout has one class only, cannot score"
    if not r["null"]["beats_null"]:
        return "not deployable: train CV does not beat shuffled labels"
    if base and test["balanced_accuracy"] <= base["balanced_accuracy"]:
        return "not deployable: holdout does not beat the majority baseline"
    return "deployable"


def train_check(
    frames: pd.DataFrame,
    embeddings: pd.DataFrame,
    check: str,
    out_dir: Path,
) -> dict[str, Any]:
    """All heads of one check plus the check-level card in out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = failed_rows(frames, embeddings, check)
    features = embedding_columns(embeddings)
    classes = modelled_classes(rows, check)
    split = rows["split"].to_numpy()
    train, test = split == TRAIN, split == TEST
    card: dict[str, Any] = {
        "check": check,
        "rows": "failed frames of this check with an embedding",
        "design": "multi-label: one binary head per reason class",
        "class_map": REASON_CLASSES[check],
        "min_support": MIN_SUPPORT,
        "support": support(rows, check),
        "robots": {
            side: rows[split == side]["robot"].value_counts().to_dict()
            for side in (TRAIN, TEST)
        },
        "modelled": classes,
        "not_modelled": sorted(
            set(REASON_CLASSES[check].values()) - set(classes)
        ),
        "heads": {},
    }
    if not classes:
        card["verdict"] = (
            f"no reason class has {MIN_SUPPORT} train frames on both sides; "
            "not enough data to train a head"
        )
        write_card(card, out_dir / "model.json")
        return card
    oof, p_test = {}, {}
    for cls in classes:
        head_card, oof[cls], p_test[cls] = train_head(
            rows, features, check, cls, out_dir / cls
        )
        card["heads"][cls] = {
            "verdict": head_verdict(head_card),
            "holdout": head_card["metrics"][TEST],
            "holdout_auc": head_card["reason"]["holdout_auc"],
            "train_cv_auc": head_card["reason"]["train_cv"]["auc"],
            "null": head_card["reason"]["null"],
            "majority_baseline": head_card["reason"]["majority_baseline"],
        }
    truth_train = truth_sets(rows["codes"][train], classes)
    threshold, scores = choose_threshold(truth_train, oof)
    truth_test = truth_sets(rows["codes"][test], classes)
    pred_test = predict_sets(p_test, threshold)
    card["abstain"] = {
        "threshold": threshold,
        "chosen_by": "best train-CV set F1 (run-grouped), ties to higher",
        "train_cv_set_f1": scores,
    }
    card["holdout"] = set_metrics(truth_test, pred_test, classes)
    card["holdout_majority"] = set_metrics(
        truth_test,
        [_majority_set(truth_train)] * len(truth_test),
        classes,
    )
    deploy = [
        c for c, h in card["heads"].items() if h["verdict"] == "deployable"
    ]
    card["verdict"] = (
        f"deployable heads: {', '.join(deploy)}"
        if deploy
        else "research only: no head passes the null and majority checks"
    )
    write_card(card, out_dir / "model.json")
    return card


def _majority_set(truth: list[list[str]]) -> list[str]:
    counts = pd.Series([tuple(t) for t in truth]).value_counts()
    return list(counts.index[0])
