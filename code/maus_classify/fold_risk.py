"""Fold risk: predict from the spread check whether a later fold fails.

One row per run that has a spread check and at least one fold check. The
target is the robot's own verdict (any fold1 or fold2 attempt not ok), so it
is a weak label. Features are known when the spread check finishes: the
overlay scores of the last spread attempt (`vision.parquet`, the same
last-attempt rule) and that attempt's verdict, plus optionally its frame
embedding. The split is the shared frame holdout, one side per run.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .check_verdict import embedding_columns
from .split import TEST

SPREAD = "spread"
FOLDS = ("fold1", "fold2")
VISION_PREFIX = "vision.spread."
# Crepe shape and colour scores. Left out: zone_r_px (camera geometry, not
# the crepe) and circle_* (missing on some runs, close to diameter/offset).
SCORE_FEATURES = [
    "area_rel",
    "diameter_rel",
    "offset_rel",
    "circularity",
    "hole_frac",
    "uncovered_frac",
    "l_mean",
    "l_std",
    "residue_frac",
]
VERDICT_FEATURE = "spread_ok"
FAIL = "fail"
PASS = "ok"
N_BOOT = 2000
CI_LEVEL = 0.95
TOP_FRAC = 0.25  # share of runs flagged as highest risk for top-k lift
WILSON_Z = 1.959964  # two-sided 95 % normal quantile


def fold_target(frames: pd.DataFrame) -> pd.Series:
    """Run id -> True when any fold1 or fold2 attempt failed. Runs without
    a fold check are absent (unknown, not a pass)."""
    folds = frames[frames["check"].isin(FOLDS)]
    return (~folds["ok"].astype(bool)).groupby(folds["run_id"]).any()


def last_spread(frames: pd.DataFrame) -> pd.DataFrame:
    """The last spread attempt of each run, indexed by run id."""
    spread = frames[(frames["check"] == SPREAD) & frames["is_last_attempt"]]
    return spread.set_index("run_id")


def fold_risk_table(
    frames: pd.DataFrame,
    vision: pd.DataFrame,
    embeddings: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Model rows indexed by run id: score features, `spread_ok`, optional
    embedding columns, `fold_fail`, `label`, `split`, `robot` and `ts_utc`.
    Runs
    missing a target or any score are dropped."""
    spread = last_spread(frames)
    scores = vision.set_index("id")[[VISION_PREFIX + f for f in SCORE_FEATURES]]
    scores.columns = SCORE_FEATURES
    cols = ["ok", "split", "ts_utc", "robot", "check", "attempt"]
    table = spread[cols].join(scores, how="inner")
    table = table.join(fold_target(frames).rename("fold_fail"), how="inner")
    table[VERDICT_FEATURE] = table.pop("ok").astype(float)
    if embeddings is not None:
        emb = embeddings.set_index(["run_id", "check", "attempt"])
        cols = embedding_columns(embeddings)
        keys = pd.MultiIndex.from_arrays(
            [table.index, table["check"], table["attempt"]]
        )
        vectors = pd.DataFrame(
            emb.reindex(keys)[cols].to_numpy(), index=table.index, columns=cols
        )
        table = pd.concat([table, vectors], axis=1)
    table = table.drop(columns=["check", "attempt"])
    feats = SCORE_FEATURES + (
        embedding_columns(embeddings) if embeddings is not None else []
    )
    table = table[np.isfinite(table[feats].to_numpy(float)).all(axis=1)]
    table["fold_fail"] = table["fold_fail"].astype(bool)
    table["label"] = np.where(table["fold_fail"], FAIL, PASS)
    return table.sort_index()


def wilson(k: int, n: int, z: float = WILSON_Z) -> list[float]:
    """Wilson score interval for a proportion k / n."""
    if n == 0:
        return [math.nan, math.nan]
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = (
        z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    )
    return [max(0.0, centre - half), min(1.0, centre + half)]


def _top_lift(y: np.ndarray, p: np.ndarray, top_frac: float) -> float:
    k = max(1, math.ceil(len(y) * top_frac))
    order = np.argsort(-p, kind="stable")[:k]
    base = y.mean()
    return float(y[order].mean() / base) if base > 0 else math.nan


def _stats(y: np.ndarray, p: np.ndarray, top_frac: float) -> dict[str, float]:
    base = y.mean()
    both = 0 < y.sum() < len(y)
    ap = average_precision_score(y, p) if both else math.nan
    brier = float(np.mean((p - y) ** 2))
    brier_base = float(np.mean((base - y) ** 2))
    return {
        "roc_auc": float(roc_auc_score(y, p)) if both else math.nan,
        "average_precision": float(ap),
        "ap_lift": float(ap / base) if base > 0 else math.nan,
        "top_lift": _top_lift(y, p, top_frac),
        "brier_skill": 1 - brier / brier_base if brier_base > 0 else math.nan,
    }


def lift_report(
    y: np.ndarray,
    p: np.ndarray,
    *,
    n_boot: int = N_BOOT,
    seed: int = 0,
    top_frac: float = TOP_FRAC,
    level: float = CI_LEVEL,
) -> dict[str, Any]:
    """Holdout lift over the base rate with bootstrap intervals.

    `y` is 1 for a fold failure, `p` the predicted failure probability, one
    entry per run (so resampling rows resamples runs). Chance is 0.5 for
    ROC AUC, 1 for the lifts and 0 for Brier skill (against always
    predicting the holdout base rate).
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    point = _stats(y, p, top_frac)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {k: [] for k in point}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if not 0 < y[idx].sum() < len(idx):
            continue
        for k, v in _stats(y[idx], p[idx], top_frac).items():
            draws[k].append(v)
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    out: dict[str, Any] = {
        "n_runs": len(y),
        "n_fail": int(y.sum()),
        "base_rate": float(y.mean()),
        "base_rate_ci": wilson(int(y.sum()), len(y)),
        "top_frac": top_frac,
        "ci_level": level,
        "n_boot_used": len(draws["roc_auc"]),
    }
    for k, v in point.items():
        vals = np.asarray(draws[k], dtype=float)
        vals = vals[np.isfinite(vals)]
        ci = (
            [float(np.quantile(vals, lo)), float(np.quantile(vals, hi))]
            if len(vals)
            else [math.nan, math.nan]
        )
        out[k] = {"value": v, "ci": ci}
    return out


def verdict_baseline(table: pd.DataFrame) -> np.ndarray:
    """Holdout risk from the robot's spread verdict alone: the train fail
    rate among runs with the same verdict."""
    train = table[table["split"] != TEST]
    rate = train.groupby(VERDICT_FEATURE)["fold_fail"].mean()
    test = table[table["split"] == TEST]
    fallback = train["fold_fail"].mean()
    return test[VERDICT_FEATURE].map(rate).fillna(fallback).to_numpy(float)


def robot_breakdown(table: pd.DataFrame) -> list[dict[str, Any]]:
    """Runs and fold-fail rate per split and robot. Robots are relabelled
    A, B, ... by train run count, so the card holds no machine names."""
    train = table[table["split"] != TEST]
    order = list(train["robot"].value_counts().index)
    order += [r for r in table["robot"].unique() if r not in order]
    names = {r: chr(ord("A") + i) for i, r in enumerate(order)}
    rows = []
    for (split, robot), g in table.groupby(["split", "robot"]):
        rows.append(
            {
                "split": split,
                "robot": names[robot],
                "n_runs": len(g),
                "fold_fail_rate": float(g["fold_fail"].mean()),
            }
        )
    return sorted(rows, key=lambda r: (r["split"], r["robot"]))


def signal_verdict(report: dict[str, Any]) -> str:
    """Plain reading of a lift report."""
    lo = report["roc_auc"]["ci"][0]
    if not np.isfinite(lo):
        return "not enough holdout failures to judge"
    if lo > 0.5:
        return "signal: ROC AUC interval is above chance"
    return "no clear signal: ROC AUC interval includes chance (0.5)"
