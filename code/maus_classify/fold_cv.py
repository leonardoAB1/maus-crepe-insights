"""Model checks for fold risk that a robot confound cannot pass.

- Pooled cross validation with a robot term, stratified by robot and label
  (one row per run, so folds are run-grouped). The baseline is robot-only:
  each robot's training-fold failure rate. Delta AUC is the run-weighted mean
  of within-robot AUC minus 0.5, the AUC of any robot-only score inside one
  robot. Brier skill is against the robot-only baseline.
- Cross validation inside each robot separately.
- Leave one robot out, features only, as a transfer check.

Intervals bootstrap runs within robot over fixed out-of-fold predictions, so
they exclude the variance of refitting in CV and are too narrow; pooled
out-of-fold AUC on small null data is also biased below zero. Whether the
model beats robot-only is therefore judged by `permutation_test`: labels are
shuffled within robot and the whole CV is rerun.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROBOT = "robot"
N_SPLITS = 5
N_REPEATS = 20
N_BOOT = 2000
N_PERM = 1000
PERM_REPEATS = 5
ALPHA = 0.05
C = 1.0
MAX_ITER = 1000
CI = (0.025, 0.975)
MIN_CLASS = 2  # fewer events than this in a robot: no CV for it


def _model():
    return make_pipeline(
        StandardScaler(), LogisticRegression(C=C, max_iter=MAX_ITER)
    )


def _robot_onehot(robot: pd.Series, levels: list[str]) -> np.ndarray:
    return np.stack([(robot == r).to_numpy(float) for r in levels], axis=1)


def oof_predictions(
    rows: pd.DataFrame,
    target: str,
    features: list[str],
    *,
    with_robot: bool = True,
    n_splits: int = N_SPLITS,
    n_repeats: int = N_REPEATS,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold failure probabilities averaged over repeats: the model
    (features plus robot one-hot) and the robot-only baseline."""
    y = rows[target].to_numpy(float)
    robot = rows[ROBOT].astype(str)
    levels = sorted(robot.unique())
    X = rows[features].to_numpy(float)
    if with_robot and len(levels) > 1:
        X = np.hstack([X, _robot_onehot(robot, levels)])
    strata = (
        robot + "|" + pd.Series(y, index=rows.index).astype(str)
    ).to_numpy()
    smallest = pd.Series(strata).value_counts().min()
    k = int(min(n_splits, smallest))
    if k < 2:
        raise ValueError("too few runs per robot and label for CV")
    p_model = np.zeros(len(y))
    p_base = np.zeros(len(y))
    for rep in range(n_repeats):
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed + rep)
        for tr, te in cv.split(X, strata):
            m = _model().fit(X[tr], y[tr])
            p_model[te] += m.predict_proba(X[te])[:, 1]
            rate = pd.Series(y[tr]).groupby(robot.iloc[tr].to_numpy()).mean()
            p_base[te] += (
                robot.iloc[te].map(rate).fillna(y[tr].mean()).to_numpy()
            )
    return p_model / n_repeats, p_base / n_repeats


def within_auc(y: np.ndarray, p: np.ndarray, robot: np.ndarray) -> float:
    """Run-weighted mean AUC inside each robot (robots with both classes)."""
    num = den = 0.0
    for r in np.unique(robot):
        m = robot == r
        if 0 < y[m].sum() < m.sum():
            num += m.sum() * roc_auc_score(y[m], p[m])
            den += m.sum()
    return num / den if den else math.nan


def _brier_skill(y: np.ndarray, p: np.ndarray, base: np.ndarray) -> float:
    b0 = float(np.mean((base - y) ** 2))
    return 1 - float(np.mean((p - y) ** 2)) / b0 if b0 > 0 else math.nan


def _ci(vals: list[float]) -> list[float]:
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return [math.nan, math.nan]
    return [float(np.quantile(v, CI[0])), float(np.quantile(v, CI[1]))]


def delta_report(
    y: np.ndarray,
    p_model: np.ndarray,
    p_base: np.ndarray,
    robot: np.ndarray,
    *,
    n_boot: int = N_BOOT,
    seed: int = 0,
) -> dict[str, Any]:
    """Within-robot delta AUC over robot-only and Brier skill, with
    intervals from a bootstrap of runs within each robot."""
    y, robot = np.asarray(y, float), np.asarray(robot)

    def stats(idx: np.ndarray) -> tuple[float, float, float]:
        auc = within_auc(y[idx], p_model[idx], robot[idx])
        pooled = (
            roc_auc_score(y[idx], p_model[idx])
            - roc_auc_score(y[idx], p_base[idx])
            if 0 < y[idx].sum() < len(idx)
            else math.nan
        )
        return (
            auc - 0.5,
            pooled,
            _brier_skill(y[idx], p_model[idx], p_base[idx]),
        )

    point = stats(np.arange(len(y)))
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(robot == r) for r in np.unique(robot)]
    draws = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(g, len(g)) for g in groups])
        draws.append(stats(idx))
    draws_arr = np.array(draws)
    names = ("delta_auc_within_robot", "delta_auc_pooled", "brier_skill")
    out: dict[str, Any] = {
        "n_runs": len(y),
        "n_events": int(y.sum()),
        "note": "resample_ci95 resamples fixed out-of-fold predictions; it "
        "excludes CV refit variance. Use permutation_test for inference.",
    }
    for i, name in enumerate(names):
        out[name] = {
            "value": float(point[i]),
            "resample_ci95": _ci(list(draws_arr[:, i])),
        }
    return out


def auc_report(
    y: np.ndarray, p: np.ndarray, *, n_boot: int = N_BOOT, seed: int = 0
) -> dict[str, Any]:
    """AUC (chance 0.5) and Brier skill against the mean of `y`, with a
    bootstrap interval."""
    y = np.asarray(y, float)
    base = np.full(len(y), y.mean())

    def stats(idx):
        if not 0 < y[idx].sum() < len(idx):
            return math.nan, math.nan
        return roc_auc_score(y[idx], p[idx]), _brier_skill(
            y[idx], p[idx], base[idx]
        )

    point = stats(np.arange(len(y)))
    rng = np.random.default_rng(seed)
    draws = np.array(
        [stats(rng.integers(0, len(y), len(y))) for _ in range(n_boot)]
    )
    return {
        "n_runs": len(y),
        "n_events": int(y.sum()),
        "auc": {"value": float(point[0]), "ci95": _ci(list(draws[:, 0]))},
        "brier_skill_vs_mean": {
            "value": float(point[1]),
            "ci95": _ci(list(draws[:, 1])),
        },
    }


def within_robot_cv(
    table: pd.DataFrame,
    target: str,
    features: list[str],
    robots: list[str],
    *,
    n_boot: int = N_BOOT,
) -> dict[str, Any]:
    """CV inside each robot on its own runs; a robot with too few events
    is reported without a score."""
    out = {}
    for r in robots:
        rows = table[(table[ROBOT] == r) & table[target].notna()]
        y = rows[target].to_numpy(float)
        if min(y.sum(), len(y) - y.sum()) < MIN_CLASS:
            out[r] = {"n_runs": len(y), "n_events": int(y.sum()), "auc": None}
            continue
        p, _ = oof_predictions(rows, target, features, with_robot=False)
        out[r] = auc_report(y, p, n_boot=n_boot)
    return out


def leave_one_robot_out(
    table: pd.DataFrame,
    target: str,
    features: list[str],
    robots: list[str],
    *,
    n_boot: int = N_BOOT,
) -> dict[str, Any]:
    """Train on the other robots (features only), score the held-out one."""
    rows = table[table[target].notna() & table[ROBOT].isin(robots)]
    out = {}
    for r in robots:
        train, test = rows[rows[ROBOT] != r], rows[rows[ROBOT] == r]
        y = test[target].to_numpy(float)
        if not (0 < y.sum() < len(y)) or train[target].nunique() < 2:
            out[r] = {"n_runs": len(y), "n_events": int(y.sum()), "auc": None}
            continue
        m = _model().fit(
            train[features].to_numpy(float), train[target].to_numpy(float)
        )
        p = m.predict_proba(test[features].to_numpy(float))[:, 1]
        out[r] = auc_report(y, p, n_boot=n_boot)
    return out


def pooled_cv(
    table: pd.DataFrame,
    target: str,
    features: list[str],
    robots: list[str],
    *,
    n_boot: int = N_BOOT,
) -> dict[str, Any]:
    """Features plus robot term against robot-only, within-robot delta."""
    rows = table[table[ROBOT].isin(robots) & table[target].notna()]
    p_model, p_base = oof_predictions(rows, target, features)
    return delta_report(
        rows[target].to_numpy(float),
        p_model,
        p_base,
        rows[ROBOT].astype(str).to_numpy(),
        n_boot=n_boot,
    )


def permutation_test(
    table: pd.DataFrame,
    target: str,
    features: list[str],
    robots: list[str],
    *,
    n_perm: int = N_PERM,
    n_repeats: int = PERM_REPEATS,
    seed: int = 0,
) -> dict[str, Any]:
    """One-sided test that within-robot delta AUC beats labels shuffled
    within robot. The observed value and every shuffle run the same CV
    (`n_repeats`), so the CV bias cancels."""
    rows = table[table[ROBOT].isin(robots) & table[target].notna()].copy()
    robot = rows[ROBOT].astype(str).to_numpy()

    def delta(col: str, rep_seed: int) -> float:
        pm, _ = oof_predictions(
            rows, col, features, n_repeats=n_repeats, seed=rep_seed
        )
        return within_auc(rows[col].to_numpy(float), pm, robot) - 0.5

    observed = delta(target, seed)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        rows["_perm"] = rows.groupby(ROBOT)[target].transform(
            lambda s: rng.permutation(s.to_numpy())
        )
        null[i] = delta("_perm", seed + 1 + i)
    p = (1 + int(np.sum(null >= observed))) / (1 + n_perm)
    return {
        "observed_delta_auc": float(observed),
        "n_perm": n_perm,
        "cv_repeats": n_repeats,
        "null_mean": float(null.mean()),
        "null_q05_q95": [
            float(np.quantile(null, 0.05)),
            float(np.quantile(null, 0.95)),
        ],
        "p_one_sided": float(p),
        "beats_null": bool(p < ALPHA),
    }
