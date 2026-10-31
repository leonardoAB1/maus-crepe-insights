"""Robot-adjusted effect of spread features on fold failure, and power.

Each robot ran in its own time window, so robot also stands for era, recipe
and software. Pooled effects mix "this robot fails folds more" with "this
spread raises fold risk". Every effect here is estimated within robot: a
logistic regression with a robot fixed effect, one spread feature at a time,
on a pre-registered feature set. Continuous features are scaled by their
pooled within-robot standard deviation, so an odds ratio is per within-robot
SD. Holm corrects across the features of one failure mechanism.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import (
    ConvergenceWarning,
    PerfectSeparationError,
    PerfectSeparationWarning,
)

from .fold_risk import MECHANISMS, VERDICT_FEATURE

ROBOT = "robot"
SPREAD_FAIL = "spread_fail"  # 1 when the robot's last spread check failed
# Pre-registered in issue #94, before looking at adjusted effects
PRIMARY = [
    SPREAD_FAIL,
    "diameter_rel",
    "offset_rel",
    "circularity",
    "hole_frac",
    "uncovered_frac",
    "residue_frac",
]
BINARY = {SPREAD_FAIL}
# Brightness tracks the camera and lighting of each robot
CONFOUND = ["l_mean", "l_std"]
ALPHA = 0.05
POWER = 0.8
OR_TARGET = 2.0
Z95 = norm.ppf(1 - ALPHA / 2)
MAX_ITER = 200
# Beyond this the fit has run off to separation, not estimated an effect
MAX_ABS_LOG_OR = 10.0


def add_spread_fail(table: pd.DataFrame) -> pd.DataFrame:
    """Add `spread_fail` = 1 - `spread_ok`."""
    out = table.copy()
    out[SPREAD_FAIL] = 1.0 - out[VERDICT_FEATURE].astype(float)
    return out


def within_robot_scale(x: pd.Series, robot: pd.Series) -> pd.Series:
    """Centre on each robot's mean and divide by the pooled within-robot
    SD. The robot fixed effect absorbs the means anyway; this sets the
    unit."""
    centred = x - x.groupby(robot).transform("mean")
    dof = len(x) - robot.nunique()
    sd = math.sqrt(float((centred**2).sum()) / dof) if dof > 0 else math.nan
    return centred / sd if sd and np.isfinite(sd) else centred * math.nan


def _design(rows: pd.DataFrame, feature: str) -> pd.DataFrame:
    x = rows[feature].astype(float)
    if feature not in BINARY:
        x = within_robot_scale(x, rows[ROBOT])
    dummies = pd.get_dummies(rows[ROBOT], prefix=ROBOT, drop_first=True)
    design = pd.concat([x.rename(feature), dummies.astype(float)], axis=1)
    return sm.add_constant(design, has_constant="add")


def adjusted_effect(
    table: pd.DataFrame, target: str, feature: str, robots: list[str]
) -> dict[str, Any]:
    """Odds ratio of `target` per within-robot SD of `feature` (or for
    feature = 1 if binary), adjusted for robot. Fit on `robots` only."""
    rows = table[table[ROBOT].isin(robots) & table[target].notna()]
    y = rows[target].astype(float)
    out: dict[str, Any] = {
        "feature": feature,
        "n_runs": len(rows),
        "n_events": int(y.sum()),
        "unit": "present vs absent" if feature in BINARY else "per SD",
    }
    if feature in BINARY:
        x = rows[feature].astype(float)
        out["events_if_present"] = [int(y[x == 1].sum()), int((x == 1).sum())]
        out["events_if_absent"] = [int(y[x == 0].sum()), int((x == 0).sum())]
    X = _design(rows, feature)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", PerfectSeparationWarning)
            warnings.simplefilter("ignore", ConvergenceWarning)
            fit = sm.Logit(y, X).fit(disp=0, maxiter=MAX_ITER)
    except (
        PerfectSeparationError,
        PerfectSeparationWarning,
        np.linalg.LinAlgError,
    ):
        return out | {"estimable": False}
    b, se = float(fit.params[feature]), float(fit.bse[feature])
    if not (np.isfinite(b) and np.isfinite(se)) or abs(b) > MAX_ABS_LOG_OR:
        return out | {"estimable": False}
    return out | {
        "estimable": True,
        "log_or": b,
        "se": se,
        "odds_ratio": math.exp(b),
        "ci95": [math.exp(b - Z95 * se), math.exp(b + Z95 * se)],
        "p": float(fit.pvalues[feature]),
    }


def holm(results: list[dict[str, Any]], alpha: float = ALPHA) -> None:
    """Add `p_holm` and `significant` in place, over estimable rows."""
    ok = [r for r in results if r.get("estimable")]
    if not ok:
        return
    reject, p_adj, _, _ = multipletests(
        [r["p"] for r in ok], alpha=alpha, method="holm"
    )
    for r, rej, p in zip(ok, reject, p_adj, strict=True):
        r["p_holm"] = float(p)
        r["significant"] = bool(rej)


def pooled_verdict(table: pd.DataFrame, target: str) -> dict[str, Any]:
    """Unadjusted fold-fail rate after a failed and a passed spread check,
    over all robots, to compare with the adjusted odds ratio."""
    rows = table[table[target].notna()]
    out = {}
    for name, val in (("after_spread_fail", 1.0), ("after_spread_ok", 0.0)):
        g = rows[rows[SPREAD_FAIL] == val][target]
        out[name] = [int(g.sum()), len(g)]
    return out


def effects(
    table: pd.DataFrame, robots: list[str]
) -> dict[str, list[dict[str, Any]]]:
    """Adjusted effects of every primary feature, per mechanism, Holm
    corrected within mechanism."""
    out = {}
    for mech, target in MECHANISMS.items():
        rows = [adjusted_effect(table, target, f, robots) for f in PRIMARY]
        holm(rows)
        out[mech] = rows
    return out


def confound_check(table: pd.DataFrame, robots: list[str]) -> dict[str, Any]:
    """How much of each brightness feature is robot, and its pooled versus
    robot-adjusted effect on any fold failure. Not a predictor."""
    target = MECHANISMS["any"]
    out = {}
    for f in CONFOUND:
        rows = table[table[ROBOT].isin(robots)]
        x = rows[f].astype(float)
        between = x.groupby(rows[ROBOT]).transform("mean")
        eta2 = float(
            ((between - x.mean()) ** 2).sum() / ((x - x.mean()) ** 2).sum()
        )
        sd = float(x.std())
        pooled = sm.Logit(
            rows[target].astype(float),
            sm.add_constant(((x - x.mean()) / sd).rename(f)),
        ).fit(disp=0)
        b = float(pooled.params[f])
        out[f] = {
            "robot_share_of_variance": eta2,
            "robot_means": {
                str(k): float(v)
                for k, v in x.groupby(rows[ROBOT]).mean().items()
            },
            "pooled_odds_ratio_per_sd": math.exp(b),
            "adjusted": adjusted_effect(table, target, f, robots),
        }
    return out


def event_variance(
    table: pd.DataFrame, target: str, robots: list[str]
) -> float:
    """Mean within-robot p(1 - p) of the target (the information per run
    left after the robot effect)."""
    rows = table[table[ROBOT].isin(robots) & table[target].notna()]
    p = rows[target].astype(float).groupby(rows[ROBOT]).transform("mean")
    return float((p * (1 - p)).mean())


def binary_variance(
    table: pd.DataFrame, feature: str, robots: list[str]
) -> float:
    """Mean within-robot q(1 - q) of a 0/1 feature."""
    rows = table[table[ROBOT].isin(robots)]
    q = rows[feature].astype(float).groupby(rows[ROBOT]).transform("mean")
    return float((q * (1 - q)).mean())


def _z(n_tests: int, power: float, alpha: float) -> float:
    return norm.ppf(1 - alpha / (2 * n_tests)) + norm.ppf(power)


def power_analysis(
    table: pd.DataFrame,
    robots: list[str],
    *,
    n_tests: int = len(PRIMARY),
    power: float = POWER,
    alpha: float = ALPHA,
    or_target: float = OR_TARGET,
) -> dict[str, Any]:
    """Wald approximation for a logistic slope (Hsieh 1989), within robot.

    Var(log OR) ~ 1 / (n * v_y * v_x), with v_y the within-robot event
    variance and v_x the within-robot feature variance (1 for a feature in
    within-robot SD units). Alpha is Bonferroni over the features, the
    strictest step of Holm.
    """
    z = _z(n_tests, power, alpha)
    out: dict[str, Any] = {
        "method": "Wald approximation, within robot, Bonferroni alpha "
        f"{alpha}/{n_tests}, power {power}",
        "or_target": or_target,
    }
    for mech, target in MECHANISMS.items():
        n = int((table[ROBOT].isin(robots) & table[target].notna()).sum())
        vy = event_variance(table, target, robots)
        vx_bin = binary_variance(table, SPREAD_FAIL, robots)
        ln_t = math.log(or_target)
        out[mech] = {
            "n_runs": n,
            "event_variance": vy,
            "min_detectable_or_per_sd": math.exp(z / math.sqrt(n * vy))
            if n and vy
            else math.nan,
            "runs_needed_or_per_sd": math.ceil(z**2 / (vy * ln_t**2))
            if vy
            else None,
            "spread_fail_variance": vx_bin,
            "min_detectable_or_spread_fail": math.exp(
                z / math.sqrt(n * vy * vx_bin)
            )
            if n and vy and vx_bin
            else math.nan,
            "runs_needed_or_spread_fail": math.ceil(
                z**2 / (vy * vx_bin * ln_t**2)
            )
            if vy and vx_bin
            else None,
        }
    return out


LABELS = {
    SPREAD_FAIL: "spread check failed (yes vs no)",
    "diameter_rel": "diameter",
    "offset_rel": "offset",
    "circularity": "circularity",
    "hole_frac": "holes",
    "uncovered_frac": "uncovered area",
    "residue_frac": "residue",
}
# Reference palette (dataviz skill): light surface, text inks, series slot 1
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = "#2a78d6"
OR_LIMITS = (0.1, 20.0)
FIG_SIZE = (10.0, 3.6)
DPI = 160


def plot_effects(eff: dict[str, list[dict[str, Any]]], path: Any) -> None:
    """Forest plot: robot-adjusted odds ratio and 95% CI per feature, one
    panel per mechanism. Aggregates only."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1, len(eff), figsize=FIG_SIZE, sharey=True, facecolor=SURFACE
    )
    names = [LABELS[f] for f in PRIMARY]
    ypos = np.arange(len(PRIMARY))[::-1]
    for ax, (mech, rows) in zip(axes, eff.items(), strict=True):
        ax.set_facecolor(SURFACE)
        ax.set_xscale("log")
        ax.set_xlim(*OR_LIMITS)
        ax.axvline(1.0, color=INK_2, linewidth=1, zorder=1)
        ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
        for y, r in zip(ypos, rows, strict=True):
            if not r.get("estimable"):
                ax.text(
                    1.0,
                    y,
                    "not estimable",
                    color=INK_2,
                    fontsize=8,
                    ha="center",
                    va="center",
                )
                continue
            lo, hi = (np.clip(v, *OR_LIMITS) for v in r["ci95"])
            ax.plot(
                [lo, hi],
                [y, y],
                color=SERIES,
                linewidth=2,
                solid_capstyle="round",
                zorder=2,
            )
            ax.plot(
                r["odds_ratio"],
                y,
                "o",
                color=SERIES,
                markersize=8,
                markeredgecolor=SURFACE,
                markeredgewidth=2,
                zorder=3,
            )
        n, k = rows[0]["n_runs"], rows[0]["n_events"]
        ax.set_title(
            f"{mech} fold failure ({k} of {n} runs)",
            color=INK,
            fontsize=10,
            loc="left",
        )
        ax.set_xlabel("odds ratio (log scale)", color=INK_2, fontsize=9)
        ax.tick_params(colors=INK_2, labelsize=8, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    axes[0].set_yticks(ypos, names, color=INK, fontsize=9)
    fig.suptitle(
        "Robot-adjusted effect of the spread check on fold failure "
        "(per within-robot SD, 95% CI)",
        color=INK,
        fontsize=11,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
