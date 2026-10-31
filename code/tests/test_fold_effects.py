"""Robot-adjusted effects, power and model checks on synthetic runs that
plant a within-robot effect next to a robot confound."""

import math

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")  # analysis runs in the ml environment
pytest.importorskip("sklearn")

from maus_classify import fold_cv, fold_effects  # noqa: E402
from maus_classify.fold_effects import (  # noqa: E402
    PRIMARY,
    ROBOT,
    SPREAD_FAIL,
)

TRUE_LOG_OR = 1.0  # planted within-robot effect of diameter, per SD
ROBOTS = {"r-low": -2.0, "r-high": 0.5}  # robot intercepts (the confound)
CONFOUND_SHIFT = 3.0  # offset differs by robot only, no effect inside
TARGET = "fail_any"


def synthetic(n_per_robot: int, seed: int = 0, effect: float = TRUE_LOG_OR):
    rng = np.random.default_rng(seed)
    rows = []
    for i, (robot, alpha) in enumerate(ROBOTS.items()):
        x = rng.normal(size=n_per_robot)
        logit = alpha + effect * x
        y = rng.random(n_per_robot) < 1 / (1 + np.exp(-logit))
        for k in range(n_per_robot):
            row = {f: rng.normal() for f in PRIMARY}
            row[SPREAD_FAIL] = float(rng.random() < 0.2)
            row["diameter_rel"] = x[k]
            row["offset_rel"] = CONFOUND_SHIFT * i + rng.normal()
            row[ROBOT] = robot
            row[TARGET] = float(y[k])
            row["fail_residue"] = float(y[k])
            row["fail_shape"] = float(y[k])
            rows.append(row)
    return pd.DataFrame(rows)


def test_adjusted_effect_recovers_planted_effect() -> None:
    t = synthetic(400)
    r = fold_effects.adjusted_effect(t, TARGET, "diameter_rel", list(ROBOTS))
    assert r["estimable"]
    lo, hi = r["ci95"]
    assert lo < math.exp(TRUE_LOG_OR) < hi
    assert lo > 1


def test_robot_confound_is_not_an_effect() -> None:
    t = synthetic(400)
    # Pooled, offset tracks the high-failure robot
    pooled = t.groupby(t["offset_rel"] > CONFOUND_SHIFT / 2)[TARGET].mean()
    assert pooled[True] > pooled[False] + 0.2
    r = fold_effects.adjusted_effect(t, TARGET, "offset_rel", list(ROBOTS))
    lo, hi = r["ci95"]
    assert lo < 1 < hi


def test_holm_and_binary_feature() -> None:
    t = synthetic(200)
    eff = fold_effects.effects(t, list(ROBOTS))
    rows = eff["any"]
    assert [r["feature"] for r in rows] == PRIMARY
    assert all(r["p_holm"] >= r["p"] for r in rows if r["estimable"])
    binary = rows[0]
    assert binary["unit"] == "present vs absent"
    events = binary["events_if_present"][0] + binary["events_if_absent"][0]
    runs = binary["events_if_present"][1] + binary["events_if_absent"][1]
    assert (events, runs) == (binary["n_events"], binary["n_runs"])


def test_perfect_separation_is_not_estimable() -> None:
    t = synthetic(50)
    t[TARGET] = (t["diameter_rel"] > 0).astype(float)
    r = fold_effects.adjusted_effect(t, TARGET, "diameter_rel", list(ROBOTS))
    assert r["estimable"] is False


def test_within_robot_scale_removes_robot_means() -> None:
    t = synthetic(100)
    z = fold_effects.within_robot_scale(t["offset_rel"], t[ROBOT])
    assert z.groupby(t[ROBOT]).mean().abs().max() < 1e-9
    assert z.std() == pytest.approx(1.0, abs=0.05)


def test_power_formula_matches_simulation() -> None:
    sm = pytest.importorskip("statsmodels.api")
    t = synthetic(200)
    p = fold_effects.power_analysis(t, list(ROBOTS), n_tests=1)
    n = p["any"]["runs_needed_or_per_sd"]
    alpha = fold_effects.ALPHA
    hits = 0
    sims = 200
    for s in range(sims):
        sim = synthetic(n // 2, seed=100 + s, effect=math.log(2))
        X = fold_effects._design(sim, "diameter_rel")
        fit = sm.Logit(sim[TARGET], X).fit(disp=0)
        hits += fit.pvalues["diameter_rel"] < alpha
    assert abs(hits / sims - fold_effects.POWER) < 0.12


def test_cv_finds_within_robot_signal_not_confound() -> None:
    t = synthetic(150)
    robots = list(ROBOTS)
    signal = fold_cv.pooled_cv(t, TARGET, ["diameter_rel"], robots, n_boot=200)
    assert signal["delta_auc_within_robot"]["ci95"][0] > 0
    confound = fold_cv.pooled_cv(t, TARGET, ["offset_rel"], robots, n_boot=200)
    assert confound["delta_auc_within_robot"]["ci95"][0] <= 0


def test_leave_one_robot_out_and_within_robot() -> None:
    t = synthetic(150)
    robots = list(ROBOTS)
    loro = fold_cv.leave_one_robot_out(
        t, TARGET, ["diameter_rel"], robots, n_boot=100
    )
    assert set(loro) == set(robots)
    assert all(v["auc"]["value"] > 0.6 for v in loro.values())
    within = fold_cv.within_robot_cv(
        t, TARGET, ["diameter_rel"], robots, n_boot=100
    )
    assert all(v["auc"]["value"] > 0.6 for v in within.values())
