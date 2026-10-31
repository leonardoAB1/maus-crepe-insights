"""Robot-adjusted effects, power and model checks on synthetic runs that
plant a within-robot effect next to a robot confound."""

import math

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")  # analysis runs in the ml environment
pytest.importorskip("sklearn")

from maus_classify import fold_cv, fold_effects  # noqa: E402
from maus_classify.firth import firth_logit  # noqa: E402
from maus_classify.fold_effects import (  # noqa: E402
    PRIMARY,
    ROBOT,
    SPREAD_FAIL,
)

TRUE_LOG_OR = 1.0  # planted within-robot effect of diameter, per SD
ROBOTS = {"r-low": -2.0, "r-high": 0.5}  # robot intercepts (the confound)
CONFOUND_SHIFT = 3.0  # offset differs by robot only, no effect inside
TARGET = "fail_any"
PERM_SMALL = 99
PERM_REPEATS_SMALL = 2


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


def test_perfect_separation_stays_finite_with_firth() -> None:
    t = synthetic(50)
    t[TARGET] = (t["diameter_rel"] > 0).astype(float)
    r = fold_effects.adjusted_effect(t, TARGET, "diameter_rel", list(ROBOTS))
    assert r["estimable"]
    assert math.isfinite(r["odds_ratio"])
    assert r["wald"]["estimable"] is False


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


def test_permutation_test_separates_signal_from_confound() -> None:
    t = synthetic(150)
    robots = list(ROBOTS)
    kw = {"n_perm": PERM_SMALL, "n_repeats": PERM_REPEATS_SMALL}
    signal = fold_cv.permutation_test(t, TARGET, ["diameter_rel"], robots, **kw)
    assert signal["beats_null"]
    assert signal["observed_delta_auc"] > signal["null_q05_q95"][1]
    confound = fold_cv.permutation_test(t, TARGET, ["offset_rel"], robots, **kw)
    assert not confound["beats_null"]


def test_delta_report_labels_resample_interval() -> None:
    t = synthetic(80)
    r = fold_cv.pooled_cv(t, TARGET, ["diameter_rel"], list(ROBOTS), n_boot=50)
    assert "resample_ci95" in r["delta_auc_within_robot"]
    assert "excludes CV refit variance" in r["note"]


def test_firth_matches_ml_on_large_data() -> None:
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(0)
    X = sm.add_constant(rng.normal(size=(2000, 2)))
    y = (rng.random(2000) < 1 / (1 + np.exp(-(X @ [-1, 0.8, -0.5])))).astype(
        float
    )
    f = firth_logit(X, y)
    ml = sm.Logit(y, X).fit(disp=0)
    np.testing.assert_allclose(f.beta, ml.params, atol=0.01)
    np.testing.assert_allclose(f.profile_ci(1), ml.conf_int()[1], atol=0.01)


def test_firth_separated_table_is_finite() -> None:
    # Firth on a 2x2 table equals adding 0.5 to each cell
    x = np.r_[np.zeros(10), np.ones(10)]
    X = np.column_stack([np.ones(20), x])
    f = firth_logit(X, x)
    assert f.beta[1] == pytest.approx(math.log(10.5**2 / 0.5**2), abs=1e-4)
    lo, hi = f.profile_ci(1)
    assert 0 < lo < f.beta[1] < hi < math.inf


def test_adjusted_effect_reports_firth_and_wald() -> None:
    r = fold_effects.adjusted_effect(
        synthetic(200), TARGET, "diameter_rel", list(ROBOTS)
    )
    assert r["method"].startswith("Firth")
    assert r["wald"]["estimable"]
    # Firth shrinks toward 1 relative to plain maximum likelihood
    assert abs(math.log(r["odds_ratio"])) <= abs(
        math.log(r["wald"]["odds_ratio"])
    )


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
