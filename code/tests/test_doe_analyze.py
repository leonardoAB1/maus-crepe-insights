"""Offline tests for the DSD analysis: plant a known response surface on a
synthetic run sheet and check it comes back (no real data)."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")  # analysis runs in the ml environment

from maus_export.doe import RUN_ID  # noqa: E402
from maus_export.doe_analyze import (  # noqa: E402
    ALPHA,
    MAIN,
    SQUARE,
    analyze_sheet,
    best_level,
    load_results,
    predict,
    to_onnx,
)
from maus_export.doe_figures import (  # noqa: E402
    SIM_YC,
    simulated_batch,
    simulated_factors,
)

FACTORS = simulated_factors()
filled_sheet = simulated_batch


def term(effects: pd.DataFrame, kind: str, factor: str) -> pd.Series:
    sub = effects[
        (effects["outcome"] == "spread_diameter")
        & (effects["kind"] == kind)
        & (effects["factor"] == factor)
    ]
    return sub.iloc[0]


def test_planted_surface_is_recovered() -> None:
    sheet, runs, vision = filled_sheet()
    effects = analyze_sheet(sheet, runs, vision)["effects"]
    fan = term(effects, MAIN, FACTORS[0].name)
    yc = term(effects, MAIN, FACTORS[2].name)
    curve = term(effects, SQUARE, FACTORS[4].name)
    assert fan["coef"] > 0 and fan["p"] < ALPHA
    assert yc["coef"] < 0 and yc["p"] < ALPHA
    assert curve["coef"] < 0 and curve["p"] < ALPHA
    idle = term(effects, MAIN, FACTORS[3].name)
    assert abs(idle["coef"]) < abs(SIM_YC) / 2


def test_constant_outcomes_are_skipped() -> None:
    sheet, runs, vision = filled_sheet()
    effects = analyze_sheet(sheet, runs, vision)["effects"]
    assert set(effects["outcome"]) == {"spread_diameter"}


def test_mismatched_settings_are_reported() -> None:
    sheet, runs, vision = filled_sheet()
    runs.loc[0, FACTORS[0].column] = 0.5
    _, mismatches = load_results(sheet, runs, vision)
    assert len(mismatches) == 1
    assert mismatches["factor"].iloc[0] == FACTORS[0].name


def test_partial_batch_is_analyzed_and_unknown_runs_raise() -> None:
    sheet, runs, vision = filled_sheet()
    sheet.loc[sheet.index[:3], RUN_ID] = pd.NA
    result = analyze_sheet(sheet, runs, vision)
    assert len(result["run_ids"]) == len(sheet) - 3
    with pytest.raises(ValueError):
        load_results(sheet, runs.iloc[5:], vision)


def test_best_level() -> None:
    assert best_level(0.1, 0.0, 1) == 1.0
    assert best_level(0.1, 0.0, -1) == -1.0
    assert best_level(0.0, -0.2, 1) == 0.0  # peak at the centre


def test_onnx_matches_numpy() -> None:
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    sheet, runs, vision = filled_sheet()
    effects = analyze_sheet(sheet, runs, vision)["effects"]
    outcomes = list(effects["outcome"].unique())
    model = to_onnx(effects, FACTORS, outcomes)
    sess = ort.InferenceSession(model.SerializeToString())
    coded = np.random.default_rng(1).uniform(-1, 1, (10, len(FACTORS)))
    got = sess.run(None, {"x": coded.astype(np.float32)})[0]
    want = predict(effects, FACTORS, outcomes, coded)
    np.testing.assert_allclose(got, want, atol=1e-4)
