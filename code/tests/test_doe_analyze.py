"""Offline tests for the DSD analysis: plant a known response surface on a
synthetic run sheet and check it comes back (no real data)."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")  # analysis runs in the ml environment

from maus_export.doe import (  # noqa: E402
    CODED_SUFFIX,
    RUN_ID,
    Factor,
    dsd,
    run_sheet,
)
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

FACTORS = [
    Factor("dispense.batter.fan_power", 0.85, 1.0),
    Factor("dispense.batter.concentric_inner_fan_power", 0.82, 0.97),
    Factor("machine.y_center", -46.0, -42.0),
    Factor("machine.z_scrape_start", 38.9, 41.5),
    Factor("dispense.batter.concentric_inner_radius_offset_mm", 2.0, 25.0),
    Factor("dispense.batter.concentric_outer_radius_offset_mm", -10.0, 10.0),
]
DIAMETER = "vision.spread.diameter_rel"
# Planted surface on the coded scale: fan raises the diameter, y_center
# lowers it, inner radius offset is curved, the rest do nothing
BASE, FAN_EFFECT, YC_EFFECT, CURVE = 0.8, 0.05, -0.03, -0.04
NOISE = 0.005
FIRST_RUN = 1000


def filled_sheet(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    sheet = run_sheet(dsd(len(FACTORS)), FACTORS, "r1", seed=seed)
    sheet[RUN_ID] = np.arange(FIRST_RUN, FIRST_RUN + len(sheet))
    x = sheet[[f.name + CODED_SUFFIX for f in FACTORS]].to_numpy(float)
    rng = np.random.default_rng(seed)
    y = (
        BASE
        + FAN_EFFECT * x[:, 0]
        + YC_EFFECT * x[:, 2]
        + CURVE * x[:, 4] ** 2
        + rng.normal(0, NOISE, len(x))
    )
    runs = pd.DataFrame(
        {
            "id": sheet[RUN_ID],
            "robot": "r1",
            "topping": "plain",
            "status": "succeeded",
            "check.spread.ok": True,
            **{f.column: sheet[f.name] for f in FACTORS},
        }
    )
    vision = pd.DataFrame({"id": sheet[RUN_ID], DIAMETER: y})
    return sheet, runs, vision


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
    assert abs(idle["coef"]) < abs(YC_EFFECT) / 2


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
