"""Offline tests for the DSD run sheet generator (synthetic data only)."""

import numpy as np
import pandas as pd
import pytest

from maus_export.doe import (
    CENTRE,
    CODED_SUFFIX,
    RUN_ID,
    Factor,
    build_run_sheet,
    conference_matrix,
    design_report,
    dsd,
    factors_from_history,
    run_sheet,
    sheet_factors,
)

ORDERS = (4, 6, 8, 12, 14, 18)
FAN = "dispense.batter.fan_power"
YC = "machine.y_center"
NAMES = (FAN, YC)
TOL = 1e-9


def history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            f"param.{FAN}": [0.8, 0.85, 1.0, 0.9],
            f"param.{YC}": [-47.0, -46.0, -42.0, -44.0],
        }
    )


def shortlist(fan: list[float]) -> dict:
    return {"shortlist": [{"safe_range": {FAN: fan}}]}


@pytest.mark.parametrize("n", ORDERS)
def test_conference_matrix_is_orthogonal(n: int) -> None:
    c = conference_matrix(n)
    assert (np.diag(c) == 0).all()
    assert (c.T @ c == (n - 1) * np.eye(n)).all()


def test_conference_matrix_rejects_unsupported_order() -> None:
    with pytest.raises(ValueError):
        conference_matrix(10)  # 9 is not prime


def test_dsd_is_balanced_and_orthogonal() -> None:
    design = dsd(6, n_fake=2, n_centre=3)
    assert design.shape == (19, 6)  # 2 * 8 + 3
    rep = design_report(design)
    for counts in rep["balance"]:
        assert counts["-1"] == counts["1"]
    assert rep["max_corr_main"] < TOL
    assert rep["max_corr_main_square"] < TOL
    assert ((design == 0).all(axis=1)).sum() == 3


def test_decode_maps_coded_levels_to_range() -> None:
    f = Factor(FAN, 0.85, 1.0)
    assert list(f.decode(np.array([-1, 0, 1]))) == [0.85, 0.925, 1.0]
    np.testing.assert_allclose(f.encode(f.decode(np.array([-1, 1]))), [-1, 1])


def test_levels_come_from_shortlist_else_history() -> None:
    factors = factors_from_history(history(), shortlist([0.85, 1.0]), NAMES)
    assert (factors[0].low, factors[0].high) == (0.85, 1.0)
    assert (factors[1].low, factors[1].high) == (-47.0, -42.0)


def test_levels_outside_history_raise() -> None:
    with pytest.raises(ValueError):
        factors_from_history(history(), shortlist([0.5, 1.0]), NAMES)


def test_single_value_factor_raises() -> None:
    with pytest.raises(ValueError):
        factors_from_history(history(), shortlist([0.9, 0.9]), NAMES)


def test_run_sheet_is_random_but_reproducible() -> None:
    design = dsd(2, n_fake=2)
    factors = [Factor(FAN, 0.85, 1.0), Factor(YC, -46.0, -42.0)]
    a = run_sheet(design, factors, "r1", seed=1)
    b = run_sheet(design, factors, "r1", seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert list(a["std_order"]) != sorted(a["std_order"])
    assert list(a["run_order"]) == list(range(1, len(a) + 1))
    assert a[RUN_ID].isna().all()
    assert (a["point_type"] == CENTRE).sum() == 3
    assert a[FAN].between(0.85, 1.0).all()
    assert sorted(a[FAN + CODED_SUFFIX].unique()) == [-1, 0, 1]
    assert sheet_factors(a) == factors


def test_build_run_sheet_writes_parquet_and_csv(tmp_path) -> None:
    runs = history()
    for name in (
        "dispense.batter.concentric_inner_fan_power",
        "machine.z_scrape_start",
        "dispense.batter.concentric_inner_radius_offset_mm",
        "dispense.batter.concentric_outer_radius_offset_mm",
    ):
        runs[f"param.{name}"] = [0.0, 1.0, 2.0, 3.0]
    runs.to_parquet(tmp_path / "runs.parquet")
    sheet = build_run_sheet(tmp_path, "r1")
    doe_id = sheet["doe_id"].iloc[0]
    assert (tmp_path / "doe" / f"{doe_id}.csv").is_file()
    back = pd.read_parquet(tmp_path / "doe" / f"{doe_id}.parquet")
    assert len(back) == 19
