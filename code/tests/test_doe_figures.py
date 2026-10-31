"""Offline tests for the DoE figures, on the simulated batch (no real data)."""

import pandas as pd
import pytest

pytest.importorskip("scipy")  # analysis runs in the ml environment

from maus_export.doe import dsd  # noqa: E402
from maus_export.doe_analyze import analyze, analyze_sheet  # noqa: E402
from maus_export.doe_figures import (  # noqa: E402
    SIM_OUTCOME,
    design_figure,
    effects_figure,
    simulated_batch,
    simulated_factors,
)


def test_design_figure_writes_png(tmp_path) -> None:
    factors = simulated_factors()
    path = design_figure(dsd(len(factors)), factors, tmp_path / "d.png")
    assert path.stat().st_size > 0


@pytest.mark.parametrize("simulated", [True, False])
def test_effects_figure_writes_png(tmp_path, simulated: bool) -> None:
    result = analyze_sheet(*simulated_batch())
    path = effects_figure(result, SIM_OUTCOME, tmp_path / "e.png", simulated)
    assert path.stat().st_size > 0


def test_analyze_writes_one_figure_per_outcome(tmp_path) -> None:
    sheet, runs, vision = simulated_batch()
    runs.to_parquet(tmp_path / "runs.parquet")
    vision.to_parquet(tmp_path / "vision.parquet")
    sheet.to_parquet(tmp_path / "sheet.parquet")
    result = analyze(tmp_path, tmp_path / "sheet.parquet")
    for name in result["effects"]["outcome"].unique():
        assert (tmp_path / "doe" / f"{result['doe_id']}-{name}.png").is_file()
    assert isinstance(result["frame"], pd.DataFrame)
