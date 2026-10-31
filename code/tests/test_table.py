"""Offline tests for the flat run table, on synthetic runs."""

from pathlib import Path

import pandas as pd
import pytest

from maus_export.to_table import build_table, run_row, runs_to_table

RADIUS = "dispense.batter.concentric_outer_radius_offset_mm"
OIL = "dispense.oil.grams"
LIST_PARAM = "machine.y_offsets"
TEXT_PARAM = "dispense.batter.mode"


def make_run(
    run_id: int,
    robot: str = "maus-a",
    radius: float = 5.0,
    phases: tuple[str, ...] = ("init", "spread_batter", "cook", "plate_load"),
    checks: list[dict] | None = None,
    extra: dict | None = None,
) -> dict:
    params = {RADIUS: radius, OIL: 2, LIST_PARAM: [1, 2]}
    params.update(extra or {})
    return {
        "id": run_id,
        "machineName": robot,
        "machineId": 1,
        "tsStart": "2026-10-30T16:00:00.000Z",
        "tsEnd": "2026-10-30T16:03:00.000Z",
        "terminal_status": "succeeded",
        "phases": [
            {"phase": p, "elapsed_sec": 10.0 * i} for i, p in enumerate(phases)
        ],
        "visionChecks": checks or [],
        "logs": [],
        "parameters": params,
    }


@pytest.mark.parametrize("phase_state", ["missing", "null", "empty"])
def test_run_without_phases_remains_in_table(phase_state: str) -> None:
    run = make_run(1)
    if phase_state == "missing":
        del run["phases"]
    elif phase_state == "null":
        run["phases"] = None
    else:
        run["phases"] = []
    table = runs_to_table([run])
    assert len(table) == 1
    assert table.at[0, "last_phase"] is None
    assert not table.at[0, "is_full_cycle"]
    assert not any(c.startswith("phase_s.") for c in table.columns)


def test_recorded_phase_durations_are_preserved() -> None:
    row = run_row(make_run(1))
    assert row["phase_s.init"] == 10.0
    assert row["phase_s.spread_batter"] == 10.0
    assert row["phase_s.cook"] == 10.0
    assert row["phase_s.plate_load"] == 150.0
    assert row["last_phase"] == "plate_load"


def test_empty_export_reports_missing_runs_without_overwriting(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "runs.csv"
    parquet_path = tmp_path / "runs.parquet"
    csv_path.write_bytes(b"existing csv")
    parquet_path.write_bytes(b"existing parquet")

    with pytest.raises(ValueError, match="No raw run JSON.*fetch"):
        build_table(tmp_path)

    assert csv_path.read_bytes() == b"existing csv"
    assert parquet_path.read_bytes() == b"existing parquet"


def test_per_check_columns_use_last_attempt() -> None:
    checks = [
        {"phase": "spread", "ok": False, "errors": ["crepe too small"]},
        {"phase": "push_off", "ok": False, "errors": ["still on plate"]},
        {"phase": "push_off", "ok": True, "errors": []},
    ]
    row = run_row(make_run(1, checks=checks))
    assert row["check.spread.ok"] is False
    assert row["check.spread.error"] == "crepe too small"
    assert row["check.push_off.n"] == 2
    assert row["check.push_off.ok"] is True
    assert row["check.push_off.error"] is None
    assert row["check.fold1.n"] == 0 and row["check.fold1.ok"] is None


def test_full_cycle_flag() -> None:
    assert run_row(make_run(1))["is_full_cycle"] is True
    short = make_run(2, phases=("init", "plate_load"))
    assert run_row(short)["is_full_cycle"] is False


def test_constant_params_dropped_and_varying_kept() -> None:
    runs = [make_run(1, radius=5.0), make_run(2, radius=10.0)]
    df = runs_to_table(runs)
    assert f"param.{RADIUS}" in df.columns
    assert f"param.{OIL}" not in df.columns
    assert f"param.{LIST_PARAM}" not in df.columns


def test_key_unset_in_some_runs_counts_as_varying() -> None:
    runs = [make_run(1), make_run(2, extra={"fan_power": 0.9})]
    df = runs_to_table(runs)
    assert "param.fan_power" in df.columns
    assert df["config_id"].nunique() == 2


def test_segments_split_on_config_change_and_robot() -> None:
    runs = [
        make_run(1, radius=5.0),
        make_run(2, radius=5.0),
        make_run(3, radius=10.0),
        make_run(4, radius=5.0),
        make_run(5, robot="maus-b", radius=5.0),
    ]
    df = runs_to_table(runs).set_index("id")
    seg = df["segment_id"]
    assert seg[1] == seg[2]
    assert len({seg[2], seg[3], seg[4], seg[5]}) == 4
    # Same settings on another robot share config but not segment
    assert df.at[1, "config_id"] == df.at[5, "config_id"]


def test_mixed_type_param_writes_to_parquet(tmp_path: Path) -> None:
    runs = [
        make_run(1, extra={TEXT_PARAM: 1}),
        make_run(2, extra={TEXT_PARAM: "concentric"}),
    ]
    out = tmp_path / "runs.parquet"
    runs_to_table(runs).to_parquet(out, index=False)
    back = pd.read_parquet(out)
    assert set(back[f"param.{TEXT_PARAM}"]) == {"1", "concentric"}
