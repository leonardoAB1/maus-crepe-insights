"""Offline tests for the flat run table, on synthetic runs."""

import json
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


def test_stable_parameter_id_is_independent_of_export_subset() -> None:
    first = make_run(1, radius=5.0)
    second = make_run(2, radius=10.0)
    full = runs_to_table([first, second]).set_index("id")
    first_only = runs_to_table([first]).set_index("id")
    second_only = runs_to_table([second]).set_index("id")
    assert first_only.at[1, "parameter_id"] == full.at[1, "parameter_id"]
    assert second_only.at[2, "parameter_id"] == full.at[2, "parameter_id"]
    assert first_only.at[1, "parameter_id"] != second_only.at[2, "parameter_id"]


def test_parameter_id_has_versioned_canonical_json_digest() -> None:
    run = make_run(1)
    run["parameters"] = {"b": [True, None], "a": 1}
    expected = (
        "v1:sha256:"
        "1cc69c7fa23616ca2ec3ee70d24390a6225c8832db8a4c814c7e0e7f942f8668"
    )
    assert runs_to_table([run]).at[0, "parameter_id"] == expected


def test_parameter_id_sorts_nested_mapping_keys_but_not_lists() -> None:
    runs = [make_run(i) for i in range(1, 5)]
    runs[0]["parameters"] = {"z": [{"b": 2, "a": 1}], "a": 3}
    runs[1]["parameters"] = {"a": 3, "z": [{"a": 1, "b": 2}]}
    runs[2]["parameters"] = {"a": 3, "z": [{"a": 1, "b": 2}, 0]}
    runs[3]["parameters"] = {"a": 3, "z": [0, {"a": 1, "b": 2}]}
    ids = runs_to_table(runs).set_index("id")["parameter_id"]
    assert ids[1] == ids[2]
    assert ids[1] != ids[3]
    assert ids[3] != ids[4]


def test_parameter_id_missing_null_and_empty_mapping() -> None:
    runs = [make_run(i) for i in range(1, 6)]
    del runs[0]["parameters"]
    runs[1]["parameters"] = None
    runs[2]["parameters"] = {}
    runs[3]["parameters"] = {"value": None}
    runs[4]["parameters"] = {"value": {}}
    ids = runs_to_table(runs).set_index("id")["parameter_id"]
    assert ids[1] == ids[2]
    assert len({ids[1], ids[3], ids[4], ids[5]}) == 4


def test_parameter_id_preserves_json_numeric_types_before_coercion() -> None:
    runs = [make_run(i) for i in range(1, 6)]
    for run, value in zip(runs, [1, 1.0, "1", True, None], strict=True):
        run["parameters"] = {"value": value}
    ids = runs_to_table(runs).set_index("id")["parameter_id"]
    assert ids.nunique() == len(runs)
    for run in runs:
        assert runs_to_table([run]).at[0, "parameter_id"] == ids[run["id"]]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_parameter_id_rejects_nonfinite_numbers(value: float) -> None:
    run = make_run(1)
    run["parameters"] = {"value": value}
    with pytest.raises(ValueError, match="Out of range float"):
        runs_to_table([run])


def test_parameter_id_ignores_robot_and_other_run_metadata() -> None:
    first = make_run(1)
    second = make_run(2, robot="maus-b")
    second.update(
        {
            "machineId": 42,
            "terminal_status": "aborted",
            "topping": "synthetic",
            "tsStart": "2026-10-29T16:00:00.000Z",
            "logs": [{"level": 40}],
            "visionChecks": [{"phase": "spread", "ok": False}],
        }
    )
    table = runs_to_table([first, second]).set_index("id")
    assert table.at[1, "parameter_id"] == table.at[2, "parameter_id"]
    assert table.at[1, "segment_id"] != table.at[2, "segment_id"]


def test_legacy_config_and_segment_ids_remain_unchanged() -> None:
    runs = [
        make_run(1),
        make_run(2, radius=10),
        make_run(3),
        make_run(4, robot="maus-b"),
        make_run(5, extra={"v": "x"}),
    ]
    table = runs_to_table(runs).set_index("id")
    assert table["config_id"].tolist() == [
        "ff7aa6995a",
        "8d1e87488f",
        "ff7aa6995a",
        "ff7aa6995a",
        "85a87db64a",
    ]
    assert table["segment_id"].tolist() == [0, 1, 2, 4, 3]


def test_build_table_writes_parameter_identity_to_both_files(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "raw" / "runs"
    directory.mkdir(parents=True)
    runs = [make_run(1), make_run(2, radius=10)]
    for run in runs:
        (directory / f"{run['id']}.json").write_text(
            json.dumps(run), encoding="utf-8"
        )
    expected = build_table(tmp_path)["parameter_id"].tolist()
    csv = pd.read_csv(tmp_path / "runs.csv")
    assert csv["parameter_id"].tolist() == expected
    back = pd.read_parquet(tmp_path / "runs.parquet")
    assert back["parameter_id"].tolist() == expected


def test_mixed_type_param_writes_to_parquet(tmp_path: Path) -> None:
    runs = [
        make_run(1, extra={TEXT_PARAM: 1}),
        make_run(2, extra={TEXT_PARAM: "concentric"}),
    ]
    out = tmp_path / "runs.parquet"
    table = runs_to_table(runs)
    table.to_parquet(out, index=False)
    back = pd.read_parquet(out)
    assert set(back[f"param.{TEXT_PARAM}"]) == {"1", "concentric"}
    assert back["parameter_id"].tolist() == table["parameter_id"].tolist()
