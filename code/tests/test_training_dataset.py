"""Offline tests for the candidate run-start training export."""

import json
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from maus_export.training_dataset import build_training_dataset


def test_only_fixed_recipe_candidates_are_exported(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "runs"
    raw.mkdir(parents=True)
    parameters = {
        "dispense.batter.cook_time": 4,
        "terminal_status": "failed",
        "duration": 100,
        "checks": [{"ok": False}],
        "post_run": {
            "terminal_status": "failed",
            "duration": 100,
            "checks": [{"ok": False}],
        },
        "unknown_scalar": "private",
    }
    (raw / "run.json").write_text(
        json.dumps(
            {
                "id": 1,
                "tsStart": "2026-01-01T00:00:00Z",
                "terminal_status": "succeeded",
                "parameters": parameters,
            }
        )
    )
    build_training_dataset(tmp_path)
    table = pq.read_table(tmp_path / "training" / "run_start.parquet")
    expected = {
        "dispense.batter.cook_time",
        "dispense.batter.centre_dispense_time",
        "dispense.batter.centre_flowrate",
        "dispense.batter.fan_power",
        "dispense.batter.concentric_inner_radius_mm",
        "dispense.batter.concentric_outer_radius_mm",
        "dispense.batter.concentric_outer_fan_tail_deg",
        "dispense.batter.concentric_inner_flowrate_g_s",
        "dispense.batter.concentric_fan_inset_mm",
        "dispense.oil.grams",
    }
    assert {c for c in table.column_names if c.startswith("param.")} == {
        f"param.{key}" for key in expected
    }
    manifest = json.loads(table.schema.metadata[b"maus.training.manifest"])
    manifest_params = {
        c for c in manifest["candidate_features"] if c.startswith("param.")
    }
    assert manifest_params == {f"param.{key}" for key in expected}
    assert table.column("terminal_status").to_pylist() == ["succeeded"]
    assert table.column("param.dispense.batter.cook_time").to_pylist() == ["4"]


@pytest.mark.parametrize(
    "key",
    [
        "dispense.batter.cook_time",
        "dispense.batter.centre_dispense_time",
        "dispense.batter.centre_flowrate",
        "dispense.batter.fan_power",
        "dispense.batter.concentric_inner_radius_mm",
        "dispense.batter.concentric_outer_radius_mm",
        "dispense.batter.concentric_outer_fan_tail_deg",
        "dispense.batter.concentric_inner_flowrate_g_s",
        "dispense.batter.concentric_fan_inset_mm",
        "dispense.oil.grams",
    ],
)
@pytest.mark.parametrize("value", [[1, {"private": "x"}], {"private": "x"}])
def test_nested_candidate_is_rejected_without_replacing_artifact(
    tmp_path: Path, key: str, value: object
) -> None:
    raw = tmp_path / "raw" / "runs"
    raw.mkdir(parents=True)
    (raw / "run.json").write_text(
        json.dumps(
            {
                "id": 1,
                "tsStart": "2026-01-01T00:00:00Z",
                "parameters": {key: value},
            }
        )
    )
    output = tmp_path / "training" / "run_start.parquet"
    output.parent.mkdir()
    output.write_bytes(b"old export")
    with pytest.raises(ValueError, match="scalar"):
        build_training_dataset(tmp_path)
    assert output.read_bytes() == b"old export"
    assert list(output.parent.iterdir()) == [output]


def test_schema_does_not_depend_on_later_input_rows(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "runs"
    raw.mkdir(parents=True)
    (raw / "earlier.json").write_text(
        json.dumps({"id": 1, "tsStart": "2026-01-01T00:00:00Z"})
    )
    build_training_dataset(tmp_path)
    output = tmp_path / "training" / "run_start.parquet"
    earlier_schema = pq.read_schema(output)
    (raw / "later.json").write_text(
        json.dumps(
            {
                "id": 2,
                "tsStart": "2026-02-01T00:00:00Z",
                "parameters": {
                    "dispense.oil.grams": 3,
                    "later_only": {"checks": [False], "duration": 999},
                },
            }
        )
    )
    build_training_dataset(tmp_path)
    later_schema = pq.read_schema(output)
    assert earlier_schema.equals(later_schema, check_metadata=True)
    assert "param.later_only" not in later_schema.names
    assert "param.dispense.oil.grams" in earlier_schema.names


def test_cli_exports_candidate_features_and_manifest(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "runs"
    raw.mkdir(parents=True)
    run = {
        "id": 1,
        "tsStart": "2026-10-30T16:00:00Z",
        "machineName": "robot-a",
        "topping": "apple",
        "isTest": False,
        "hotplateTemp": 190,
        "envTempStart": 22.5,
        "envHumidityStart": 45,
        "parameters": {
            "dispense.batter.cook_time": 5,
            "nested": {"b": 2, "a": [1]},
        },
        "terminal_status": "canceled",
        "tsEnd": "2026-10-30T16:03:00Z",
        "phases": [{"phase": "cook"}],
        "visionChecks": [{"ok": False}],
        "logs": [{"message": "private"}],
        "successFold": True,
        "config_id": "leak",
        "segment_id": 99,
    }
    (raw / "one.json").write_text(json.dumps(run))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "maus_export.cli",
            "--data-dir",
            str(tmp_path),
            "training",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "robot-a" not in result.stdout + result.stderr
    assert "apple" not in result.stdout + result.stderr
    table = pq.read_table(tmp_path / "training" / "run_start.parquet")
    row = table.to_pylist()[0]
    assert set(row) == {
        "id",
        "ts_start",
        "terminal_status",
        "robot",
        "topping",
        "is_test",
        "hotplate_temp",
        "env_temp_start",
        "env_humidity_start",
        "param.dispense.batter.cook_time",
        "param.dispense.batter.centre_dispense_time",
        "param.dispense.batter.centre_flowrate",
        "param.dispense.batter.fan_power",
        "param.dispense.batter.concentric_inner_radius_mm",
        "param.dispense.batter.concentric_outer_radius_mm",
        "param.dispense.batter.concentric_outer_fan_tail_deg",
        "param.dispense.batter.concentric_inner_flowrate_g_s",
        "param.dispense.batter.concentric_fan_inset_mm",
        "param.dispense.oil.grams",
    }
    assert row["terminal_status"] == "canceled"
    assert row["param.dispense.batter.cook_time"] == "5"
    assert "param.nested" not in row
    manifest = json.loads(table.schema.metadata[b"maus.training.manifest"])
    assert manifest["metadata"] == ["id", "ts_start"]
    assert manifest["target"] == "terminal_status"
    assert "param.dispense.batter.cook_time" in manifest["candidate_features"]


@pytest.mark.parametrize(
    "runs",
    [
        [],
        [{"id": 1, "tsStart": "2026-01-01T00:00:00Z"}] * 2,
        [{"id": 1, "tsStart": "2026-01-01T00:00:00Z", "parameters": []}],
        [{"id": 1, "tsStart": "2026-01-01T00:00:00"}],
        [
            {
                "id": 1,
                "tsStart": "2026-01-01T00:00:00Z",
                "parameters": {"bad": float("nan")},
            }
        ],
    ],
)
def test_invalid_input_preserves_previous_export(
    tmp_path: Path, runs: list[dict]
) -> None:
    raw = tmp_path / "raw" / "runs"
    raw.mkdir(parents=True)
    for index, run in enumerate(runs):
        (raw / f"{index}.json").write_text(json.dumps(run))
    output = tmp_path / "training" / "run_start.parquet"
    output.parent.mkdir()
    output.write_bytes(b"previous complete output")
    with pytest.raises(ValueError):
        build_training_dataset(tmp_path)
    assert output.read_bytes() == b"previous complete output"


def test_interrupted_write_does_not_replace_previous_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "raw" / "runs"
    raw.mkdir(parents=True)
    (raw / "run.json").write_text(
        json.dumps({"id": 1, "tsStart": "2026-01-01T00:00:00Z"})
    )
    output = tmp_path / "training" / "run_start.parquet"
    output.parent.mkdir()
    output.write_bytes(b"previous complete output")

    def interrupted_write(table: object, path: Path) -> None:
        Path(path).write_bytes(b"incomplete")
        raise OSError("simulated write failure")

    monkeypatch.setattr(pq, "write_table", interrupted_write)
    with pytest.raises(OSError, match="simulated write failure"):
        build_training_dataset(tmp_path)
    assert output.read_bytes() == b"previous complete output"
    assert list(output.parent.iterdir()) == [output]


def test_statuses_and_mixed_recipe_values_remain_distinct(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw" / "runs"
    raw.mkdir(parents=True)
    statuses = [
        "succeeded",
        "not_succeeded",
        "canceled",
        "aborted",
        None,
        "unknown",
        "future_status",
    ]
    values = [1, "1", True, False, None, "text", 3.5]
    for index, (status, value) in enumerate(zip(statuses, values, strict=True)):
        run = {
            "id": index + 1,
            "tsStart": "2026-01-01T00:00:00Z",
            "terminal_status": status,
            "parameters": {"dispense.batter.cook_time": value},
        }
        (raw / f"{index}.json").write_text(json.dumps(run))
    build_training_dataset(tmp_path)
    table = pq.read_table(tmp_path / "training" / "run_start.parquet")
    rows = table.to_pylist()
    assert [r["terminal_status"] for r in rows] == statuses
    assert [r["param.dispense.batter.cook_time"] for r in rows] == [
        "1",
        '"1"',
        "true",
        "false",
        None,
        '"text"',
        "3.5",
    ]
    assert all(r["robot"] is None and r["is_test"] is None for r in rows)
    manifest = json.loads(table.schema.metadata[b"maus.training.manifest"])
    assert manifest["parameter_encoding"] == "canonical JSON; null is missing"
    assert (
        manifest["target_semantics"] == "recorded terminal status, not quality"
    )


def test_cli_rejects_duplicate_json_keys_without_touching_old_output(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw" / "runs"
    raw.mkdir(parents=True)
    (raw / "run.json").write_text(
        '{"id":1,"id":2,"tsStart":"2026-01-01T00:00:00Z"}'
    )
    output = tmp_path / "training" / "run_start.parquet"
    output.parent.mkdir()
    output.write_bytes(b"old export")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "maus_export.cli",
            "--data-dir",
            str(tmp_path),
            "training",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "previous output" in result.stderr.lower()
    assert output.read_bytes() == b"old export"


@pytest.mark.parametrize(
    "sensor", ["hotplateTemp", "envTempStart", "envHumidityStart"]
)
@pytest.mark.parametrize("value", [True, False, "private-sentinel", [], {}])
def test_invalid_sensor_is_rejected_without_replacing_output(
    tmp_path: Path, sensor: str, value: object
) -> None:
    raw = tmp_path / "raw" / "runs"
    raw.mkdir(parents=True)
    run = {"id": 1, "tsStart": "2026-01-01T00:00:00Z", sensor: value}
    (raw / "run.json").write_text(json.dumps(run))
    output = tmp_path / "training" / "run_start.parquet"
    output.parent.mkdir()
    output.write_bytes(b"old export")
    with pytest.raises(ValueError, match="sensor"):
        build_training_dataset(tmp_path)
    assert output.read_bytes() == b"old export"
    assert list(output.parent.iterdir()) == [output]


@pytest.mark.parametrize(
    "sensor", ["hotplateTemp", "envTempStart", "envHumidityStart"]
)
@pytest.mark.parametrize("value", [True, False])
def test_cli_rejects_boolean_sensor_privately(
    tmp_path: Path, sensor: str, value: bool
) -> None:
    raw = tmp_path / "raw" / "runs"
    raw.mkdir(parents=True)
    run = {
        "id": 1,
        "tsStart": "2026-01-01T00:00:00Z",
        "machineName": "private-sentinel",
        sensor: value,
    }
    (raw / "run.json").write_text(json.dumps(run))
    output = tmp_path / "training" / "run_start.parquet"
    output.parent.mkdir()
    output.write_bytes(b"old export")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "maus_export.cli",
            "--data-dir",
            str(tmp_path),
            "training",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "private-sentinel" not in result.stdout + result.stderr
    assert "previous output" in result.stderr.lower()
    assert output.read_bytes() == b"old export"
    assert list(output.parent.iterdir()) == [output]


@pytest.mark.parametrize(
    ("sensor", "column"),
    [
        ("hotplateTemp", "hotplate_temp"),
        ("envTempStart", "env_temp_start"),
        ("envHumidityStart", "env_humidity_start"),
    ],
)
def test_numeric_and_null_sensors_are_accepted(
    tmp_path: Path, sensor: str, column: str
) -> None:
    raw = tmp_path / "raw" / "runs"
    raw.mkdir(parents=True)
    values = [0, -3, 2.5, None]
    for index, value in enumerate(values, start=1):
        run = {"id": index, "tsStart": "2026-01-01T00:00:00Z", sensor: value}
        (raw / f"{index}.json").write_text(json.dumps(run))
    build_training_dataset(tmp_path)
    table = pq.read_table(tmp_path / "training" / "run_start.parquet")
    assert table.column(column).to_pylist() == [0.0, -3.0, 2.5, None]


@pytest.mark.parametrize(
    "sensor", ["hotplateTemp", "envTempStart", "envHumidityStart"]
)
@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_nonfinite_sensor_value_is_rejected(
    tmp_path: Path, sensor: str, value: float
) -> None:
    raw = tmp_path / "raw" / "runs"
    raw.mkdir(parents=True)
    run = {"id": 1, "tsStart": "2026-01-01T00:00:00Z", sensor: value}
    (raw / "run.json").write_text(json.dumps(run))
    output = tmp_path / "training" / "run_start.parquet"
    output.parent.mkdir()
    output.write_bytes(b"old export")
    with pytest.raises(ValueError):
        build_training_dataset(tmp_path)
    assert output.read_bytes() == b"old export"
    assert list(output.parent.iterdir()) == [output]


def test_cli_failure_does_not_echo_private_field_values(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "runs"
    raw.mkdir(parents=True)
    (raw / "run.json").write_text(
        json.dumps(
            {
                "id": 1,
                "tsStart": "2026-01-01T00:00:00Z",
                "hotplateTemp": "private-sentinel",
            }
        )
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "maus_export.cli",
            "--data-dir",
            str(tmp_path),
            "training",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "private-sentinel" not in result.stderr + result.stdout
    assert "run.json" in result.stderr
    assert "sensor" in result.stderr


def test_duplicate_run_id_names_both_files(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "runs"
    raw.mkdir(parents=True)
    run = {"id": 1, "tsStart": "2026-01-01T00:00:00Z"}
    for name in ("a.json", "b.json"):
        (raw / name).write_text(json.dumps(run))
    with pytest.raises(ValueError, match=r"b\.json: .*also in a\.json"):
        build_training_dataset(tmp_path)


def test_malformed_json_names_the_file(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "runs"
    raw.mkdir(parents=True)
    (raw / "broken.json").write_text('{"id": 1, "machineName": "private"')
    with pytest.raises(ValueError, match=r"broken\.json: invalid JSON"):
        build_training_dataset(tmp_path)


def test_absent_recipe_key_is_null_and_manifest_warns_about_availability(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw" / "runs"
    raw.mkdir(parents=True)
    for index, parameters in enumerate(
        [{"dispense.oil.grams": 0}, {}, None], start=1
    ):
        run = {
            "id": index,
            "tsStart": "2026-01-01T00:00:00Z",
            "parameters": parameters,
        }
        (raw / f"{index}.json").write_text(json.dumps(run))
    build_training_dataset(tmp_path)
    table = pq.read_table(tmp_path / "training" / "run_start.parquet")
    assert table.column("param.dispense.oil.grams").to_pylist() == [
        "0",
        None,
        None,
    ]
    manifest = json.loads(table.schema.metadata[b"maus.training.manifest"])
    assert manifest["availability"] == "run-start availability not verified"


def test_overflowing_json_number_is_rejected(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "runs"
    raw.mkdir(parents=True)
    (raw / "run.json").write_text(
        '{"id":1,"tsStart":"2026-01-01T00:00:00Z","envTempStart":1e999}'
    )
    with pytest.raises(ValueError):
        build_training_dataset(tmp_path)


def test_config_and_segment_identifiers_are_never_recipe_features(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw" / "runs"
    raw.mkdir(parents=True)
    (raw / "run.json").write_text(
        json.dumps(
            {
                "id": 1,
                "tsStart": "2026-01-01T00:00:00Z",
                "parameters": {
                    "config_id": "x",
                    "segment_id": 9,
                    "dispense.oil.grams": 2,
                },
            }
        )
    )
    build_training_dataset(tmp_path)
    table = pq.read_table(tmp_path / "training" / "run_start.parquet")
    assert "param.config_id" not in table.column_names
    assert "param.segment_id" not in table.column_names
    assert "param.dispense.oil.grams" in table.column_names


def test_renamed_candidate_is_read_from_its_old_name(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "runs"
    raw.mkdir(parents=True)
    (raw / "run.json").write_text(
        json.dumps(
            {
                "id": 1,
                "tsStart": "2026-01-01T00:00:00Z",
                "parameters": {"dispense.batter.concentric_fan_tail_deg": 20},
            }
        )
    )
    build_training_dataset(tmp_path)
    table = pq.read_table(tmp_path / "training" / "run_start.parquet")
    column = "param.dispense.batter.concentric_outer_fan_tail_deg"
    assert table.column(column).to_pylist() == ["20"]
