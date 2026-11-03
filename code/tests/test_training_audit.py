"""Offline training audit tests using synthetic records."""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import TextIO

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from maus_export.training_audit import build_training_audit
from maus_export.training_dataset import build_training_dataset


def _inputs(data: Path) -> None:
    raw = data / "raw" / "runs"
    raw.mkdir(parents=True)
    rows = [
        {
            "id": 1,
            "tsStart": "2026-01-01T00:00:00Z",
            "machineName": "maus-a",
            "isTest": True,
            "terminal_status": "succeeded",
            "hotplateTemp": 100,
        },
        {
            "id": 2,
            "tsStart": "2026-02-01T00:00:00Z",
            "machineName": "maus-b",
            "isTest": False,
            "terminal_status": "canceled",
        },
        {
            "id": 3,
            "tsStart": "2026-02-02T00:00:00Z",
            "machineName": None,
            "isTest": None,
            "terminal_status": None,
        },
    ]
    for row in rows:
        (raw / f"{row['id']}.json").write_text(json.dumps(row))
    build_training_dataset(data)
    pd.DataFrame(
        {"id": [1, 4], "vision.spread.area_rel": [0.5, 0.7]}
    ).to_parquet(data / "vision.parquet", index=False)


def _cli(data: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "maus_export.cli",
            "--data-dir",
            str(data),
            "audit",
        ],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.slow
def test_cli_writes_aggregate_join_and_coverage(tmp_path: Path) -> None:
    _inputs(tmp_path)
    result = _cli(tmp_path)
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "training" / "audit.json").read_text())
    assert report["join"] == {
        "training_rows": 3,
        "training_unique_ids": 3,
        "vision_rows": 2,
        "vision_unique_ids": 2,
        "matched_training_runs": 1,
        "training_runs_without_vision_row": 2,
        "vision_rows_without_training_run": 1,
    }
    assert report["vision_metrics"]["vision.spread.area_rel"] == {
        "denominator_training_runs": 3,
        "non_null_runs": 1,
        "null_runs": 2,
        "coverage": 1 / 3,
    }
    assert report["features"]["hotplate_temp"]["null_runs"] == 2
    assert report["overall"]["status_counts"]["unknown"] == 1
    assert report["overall"]["test_counts"] == {
        "true": 1,
        "false": 1,
        "unknown": 1,
    }
    assert report["breakdowns"]["month"]["2026-02"]["runs"] == 2


@pytest.mark.parametrize("source", ["training", "vision"])
@pytest.mark.parametrize(
    "kind", ["duplicate", "null", "zero", "string", "float"]
)
@pytest.mark.slow
def test_invalid_ids_fail_privately_and_preserve_report(
    tmp_path: Path, source: str, kind: str
) -> None:
    _inputs(tmp_path)
    file = (
        tmp_path / "training" / "run_start.parquet"
        if source == "training"
        else tmp_path / "vision.parquet"
    )
    table = pq.read_table(file)
    values = table.column("id").to_pylist()
    if kind == "duplicate":
        values[1] = values[0]
    elif kind == "null":
        values[0] = None
    elif kind == "zero":
        values[0] = 0
    elif kind == "string":
        values = ["private-sentinel"] * len(values)
    else:
        values = [value + 0.5 for value in values]
    table = table.set_column(
        table.column_names.index("id"), "id", pa.array(values)
    )
    pq.write_table(table, file)
    output = tmp_path / "training" / "audit.json"
    output.write_text("old report")
    result = _cli(tmp_path)
    assert result.returncode != 0
    assert "stale" in result.stderr
    assert "private-sentinel" not in result.stdout + result.stderr
    assert output.read_text() == "old report"
    assert not list(output.parent.glob(".audit-*"))


@pytest.mark.parametrize(
    "encoded",
    [
        "null",
        "[1,2]",
        '{"private-sentinel":1}',
        "invalid-json-private-sentinel",
        "NaN",
        "Infinity",
        "-Infinity",
        "1e9999",
        " true ",
        "1.00",
        '"\\u0061"',
    ],
)
@pytest.mark.slow
def test_invalid_encoded_parameter_preserves_report_privately(
    tmp_path: Path, encoded: str
) -> None:
    _inputs(tmp_path)
    file = tmp_path / "training" / "run_start.parquet"
    table = pq.read_table(file)
    name = "param.dispense.oil.grams"
    values = [encoded, None, None]
    table = table.set_column(
        table.column_names.index(name), name, pa.array(values, pa.string())
    )
    pq.write_table(table, file)
    output = tmp_path / "training" / "audit.json"
    output.write_text("old report")
    result = _cli(tmp_path)
    assert result.returncode != 0
    assert "stale" in result.stderr
    assert "private-sentinel" not in result.stdout + result.stderr
    assert output.read_text() == "old report"
    assert not list(output.parent.glob(".audit-*"))


@pytest.mark.slow
def test_canonical_scalar_parameters_count_as_present(tmp_path: Path) -> None:
    _inputs(tmp_path)
    file = tmp_path / "training" / "run_start.parquet"
    table = pq.read_table(file)
    values_by_name = {
        "param.dispense.oil.grams": ['"private-sentinel"', "42", None],
        "param.dispense.batter.fan_power": ["true", "false", None],
        "param.dispense.batter.cook_time": ["1.25", "-1e-06", "-3"],
        "param.dispense.batter.centre_flowrate": [
            '"null"',
            '"[1,2]"',
            '"\\u00e9"',
        ],
    }
    for name, values in values_by_name.items():
        table = table.set_column(
            table.column_names.index(name), name, pa.array(values, pa.string())
        )
    pq.write_table(table, file)
    result = _cli(tmp_path)
    assert result.returncode == 0, result.stderr
    text = (tmp_path / "training" / "audit.json").read_text()
    assert "private-sentinel" not in text + result.stdout + result.stderr
    features = json.loads(text)["features"]
    assert features["param.dispense.oil.grams"]["non_null_runs"] == 2
    assert features["param.dispense.batter.fan_power"]["non_null_runs"] == 2
    assert features["param.dispense.batter.cook_time"]["non_null_runs"] == 3
    assert (
        features["param.dispense.batter.centre_flowrate"]["non_null_runs"] == 3
    )


@pytest.mark.parametrize(
    "case",
    [
        "unknown_metric",
        "extra_column",
        "string_metric",
        "infinite_metric",
        "invalid_robot",
        "invalid_test",
        "naive_time",
        "empty_training",
    ],
)
@pytest.mark.slow
def test_incompatible_inputs_fail_without_exposing_values(
    tmp_path: Path, case: str
) -> None:
    _inputs(tmp_path)
    training_file = tmp_path / "training" / "run_start.parquet"
    vision_file = tmp_path / "vision.parquet"
    file = (
        vision_file
        if case
        in {
            "unknown_metric",
            "extra_column",
            "string_metric",
            "infinite_metric",
        }
        else training_file
    )
    table = pq.read_table(file)
    if case == "unknown_metric":
        table = table.append_column(
            "vision.https://private-sentinel", pa.array([1.0] * table.num_rows)
        )
    elif case == "extra_column":
        table = table.append_column(
            "url", pa.array(["https://private-sentinel"] * table.num_rows)
        )
    elif case == "empty_training":
        table = table.slice(0, 0)
    else:
        if case in {"string_metric", "infinite_metric"}:
            name = "vision.spread.area_rel"
            values = ["private-sentinel"] * table.num_rows
            if case == "infinite_metric":
                values = [float("inf")] * table.num_rows
            replacement = pa.array(values)
        elif case in {"invalid_robot", "invalid_test"}:
            name = "robot" if case == "invalid_robot" else "is_test"
            replacement = pa.array(
                ["https://private-sentinel"] * table.num_rows
            )
        else:
            name = "ts_start"
            replacement = table.column(name).cast(pa.timestamp("us"))
        table = table.set_column(
            table.column_names.index(name), name, replacement
        )
    pq.write_table(table, file)
    output = tmp_path / "training" / "audit.json"
    output.write_text("old report")
    result = _cli(tmp_path)
    assert result.returncode != 0
    assert "stale" in result.stderr
    assert "private-sentinel" not in result.stdout + result.stderr
    assert output.read_text() == "old report"


@pytest.mark.slow
def test_breakdowns_include_group_feature_and_vision_denominators(
    tmp_path: Path,
) -> None:
    _inputs(tmp_path)
    result = _cli(tmp_path)
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "training" / "audit.json").read_text())
    group = report["breakdowns"]["month"]["2026-02"]
    assert group["vision_row_coverage"] == {
        "denominator_training_runs": 2,
        "non_null_runs": 0,
        "null_runs": 2,
        "coverage": 0.0,
    }
    assert group["features"]["hotplate_temp"]["null_runs"] == 2
    assert group["vision_metrics"]["vision.spread.area_rel"]["null_runs"] == 2
    assert report["breakdowns"]["test"]["unknown"]["runs"] == 1


@pytest.mark.slow
def test_report_identifies_inputs_and_utc_generation_time(
    tmp_path: Path,
) -> None:
    _inputs(tmp_path)
    result = _cli(tmp_path)
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "training" / "audit.json").read_text())
    assert report["sources"] == {
        "training": "training/run_start.parquet",
        "vision": "vision.parquet",
    }
    assert pd.Timestamp(report["generated_at_utc"]).tzname() == "UTC"


@pytest.mark.slow
def test_full_stdout_does_not_invalidate_completed_report(
    tmp_path: Path,
) -> None:
    if not Path("/dev/full").exists():
        pytest.skip("/dev/full is unavailable")
    _inputs(tmp_path)
    output = tmp_path / "training" / "audit.json"
    output.write_text("old report")
    with Path("/dev/full").open("w") as full:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "maus_export.cli",
                "--data-dir",
                str(tmp_path),
                "audit",
            ],
            stdout=full,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            check=False,
        )
    assert result.returncode == 0, result.stderr
    assert "stale" not in result.stderr
    assert json.loads(output.read_text())["join"]["training_rows"] == 3


@pytest.mark.slow
def test_success_does_not_echo_private_path(tmp_path: Path) -> None:
    _inputs(tmp_path)
    result = _cli(tmp_path)
    assert result.returncode == 0, result.stderr
    assert str(tmp_path) not in result.stdout + result.stderr


@pytest.mark.slow
def test_missing_vision_is_an_error_not_zero_coverage(tmp_path: Path) -> None:
    _inputs(tmp_path)
    (tmp_path / "vision.parquet").unlink()
    output = tmp_path / "training" / "audit.json"
    output.write_text("old report")
    result = _cli(tmp_path)
    assert result.returncode != 0
    assert "stale" in result.stderr
    assert output.read_text() == "old report"


def test_interrupted_atomic_write_preserves_old_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _inputs(tmp_path)
    output = tmp_path / "training" / "audit.json"
    output.write_text("old report")

    def interrupted(report: object, file: TextIO, **kwargs: object) -> None:
        file.write("incomplete")
        raise OSError("simulated failure")

    monkeypatch.setattr("maus_export.training_audit.json.dump", interrupted)
    with pytest.raises(OSError, match="simulated failure"):
        build_training_audit(tmp_path)
    assert output.read_text() == "old report"
    assert not list(output.parent.glob(".audit-*"))


@pytest.mark.slow
def test_private_values_never_appear_in_report_or_cli(tmp_path: Path) -> None:
    _inputs(tmp_path)
    file = tmp_path / "training" / "run_start.parquet"
    table = pq.read_table(file)
    for name in ("terminal_status", "topping", "param.dispense.oil.grams"):
        value = "https://private-sentinel/run/999999"
        if name.startswith("param."):
            value = json.dumps(value)
        values = [value] * table.num_rows
        table = table.set_column(
            table.column_names.index(name), name, pa.array(values)
        )
    pq.write_table(table, file)
    result = _cli(tmp_path)
    assert result.returncode == 0, result.stderr
    text = (tmp_path / "training" / "audit.json").read_text()
    assert "private-sentinel" not in text + result.stdout + result.stderr
    assert "999999" not in text + result.stdout + result.stderr
    report = json.loads(text)
    assert report["overall"]["status_counts"]["other"] == 3


@pytest.mark.slow
def test_null_metric_on_matched_row_is_not_missing_vision(
    tmp_path: Path,
) -> None:
    _inputs(tmp_path)
    pd.DataFrame(
        {
            "id": [1, 2, 4],
            "vision.spread.area_rel": [0.5, None, 0.7],
        }
    ).to_parquet(tmp_path / "vision.parquet", index=False)
    result = _cli(tmp_path)
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "training" / "audit.json").read_text())
    assert report["join"]["matched_training_runs"] == 2
    assert report["join"]["training_runs_without_vision_row"] == 1
    assert report["vision_metrics"]["vision.spread.area_rel"]["null_runs"] == 2


@pytest.mark.slow
def test_absent_metric_column_is_reported_as_unavailable(
    tmp_path: Path,
) -> None:
    _inputs(tmp_path)
    result = _cli(tmp_path)
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "training" / "audit.json").read_text())
    assert "vision.fold1.iou" in report["unavailable_vision_metrics"]
    assert "vision.spread.area_rel" not in report["unavailable_vision_metrics"]


@pytest.mark.slow
def test_status_breakdown_includes_vision_coverage(tmp_path: Path) -> None:
    _inputs(tmp_path)
    result = _cli(tmp_path)
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "training" / "audit.json").read_text())
    statuses = report["breakdowns"]["status"]
    assert statuses["succeeded"]["vision_row_coverage"]["non_null_runs"] == 1
    assert statuses["unknown"]["vision_row_coverage"]["null_runs"] == 1
