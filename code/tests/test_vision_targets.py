"""Offline tests for the separate vision-target training export."""

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from maus_export.training_dataset import build_training_dataset

CLI_TIMEOUT_SECONDS = 30

TARGETS = [
    "vision.spread.area_rel",
    "vision.spread.diameter_rel",
    "vision.spread.offset_rel",
    "vision.spread.circularity",
    "vision.spread.hole_frac",
    "vision.spread.uncovered_frac",
    "vision.spread.l_mean",
    "vision.spread.l_std",
    "vision.spread.circle_diameter_rel",
    "vision.spread.circle_offset_rel",
    "vision.spread.residue_frac",
    "vision.fold1.area_rel",
    "vision.fold1.l_mean",
    "vision.fold1.l_std",
    "vision.fold1.iou",
    "vision.fold2.area_rel",
    "vision.fold2.l_mean",
    "vision.fold2.l_std",
    "vision.fold2.iou",
    "vision.fold2.area_ratio",
]


def make_run_start(path: Path) -> None:
    raw = path / "raw" / "runs"
    raw.mkdir(parents=True)
    for number in (1, 2):
        run = {
            "id": number,
            "tsStart": "2026-01-01T00:00:00Z",
            "terminal_status": "not_succeeded",
            "machineName": "private-sentinel",
            "parameters": {"dispense.oil.grams": 2},
        }
        (raw / f"{number}.json").write_text(json.dumps(run))
    build_training_dataset(path)


def cli(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "maus_export.cli",
            "--data-dir",
            str(path),
            "vision-targets",
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=CLI_TIMEOUT_SECONDS,
    )


@pytest.mark.slow
def test_cli_left_joins_vision_targets_with_fixed_roles(tmp_path: Path) -> None:
    make_run_start(tmp_path)
    source = pa.table(
        {
            "id": pa.array([1], type=pa.int64()),
            "vision.spread.area_rel": pa.array([0.75], type=pa.float64()),
            "vision.spread.zone_r_px": pa.array([100.0], type=pa.float64()),
        }
    )
    pq.write_table(source, tmp_path / "vision.parquet")
    result = cli(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "private-sentinel" not in result.stdout + result.stderr
    table = pq.read_table(tmp_path / "training" / "vision_targets.parquet")
    run_start = pq.read_table(tmp_path / "training" / "run_start.parquet")
    old_manifest = json.loads(
        run_start.schema.metadata[b"maus.training.manifest"]
    )
    manifest = json.loads(table.schema.metadata[b"maus.training.manifest"])
    assert manifest["metadata"] == ["id", "ts_start"]
    assert manifest["candidate_features"] == old_manifest["candidate_features"]
    assert manifest["targets"] == TARGETS
    assert table.column_names == [
        "id",
        "ts_start",
        *manifest["candidate_features"],
        *TARGETS,
    ]
    assert len(table) == 2
    assert table.column("vision.spread.area_rel").to_pylist() == [0.75, None]
    assert table.column("vision.fold2.iou").to_pylist() == [None, None]
    assert all(
        table.schema.field(name).type == pa.float64() for name in TARGETS
    )
    assert "terminal_status" not in table.column_names
    assert "vision.spread.zone_r_px" not in table.column_names


@pytest.mark.parametrize("ids", [[1, 1], [1, None], [0], [-5], [999999]])
@pytest.mark.slow
def test_invalid_vision_ids_preserve_previous_output_privately(
    tmp_path: Path, ids: list[int | None]
) -> None:
    make_run_start(tmp_path)
    pq.write_table(
        pa.table({"id": pa.array(ids, type=pa.int64())}),
        tmp_path / "vision.parquet",
    )
    output = tmp_path / "training" / "vision_targets.parquet"
    output.write_bytes(b"stale")
    result = cli(tmp_path)
    assert result.returncode != 0
    assert "stale" in result.stderr
    assert "999999" not in result.stdout + result.stderr
    assert "private-sentinel" not in result.stdout + result.stderr
    assert output.read_bytes() == b"stale"
    assert list(output.parent.glob(".vision_targets-*")) == []


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_id",
        "null_id",
        "negative_id",
        "null_timestamp",
        "empty",
        "bad_type",
        "extra_column",
        "bad_manifest",
        "duplicate_manifest_key",
        "feature_leak",
        "missing_manifest",
    ],
)
@pytest.mark.slow
def test_invalid_run_start_schema_or_manifest_is_rejected(
    tmp_path: Path, mutation: str
) -> None:
    make_run_start(tmp_path)
    path = tmp_path / "training" / "run_start.parquet"
    start = pq.read_table(path)
    if mutation == "duplicate_id":
        start = start.set_column(0, "id", pa.array([1, 1], type=pa.int64()))
    elif mutation == "null_id":
        start = start.set_column(0, "id", pa.array([1, None], type=pa.int64()))
    elif mutation == "negative_id":
        start = start.set_column(0, "id", pa.array([1, -1], type=pa.int64()))
    elif mutation == "null_timestamp":
        start = start.set_column(
            1,
            start.schema.field(1),
            pa.array([None, None], type=start.schema.field(1).type),
        )
    elif mutation == "empty":
        start = start.slice(0, 0)
    elif mutation == "bad_type":
        start = start.set_column(
            start.schema.get_field_index("is_test"),
            "is_test",
            pa.array(["true", None]),
        )
    elif mutation == "extra_column":
        start = start.append_column("check.secret", pa.array([0, 1]))
    elif mutation == "missing_manifest":
        start = start.replace_schema_metadata(None)
    else:
        manifest = json.loads(start.schema.metadata[b"maus.training.manifest"])
        if mutation == "bad_manifest":
            manifest["target"] = "private-sentinel"
        if mutation == "feature_leak":
            manifest["candidate_features"][0] = "terminal_status"
        encoded = json.dumps(manifest).encode()
        if mutation == "duplicate_manifest_key":
            encoded = b'{"schema_version":1,"schema_version":1}'
        start = start.replace_schema_metadata(
            {b"maus.training.manifest": encoded}
        )
    pq.write_table(start, path)
    pq.write_table(
        pa.table({"id": pa.array([1], type=pa.int64())}),
        tmp_path / "vision.parquet",
    )
    output = tmp_path / "training" / "vision_targets.parquet"
    output.write_bytes(b"stale")
    result = cli(tmp_path)
    assert result.returncode != 0
    assert "private-sentinel" not in result.stdout + result.stderr
    assert "stale" in result.stderr
    assert output.read_bytes() == b"stale"


@pytest.mark.parametrize(
    "vision",
    [
        pa.table({"vision.spread.area_rel": [1.0]}),
        pa.table({"id": [1], "vision.private-sentinel": [0.0]}),
        pa.table({"id": [1], "vision.spread.area_rel": [True]}),
        pa.table({"id": [1], "vision.spread.area_rel": ["secret"]}),
        pa.table({"id": [1], "vision.spread.area_rel": [2]}),
        pa.table({"id": [1], "vision.spread.area_rel": [float("inf")]}),
        pa.table({"id": [1], "vision.spread.zone_r_px": [float("-inf")]}),
        pa.Table.from_arrays(
            [pa.array([1]), pa.array([1])], names=["id", "id"]
        ),
    ],
)
@pytest.mark.slow
def test_invalid_vision_schema_and_values_preserve_output(
    tmp_path: Path, vision: pa.Table
) -> None:
    make_run_start(tmp_path)
    pq.write_table(vision, tmp_path / "vision.parquet")
    output = tmp_path / "training" / "vision_targets.parquet"
    output.write_bytes(b"stale")
    result = cli(tmp_path)
    assert result.returncode != 0
    assert "private-sentinel" not in result.stdout + result.stderr
    assert "secret" not in result.stdout + result.stderr
    assert "stale" in result.stderr
    assert output.read_bytes() == b"stale"


@pytest.mark.parametrize("empty", [True, False])
@pytest.mark.slow
def test_nan_and_empty_vision_produce_nulls(
    tmp_path: Path, empty: bool
) -> None:
    make_run_start(tmp_path)
    vision = (
        pa.table({})
        if empty
        else pa.table({"id": [1], "vision.spread.area_rel": [float("nan")]})
    )
    pq.write_table(vision, tmp_path / "vision.parquet")
    result = cli(tmp_path)
    assert result.returncode == 0, result.stderr
    table = pq.read_table(tmp_path / "training" / "vision_targets.parquet")
    assert table.column("vision.spread.area_rel").to_pylist() == [None, None]


@pytest.mark.slow
def test_manifest_records_source_hashes_and_caveats(tmp_path: Path) -> None:
    make_run_start(tmp_path)
    pq.write_table(pa.table({"id": [1]}), tmp_path / "vision.parquet")
    result = cli(tmp_path)
    assert result.returncode == 0, result.stderr
    schema = pq.read_schema(tmp_path / "training" / "vision_targets.parquet")
    manifest = json.loads(schema.metadata[b"maus.training.manifest"])
    for role, relative in (
        ("run_start", "training/run_start.parquet"),
        ("vision", "vision.parquet"),
    ):
        assert manifest["provenance"][role]["path"] == relative
        assert (
            manifest["provenance"][role]["sha256"]
            == hashlib.sha256((tmp_path / relative).read_bytes()).hexdigest()
        )
    assert "unverified" in manifest["provenance"]["raw_snapshot"]
    assert "not verified" in manifest["availability"]
    assert "last" in manifest["target_semantics"]
    assert "not human" in manifest["target_semantics"]


@pytest.mark.parametrize("failure", ["write", "changed_input"])
def test_interrupted_or_racing_write_preserves_previous_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    from maus_export.vision_targets import build_vision_targets

    make_run_start(tmp_path)
    path = tmp_path / "vision.parquet"
    pq.write_table(pa.table({"id": [1]}), path)
    output = tmp_path / "training" / "vision_targets.parquet"
    output.write_bytes(b"previous export")
    original = pq.write_table

    def interrupted(table: pa.Table, target: Path) -> None:
        if failure == "write":
            Path(target).write_bytes(b"incomplete")
            raise OSError("write failed")
        original(table, target)
        path.write_bytes(b"changed")

    monkeypatch.setattr(pq, "write_table", interrupted)
    with pytest.raises((ValueError, OSError)):
        build_vision_targets(tmp_path)
    assert output.read_bytes() == b"previous export"
    assert list(output.parent.glob(".vision_targets-*")) == []


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.slow
def test_start_nonfinite_sensors_are_not_passed_through(
    tmp_path: Path, value: float
) -> None:
    make_run_start(tmp_path)
    raw_path = tmp_path / "raw" / "runs" / "2.json"
    raw = json.loads(raw_path.read_text())
    raw["envTempStart"] = 3.0
    raw_path.write_text(json.dumps(raw))
    start_path = tmp_path / "training" / "run_start.parquet"
    start = pq.read_table(start_path)
    index = start.schema.get_field_index("env_temp_start")
    start = start.set_column(index, "env_temp_start", pa.array([value, 3.0]))
    pq.write_table(start, start_path)
    pq.write_table(pa.table({"id": [1]}), tmp_path / "vision.parquet")
    result = cli(tmp_path)
    output = tmp_path / "training" / "vision_targets.parquet"
    if value != value:
        assert result.returncode == 0, result.stderr
        assert pq.read_table(output)["env_temp_start"].to_pylist() == [
            None,
            3.0,
        ]
    else:
        assert result.returncode != 0
        assert not output.exists()


@pytest.mark.parametrize(
    "encoded",
    ['{"secret":1}', "[1]", "null", "NaN", "1e999", "not-json", '"secret" '],
)
@pytest.mark.slow
def test_corrupt_recipe_encodings_cannot_smuggle_nested_outcomes(
    tmp_path: Path, encoded: str
) -> None:
    make_run_start(tmp_path)
    start_path = tmp_path / "training" / "run_start.parquet"
    start = pq.read_table(start_path)
    index = start.schema.get_field_index("param.dispense.oil.grams")
    start = start.set_column(
        index, start.schema.field(index), pa.array([encoded, "2"])
    )
    pq.write_table(start, start_path)
    pq.write_table(pa.table({"id": [1]}), tmp_path / "vision.parquet")
    output = tmp_path / "training" / "vision_targets.parquet"
    output.write_bytes(b"stale")
    result = cli(tmp_path)
    assert result.returncode != 0
    assert "secret" not in result.stdout + result.stderr
    assert output.read_bytes() == b"stale"


@pytest.mark.slow
def test_vision_accepts_positive_int32_ids(tmp_path: Path) -> None:
    make_run_start(tmp_path)
    pq.write_table(
        pa.table({"id": pa.array([1], type=pa.int32())}),
        tmp_path / "vision.parquet",
    )
    result = cli(tmp_path)
    assert result.returncode == 0, result.stderr


def test_upstream_new_feature_cannot_silently_change_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from maus_export import vision_targets

    make_run_start(tmp_path)
    path = tmp_path / "training" / "run_start.parquet"
    start = pq.read_table(path)
    start = start.append_column("param.outcome", pa.array(["1", "2"]))
    manifest = json.loads(start.schema.metadata[b"maus.training.manifest"])
    manifest["candidate_features"].append("param.outcome")
    start = start.replace_schema_metadata(
        {b"maus.training.manifest": json.dumps(manifest).encode()}
    )
    pq.write_table(start, path)
    pq.write_table(pa.table({"id": [1]}), tmp_path / "vision.parquet")
    monkeypatch.setattr(
        vision_targets,
        "CANDIDATE_PARAMETERS",
        (*vision_targets.CANDIDATE_PARAMETERS, "outcome"),
    )
    with pytest.raises(vision_targets.VisionTargetsError):
        vision_targets.build_vision_targets(tmp_path)


def test_upstream_source_mapping_drift_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from maus_export import vision_targets

    make_run_start(tmp_path)
    pq.write_table(pa.table({"id": [1]}), tmp_path / "vision.parquet")
    changed = dict(vision_targets.CONTEXT)
    changed["robot"] = ("terminal_status", pa.string())
    monkeypatch.setattr(vision_targets, "CONTEXT", changed)
    with pytest.raises(vision_targets.VisionTargetsError):
        vision_targets.build_vision_targets(tmp_path)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("robot", "not_succeeded"),
        ("robot", None),
        ("topping", "private-sentinel"),
        ("is_test", True),
        ("hotplate_temp", 3.0),
        ("env_temp_start", 3.0),
        ("env_humidity_start", 3.0),
        ("param.dispense.oil.grams", "3"),
        ("param.dispense.batter.cook_time", "1"),
        ("terminal_status", "forged"),
    ],
)
@pytest.mark.slow
def test_forged_start_values_are_rejected_privately(
    tmp_path: Path, column: str, value: object
) -> None:
    make_run_start(tmp_path)
    path = tmp_path / "training" / "run_start.parquet"
    start = pq.read_table(path)
    index = start.schema.get_field_index(column)
    values = start[column].to_pylist()
    values[0] = value
    start = start.set_column(
        index,
        start.schema.field(index),
        pa.array(values, type=start[column].type),
    )
    pq.write_table(start, path)
    pq.write_table(pa.table({"id": [1]}), tmp_path / "vision.parquet")
    output = tmp_path / "training" / "vision_targets.parquet"
    output.write_bytes(b"previous")
    result = cli(tmp_path)
    assert result.returncode != 0
    assert "private-sentinel" not in result.stdout + result.stderr
    assert "not_succeeded" not in result.stdout + result.stderr
    assert output.read_bytes() == b"previous"


@pytest.mark.slow
def test_forged_timestamp_is_rejected(tmp_path: Path) -> None:
    make_run_start(tmp_path)
    path = tmp_path / "training" / "run_start.parquet"
    start = pq.read_table(path)
    timestamp = start["ts_start"].cast(pa.int64()).to_pylist()
    timestamp[1] += 1
    start = start.set_column(
        1,
        start.schema.field(1),
        pa.array(timestamp, type=pa.int64()).cast(start["ts_start"].type),
    )
    pq.write_table(start, path)
    pq.write_table(pa.table({"id": [1]}), tmp_path / "vision.parquet")
    output = tmp_path / "training" / "vision_targets.parquet"
    output.write_bytes(b"previous")
    assert cli(tmp_path).returncode != 0
    assert output.read_bytes() == b"previous"


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "malformed"])
@pytest.mark.slow
def test_missing_or_invalid_raw_is_rejected_privately(
    tmp_path: Path, mutation: str
) -> None:
    make_run_start(tmp_path)
    raw = tmp_path / "raw" / "runs"
    if mutation == "missing":
        (raw / "1.json").unlink()
    elif mutation == "duplicate":
        (raw / "private-sentinel.json").write_bytes(
            (raw / "1.json").read_bytes()
        )
    else:
        (raw / "1.json").write_text('{"private-sentinel":')
    pq.write_table(pa.table({"id": [1]}), tmp_path / "vision.parquet")
    output = tmp_path / "training" / "vision_targets.parquet"
    output.write_bytes(b"previous")
    result = cli(tmp_path)
    assert result.returncode != 0
    assert "private-sentinel" not in result.stdout + result.stderr
    assert output.read_bytes() == b"previous"


@pytest.mark.parametrize(
    "mutation", ["missing", "additional", "replaced", "duplicate"]
)
@pytest.mark.slow
def test_raw_id_set_mismatch_fails_without_replacing_valid_output(
    tmp_path: Path, mutation: str
) -> None:
    make_run_start(tmp_path)
    pq.write_table(pa.table({"id": [1]}), tmp_path / "vision.parquet")
    assert cli(tmp_path).returncode == 0
    output = tmp_path / "training" / "vision_targets.parquet"
    previous = output.read_bytes()
    assert len(pq.read_table(output)) == 2

    raw = tmp_path / "raw" / "runs"
    if mutation in ("missing", "replaced"):
        (raw / "2.json").unlink()
    if mutation in ("additional", "replaced", "duplicate"):
        run = json.loads((raw / "1.json").read_text())
        if mutation != "duplicate":
            run["id"] = 92731941
        (raw / "private-sentinel.json").write_text(json.dumps(run))

    result = cli(tmp_path)
    assert result.returncode != 0
    assert result.stdout == ""
    assert "stale" in result.stderr
    assert "private-sentinel" not in result.stderr
    assert "92731941" not in result.stderr
    if mutation != "duplicate":
        missing = int(mutation in ("missing", "replaced"))
        additional = int(mutation in ("additional", "replaced"))
        assert (
            f"missing raw: {missing}, additional raw: {additional}"
            in result.stderr
        )
    assert output.read_bytes() == previous
    assert list(output.parent.glob(".vision_targets-*")) == []


@pytest.mark.parametrize("mutation", ["changed", "added", "removed"])
def test_raw_snapshot_race_preserves_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    from maus_export.vision_targets import build_vision_targets

    make_run_start(tmp_path)
    raw = tmp_path / "raw" / "runs"
    pq.write_table(pa.table({"id": [1]}), tmp_path / "vision.parquet")
    output = tmp_path / "training" / "vision_targets.parquet"
    output.write_bytes(b"previous")
    original = pq.write_table

    def racing(table: pa.Table, target: Path) -> None:
        original(table, target)
        if mutation == "changed":
            (raw / "1.json").write_text("{}")
        elif mutation == "added":
            (raw / "extra.json").write_text("{}")
        else:
            (raw / "1.json").unlink()

    monkeypatch.setattr(pq, "write_table", racing)
    with pytest.raises((ValueError, OSError)):
        build_vision_targets(tmp_path)
    assert output.read_bytes() == b"previous"
    assert list(output.parent.glob(".vision_targets-*")) == []


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("vision.fold1.iou", -0.01),
        ("vision.fold2.iou", 1.01),
        ("vision.spread.l_mean", -0.01),
        ("vision.fold1.l_mean", 100.01),
        ("vision.fold2.l_mean", 100.01),
        ("vision.spread.l_std", -0.01),
        ("vision.spread.circularity", -0.01),
        ("vision.spread.zone_r_px", -0.01),
        ("vision.fold2.area_ratio", -0.01),
        ("vision.spread.hole_frac", 1.01),
        ("vision.spread.uncovered_frac", 1.01),
    ],
)
@pytest.mark.slow
def test_invalid_vision_domains_preserve_output(
    tmp_path: Path, name: str, value: float
) -> None:
    make_run_start(tmp_path)
    pq.write_table(
        pa.table({"id": [1], name: [value]}), tmp_path / "vision.parquet"
    )
    output = tmp_path / "training" / "vision_targets.parquet"
    output.write_bytes(b"previous")
    result = cli(tmp_path)
    assert result.returncode != 0
    assert output.read_bytes() == b"previous"


@pytest.mark.parametrize("stage", ["spread", "fold1", "fold2"])
@pytest.mark.parametrize(
    ("mean", "std"),
    [
        (None, 51.0),
        (None, 100.0),
        (None, 1e30),
        (None, 1e308),
        (None, math.nextafter(50.0, math.inf)),
        (0.0, 1.0),
        (100.0, 1.0),
        (25.0, 44.0),
    ],
)
@pytest.mark.slow
def test_impossible_lightness_std_preserves_output(
    tmp_path: Path, stage: str, mean: float | None, std: float
) -> None:
    make_run_start(tmp_path)
    pq.write_table(
        pa.table(
            {
                "id": [1],
                f"vision.{stage}.l_mean": pa.array([mean], type=pa.float64()),
                f"vision.{stage}.l_std": [std],
            }
        ),
        tmp_path / "vision.parquet",
    )
    output = tmp_path / "training" / "vision_targets.parquet"
    output.write_bytes(b"previous")
    result = cli(tmp_path)
    assert result.returncode != 0
    assert "stale" in result.stderr
    assert "private-sentinel" not in result.stdout + result.stderr
    assert output.read_bytes() == b"previous"


@pytest.mark.parametrize("stage", ["spread", "fold1", "fold2"])
@pytest.mark.slow
def test_lightness_boundaries_tolerance_and_missingness(
    tmp_path: Path, stage: str
) -> None:
    make_run_start(tmp_path)
    mean_name = f"vision.{stage}.l_mean"
    std_name = f"vision.{stage}.l_std"
    near_zero = math.sqrt(0.5e-9)
    pq.write_table(
        pa.table(
            {"id": [1, 2], mean_name: [50.0, 0.0], std_name: [50.0, near_zero]}
        ),
        tmp_path / "vision.parquet",
    )
    result = cli(tmp_path)
    assert result.returncode == 0, result.stderr
    result_table = pq.read_table(
        tmp_path / "training" / "vision_targets.parquet"
    )
    assert result_table[mean_name].to_pylist() == [50.0, 0.0]
    assert result_table[std_name].to_pylist() == [50.0, near_zero]

    pq.write_table(
        pa.table(
            {"id": [1, 2], mean_name: [None, 100.0], std_name: [30.0, None]}
        ),
        tmp_path / "vision.parquet",
    )
    result = cli(tmp_path)
    assert result.returncode == 0, result.stderr
    result_table = pq.read_table(
        tmp_path / "training" / "vision_targets.parquet"
    )
    assert result_table[mean_name].to_pylist() == [None, 100.0]
    assert result_table[std_name].to_pylist() == [30.0, None]

    boundary_std = math.sqrt(25.0 * 75.0)
    pq.write_table(
        pa.table({"id": [1], mean_name: [25.0], std_name: [boundary_std]}),
        tmp_path / "vision.parquet",
    )
    result = cli(tmp_path)
    assert result.returncode == 0, result.stderr
    result_table = pq.read_table(
        tmp_path / "training" / "vision_targets.parquet"
    )
    assert result_table[std_name].to_pylist() == [boundary_std, None]


@pytest.mark.parametrize("stage", ["spread", "fold1", "fold2"])
@pytest.mark.parametrize("missing", ["mean_column", "std_column", "nan"])
@pytest.mark.slow
def test_lightness_missing_components_stay_independently_unknown(
    tmp_path: Path, stage: str, missing: str
) -> None:
    make_run_start(tmp_path)
    mean_name = f"vision.{stage}.l_mean"
    std_name = f"vision.{stage}.l_std"
    vision: dict[str, list[int] | list[float]] = {"id": [1, 2]}
    if missing != "mean_column":
        vision[mean_name] = (
            [float("nan"), 25.0] if missing == "nan" else [25.0, 25.0]
        )
    if missing != "std_column":
        vision[std_name] = (
            [20.0, float("nan")] if missing == "nan" else [20.0, 20.0]
        )
    pq.write_table(pa.table(vision), tmp_path / "vision.parquet")
    result = cli(tmp_path)
    assert result.returncode == 0, result.stderr
    table = pq.read_table(tmp_path / "training" / "vision_targets.parquet")
    expected_mean = {
        "mean_column": [None, None],
        "std_column": [25.0, 25.0],
        "nan": [None, 25.0],
    }
    expected_std = {
        "mean_column": [20.0, 20.0],
        "std_column": [None, None],
        "nan": [20.0, None],
    }
    assert table[mean_name].to_pylist() == expected_mean[missing]
    assert table[std_name].to_pylist() == expected_std[missing]


@pytest.mark.parametrize("stage", ["spread", "fold1", "fold2"])
@pytest.mark.slow
def test_lightness_variance_exceeding_tolerance_is_rejected(
    tmp_path: Path, stage: str
) -> None:
    make_run_start(tmp_path)
    pq.write_table(
        pa.table(
            {
                "id": [1],
                f"vision.{stage}.l_mean": [0.0],
                f"vision.{stage}.l_std": [math.sqrt(2e-9)],
            }
        ),
        tmp_path / "vision.parquet",
    )
    assert cli(tmp_path).returncode != 0


@pytest.mark.slow
def test_vision_domain_boundaries_and_unbounded_ratios(tmp_path: Path) -> None:
    make_run_start(tmp_path)
    pq.write_table(
        pa.table(
            {
                "id": [1, 2],
                "vision.fold1.iou": [0.0, 1.0],
                "vision.fold2.iou": [1.0, 0.0],
                "vision.spread.hole_frac": [0.0, 1.0],
                "vision.spread.uncovered_frac": [1.0, 0.0],
                "vision.spread.l_mean": [0.0, 100.0],
                "vision.fold1.l_mean": [100.0, 0.0],
                "vision.fold2.l_mean": [0.0, 100.0],
                "vision.fold2.area_ratio": [0.0, 8.0],
                "vision.spread.circularity": [2.0, 0.0],
                "vision.spread.offset_rel": [3.0, 0.0],
            }
        ),
        tmp_path / "vision.parquet",
    )
    assert cli(tmp_path).returncode == 0
    table = pq.read_table(tmp_path / "training" / "vision_targets.parquet")
    assert table["vision.fold2.area_ratio"].to_pylist() == [0.0, 8.0]
    assert table["vision.spread.circularity"].to_pylist() == [2.0, 0.0]


@pytest.mark.parametrize("name", [*TARGETS, "vision.spread.zone_r_px"])
@pytest.mark.parametrize("value", [-1.0, float("inf"), float("-inf")])
def test_every_metric_rejects_negative_and_infinite_values(
    name: str, value: float
) -> None:
    from maus_export.vision_targets import VisionTargetsError, _validate_vision

    with pytest.raises(VisionTargetsError):
        _validate_vision(pa.table({"id": [1], name: [value]}))


@pytest.mark.skipif(
    sys.platform != "linux", reason="/dev/full is a Linux device"
)
@pytest.mark.slow
def test_full_stdout_does_not_turn_success_into_failure(tmp_path: Path) -> None:
    make_run_start(tmp_path)
    pq.write_table(pa.table({"id": [1]}), tmp_path / "vision.parquet")
    output = tmp_path / "training" / "vision_targets.parquet"
    output.write_bytes(b"previous")
    with Path("/dev/full").open("wb") as full:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "maus_export.cli",
                "--data-dir",
                str(tmp_path),
                "vision-targets",
            ],
            stdout=full,
            stderr=subprocess.PIPE,
            check=False,
            timeout=CLI_TIMEOUT_SECONDS,
        )
    assert result.returncode == 0, result.stderr
    assert len(pq.read_table(output)) == 2


def test_no_cleanup_after_successful_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from maus_export.vision_targets import build_vision_targets

    make_run_start(tmp_path)
    pq.write_table(pa.table({"id": [1]}), tmp_path / "vision.parquet")
    original = Path.unlink

    def forbid_post_replace(
        path: Path, *args: object, **kwargs: object
    ) -> None:
        if path.name.startswith(".vision_targets-") and not path.exists():
            raise OSError("post-replacement cleanup")
        original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", forbid_post_replace)
    build_vision_targets(tmp_path)
    assert (
        len(pq.read_table(tmp_path / "training" / "vision_targets.parquet"))
        == 2
    )


@pytest.mark.slow
def test_success_is_silent(tmp_path: Path) -> None:
    make_run_start(tmp_path)
    pq.write_table(pa.table({"id": [1]}), tmp_path / "vision.parquet")
    result = cli(tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize("scored", [False, True])
@pytest.mark.slow
def test_manifest_coverage_counts_non_null_target_rows(
    tmp_path: Path, scored: bool
) -> None:
    make_run_start(tmp_path)
    vision = {"id": [1, 2]}
    if scored:
        vision["vision.fold1.iou"] = [float("nan"), 0.0]
    pq.write_table(pa.table(vision), tmp_path / "vision.parquet")
    result = cli(tmp_path)
    assert result.returncode == 0, result.stderr
    schema = pq.read_schema(tmp_path / "training" / "vision_targets.parquet")
    manifest = json.loads(schema.metadata[b"maus.training.manifest"])
    assert manifest["coverage"] == {
        "rows": 2,
        "rows_with_any_target": 1 if scored else 0,
    }


@pytest.mark.parametrize("version", [True, 1.0])
@pytest.mark.slow
def test_manifest_version_requires_integer_type(
    tmp_path: Path, version: bool | float
) -> None:
    make_run_start(tmp_path)
    path = tmp_path / "training" / "run_start.parquet"
    start = pq.read_table(path)
    manifest = json.loads(start.schema.metadata[b"maus.training.manifest"])
    manifest["schema_version"] = version
    start = start.replace_schema_metadata(
        {b"maus.training.manifest": json.dumps(manifest).encode()}
    )
    pq.write_table(start, path)
    pq.write_table(pa.table({"id": [1]}), tmp_path / "vision.parquet")
    output = tmp_path / "training" / "vision_targets.parquet"
    output.write_bytes(b"stale")
    result = cli(tmp_path)
    assert result.returncode != 0
    assert "stale" in result.stderr
    assert output.read_bytes() == b"stale"
