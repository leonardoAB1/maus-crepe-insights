"""Join run-start candidates to post-run vision scores."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .to_table import canonical_json
from .training_dataset import (
    CANDIDATE_PARAMETERS,
    CONTEXT,
    _finite_float,
    _invalid_constant,
    _unique_object,
    _validate_run,
)

FEATURE_CONTEXT = (
    ("robot", "machineName", pa.string()),
    ("topping", "topping", pa.string()),
    ("is_test", "isTest", pa.bool_()),
    ("hotplate_temp", "hotplateTemp", pa.float64()),
    ("env_temp_start", "envTempStart", pa.float64()),
    ("env_humidity_start", "envHumidityStart", pa.float64()),
)
FEATURE_PARAMETERS = (
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
)
RUN_START_SCHEMA = 2  # run-start dataset version this join accepts

TARGETS = (
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
)
UNIT_INTERVAL = {
    "vision.fold1.iou",
    "vision.fold2.iou",
    "vision.spread.hole_frac",
    "vision.spread.uncovered_frac",
}
LIGHTNESS_MEANS = {
    "vision.spread.l_mean",
    "vision.fold1.l_mean",
    "vision.fold2.l_mean",
}
LIGHTNESS_MAX = 100.0
LIGHTNESS_STD_MAX = 50.0
# Absolute tolerance in squared L* units for floating-point roundoff.
LIGHTNESS_VARIANCE_TOLERANCE = 1e-9
LIGHTNESS_STDS = {name.replace(".l_mean", ".l_std") for name in LIGHTNESS_MEANS}


class VisionTargetsError(ValueError):
    """A privacy-safe validation failure."""


def _ids(table: pa.Table, *, allow_empty_without_id: bool = False) -> set[int]:
    if (
        allow_empty_without_id
        and len(table) == 0
        and "id" not in table.schema.names
    ):
        return set()
    if "id" not in table.schema.names or not pa.types.is_integer(
        table.schema.field("id").type
    ):
        raise VisionTargetsError("IDs must have integer type")
    values = table["id"].to_pylist()
    if any(value is None or value < 1 for value in values):
        raise VisionTargetsError("IDs must be positive and non-null")
    if len(set(values)) != len(values):
        raise VisionTargetsError("Duplicate IDs")
    return set(values)


def _validate_start(table: pa.Table) -> list[str]:
    if (
        tuple((name, source, typ) for name, (source, typ) in CONTEXT.items())
        != FEATURE_CONTEXT
        or CANDIDATE_PARAMETERS != FEATURE_PARAMETERS
    ):
        raise VisionTargetsError("Incompatible run-start feature definition")
    features = [name for name, _, _ in FEATURE_CONTEXT] + [
        f"param.{key}" for key in FEATURE_PARAMETERS
    ]
    fields = [
        pa.field("id", pa.int64()),
        pa.field("ts_start", pa.timestamp("us", tz="UTC")),
        pa.field("terminal_status", pa.string()),
        *[pa.field(name, typ) for name, (_, typ) in CONTEXT.items()],
        *[
            pa.field(f"param.{key}", pa.string())
            for key in CANDIDATE_PARAMETERS
        ],
    ]
    if not table.schema.remove_metadata().equals(pa.schema(fields)) or not len(
        table
    ):
        raise VisionTargetsError("Invalid run-start schema or empty input")
    if table["ts_start"].null_count:
        raise VisionTargetsError("Missing run-start timestamps")
    try:
        manifest = json.loads(
            table.schema.metadata[b"maus.training.manifest"],
            object_pairs_hook=_unique_object,
            parse_constant=_invalid_constant,
        )
    except (TypeError, KeyError, ValueError):
        raise VisionTargetsError("Invalid run-start manifest") from None
    expected = {
        "schema_version": RUN_START_SCHEMA,
        "metadata": ["id", "ts_start"],
        "target": "terminal_status",
        "candidate_features": features,
        "parameter_encoding": "canonical JSON; null is missing",
        "target_semantics": "recorded terminal status, not quality",
        "availability": "run-start availability not verified",
    }
    if manifest != expected or type(manifest["schema_version"]) is not int:
        raise VisionTargetsError("Incompatible run-start manifest")
    for key in CANDIDATE_PARAMETERS:
        for encoded in table[f"param.{key}"].to_pylist():
            if encoded is None:
                continue
            try:
                value = json.loads(
                    encoded,
                    parse_constant=_invalid_constant,
                    parse_float=_finite_float,
                    object_pairs_hook=_unique_object,
                )
                canonical = json.dumps(
                    value,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            except (TypeError, ValueError):
                raise VisionTargetsError("Invalid recipe encoding") from None
            if (
                type(value) not in (str, int, float, bool)
                or canonical != encoded
            ):
                raise VisionTargetsError("Invalid recipe encoding")
    return features


def _validate_vision(table: pa.Table) -> None:
    names = table.schema.names
    allowed = {"id", "vision.spread.zone_r_px", *TARGETS}
    if len(names) != len(set(names)) or set(names) - allowed:
        raise VisionTargetsError("Unexpected or duplicate vision fields")
    for field in table.schema:
        if field.name == "id":
            continue
        if field.type != pa.float64() and not pa.types.is_null(field.type):
            raise VisionTargetsError("Vision metrics must be float64 or null")
        if any(
            value is not None and math.isinf(value)
            for value in table[field.name].to_pylist()
        ):
            raise VisionTargetsError("Infinite vision metric")
        for value in table[field.name].to_pylist():
            if value is None or math.isnan(value):
                continue
            if (
                value < 0
                or (field.name in UNIT_INTERVAL and value > 1)
                or (field.name in LIGHTNESS_MEANS and value > LIGHTNESS_MAX)
                or (field.name in LIGHTNESS_STDS and value > LIGHTNESS_STD_MAX)
            ):
                raise VisionTargetsError("Vision metric outside valid domain")

    for mean_name in LIGHTNESS_MEANS:
        std_name = mean_name.replace(".l_mean", ".l_std")
        if mean_name not in names or std_name not in names:
            continue
        for mean, std in zip(
            table[mean_name].to_pylist(),
            table[std_name].to_pylist(),
            strict=True,
        ):
            if (
                mean is None
                or std is None
                or math.isnan(mean)
                or math.isnan(std)
            ):
                continue
            if std * std > mean * (LIGHTNESS_MAX - mean) + (
                LIGHTNESS_VARIANCE_TOLERANCE
            ):
                raise VisionTargetsError("Vision metric outside valid domain")


def _normalise_sensor(column: pa.ChunkedArray) -> pa.Array:
    values = column.to_pylist()
    if any(value is not None and math.isinf(value) for value in values):
        raise VisionTargetsError("Infinite run-start sensor")
    return pa.array(
        [
            None if value is None or math.isnan(value) else value
            for value in values
        ],
        type=pa.float64(),
    )


def _raw_snapshot(data_dir: Path, start: pa.Table) -> dict[Path, str]:
    files = sorted((data_dir / "raw" / "runs").glob("*.json"))
    if not files:
        raise VisionTargetsError("Raw snapshots unavailable")
    fingerprints = {}
    runs = {}
    seen: dict[int, str] = {}
    try:
        for path in files:
            content = path.read_bytes()
            fingerprints[path] = hashlib.sha256(content).hexdigest()
            run = json.loads(
                content,
                object_pairs_hook=_unique_object,
                parse_constant=_invalid_constant,
                parse_float=_finite_float,
            )
            _validate_run(run, "raw snapshot", seen)
            runs[run["id"]] = run
    except Exception:
        raise VisionTargetsError("Invalid or missing raw snapshot") from None
    start_ids = set(start["id"].to_pylist())
    raw_ids = set(runs)
    missing = len(start_ids - raw_ids)
    additional = len(raw_ids - start_ids)
    if missing or additional:
        raise VisionTargetsError(
            "Raw/run-start ID sets differ "
            f"(missing raw: {missing}, additional raw: {additional})"
        )
    try:
        rows = []
        for run_id in start["id"].to_pylist():
            run = runs[run_id]
            row = {
                "id": run_id,
                "ts_start": pd.Timestamp(run["tsStart"]).to_pydatetime(),
                "terminal_status": run.get("terminal_status"),
            }
            row.update(
                {name: run.get(source) for name, source, _ in FEATURE_CONTEXT}
            )
            params = run.get("parameters") or {}
            for key in FEATURE_PARAMETERS:
                value = params.get(key)
                row[f"param.{key}"] = (
                    canonical_json(value) if value is not None else None
                )
            rows.append(row)
        expected = pa.Table.from_pylist(rows, schema=start.schema)
    except Exception:
        raise VisionTargetsError("Invalid or missing raw snapshot") from None
    columns = [
        _normalise_sensor(start[name])
        if pa.types.is_floating(start.schema.field(name).type)
        else start[name]
        for name in start.column_names
    ]
    if not pa.Table.from_arrays(columns, schema=start.schema).equals(expected):
        raise VisionTargetsError("Run-start differs from raw snapshot")
    return fingerprints


def build_vision_targets(data_dir: Path) -> None:
    """Write one private row per run with nullable vision score targets."""
    paths = {
        "run_start": "training/run_start.parquet",
        "vision": "vision.parquet",
    }
    contents = {
        role: (data_dir / path).read_bytes() for role, path in paths.items()
    }
    hashes = {
        role: hashlib.sha256(value).hexdigest()
        for role, value in contents.items()
    }
    start = pq.read_table(pa.BufferReader(contents["run_start"]))
    vision = pq.read_table(pa.BufferReader(contents["vision"]))
    features = _validate_start(start)
    start_ids = _ids(start)
    raw_hashes = _raw_snapshot(data_dir, start)
    _validate_vision(vision)
    vision_ids = _ids(vision, allow_empty_without_id=True)
    orphans = vision_ids - start_ids
    if orphans:
        raise VisionTargetsError(f"Orphan vision IDs: {len(orphans)}")
    source = {row["id"]: row for row in vision.to_pylist()}
    arrays = [
        _normalise_sensor(start[name])
        if pa.types.is_floating(start.schema.field(name).type)
        else start[name]
        for name in ["id", "ts_start", *features]
    ]
    rows = [source.get(run_id, {}) for run_id in start["id"].to_pylist()]
    arrays.extend(
        pa.array(
            [
                None
                if (value := row.get(name)) is None or math.isnan(value)
                else value
                for row in rows
            ],
            type=pa.float64(),
        )
        for name in TARGETS
    )
    manifest = {
        "schema_version": 1,
        "metadata": ["id", "ts_start"],
        "candidate_features": features,
        "targets": list(TARGETS),
        "coverage": {
            "rows": len(rows),
            "rows_with_any_target": sum(
                any(
                    (value := row.get(name)) is not None
                    and not math.isnan(value)
                    for name in TARGETS
                )
                for row in rows
            ),
        },
        "parameter_encoding": "canonical JSON; null is missing",
        "availability": "run-start availability not verified",
        "target_semantics": (
            "post-run scores from robot overlay annotations, last listed "
            "attempt per check; not human-verified labels"
        ),
        "provenance": {
            role: {"path": path, "sha256": hashes[role]}
            for role, path in paths.items()
        }
        | {
            "raw_snapshot": (
                "run-start IDs and values checked against all cached raw "
                "snapshots; vision snapshot consistency unverified"
            )
        },
    }
    names = ["id", "ts_start", *features, *TARGETS]
    table = pa.Table.from_arrays(arrays, names=names).replace_schema_metadata(
        {b"maus.training.manifest": json.dumps(manifest).encode()}
    )
    output = data_dir / "training" / "vision_targets.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".vision_targets-", dir=output.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        pq.write_table(table, temporary)
        for role, path in paths.items():
            current = hashlib.sha256((data_dir / path).read_bytes()).hexdigest()
            if current != hashes[role]:
                raise VisionTargetsError("Input changed during export")
        if sorted((data_dir / "raw" / "runs").glob("*.json")) != sorted(
            raw_hashes
        ) or any(
            hashlib.sha256(path.read_bytes()).hexdigest() != digest
            for path, digest in raw_hashes.items()
        ):
            raise VisionTargetsError("Input changed during export")
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
