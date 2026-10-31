"""Export candidate run-start inputs and recorded terminal status."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .to_table import canonical_json

CONTEXT = {
    "robot": ("machineName", pa.string()),
    "topping": ("topping", pa.string()),
    "is_test": ("isTest", pa.bool_()),
    "hotplate_temp": ("hotplateTemp", pa.float64()),
    "env_temp_start": ("envTempStart", pa.float64()),
    "env_humidity_start": ("envHumidityStart", pa.float64()),
}
CANDIDATE_PARAMETERS = (
    "dispense.batter.cook_time",
    "dispense.batter.centre_dispense_time",
    "dispense.batter.centre_flowrate",
    "dispense.batter.fan_power",
    "dispense.batter.concentric_outer_radius_offset_mm",
    "dispense.batter.concentric_inner_radius_offset_mm",
    "dispense.batter.concentric_ring_count",
    "dispense.batter.concentric_rings_enabled",
    "dispense.oil.grams",
)
SCHEMA_VERSION = 1
SCALAR_TYPES = (str, int, float, bool)
OUTPUT = Path("training") / "run_start.parquet"


class TrainingInputError(ValueError):
    """A raw run file is unusable. The message names the file and a fixed
    reason, never field values, so it is safe to print."""

    def __init__(self, name: str, reason: str) -> None:
        super().__init__(f"{name}: {reason}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError("Non-finite JSON number")


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Non-finite JSON number")
    return number


def _load_run(path: Path) -> Any:
    """Parse one raw run strictly (no duplicate keys or non-finite numbers)."""
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_invalid_constant,
            parse_float=_finite_float,
        )
    except json.JSONDecodeError as exc:
        # The decoder message has only a position, no file content
        raise TrainingInputError(
            path.name, f"invalid JSON ({exc.msg})"
        ) from None
    except ValueError as exc:
        raise TrainingInputError(path.name, str(exc)) from None


def _is_bad_sensor(value: Any) -> bool:
    if value is None:
        return False
    if type(value) not in (int, float):
        return True
    return type(value) is float and not math.isfinite(value)


def _validate_run(run: Any, name: str, seen: dict[int, str]) -> None:
    """Raise TrainingInputError if the run cannot become a row. `seen` maps
    run ids to the file they came from, to report duplicates."""
    if not isinstance(run, dict):
        raise TrainingInputError(name, "Run must be an object")
    run_id = run.get("id")
    if type(run_id) is not int or run_id < 1:
        raise TrainingInputError(name, "Missing or invalid run id")
    if run_id in seen:
        raise TrainingInputError(
            name, f"Duplicate run id, also in {seen[run_id]}"
        )
    seen[run_id] = name
    for source, typ in CONTEXT.values():
        if pa.types.is_floating(typ) and _is_bad_sensor(run.get(source)):
            raise TrainingInputError(
                name, "Numeric sensor must be finite or null"
            )
    params = run.get("parameters")
    if params is not None and not isinstance(params, dict):
        raise TrainingInputError(name, "Parameters must be an object or null")
    for key in CANDIDATE_PARAMETERS:
        value = (params or {}).get(key)
        if value is not None and type(value) not in SCALAR_TYPES:
            raise TrainingInputError(
                name, "Candidate parameter must be a JSON scalar"
            )
    start = pd.Timestamp(run.get("tsStart"))
    if pd.isna(start) or start.tzinfo is None:
        raise TrainingInputError(
            name, "tsStart must be a timestamp with timezone"
        )


def _schema() -> pa.Schema:
    """Fixed output schema, with the role manifest in its metadata."""
    fields = [
        pa.field("id", pa.int64()),
        pa.field("ts_start", pa.timestamp("us", tz="UTC")),
        pa.field("terminal_status", pa.string()),
    ]
    fields.extend(pa.field(name, typ) for name, (_, typ) in CONTEXT.items())
    fields.extend(
        pa.field(f"param.{key}", pa.string()) for key in CANDIDATE_PARAMETERS
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "metadata": ["id", "ts_start"],
        "target": "terminal_status",
        "candidate_features": list(CONTEXT)
        + [f"param.{k}" for k in CANDIDATE_PARAMETERS],
        "parameter_encoding": "canonical JSON; null is missing",
        "target_semantics": "recorded terminal status, not quality",
        "availability": "run-start availability not verified",
    }
    return pa.schema(
        fields,
        metadata={
            b"maus.training.manifest": json.dumps(manifest).encode("utf-8")
        },
    )


def _row(run: dict[str, Any]) -> dict[str, Any]:
    """One output row for a validated run."""
    row = {
        "id": run["id"],
        "ts_start": pd.Timestamp(run["tsStart"]).to_pydatetime(),
        "terminal_status": run.get("terminal_status"),
    }
    row.update({name: run.get(src) for name, (src, _) in CONTEXT.items()})
    params = run.get("parameters") or {}
    for key in CANDIDATE_PARAMETERS:
        value = params.get(key)
        row[f"param.{key}"] = (
            canonical_json(value) if value is not None else None
        )
    return row


def _write_atomic(table: pa.Table, output: Path) -> None:
    """Write to a temp file and swap it in, so a failed write keeps the
    previous output."""
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".run_start-", dir=output.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        pq.write_table(table, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def build_training_dataset(data_dir: Path) -> None:
    """Build a run-start candidate dataset from raw JSON runs."""
    files = sorted((data_dir / "raw" / "runs").glob("*.json"))
    if not files:
        raise ValueError("No raw runs found; previous output was not updated")
    runs = [_load_run(path) for path in files]
    seen: dict[int, str] = {}
    for path, run in zip(files, runs, strict=True):
        _validate_run(run, path.name, seen)
    table = pa.Table.from_pylist([_row(r) for r in runs], schema=_schema())
    output = data_dir / OUTPUT
    _write_atomic(table, output)
    print(f"{len(runs)} runs -> {output}")
