"""Export candidate run-start inputs and recorded terminal status."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

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


def build_training_dataset(data_dir: Path) -> None:
    """Build a run-start candidate dataset from raw JSON runs."""
    files = sorted((data_dir / "raw" / "runs").glob("*.json"))
    runs = [
        json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_invalid_constant,
            parse_float=_finite_float,
        )
        for path in files
    ]
    if not runs:
        raise ValueError("No raw runs found; previous output was not updated")
    ids: set[int] = set()
    for run in runs:
        if not isinstance(run, dict):
            raise ValueError("Run must be an object")
        run_id = run.get("id")
        if type(run_id) is not int or run_id < 1 or run_id in ids:
            raise ValueError("Missing, invalid or duplicate run id")
        ids.add(run_id)
        for source, typ in CONTEXT.values():
            if pa.types.is_floating(typ):
                value = run.get(source)
                if value is not None and (
                    type(value) not in (int, float)
                    or (type(value) is float and not math.isfinite(value))
                ):
                    raise ValueError("Numeric sensor must be finite or null")
        params = run.get("parameters")
        if params is not None and not isinstance(params, dict):
            raise ValueError("Parameters must be an object or null")
        for key in CANDIDATE_PARAMETERS:
            value = (params or {}).get(key)
            if value is not None and type(value) not in (str, int, float, bool):
                raise ValueError("Candidate parameter must be a JSON scalar")
        start = pd.Timestamp(run.get("tsStart"))
        if pd.isna(start) or start.tzinfo is None:
            raise ValueError("tsStart must be a timestamp with timezone")
    keys = CANDIDATE_PARAMETERS
    fields = [
        pa.field("id", pa.int64()),
        pa.field("ts_start", pa.timestamp("us", tz="UTC")),
        pa.field("terminal_status", pa.string()),
    ]
    fields.extend(pa.field(name, typ) for name, (_, typ) in CONTEXT.items())
    fields.extend(pa.field(f"param.{key}", pa.string()) for key in keys)
    manifest = {
        "schema_version": 1,
        "metadata": ["id", "ts_start"],
        "target": "terminal_status",
        "candidate_features": list(CONTEXT) + [f"param.{k}" for k in keys],
        "parameter_encoding": "canonical JSON; null is missing",
        "target_semantics": "recorded terminal status, not quality",
        "availability": "run-start availability not verified",
    }
    schema = pa.schema(
        fields,
        metadata={
            b"maus.training.manifest": json.dumps(manifest).encode("utf-8")
        },
    )

    rows = []
    for run in runs:
        row = {
            "id": run["id"],
            "ts_start": pd.Timestamp(run["tsStart"]).to_pydatetime(),
            "terminal_status": run.get("terminal_status"),
        }
        row.update(
            {name: run.get(source) for name, (source, _) in CONTEXT.items()}
        )
        params = run.get("parameters") or {}
        row.update(
            {
                f"param.{key}": json.dumps(
                    params[key],
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                if params.get(key) is not None
                else None
                for key in keys
            }
        )
        rows.append(row)
    table = pa.Table.from_pylist(rows, schema=schema)
    output = data_dir / "training" / "run_start.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".run_start-", dir=output.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        pq.write_table(table, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"{len(rows)} runs -> {output}")
