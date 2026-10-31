"""Aggregate-only audit of the local run-start and vision tables."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .to_table import canonical_json
from .training_dataset import CANDIDATE_PARAMETERS, CONTEXT, SCALAR_TYPES

STATUS = ("succeeded", "not_succeeded", "aborted", "canceled")
FEATURES = tuple(CONTEXT) + tuple(f"param.{p}" for p in CANDIDATE_PARAMETERS)
SPREAD_METRICS = (
    "zone_r_px",
    "area_rel",
    "diameter_rel",
    "offset_rel",
    "circularity",
    "hole_frac",
    "uncovered_frac",
    "l_mean",
    "l_std",
    "circle_diameter_rel",
    "circle_offset_rel",
    "residue_frac",
)
FOLD_METRICS = ("area_rel", "l_mean", "l_std", "iou")
VISION_METRICS = {
    *(f"vision.spread.{name}" for name in SPREAD_METRICS),
    *(
        f"vision.{fold}.{name}"
        for fold in ("fold1", "fold2")
        for name in FOLD_METRICS
    ),
    "vision.fold2.area_ratio",
}


def _coverage(values: pd.Series) -> dict:
    total = len(values)
    non_null = int(values.notna().sum())
    return {
        "denominator_training_runs": total,
        "non_null_runs": non_null,
        "null_runs": total - non_null,
        "coverage": non_null / total if total else None,
    }


def _validate_ids(frame: pd.DataFrame) -> None:
    ids = frame["id"]
    if (
        not pd.api.types.is_integer_dtype(ids.dtype)
        or ids.isna().any()
        or (ids <= 0).any()
        or ids.duplicated().any()
    ):
        raise ValueError("Invalid run identifiers")


def _validate_schema(training: pa.Table, vision: pa.Table) -> None:
    expected = {"id", "ts_start", "terminal_status", *FEATURES}
    if set(training.column_names) != expected or not training.num_rows:
        raise ValueError("Invalid training schema or empty training data")
    if len(training.column_names) != len(expected):
        raise ValueError("Duplicate training columns")
    types = {
        "terminal_status": pa.string(),
        **{name: typ for name, (_, typ) in CONTEXT.items()},
        **{f"param.{p}": pa.string() for p in CANDIDATE_PARAMETERS},
    }
    if any(training.schema.field(c).type != typ for c, typ in types.items()):
        raise ValueError("Incompatible training fields")
    stamp = training.schema.field("ts_start").type
    if not pa.types.is_timestamp(stamp) or stamp.tz != "UTC":
        raise ValueError("Training timestamps must be UTC")
    if training.column("ts_start").null_count:
        raise ValueError("Training timestamps must be present")
    for key in CANDIDATE_PARAMETERS:
        for encoded in training.column(f"param.{key}").to_pylist():
            if encoded is None:
                continue
            try:
                value = json.loads(encoded)
                if type(value) not in SCALAR_TYPES:
                    raise ValueError
                if canonical_json(value) != encoded:
                    raise ValueError
            except (ValueError, RecursionError):
                raise ValueError("Invalid candidate parameter") from None
    robots = training.column("robot").to_pylist()
    if any(
        r is not None and not re.fullmatch(r"maus-[a-z]{1,24}", r)
        for r in robots
    ):
        raise ValueError("Invalid robot name")
    for c, (_, typ) in CONTEXT.items():
        if pa.types.is_floating(typ) and any(
            value is not None and not math.isfinite(value)
            for value in training.column(c).to_pylist()
        ):
            raise ValueError("Invalid training numeric field")
    if (
        "id" not in vision.column_names
        or len(set(vision.column_names)) != len(vision.column_names)
        or not set(vision.column_names) - {"id"}
        or set(vision.column_names) - {"id", *VISION_METRICS}
    ):
        raise ValueError("Invalid vision columns")
    for c in set(vision.column_names) - {"id"}:
        typ = vision.schema.field(c).type
        if not (
            pa.types.is_integer(typ)
            or pa.types.is_floating(typ)
            or pa.types.is_null(typ)
        ):
            raise ValueError("Invalid vision metric type")
        if any(
            value is not None and not math.isfinite(value)
            for value in vision.column(c).to_pylist()
        ):
            raise ValueError("Invalid vision metric")


def _status_labels(values: pd.Series) -> pd.Series:
    return values.where(values.isin(STATUS), "other").mask(
        values.isna(), "unknown"
    )


def _summary(frame: pd.DataFrame, metrics: list[str]) -> dict:
    status = _status_labels(frame["terminal_status"])
    tests = (
        frame["is_test"].map({True: "true", False: "false"}).fillna("unknown")
    )
    return {
        "runs": len(frame),
        "status_counts": {
            k: int((status == k).sum()) for k in (*STATUS, "other", "unknown")
        },
        "test_counts": {
            k: int((tests == k).sum()) for k in ("true", "false", "unknown")
        },
        "vision_row_coverage": _coverage(frame["_vision_match"]),
        "vision_metrics": {c: _coverage(frame[c]) for c in metrics},
        "features": {c: _coverage(frame[c]) for c in FEATURES},
    }


def build_training_audit(data_dir: Path) -> None:
    """Write an aggregate report, preserving the previous one on failure."""
    training_table = pq.read_table(data_dir / "training" / "run_start.parquet")
    vision_table = pq.read_table(data_dir / "vision.parquet")
    _validate_schema(training_table, vision_table)
    training = training_table.to_pandas()
    vision = vision_table.to_pandas()
    _validate_ids(training)
    _validate_ids(vision)
    metrics = [c for c in vision if c.startswith("vision.")]
    training_ids = set(training["id"])
    vision_ids = set(vision["id"])
    merged = training.merge(vision, on="id", how="left", validate="one_to_one")
    merged["_vision_match"] = merged["id"].map(
        lambda value: True if value in vision_ids else None
    )
    groups = {
        "month": merged["ts_start"].dt.strftime("%Y-%m").fillna("unknown"),
        "robot": merged["robot"].fillna("unknown"),
        "status": _status_labels(merged["terminal_status"]),
        "test": merged["is_test"]
        .map({True: "true", False: "false"})
        .fillna("unknown"),
    }
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "sources": {
            "training": "training/run_start.parquet",
            "vision": "vision.parquet",
        },
        "join": {
            "training_rows": len(training),
            "training_unique_ids": len(training_ids),
            "vision_rows": len(vision),
            "vision_unique_ids": len(vision_ids),
            "matched_training_runs": len(training_ids & vision_ids),
            "training_runs_without_vision_row": len(training_ids - vision_ids),
            "vision_rows_without_training_run": len(vision_ids - training_ids),
        },
        "overall": _summary(merged, metrics),
        "vision_metrics": {c: _coverage(merged[c]) for c in metrics},
        "unavailable_vision_metrics": sorted(VISION_METRICS - set(metrics)),
        "features": {c: _coverage(merged[c]) for c in FEATURES},
        "breakdowns": {
            name: {
                key: _summary(merged.loc[labels == key], metrics)
                for key in sorted(labels.unique())
            }
            for name, labels in groups.items()
        },
    }
    output = data_dir / "training" / "audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".audit-", dir=output.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, allow_nan=False)
            file.write("\n")
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
