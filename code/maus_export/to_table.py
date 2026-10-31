"""Flatten raw runs into one row per run (CSV + Parquet) for analysis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

PARAM_PREFIX = "param."
CHECK_PREFIX = "check."
VISION_CHECKS = [
    "plate_present",
    "spread",
    "fold1",
    "fold2",
    "push_off",
    "plate_clean",
    "plate_loaded",
]
# A run that reached this phase made a real crepe, not a short test cycle
FULL_CYCLE_PHASE = "cook"
CONFIG_ID_LEN = 10
LOG_WARN = 30
LOG_ERROR = 40


def _cell(value: Any) -> Any:
    """Make a parameter value fit a flat column (lists become JSON)."""
    if isinstance(value, list | dict):
        return json.dumps(value, sort_keys=True)
    return value


def run_row(run: dict[str, Any]) -> dict[str, Any]:
    """One flat row for a raw `crepe_get` run, with every parameter."""
    start = pd.Timestamp(run["tsStart"])
    end = pd.Timestamp(run["tsEnd"]) if run.get("tsEnd") else None
    logs = run.get("logs") or []
    checks = run.get("visionChecks") or []
    failed = [c for c in checks if not c.get("ok")]
    phases = run.get("phases") or []
    phase_names = {p.get("phase") for p in phases}

    row: dict[str, Any] = {
        "id": run["id"],
        "parameter_id": "v1:sha256:"
        + hashlib.sha256(
            json.dumps(
                run.get("parameters"),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
        "robot": run.get("machineName"),
        "machine_id": run.get("machineId"),
        "ts_start": start,
        "duration_s": (end - start).total_seconds()
        if end is not None
        else None,
        "topping": run.get("topping"),
        "is_test": run.get("isTest"),
        "order_id": run.get("orderId") or None,
        "status": run.get("terminal_status"),
        "last_phase": phases[-1]["phase"] if phases else None,
        "is_full_cycle": FULL_CYCLE_PHASE in phase_names,
        "hotplate_temp": run.get("hotplateTemp"),
        "env_temp_start": run.get("envTempStart"),
        "env_humidity_start": run.get("envHumidityStart"),
        "n_checks": len(checks),
        "n_checks_failed": len(failed),
        "first_failed_check": failed[0].get("phase") if failed else None,
        "first_failed_error": "; ".join(failed[0].get("errors") or [])
        if failed
        else None,
        "last_check": checks[-1].get("phase") if checks else None,
        "last_check_ok": checks[-1].get("ok") if checks else None,
        "n_log_warn": sum(1 for e in logs if (e.get("level") or 0) == LOG_WARN),
        "n_log_error": sum(
            1 for e in logs if (e.get("level") or 0) >= LOG_ERROR
        ),
        "success_spread": run.get("successSpread"),
        "success_topping": run.get("successTopping"),
        "success_fold": run.get("successFold"),
        "success_handover": run.get("successHandover"),
        "has_batch": run.get("batch") is not None,
    }
    # Seconds spent in each phase (until the next phase starts)
    for cur, nxt in zip(phases, phases[1:] + [None], strict=True):
        t0 = float(cur.get("elapsed_sec") or 0.0)
        t1 = float(nxt["elapsed_sec"]) if nxt else (row["duration_s"] or t0)
        row[f"phase_s.{cur['phase']}"] = row.get(
            f"phase_s.{cur['phase']}", 0.0
        ) + max(t1 - t0, 0.0)
    row.update(check_columns(checks))
    for k, v in (run.get("parameters") or {}).items():
        row[f"{PARAM_PREFIX}{k}"] = _cell(v)
    return row


def check_columns(checks: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-check result columns. A check can repeat (push_off retries by
    design), so ok/error come from the last attempt and n counts attempts."""
    out: dict[str, Any] = {}
    for name in VISION_CHECKS:
        attempts = [c for c in checks if c.get("phase") == name]
        last = attempts[-1] if attempts else None
        out[f"{CHECK_PREFIX}{name}.n"] = len(attempts)
        ok = error = None
        if last is not None:
            ok = last.get("ok")
            error = "; ".join(last.get("errors") or []) or None
        out[f"{CHECK_PREFIX}{name}.ok"] = ok
        out[f"{CHECK_PREFIX}{name}.error"] = error
    return out


def fix_mixed_types(df: pd.DataFrame) -> pd.DataFrame:
    """Parquet needs one type per column. A parameter that was a number in
    some runs and a string in others is stored as text."""
    df = df.copy()
    for c in df.columns:
        if df[c].dtype != object:
            continue
        kinds = {type(v) for v in df[c].dropna()}
        if len(kinds) > 1 and not kinds <= {int, float}:
            df[c] = df[c].map(lambda v: None if pd.isna(v) else str(v))
    return df


def drop_constant_params(df: pd.DataFrame) -> pd.DataFrame:
    """Drop parameter columns that hold one value in every run. A key that
    is unset in some runs counts as varying."""
    params = [c for c in df.columns if c.startswith(PARAM_PREFIX)]
    const = [c for c in params if df[c].astype(str).nunique(dropna=False) <= 1]
    return df.drop(columns=const)


def _config_hash(values: dict[str, Any]) -> str:
    clean = {k: (None if pd.isna(v) else v) for k, v in values.items()}
    blob = json.dumps(clean, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:CONFIG_ID_LEN]


def add_config_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Add `config_id` (hash of the varying parameters) and `segment_id`
    (consecutive runs on one robot with the same config)."""
    params = sorted(c for c in df.columns if c.startswith(PARAM_PREFIX))
    df = df.copy()
    df["config_id"] = [
        _config_hash(rec) for rec in df[params].to_dict("records")
    ]
    order = df.sort_values(["robot", "id"], na_position="last").index
    seg = 0
    prev: tuple[Any, str] | None = None
    segments: dict[int, int] = {}
    for i in order:
        key = (df.at[i, "robot"], df.at[i, "config_id"])
        if prev is not None and key != prev:
            seg += 1
        segments[i] = seg
        prev = key
    df["segment_id"] = pd.Series(segments)
    return df


def runs_to_table(runs: list[dict[str, Any]]) -> pd.DataFrame:
    """Raw runs -> flat analysis table, sorted by id."""
    df = (
        pd.DataFrame([run_row(r) for r in runs])
        .sort_values("id")
        .reset_index(drop=True)
    )
    return fix_mixed_types(add_config_ids(drop_constant_params(df)))


def build_table(data_dir: Path) -> pd.DataFrame:
    runs = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in (data_dir / "raw" / "runs").glob("*.json")
    ]
    df = runs_to_table(runs)
    df.to_csv(data_dir / "runs.csv", index=False)
    df.to_parquet(data_dir / "runs.parquet", index=False)
    print(f"{len(df)} runs -> {data_dir / 'runs.csv'} and runs.parquet")
    return df
