"""Flatten raw runs into one row per run (CSV + Parquet) for analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

# A few parameters worth having as columns; the rest stay in the MCAP metadata.
KEY_PARAMS = [
    "dispense.batter.total_cook_time",
    "dispense.batter.fan_power",
    "dispense.batter.concentric_inner_radius_mm",
    "dispense.batter.concentric_outer_radius_mm",
    "dispense.batter.concentric_inner_flowrate_g_s",
    "dispense.batter.concentric_outer_flowrate_g_s",
    "dispense.batter.concentric_inner_speed_mm_s",
    "dispense.batter.concentric_outer_speed_mm_s",
    "dispense.batter.gluten_free",
    "dispense.batter.gluten_free_krampouz",
    "dispense.oil.grams",
    "fold.spatula_gate_enabled",
]


def run_row(run: dict[str, Any]) -> dict[str, Any]:
    start = pd.Timestamp(run["tsStart"])
    end = pd.Timestamp(run["tsEnd"]) if run.get("tsEnd") else None
    logs = run.get("logs") or []
    checks = run.get("visionChecks") or []
    failed = [c for c in checks if not c.get("ok")]
    phases = run.get("phases") or []

    row: dict[str, Any] = {
        "id": run["id"],
        "robot": run.get("machineName"),
        "machine_id": run.get("machineId"),
        "ts_start": start,
        "duration_s": (end - start).total_seconds() if end is not None else None,
        "topping": run.get("topping"),
        "is_test": run.get("isTest"),
        "order_id": run.get("orderId") or None,
        "status": run.get("terminal_status"),
        "last_phase": phases[-1]["phase"] if phases else None,
        "hotplate_temp": run.get("hotplateTemp"),
        "env_temp_start": run.get("envTempStart"),
        "env_humidity_start": run.get("envHumidityStart"),
        "n_checks": len(checks),
        "n_checks_failed": len(failed),
        "first_failed_check": failed[0].get("phase") if failed else None,
        "first_failed_error": "; ".join(failed[0].get("errors") or []) if failed else None,
        "last_check": checks[-1].get("phase") if checks else None,
        "last_check_ok": checks[-1].get("ok") if checks else None,
        "n_log_warn": sum(1 for e in logs if (e.get("level") or 0) == 30),
        "n_log_error": sum(1 for e in logs if (e.get("level") or 0) >= 40),
        "success_spread": run.get("successSpread"),
        "success_topping": run.get("successTopping"),
        "success_fold": run.get("successFold"),
        "success_handover": run.get("successHandover"),
        "has_batch": run.get("batch") is not None,
    }
    # Seconds spent in each phase (until the next phase starts)
    for cur, nxt in zip(phases, phases[1:] + [None]):
        t0 = float(cur.get("elapsed_sec") or 0.0)
        t1 = float(nxt["elapsed_sec"]) if nxt else (row["duration_s"] or t0)
        row[f"phase_s.{cur['phase']}"] = row.get(f"phase_s.{cur['phase']}", 0.0) + max(t1 - t0, 0.0)
    params = run.get("parameters") or {}
    for k in KEY_PARAMS:
        row[f"param.{k}"] = params.get(k)
    return row


def build_table(data_dir: Path) -> pd.DataFrame:
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in (data_dir / "raw" / "runs").glob("*.json")]
    df = pd.DataFrame([run_row(r) for r in runs]).sort_values("id").reset_index(drop=True)
    df.to_csv(data_dir / "runs.csv", index=False)
    df.to_parquet(data_dir / "runs.parquet", index=False)
    print(f"{len(df)} runs -> {data_dir / 'runs.csv'} and runs.parquet")
    return df
