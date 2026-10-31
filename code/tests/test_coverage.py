"""Coverage report on a synthetic export shaped like the dashboard."""

import json
from pathlib import Path

import pandas as pd
import pytest

from maus_classify.coverage import (
    MIN_FIT_RUNS,
    SPREAD_SCORES,
    CoverageError,
    report,
)
from maus_classify.frames import build_frames

BIG, SMALL, IDLE = "maus-a", "maus-b", "maus-idle"
N_OLD = 3
N_BIG = MIN_FIT_RUNS + 2
N_SMALL = 3


def check(phase: str, i: int, ok: bool = True, errors=()) -> dict:
    return {
        "phase": phase,
        "ok": ok,
        "errors": list(errors),
        "message": "ok",
        "frame": f"{i:02d}_{phase}_frame.jpg",
        "overlay": f"{i:02d}_{phase}_overlay.jpg",
        "timestamp": "2026-10-13T13:25:33",
    }


def run(run_id: int, robot: str, checks: list, **extra) -> dict:
    return {
        "id": run_id,
        "machineName": robot,
        "tsStart": f"2026-10-{10 + run_id % 15:02d}T11:25:00.000Z",
        "isTest": run_id % 2 == 0,
        "terminal_status": "succeeded",
        "videoUrl": "https://example.invalid/v.mp4",
        "visionChecks": checks,
        "logs": [],
        "parameters": {},
    } | extra


FULL = [
    check("spread", 0),
    check("fold1", 1, ok=False, errors=["not a half fold"]),
    check("fold2", 2),
]


def export(tmp: Path) -> Path:
    runs = [run(i, BIG, []) for i in range(1, N_OLD + 1)]
    ids = iter(range(100, 200))
    runs += [run(next(ids), BIG, FULL) for _ in range(N_BIG)]
    runs += [run(next(ids), SMALL, FULL) for _ in range(N_SMALL)]
    runs.append(
        run(next(ids), BIG, [check("spread", 0)], terminal_status="canceled")
    )
    runs.append(run(next(ids), SMALL, [check("plate_present", 0)]))
    (tmp / "raw" / "runs").mkdir(parents=True)
    for r in runs:
        (tmp / "raw" / "runs" / f"{r['id']}.json").write_text(
            json.dumps(r), encoding="utf-8"
        )
    robots = {"items": [{"name": n} for n in (BIG, SMALL, IDLE)]}
    (tmp / "raw" / "robots.json").write_text(json.dumps(robots))
    build_frames(tmp)
    # Scores for every full run but one big-robot run
    scored = [r["id"] for r in runs if r["visionChecks"] == FULL][1:]
    vision = pd.DataFrame(
        [{"id": i} | {c: 0.5 for c in SPREAD_SCORES} for i in scored]
    )
    vision.to_parquet(tmp / "vision.parquet")
    return tmp


def table_rows(text: str, heading: str) -> list[list[str]]:
    block = text.split(f"## {heading}")[1].split("##")[0]
    lines = [ln for ln in block.strip().splitlines() if ln.startswith("|")]
    return [[c.strip() for c in ln.strip("|").split("|")] for ln in lines[2:]]


def test_funnel_counts_every_step(tmp_path: Path) -> None:
    text = report(export(tmp_path))
    rows = table_rows(text, "Funnel from all runs to the fold-risk fit")
    counts = [int(r[1]) for r in rows]
    total = N_OLD + N_BIG + N_SMALL + 2
    assert counts == [
        total,
        total - N_OLD,
        N_BIG + N_SMALL + 1,
        N_BIG + N_SMALL,
        N_BIG + N_SMALL - 1,
        N_BIG - 1,
    ]
    assert "canceled 1" in rows[3][2]
    assert f"{SMALL} {N_SMALL}" in rows[5][2]


def test_robot_tables_include_idle_robot(tmp_path: Path) -> None:
    text = report(export(tmp_path))
    windows = {r[0]: r for r in table_rows(text, "Robots and time windows")}
    assert windows[IDLE][1] == "0"
    assert int(windows[BIG][3]) == N_BIG + 1
    frames = table_rows(text, "Frames per check and robot")
    header = text.split("## Frames per check and robot")[1].splitlines()[2]
    assert IDLE in header
    assert frames[-1][0] == "total"
    old = {r[0]: r for r in table_rows(text, "Runs without camera checks")}
    assert old[BIG][1] == str(N_OLD)
    assert old["total"][3] == str(N_OLD)  # all have a video url


def test_reason_and_fold_class_support(tmp_path: Path) -> None:
    text = report(export(tmp_path))
    reasons = table_rows(text, "Failure reason support")
    assert ["fold1", "not a half fold"] == reasons[0][:2]
    fold = {r[0]: r for r in table_rows(text, "Fold class support")}
    assert set(fold) == {"flat", "quarter"}


def test_aggregates_only(tmp_path: Path) -> None:
    text = report(export(tmp_path))
    assert "example.invalid" not in text
    assert "raw/media" not in text
    for run_id in (101, 120):
        assert f"| {run_id} |" not in text


def test_missing_inputs_fail_cleanly(tmp_path: Path) -> None:
    with pytest.raises(CoverageError):
        report(tmp_path)
