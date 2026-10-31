"""Reproducible sampled live-video evaluation."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from .fold_frames import check_events, phase_clock_origin, video_time
from .fold_video import analyse_video

SAMPLE_PER_GROUP = 10
SAMPLE_SEED = 42
CHECK_DELAY_SECONDS = 1.0
EXPECTED_CHECKS = {"plate_clean", "spread", "fold1", "fold2"}


def sample_runs(
    candidates: list[dict[str, Any]], count: int, seed: int
) -> list[dict[str, Any]]:
    """Sample deterministically, round-robin over robot setups."""
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in sorted(candidates, key=lambda item: item["id"]):
        groups[candidate["robot"]].append(candidate)
    for group in groups.values():
        rng.shuffle(group)
    selected = []
    while len(selected) < count:
        added = False
        for robot in sorted(groups):
            if groups[robot] and len(selected) < count:
                selected.append(groups[robot].pop())
                added = True
        if not added:
            raise ValueError("Not enough eligible runs for live evaluation")
    return selected


def evaluate_live(data_dir: Path) -> dict[str, int]:
    """Analyze fixed balanced samples and score one second after each check."""
    root = data_dir / "fold_segmentation"
    checks_path = root / "live_checks.parquet"
    report_path = root / "live_report.md"
    if checks_path.exists() or report_path.exists():
        raise FileExistsError("Live evaluation exists; move it aside first")
    frames = pd.read_parquet(root / "frames.parquet")
    candidates: dict[str, list[dict[str, Any]]] = {"good": [], "bad": []}
    for run_id in sorted(frames["run_id"].unique()):
        run = json.loads(
            (data_dir / "raw" / "runs" / f"{run_id}.json").read_text()
        )
        events = check_events(run["logs"])
        if not EXPECTED_CHECKS.issubset({event.name for event in events}):
            continue
        group = None
        if run.get("terminal_status") == "succeeded" and all(
            e.check_ok for e in events
        ):
            group = "good"
        elif any(
            not e.check_ok and e.name in {"spread", "fold1", "fold2"}
            for e in events
        ):
            group = "bad"
        if group:
            candidates[group].append(
                {"id": int(run_id), "robot": run["machineName"]}
            )
    selected = [
        (group, item)
        for group in ("good", "bad")
        for item in sample_runs(
            candidates[group], SAMPLE_PER_GROUP, SAMPLE_SEED
        )
    ]
    rows = []
    run_medians: list[float] = []
    for group, item in selected:
        run_id = item["id"]
        live = root / "live" / str(run_id)
        if not live.exists():
            analyse_video(data_dir, run_id)
        timeline = pd.read_csv(live / "timeline.csv")
        if (
            "inference_ms" in timeline
            and timeline["inference_ms"].notna().any()
        ):
            run_medians.append(float(timeline["inference_ms"].median()))
        run = json.loads(
            (data_dir / "raw" / "runs" / f"{run_id}.json").read_text()
        )
        origin = phase_clock_origin(run["phases"])
        for event in check_events(run["logs"]):
            target = video_time(event, origin) + CHECK_DELAY_SECONDS
            available = timeline[timeline["video_t"] >= target]
            state = available.iloc[0]["state"] if len(available) else None
            rows.append(
                {
                    "run_id": run_id,
                    "robot": item["robot"],
                    "group": group,
                    "check": event.name,
                    "check_ok": event.check_ok,
                    "expected_state": event.state,
                    "video_t": target,
                    "predicted_state": state,
                    "match": state == event.state if state else None,
                }
            )
    table = pd.DataFrame(rows)
    table.to_parquet(checks_path, index=False)
    lines = [
        "# Live fold-state evaluation",
        "",
        f"Seed: {SAMPLE_SEED}.",
        "Scored at the first frame at least one second after each check.",
        "",
    ]
    for group in ("good", "bad"):
        subset = table[(table["group"] == group) & table["match"].notna()]
        correct = int(subset["match"].sum())
        value = (
            f"{correct / len(subset):.1%}" if len(subset) else "not available"
        )
        lines.append(f"- {group}: {value} ({correct}/{len(subset)} checks)")
    if run_medians:
        lines.append(
            f"- Median of per-run median inference times: "
            f"{pd.Series(run_medians).median():.1f} ms/frame."
        )
    lines.extend(
        ["", "Checks beyond the video are excluded from agreement.", ""]
    )
    report_path.write_text("\n".join(lines))
    return {
        "runs": len(selected),
        "checks": len(table),
        "available": int(table["match"].notna().sum()),
    }
