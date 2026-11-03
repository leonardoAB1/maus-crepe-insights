"""Reproducible sampled live-video evaluation."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from .fold_frames import check_starts
from .fold_video import analyse_video

SAMPLE_PER_GROUP = 10
SAMPLE_SEED = 42
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
        events = check_starts(run["logs"])
        if not EXPECTED_CHECKS.issubset({event.name for event in events}):
            continue
        group = None
        if run.get("terminal_status") == "succeeded" and all(
            e.check_ok is True for e in events
        ):
            group = "good"
        elif any(
            e.check_ok is False and e.name in {"spread", "fold1", "fold2"}
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
    dispenses: list[dict[str, object]] = []
    for group, item in selected:
        run_id = item["id"]
        live = root / "live" / str(run_id)
        if not (live / "observations.csv").is_file():
            analyse_video(data_dir, run_id, replace=live.exists())
        summary_path = live / "summary.json"
        if summary_path.is_file():
            dispense = json.loads(summary_path.read_text()).get("dispense")
            if isinstance(dispense, dict):
                dispenses.append(dispense)
        timeline = pd.read_csv(live / "timeline.csv")
        if (
            "inference_ms" in timeline
            and timeline["inference_ms"].notna().any()
        ):
            run_medians.append(float(timeline["inference_ms"].median()))
        observations = pd.read_csv(live / "observations.csv")
        for observation in observations.to_dict("records"):
            state = observation["observation"]
            available = pd.notna(state)
            rows.append(
                {
                    "run_id": run_id,
                    "robot": item["robot"],
                    "group": group,
                    "check": observation["check"],
                    "robot_verdict": observation["robot_verdict"],
                    "expected_state": observation["expected_state"],
                    "video_t": observation["observed_at"],
                    "predicted_state": state if available else None,
                    "status": observation["status"],
                    "match": state == observation["expected_state"]
                    if available
                    else None,
                }
            )
    table = pd.DataFrame(rows)
    table.to_parquet(checks_path, index=False)
    lines = [
        "# Live fold-state evaluation",
        "",
        f"Seed: {SAMPLE_SEED}.",
        "Post-check visual votes are compared with the expected check state.",
        "Robot verdicts and expected states are not ground truth.",
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
    dispensed = [
        item
        for item in dispenses
        if item.get("final_fraction") is not None and item.get("ok") is not None
    ]
    if dispensed:
        finals = [
            float(item["final_fraction"])  # type: ignore[arg-type]
            for item in dispensed
        ]
        passed = sum(bool(item["ok"]) for item in dispensed)
        lines.append(
            f"- Dispensing: {passed}/{len(dispensed)} sampled runs end with "
            f"stable dough coverage of at least 40% of the plate "
            f"(median final {pd.Series(finals).median():.0%})."
        )
    lines.extend(
        [
            "",
            "Unknown and incomplete observations are excluded from agreement.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines))
    return {
        "runs": len(selected),
        "checks": len(table),
        "available": int(table["match"].notna().sum()),
    }
