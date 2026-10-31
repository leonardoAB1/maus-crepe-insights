"""How much data each frame classifier really has, by robot and time.

`maus-classify coverage` prints Markdown tables of aggregates only: no run
ids, no rows. It reads raw/runs/*.json, raw/robots.json, raw/media,
training/frames.parquet and vision.parquet from the data directory and
writes nothing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .frames import normalize
from .split import TEST, TRAIN

SPREAD = "spread"
FOLDS = ("fold1", "fold2")
CHECK_ORDER = (
    "plate_present",
    "spread",
    "fold1",
    "fold2",
    "push_off",
    "plate_clean",
    "plate_loaded",
)
REASON_CHECKS = ("spread", "fold1", "fold2", "push_off")
# Spread overlay scores the fold-risk analysis needs (#94)
SPREAD_SCORES = [
    "vision.spread.area_rel",
    "vision.spread.diameter_rel",
    "vision.spread.offset_rel",
    "vision.spread.circularity",
    "vision.spread.hole_frac",
    "vision.spread.uncovered_frac",
    "vision.spread.l_mean",
    "vision.spread.l_std",
    "vision.spread.residue_frac",
]
# A robot needs this many runs in the fold-risk sample to be fitted with its
# own fixed effect; fewer and it is only scored as an unseen robot
MIN_FIT_RUNS = 20
VIDEO_FILE = "video.mp4"


class CoverageError(ValueError):
    """Missing input, with a fixed message that is safe to print."""


def load_runs(data_dir: Path) -> pd.DataFrame:
    """One row per raw run: id, robot, UTC start, test flag, status and
    whether a video is referenced and downloaded."""
    files = sorted((data_dir / "raw" / "runs").glob("*.json"))
    if not files:
        raise CoverageError("no raw runs in raw/runs; fetch first")
    media = data_dir / "raw" / "media"
    rows = []
    for p in files:
        run = json.loads(p.read_text(encoding="utf-8"))
        rows.append(
            {
                "run_id": run["id"],
                "robot": run.get("machineName"),
                "ts_start": pd.Timestamp(run["tsStart"]),
                "is_test": bool(run.get("isTest")),
                "status": run.get("terminal_status"),
                "video_url": bool(run.get("videoUrl")),
                "video_file": (media / str(run["id"]) / VIDEO_FILE).is_file(),
            }
        )
    df = pd.DataFrame(rows)
    df["ts_start"] = pd.to_datetime(df["ts_start"], utc=True)
    return df.set_index("run_id").sort_index()


def robot_names(data_dir: Path, runs: pd.DataFrame) -> list[str]:
    """Registered robots (raw/robots.json) plus any seen in runs."""
    path = data_dir / "raw" / "robots.json"
    names: set[str] = set(runs["robot"].dropna())
    if path.is_file():
        reg = json.loads(path.read_text(encoding="utf-8"))
        items = reg.get("items", reg) if isinstance(reg, dict) else reg
        names |= {r["name"] for r in items if r.get("name")}
    return sorted(names)


def funnel(
    runs: pd.DataFrame, frames: pd.DataFrame, vision: pd.DataFrame
) -> list[dict[str, Any]]:
    """Runs left after each step from all runs to the fold-risk fit."""
    checks = frames.groupby("run_id")["check"].agg(set)
    camera = set(checks.index)
    spread = {r for r, cs in checks.items() if SPREAD in cs}
    fold = {r for r in spread if set(FOLDS) & checks[r]}
    if set(SPREAD_SCORES) <= set(vision.columns):
        ok = np.isfinite(vision[SPREAD_SCORES].to_numpy(float)).all(axis=1)
        scored_ids = set(vision.loc[ok, "id"])
    else:
        scored_ids = set()
    scored = fold & scored_ids
    per_robot = runs.loc[sorted(scored), "robot"].value_counts()
    fit_robots = sorted(per_robot[per_robot >= MIN_FIT_RUNS].index)
    fit = {r for r in scored if runs.at[r, "robot"] in fit_robots}
    no_fold = spread - fold
    status = runs.loc[sorted(no_fold), "status"].value_counts().to_dict()
    first_camera = min(camera) if camera else None
    before = int((runs.index < first_camera).sum()) if camera else len(runs)
    return [
        {"step": "all runs", "runs": len(runs), "dropped": ""},
        {
            "step": "with any camera check",
            "runs": len(camera),
            "dropped": f"{len(runs) - len(camera)} runs have no vision check "
            f"({before} are older than the first run with one)",
        },
        {
            "step": "with a spread check",
            "runs": len(spread),
            "dropped": f"{len(camera) - len(spread)} stopped before the "
            "spread (e.g. no plate)",
        },
        {
            "step": "with a fold check after the spread",
            "runs": len(fold),
            "dropped": f"{len(no_fold)} ended after the spread ("
            + ", ".join(f"{k} {v}" for k, v in sorted(status.items()))
            + ")",
        },
        {
            "step": "with spread overlay scores",
            "runs": len(scored),
            "dropped": f"{len(fold) - len(scored)} without a usable overlay",
        },
        {
            "step": f"fold-risk fit (robots with >= {MIN_FIT_RUNS} runs)",
            "runs": len(fit),
            "dropped": ", ".join(
                f"{r} {n}" for r, n in per_robot.items() if r not in fit_robots
            )
            + " scored as an unseen robot only",
        },
    ]


def _window(runs: pd.DataFrame) -> str:
    if runs.empty:
        return "-"
    return (
        f"{runs['ts_start'].min():%Y-%m-%d} to "
        f"{runs['ts_start'].max():%Y-%m-%d}"
    )


def robot_windows(
    runs: pd.DataFrame, frames: pd.DataFrame, robots: list[str]
) -> list[dict[str, Any]]:
    """Per robot: all runs, runs with camera checks and their UTC window."""
    camera = set(frames["run_id"])
    out = []
    for r in robots:
        mine = runs[runs["robot"] == r]
        cam = mine[mine.index.isin(camera)]
        out.append(
            {
                "robot": r,
                "runs": len(mine),
                "all_runs_window_utc": _window(mine),
                "camera_runs": len(cam),
                "camera_window_utc": _window(cam),
                "test_runs": int(cam["is_test"].sum()),
                "production_runs": int((~cam["is_test"]).sum()),
            }
        )
    return out


def frames_by_robot(frames: pd.DataFrame, robots: list[str]) -> pd.DataFrame:
    """Frames per check (rows) and robot (columns)."""
    t = pd.crosstab(frames["check"], frames["robot"])
    t = t.reindex(
        index=[c for c in CHECK_ORDER if c in t.index]
        + sorted(set(t.index) - set(CHECK_ORDER)),
        columns=robots,
        fill_value=0,
    )
    t.loc["total"] = t.sum()
    return t


def split_by_robot(frames: pd.DataFrame, robots: list[str]) -> pd.DataFrame:
    """Runs and frames per robot in train and test."""
    rows = {}
    for r in robots:
        mine = frames[frames["robot"] == r]
        rows[r] = {
            f"{side} runs": int(
                mine.loc[mine["split"] == side, "run_id"].nunique()
            )
            for side in (TRAIN, TEST)
        } | {
            f"{side} frames": int((mine["split"] == side).sum())
            for side in (TRAIN, TEST)
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def fold_class_support(frames: pd.DataFrame, robots: list[str]) -> pd.DataFrame:
    """Frames per fold class (passed spread / fold1 / fold2) and robot."""
    rows = frames[frames["fold_class"].notna()]
    t = pd.crosstab(rows["fold_class"], rows["robot"])
    return t.reindex(columns=robots, fill_value=0)


def reason_support(frames: pd.DataFrame) -> pd.DataFrame:
    """Failed frames per check and robot error string (a frame with several
    errors counts once per error), with the train/test split."""
    failed = frames[
        ~frames["ok"].astype(bool) & frames["check"].isin(REASON_CHECKS)
    ]
    recs = []
    for _, f in failed.iterrows():
        errors = [normalize(e) for e in f["errors"]] or [
            normalize(f["message"] or "") or "(no error text)"
        ]
        for e in sorted(set(errors)):
            recs.append({"check": f["check"], "error": e, "split": f["split"]})
    df = pd.DataFrame(recs, columns=["check", "error", "split"])
    t = pd.crosstab([df["check"], df["error"]], df["split"])
    return t.reindex(columns=[TRAIN, TEST], fill_value=0)


def older_runs(runs: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    """Per robot: runs without camera checks, their window and how many
    still have a video (referenced by the dashboard, or downloaded)."""
    old = runs[~runs.index.isin(set(frames["run_id"]))]
    rows = {}
    for r, g in old.groupby("robot"):
        rows[r] = {
            "runs_without_camera_checks": len(g),
            "window_utc": _window(g),
            "with_video_url": int(g["video_url"].sum()),
            "with_video_downloaded": int(g["video_file"].sum()),
        }
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.loc["total"] = [
        len(old),
        _window(old),
        int(old["video_url"].sum()),
        int(old["video_file"].sum()),
    ]
    return df


def md_table(df: pd.DataFrame, index: str | None = None) -> str:
    """Plain Markdown table (no extra dependency)."""
    if index is not None:
        df = df.reset_index().rename(columns={"index": index})
    cols = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "|" + "|".join("---" for _ in cols) + "|",
    ]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def report(data_dir: Path) -> str:
    """All coverage tables as Markdown."""
    frames_path = data_dir / "training" / "frames.parquet"
    vision_path = data_dir / "vision.parquet"
    for path in (frames_path, vision_path):
        if not path.is_file():
            raise CoverageError(f"missing {path.name}; build it first")
    runs = load_runs(data_dir)
    frames = pd.read_parquet(frames_path)
    vision = pd.read_parquet(vision_path)
    robots = robot_names(data_dir, runs)
    old = older_runs(runs, frames)
    parts = [
        "## Funnel from all runs to the fold-risk fit",
        md_table(pd.DataFrame(funnel(runs, frames, vision))),
        "## Robots and time windows",
        md_table(pd.DataFrame(robot_windows(runs, frames, robots))),
        "## Frames per check and robot",
        md_table(frames_by_robot(frames, robots), index="check"),
        "## Current split per robot",
        md_table(split_by_robot(frames, robots), index="robot"),
        "## Fold class support (frames)",
        md_table(fold_class_support(frames, robots), index="fold_class"),
        "## Failure reason support (failed frames per error)",
        md_table(reason_support(frames).reset_index()),
        "## Runs without camera checks",
        md_table(old, index="robot"),
    ]
    return "\n\n".join(parts) + "\n"
