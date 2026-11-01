"""Bounded parallel processing of all locally cached run/video pairs."""

from __future__ import annotations

import math
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from .fold_backend import analysis_root
from .fold_video import DEFAULT_ANALYSIS_FPS, analyse_video

OUTPUT_FILES = (
    "summary.json",
    "timeline.csv",
    "observations.csv",
    "annotated.mp4",
    "timeline.png",
)


def discover_runs(data_dir: Path) -> list[int]:
    """Find cached videos with matching run metadata, in numeric order."""
    raw = data_dir / "raw"
    return sorted(
        {
            int(path.stem)
            for path in (raw / "runs").glob("*.json")
            if path.stem.isdecimal()
            and (raw / "media" / path.stem / "video.mp4").is_file()
        }
    )


def run_batch(
    data_dir: Path,
    *,
    threads: int = 1,
    analysis_fps: float = DEFAULT_ANALYSIS_FPS,
    replace: bool = False,
    model: str = "classical",
    run_ids: list[int] | None = None,
) -> dict[str, int]:
    """Process every cached video, resuming by skipping complete artifacts.

    Only one task per worker is submitted at a time. Failed videos do not stop
    other runs; their per-video snapshots retain the failure for the dashboard.
    Invoke only one batch against a given data directory at a time.
    """
    if threads < 1:
        raise ValueError("Thread count must be at least 1")
    if not math.isfinite(analysis_fps) or analysis_fps <= 0:
        raise ValueError("Analysis rate must be positive and finite")
    data_dir = data_dir.resolve()
    if not (data_dir / "raw" / "runs").is_dir():
        raise ValueError("Data directory must contain raw/runs")
    output_root = analysis_root(data_dir, model)
    runs = discover_runs(data_dir)
    if run_ids is not None:
        if not set(run_ids).issubset(runs):
            raise ValueError("Requested video is not cached")
        runs = sorted(set(run_ids))
    totals = {"total": len(runs), "completed": 0, "skipped": 0, "failed": 0}
    pending = []
    for run_id in runs:
        output = output_root / str(run_id)
        if not replace and all(
            (output / name).is_file() for name in OUTPUT_FILES
        ):
            totals["skipped"] += 1
        else:
            pending.append(run_id)
    print(
        f"Found {len(runs)} cached videos; {totals['skipped']} complete, "
        f"{len(pending)} to process with {threads} threads.",
        flush=True,
    )
    queue = iter(pending)
    with ThreadPoolExecutor(max_workers=threads) as pool:
        active = {}

        def submit_next() -> None:
            run_id = next(queue, None)
            if run_id is not None:
                future = pool.submit(
                    analyse_video,
                    data_dir,
                    run_id,
                    replace=True,
                    analysis_fps=analysis_fps,
                    model=model,
                )
                active[future] = run_id

        for _ in range(min(threads, len(pending))):
            submit_next()
        while active:
            finished, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in finished:
                run_id = active.pop(future)
                try:
                    result = future.result()
                except Exception as error:
                    totals["failed"] += 1
                    print(
                        f"Run {run_id}: failed ({type(error).__name__})",
                        flush=True,
                    )
                else:
                    totals["completed"] += 1
                    print(
                        f"Run {run_id}: completed ({result['frames']} frames)",
                        flush=True,
                    )
                submit_next()
    return totals
