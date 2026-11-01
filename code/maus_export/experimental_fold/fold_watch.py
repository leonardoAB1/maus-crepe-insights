"""One-command offline analysis and local review."""

from __future__ import annotations

from pathlib import Path

from .fold_backend import MODELS, analysis_root
from .fold_dashboard import serve_dashboard
from .fold_video import DEFAULT_ANALYSIS_FPS, analyse_video


def watch_videos(
    data_dir: Path,
    run_ids: list[int],
    *,
    replace: bool = False,
    fps: float = DEFAULT_ANALYSIS_FPS,
    port: int = 8766,
    open_browser: bool = True,
    model: str = "classical",
) -> None:
    """Analyze cached runs, reuse completed outputs, then serve localhost.

    This command never fetches data or authenticates. Existing outputs are
    reused unless replacement is explicitly requested. Ctrl-C stops review.
    """
    if model == "both":
        for backend in MODELS:
            _analyse_runs(data_dir, run_ids, backend, replace, fps)
    else:
        _analyse_runs(data_dir, run_ids, model, replace, fps)
    serve_dashboard(data_dir, port, open_browser=open_browser)


def _analyse_runs(
    data_dir: Path,
    run_ids: list[int],
    model: str,
    replace: bool,
    fps: float,
) -> None:
    root = analysis_root(data_dir, model)
    for run_id in dict.fromkeys(run_ids):
        output = root / str(run_id)
        complete = all(
            (output / name).is_file()
            for name in (
                "annotated.mp4",
                "timeline.csv",
                "observations.csv",
                "summary.json",
            )
        )
        if complete and not replace:
            print(f"Reusing run {run_id}; use --replace to rebuild")
            continue
        print(f"Analyzing run {run_id} at {fps:g} fps", flush=True)
        print(
            analyse_video(
                data_dir,
                run_id,
                replace=replace,
                analysis_fps=fps,
                model=model,
            )
        )
