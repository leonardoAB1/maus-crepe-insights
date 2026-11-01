"""Offline batch discovery, bounded concurrency and resumable processing."""

import sys
import threading
from pathlib import Path

import pytest

from maus_export.cli import main
from maus_export.experimental_fold.fold_batch import (
    OUTPUT_FILES,
    discover_runs,
    run_batch,
)


def cached_run(root: Path, run_id: int, *, video: bool = True) -> None:
    runs = root / "raw" / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    (runs / f"{run_id}.json").write_text("{}")
    if video:
        media = root / "raw" / "media" / str(run_id)
        media.mkdir(parents=True, exist_ok=True)
        (media / "video.mp4").write_bytes(b"synthetic")


def test_discover_only_cached_pairs_in_numeric_order(tmp_path):
    cached_run(tmp_path, 10)
    cached_run(tmp_path, 2)
    cached_run(tmp_path, 3, video=False)
    (tmp_path / "raw" / "runs" / "notes.json").write_text("{}")
    assert discover_runs(tmp_path) == [2, 10]


def test_parallel_workers_continue_after_failure(tmp_path, monkeypatch):
    for run_id in range(1, 5):
        cached_run(tmp_path, run_id)
    barrier = threading.Barrier(2, timeout=5)
    calls = []
    lock = threading.Lock()

    def analyse(root, run_id, **kwargs):
        assert root == tmp_path
        assert kwargs == {
            "replace": True,
            "analysis_fps": 5,
            "model": "classical",
        }
        with lock:
            calls.append(run_id)
        barrier.wait()  # two simultaneous workers, twice
        if run_id == 2:
            raise ValueError("bad video")
        return {"frames": 10}

    monkeypatch.setattr(
        "maus_export.experimental_fold.fold_batch.analyse_video", analyse
    )
    result = run_batch(tmp_path, threads=2, analysis_fps=5)
    assert result == {"total": 4, "completed": 3, "skipped": 0, "failed": 1}
    assert sorted(calls) == [1, 2, 3, 4]


def test_skip_complete_retry_partial_and_replace(tmp_path, monkeypatch):
    for run_id in (1, 2):
        cached_run(tmp_path, run_id)
        output = tmp_path / "fold_segmentation" / "live" / str(run_id)
        output.mkdir(parents=True)
        for name in OUTPUT_FILES if run_id == 1 else ["timeline.csv"]:
            (output / name).write_text("synthetic")
    calls = []

    def analyse(root, run_id, **kwargs):
        calls.append(run_id)
        return {"frames": 1}

    monkeypatch.setattr(
        "maus_export.experimental_fold.fold_batch.analyse_video", analyse
    )
    result = run_batch(tmp_path)
    assert calls == [2]
    assert result["skipped"] == 1
    calls.clear()
    result = run_batch(tmp_path, replace=True)
    assert calls == [1, 2]
    assert result["skipped"] == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"threads": 0},
        {"threads": -1},
        {"analysis_fps": 0},
        {"analysis_fps": float("nan")},
    ],
)
def test_invalid_options(tmp_path, kwargs):
    with pytest.raises(ValueError):
        run_batch(tmp_path, **kwargs)


@pytest.mark.parametrize("failed", [0, 1])
def test_cli_threads_and_failure_exit(tmp_path, monkeypatch, failed):
    def batch(root, **kwargs):
        assert root == tmp_path
        assert kwargs == {
            "threads": 3,
            "analysis_fps": 5.0,
            "replace": True,
            "model": "classical",
            "run_ids": None,
        }
        return {
            "total": 1,
            "completed": 1 - failed,
            "skipped": 0,
            "failed": failed,
        }

    monkeypatch.setattr(
        "maus_export.experimental_fold.fold_batch.run_batch", batch
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "maus-export",
            "fold-batch",
            "--data-dir",
            str(tmp_path),
            "--threads",
            "3",
            "--fps",
            "5",
            "--replace",
        ],
    )
    if failed:
        with pytest.raises(SystemExit) as error:
            main()
        assert error.value.code == 1
    else:
        main()


def test_selected_onnx_video_does_not_reuse_classical_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for run_id in (1, 2):
        cached_run(tmp_path, run_id)
    output = tmp_path / "fold_segmentation/live/2"
    output.mkdir(parents=True)
    for name in OUTPUT_FILES:
        (output / name).write_bytes(b"classical")
    calls = []
    monkeypatch.setattr(
        "maus_export.experimental_fold.fold_batch.analyse_video",
        lambda root, run_id, **kwargs: (
            calls.append((run_id, kwargs["model"])) or {"frames": 1}
        ),
    )
    result = run_batch(tmp_path, model="onnx", run_ids=[2])
    assert result["completed"] == 1
    assert calls == [(2, "onnx")]
    assert (output / "timeline.csv").read_bytes() == b"classical"


def test_missing_dataset_and_empty_dataset(tmp_path):
    with pytest.raises(ValueError, match="raw/runs"):
        run_batch(tmp_path)
    (tmp_path / "raw" / "runs").mkdir(parents=True)
    assert run_batch(tmp_path) == {
        "total": 0,
        "completed": 0,
        "skipped": 0,
        "failed": 0,
    }
