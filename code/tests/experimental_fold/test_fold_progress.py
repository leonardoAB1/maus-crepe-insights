"""Runner progress is observable offline and survives process boundaries."""

import json
from pathlib import Path

import pytest

from maus_export.experimental_fold.fold_progress import (
    VideoProgress,
    read_progress,
)
from maus_export.experimental_fold.fold_video import analyse_video


def test_progress_metrics_and_completion(tmp_path, monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(
        "maus_export.experimental_fold.fold_progress.time.perf_counter",
        lambda: clock[0],
    )
    progress = VideoProgress(tmp_path, 42)
    assert read_progress(tmp_path)[0]["fps"] is None
    clock[0] = 12.0
    progress.update(
        state="running", frames_done=10, frames_total=40, fraction=0.25
    )
    value = read_progress(tmp_path)[0]
    assert value["fps"] == 5
    assert value["seconds_per_frame"] == 0.2
    assert value["eta_seconds"] == 6
    progress.update(force=True, state="completed", fraction=1.0)
    assert read_progress(tmp_path)[0]["eta_seconds"] is None
    assert not list(progress.path.parent.glob(".progress-*"))


def test_stale_and_corrupt_progress(tmp_path):
    progress = VideoProgress(tmp_path, 1)
    value = json.loads(progress.path.read_text())
    value["updated_at"] = 0
    progress.path.write_text(json.dumps(value))
    progress.path.with_name("2.json").write_text("{")
    assert read_progress(tmp_path)[0]["stale"] is True


def test_runner_publishes_failure(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        analyse_video(tmp_path, 123)
    value = read_progress(tmp_path)[0]
    assert value["state"] == "failed"
    assert value["error"] == "FileNotFoundError"
    assert "No such file or directory" in value["error_message"]


def test_runner_publishes_completion(tmp_path, monkeypatch):
    def fake_runner(*args, progress, **kwargs):
        progress.update(frames_done=3)
        return {"frames": 3}

    monkeypatch.setattr(
        "maus_export.experimental_fold.fold_video._analyse_video", fake_runner
    )
    assert analyse_video(tmp_path, 1) == {"frames": 3}
    value = read_progress(tmp_path)[0]
    assert value["state"] == "completed"
    assert value["fraction"] == 1
    assert value["frames_done"] == 3
