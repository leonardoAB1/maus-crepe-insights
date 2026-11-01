"""Pi-only compression defaults, background scanning and encoder locking."""

import json
import os
from pathlib import Path

import pytest

from maus_export.experimental_fold.fold_pi import (
    PiPlaybackWorker,
    ensure_pi_playback,
    main,
)
from maus_export.experimental_fold.fold_playback import source_signature


def test_pi_analysis_automatically_prepares_playback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    monkeypatch.setattr(
        "maus_export.experimental_fold.fold_video.analyse_video",
        lambda *a, **k: calls.append("analyze"),
    )
    monkeypatch.setattr(
        "maus_export.experimental_fold.fold_pi.ensure_pi_playback",
        lambda *a, **k: calls.append("compress"),
    )
    main(["1", "--data-dir", str(tmp_path), "--analyze-only"])
    assert calls == ["analyze", "compress"]
    calls.clear()
    main(
        [
            "1",
            "--data-dir",
            str(tmp_path),
            "--analyze-only",
            "--no-compress-playback",
        ]
    )
    assert calls == ["analyze"]


def test_pi_website_starts_and_stops_background_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    class Worker:
        def __init__(self, data_dir: Path) -> None:
            assert data_dir == tmp_path

        def start(self) -> None:
            calls.append("worker")

        def stop(self) -> None:
            calls.append("stop")

    monkeypatch.setattr(
        "maus_export.experimental_fold.fold_pi.PiPlaybackWorker", Worker
    )
    monkeypatch.setattr(
        "maus_export.experimental_fold.fold_dashboard.serve_dashboard",
        lambda *a, **k: calls.append("serve"),
    )
    main(["--data-dir", str(tmp_path)])
    assert calls == ["worker", "serve", "stop"]


def test_worker_ignores_staging_and_incomplete_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("1", ".live-stage", "2"):
        directory = tmp_path / "fold_segmentation/live" / name
        directory.mkdir(parents=True)
        (directory / "annotated.mp4").write_bytes(b"synthetic")
        if name != "2":
            (directory / "summary.json").write_text("{}")
    calls = []
    monkeypatch.setattr(
        "maus_export.experimental_fold.fold_pi.ensure_pi_playback",
        lambda path, run_id, **kwargs: calls.append((run_id, kwargs)),
    )
    worker = PiPlaybackWorker(tmp_path)
    worker.scan_once()
    assert calls == [(1, {"blocking": False})]
    worker.stop_event.set()
    worker.scan_once()
    assert len(calls) == 1


@pytest.mark.skipif(os.name != "posix", reason="Pi uses POSIX file locking")
def test_proxy_is_reused_and_global_lock_excludes_another_encoder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import fcntl

    directory = tmp_path / "fold_segmentation/live/1"
    directory.mkdir(parents=True)
    original = directory / "annotated.mp4"
    original.write_bytes(b"synthetic-original")
    calls = []

    def encode(*args: object, **kwargs: object) -> None:
        calls.append("encode")
        (directory / "playback.mp4").write_bytes(b"synthetic-proxy")
        (directory / "playback.json").write_text(
            json.dumps(
                {
                    "source": source_signature(original),
                }
            )
        )

    monkeypatch.setattr(
        "maus_export.experimental_fold.fold_pi.prepare_playback", encode
    )
    with (tmp_path / "fold_segmentation/.playback.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert not ensure_pi_playback(tmp_path, 1, blocking=False)
        assert calls == []
    assert ensure_pi_playback(tmp_path, 1)
    assert ensure_pi_playback(tmp_path, 1)
    assert calls == ["encode"]
