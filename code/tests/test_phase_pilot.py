"""Offline frame-pilot tests using only synthetic runs and video."""

import hashlib
import json
import os
import subprocess
import sys
from collections import namedtuple
from fractions import Fraction
from pathlib import Path

import pytest

from maus_export.phase_pilot import (
    MIN_FREE_BYTES,
    UnsafeTiming,
    _extract,
    _probe,
    prepare,
)


def test_pilot_selects_newest_and_preserves_decoded_pts(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    runs = raw / "runs"
    runs.mkdir(parents=True)
    for index in range(52):
        run = {
            "id": index,
            "tsStart": f"2026-01-01T00:{index:02}:00Z",
            "phases": [{"phase": "cook", "elapsed_sec": 1}],
        }
        (runs / f"{index}.json").write_text(json.dumps(run))
    media = raw / "media"
    (media / "51").mkdir(parents=True)
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=32x32:rate=2:duration=11",
            "-vf",
            "setpts=PTS+3/TB",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(media / "51" / "video.mp4"),
        ],
        check=True,
    )
    (media / "50").mkdir()
    (media / "50" / "video.mp4").write_bytes(b"not a video")
    output = tmp_path / "labeling" / "pilot"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "maus_export.phase_pilot",
            "--data-dir",
            str(tmp_path),
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(manifest["runs"]) == 50
    assert {run["run_id"] for run in manifest["runs"]} == set(range(2, 52))
    counts = {
        status: sum(r["status"] == status for r in manifest["runs"])
        for status in ("ok", "missing_video", "invalid_video")
    }
    assert counts == {"ok": 1, "missing_video": 48, "invalid_video": 1}
    assert manifest["classes"] == ["cook", "unclear"]
    run = next(r for r in manifest["runs"] if r["status"] == "ok")
    assert run["duration_ticks"] * 1 / 16384 == 11
    assert [f["requested_seconds"] for f in run["frames"]] == [0, 5, 10]
    assert [f["decoded_pts"] for f in run["frames"]] == [49152, 131072, 212992]
    assert run["time_base"] == "1/16384"
    for frame in run["frames"]:
        assert (output / frame["image"]).is_file()
        assert (
            frame["image_sha256"]
            == hashlib.sha256(
                (output / frame["image"]).read_bytes()
            ).hexdigest()
        )
        assert frame["phase_suggestion"] is None
        assert frame["suggestion_status"] == "unavailable_unaligned"
        assert frame["suggestion_shown"] is False
        assert frame["phase_context_shown"] is False
        assert frame["human_label"] is None
        assert frame["review_status"] == "unreviewed"
    assert run["phase_sequence"][0]["phase"] == "cook"
    assert run["phase_alignment"] == "unverified"
    assert os.stat(output).st_mode & 0o077 == 0
    original = (output / "manifest.json").read_bytes()
    second = subprocess.run(
        [
            sys.executable,
            "-m",
            "maus_export.phase_pilot",
            "--data-dir",
            str(tmp_path),
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert second.returncode != 0
    assert (output / "manifest.json").read_bytes() == original


def test_sampling_uses_displayed_frame_before_off_grid_target(
    tmp_path: Path,
) -> None:
    video = tmp_path / "video.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=32x32:rate=3/2:duration=11",
            "-vf",
            "setpts=PTS+3/TB",
            str(video),
        ],
        check=True,
    )
    result = _extract(video, tmp_path / "frames")
    base = Fraction(result["time_base"])
    offsets = [
        (frame["decoded_pts"] - result["start_pts"]) * base
        for frame in result["frames"]
    ]
    assert [f["requested_seconds"] for f in result["frames"]] == [0, 5, 10]
    assert offsets == [Fraction(0), Fraction(14, 3), Fraction(10)]


def test_probe_rejects_synthesized_or_disagreeing_frame_pts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "video.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=32x32:rate=2:duration=1",
            str(video),
        ],
        check=True,
    )
    real_run = subprocess.run

    def inconsistent_probe(
        args: list[str], *, capture_output: bool, check: bool
    ) -> subprocess.CompletedProcess[bytes]:
        result = real_run(args, capture_output=capture_output, check=check)
        data = json.loads(result.stdout)
        data["frames"][0]["pts"] = (
            data["frames"][0]["best_effort_timestamp"] + 1
        )
        return subprocess.CompletedProcess(
            result.args,
            result.returncode,
            json.dumps(data).encode(),
            result.stderr,
        )

    monkeypatch.setattr(
        "maus_export.phase_pilot.subprocess.run", inconsistent_probe
    )
    with pytest.raises(UnsafeTiming):
        _probe(video)


def test_failure_removes_partial_pilot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = tmp_path / "raw" / "runs"
    runs.mkdir(parents=True)
    for index in range(50):
        run = {"id": index, "tsStart": f"2026-01-01T00:{index:02}:00Z"}
        (runs / f"{index}.json").write_text(json.dumps(run))
    video = tmp_path / "raw" / "media" / "49" / "video.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"synthetic source")

    def interrupted_extract(source: Path, output: Path) -> dict:
        output.mkdir()
        (output / "partial.png").write_bytes(b"partial")
        raise UnsafeTiming

    monkeypatch.setattr("maus_export.phase_pilot._extract", interrupted_extract)
    output = tmp_path / "labeling" / "pilot"
    with pytest.raises(UnsafeTiming):
        prepare(tmp_path, output)
    assert not output.exists()
    assert video.read_bytes() == b"synthetic source"


def test_extraction_reserves_space_for_decoded_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = tmp_path / "raw" / "runs"
    runs.mkdir(parents=True)
    for index in range(50):
        run = {"id": index, "tsStart": f"2026-01-01T00:{index:02}:00Z"}
        (runs / f"{index}.json").write_text(json.dumps(run))
    video = tmp_path / "raw" / "media" / "49" / "video.mp4"
    video.parent.mkdir(parents=True)
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=32x32:rate=2:duration=11",
            str(video),
        ],
        check=True,
    )
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(
        "maus_export.phase_pilot.shutil.disk_usage",
        lambda _: usage(0, 0, MIN_FREE_BYTES + 1),
    )
    output = tmp_path / "labeling" / "pilot"
    with pytest.raises(ValueError, match="insufficient"):
        prepare(tmp_path, output)
    assert not output.exists()
