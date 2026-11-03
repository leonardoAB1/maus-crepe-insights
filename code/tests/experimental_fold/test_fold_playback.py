"""Private web proxies preserve originals and reject stale or failed output."""

from pathlib import Path

import pytest

from maus_export.experimental_fold.fold_playback import (
    playback_video,
    prepare_playback,
)


def test_prepare_proxy_and_original_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "fold_segmentation/live/1"
    directory.mkdir(parents=True)
    original = directory / "annotated.mp4"
    original.write_bytes(b"synthetic-original")
    monkeypatch.setattr(
        "maus_export.experimental_fold.fold_playback.shutil.which",
        lambda _: "ffmpeg",
    )
    calls = []

    def encode(command: list[str], **kwargs: object) -> None:
        calls.append(command)
        Path(command[-1]).write_bytes(b"small-proxy")

    monkeypatch.setattr(
        "maus_export.experimental_fold.fold_playback.subprocess.run", encode
    )
    assert playback_video(directory) == original
    result = prepare_playback(tmp_path, 1)
    assert result["codec"] == "h264"
    assert "libx264" in calls[0] and "+faststart" in calls[0]
    assert "yuv420p" in calls[0] and "120k" in calls[0]
    assert playback_video(directory).name == "playback.mp4"
    assert original.read_bytes() == b"synthetic-original"
    with pytest.raises(FileExistsError):
        prepare_playback(tmp_path, 1)
    original.write_bytes(b"new-analysis")
    assert playback_video(directory) == original


def test_failed_encoding_does_not_publish_partial_proxy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "fold_segmentation/live/1"
    directory.mkdir(parents=True)
    (directory / "annotated.mp4").write_bytes(b"synthetic-original")
    monkeypatch.setattr(
        "maus_export.experimental_fold.fold_playback.shutil.which",
        lambda _: "ffmpeg",
    )

    def fail(command: list[str], **kwargs: object) -> None:
        Path(command[-1]).write_bytes(b"partial")
        raise RuntimeError("synthetic encoder failure")

    monkeypatch.setattr(
        "maus_export.experimental_fold.fold_playback.subprocess.run", fail
    )
    with pytest.raises(RuntimeError):
        prepare_playback(tmp_path, 1)
    assert not (directory / "playback.mp4").exists()
    assert not list(directory.glob(".playback-*"))


def test_onnx_proxy_uses_onnx_folder_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    classical = tmp_path / "fold_segmentation/live/1"
    onnx = tmp_path / "fold_segmentation/live-onnx/1"
    for directory in (classical, onnx):
        directory.mkdir(parents=True)
        (directory / "annotated.mp4").write_bytes(b"synthetic-original")
    monkeypatch.setattr(
        "maus_export.experimental_fold.fold_playback.shutil.which",
        lambda _: "ffmpeg",
    )
    sources = []

    def encode(command: list[str], **kwargs: object) -> None:
        sources.append(Path(command[command.index("-i") + 1]))
        Path(command[-1]).write_bytes(b"small-proxy")

    monkeypatch.setattr(
        "maus_export.experimental_fold.fold_playback.subprocess.run", encode
    )
    prepare_playback(tmp_path, 1, model="onnx")
    assert sources == [onnx / "annotated.mp4"]
    assert playback_video(onnx).name == "playback.mp4"
    assert playback_video(classical) == classical / "annotated.mp4"
    with pytest.raises(ValueError):
        prepare_playback(tmp_path, 1, model="unknown")


@pytest.mark.parametrize("width,bitrate", [(1, 120), (3, 120), (480, 0)])
def test_invalid_proxy_settings(
    tmp_path: Path, width: int, bitrate: int
) -> None:
    with pytest.raises(ValueError):
        prepare_playback(tmp_path, 1, width=width, bitrate_kbps=bitrate)
