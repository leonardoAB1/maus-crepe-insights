"""The one-command workflow is offline and preserves completed artifacts."""

from pathlib import Path

import pytest

from maus_export.experimental_fold.fold_watch import watch_videos


def test_watch_analyzes_once_and_serves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    monkeypatch.setattr(
        "maus_export.experimental_fold.fold_watch.analyse_video",
        lambda *args, **kwargs: calls.append(("analyze", args, kwargs)),
    )
    monkeypatch.setattr(
        "maus_export.experimental_fold.fold_watch.serve_dashboard",
        lambda *args, **kwargs: calls.append(("serve", args, kwargs)),
    )
    watch_videos(tmp_path, [1, 1], port=8768, open_browser=False)
    assert [call[0] for call in calls] == ["analyze", "serve"]
    assert calls[0][2]["analysis_fps"] == 10
    assert calls[0][2]["model"] == "classical"
    assert calls[1][2]["open_browser"] is False


def test_watch_reuses_complete_outputs_but_not_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "fold_segmentation/live/1"
    output.mkdir(parents=True)
    for name in (
        "annotated.mp4",
        "timeline.csv",
        "observations.csv",
        "summary.json",
    ):
        (output / name).touch()
    calls = []
    monkeypatch.setattr(
        "maus_export.experimental_fold.fold_watch.analyse_video",
        lambda *args, **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(
        "maus_export.experimental_fold.fold_watch.serve_dashboard",
        lambda *a, **k: None,
    )
    watch_videos(tmp_path, [1])
    assert calls == []
    watch_videos(tmp_path, [1], replace=True)
    assert calls[-1]["replace"] is True
    (output / "summary.json").unlink()
    watch_videos(tmp_path, [1])
    assert len(calls) == 2


def test_watch_prepares_both_models_before_serving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    monkeypatch.setattr(
        "maus_export.experimental_fold.fold_watch.analyse_video",
        lambda *args, **kwargs: calls.append(kwargs["model"]),
    )
    monkeypatch.setattr(
        "maus_export.experimental_fold.fold_watch.serve_dashboard",
        lambda *args, **kwargs: calls.append("serve"),
    )
    watch_videos(tmp_path, [1, 1], model="both")
    assert calls == ["classical", "onnx", "serve"]
    with pytest.raises(ValueError, match="Unknown"):
        watch_videos(tmp_path, [1], model="unknown")
