"""Sequential live video analysis with synthetic frames."""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from maus_export.experimental_fold.fold_plate import PlateEllipse
from maus_export.experimental_fold.fold_video import _phase_color, analyse_video


def test_dispense_phase_variants_share_the_dough_color() -> None:
    assert _phase_color("spread_batter") == "#e8a35d"
    assert _phase_color("spread_batter_concentric") == "#e8a35d"
    assert _phase_color("spread_batter_spiral") == "#e8a35d"
    assert _phase_color("cook") == "#f2cf5b"
    assert _phase_color("add_topping") == "#ec86cb"
    assert _phase_color("add_topping_chocolate") == "#ec86cb"


def test_analyse_video_writes_timeline_without_seeking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = tmp_path / "raw" / "runs"
    media = tmp_path / "raw" / "media" / "1"
    runs.mkdir(parents=True)
    media.mkdir(parents=True)
    run = {
        "tsStart": "2026-01-01T09:00:00Z",
        "tsEnd": "2026-01-01T09:00:02Z",
        "machineName": "synthetic",
        "phases": [{"timestamp": "2026-01-01T10:00:00", "elapsed_sec": 0}],
        "logs": [
            {"timestamp": "2026-01-01T10:00:01", "msg": "check_fold1 started"},
            {
                "timestamp": "2026-01-01T10:00:01.100",
                "msg": "check_fold1: first fold completed",
            },
        ],
    }
    (runs / "1.json").write_text(json.dumps(run))
    calibration = {
        "schema_version": 1,
        "robot_corrections": {
            "synthetic": {
                "source_size_px": [200, 200],
                "target_size_px": [200, 200],
                "matrix_mean": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            }
        },
    }
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(calibration))
    monkeypatch.setattr(
        "maus_export.experimental_fold.fold_video.CALIBRATION_PATH", path
    )
    monkeypatch.setattr(
        "maus_export.experimental_fold.fold_video.detect_plate",
        lambda frame: PlateEllipse(100, 100, 180, 180, 0),
    )
    writer = cv2.VideoWriter(
        str(media / "video.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10,
        (200, 200),
    )
    assert writer.isOpened()
    for _ in range(20):
        writer.write(np.full((200, 200, 3), 25, np.uint8))
    writer.release()
    result = analyse_video(tmp_path, 1, analysis_fps=10)
    assert result["frames"] == 20
    progress = json.loads(
        (tmp_path / "fold_segmentation" / "progress" / "1.json").read_text()
    )
    assert progress["state"] == "completed"
    assert progress["frames_done"] == progress["frames_total"] == 20
    assert progress["fraction"] == 1.0
    assert progress["fps"] > 0
    assert progress["seconds_per_frame"] > 0
    out = tmp_path / "fold_segmentation" / "live" / "1"
    assert (out / "annotated.mp4").is_file()
    assert (out / "timeline.png").is_file()
    timeline = pd.read_csv(out / "timeline.csv")
    assert len(timeline) == 20
    assert timeline.loc[timeline["video_t"] < 1.6, "state"].isna().all()
    assert timeline["state"].isna().all()
    checks = pd.read_csv(out / "observations.csv")
    assert pd.isna(checks.iloc[0]["observation"])
    assert checks.iloc[0]["status"] == "no_flat_reference"
    assert checks.iloc[0]["expected_state"] == "half"
    assert checks.iloc[0]["robot_verdict"] == "reported_pass"
    assert timeline.iloc[-1]["video_t"] == pytest.approx(1.9)
    with pytest.raises(FileExistsError):
        analyse_video(tmp_path, 1)
    original = (out / "timeline.csv").read_bytes()
    (out / "old-marker").write_text("old output")
    with monkeypatch.context() as patch:

        def fail(*args: object, **kwargs: object) -> None:
            raise ValueError("synthetic failure")

        patch.setattr(
            "maus_export.experimental_fold.fold_video.annotate_frame", fail
        )
        with pytest.raises(ValueError, match="synthetic failure"):
            analyse_video(tmp_path, 1, replace=True, analysis_fps=10)
    assert (out / "timeline.csv").read_bytes() == original
    assert (out / "old-marker").exists()
    replacement = analyse_video(tmp_path, 1, replace=True, analysis_fps=10)
    assert replacement["frames"] == 20
    assert "median_render_ms" in replacement
    assert (out / "annotated.mp4").is_file()
    timeline_columns = pd.read_csv(out / "timeline.csv").columns
    assert "plate_fraction" in timeline_columns
    assert "phase" in timeline_columns
    assert "raw_plate_fraction" in timeline_columns
    assert "structural_prediction" in timeline_columns
    assert "verdict" in timeline_columns
    assert "confidence" in timeline_columns
    assert checks.iloc[0]["verdict"] == "unknown"
    assert checks.iloc[0]["confidence"] == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["dispense"]["frames"] == 0
    assert summary["dispense"]["ok"] is None
    assert summary["mask_window_frames"] == 5
    assert summary["toppings_stage_logged"] is False
    assert (out / "reference" / "manifest.json").is_file()
    assert not (out / "old-marker").exists()


def _timeline_rows(with_dough: bool) -> list[dict]:
    rows = [
        {
            "video_t": t,
            "state": ("flat" if with_dough and t >= 2 else None),
            "phase": "spread_batter" if with_dough and 2 <= t < 4 else "",
            "plate_fraction": ("0.6" if with_dough and 2 <= t < 4 else ""),
        }
        for t in (0.0, 1.0, 2.0, 3.0, 5.0)
    ]
    return rows


def test_timeline_plot_dough_band_toggle(tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from maus_export.experimental_fold.fold_video import _timeline_plot

    with_dough = tmp_path / "dough.png"
    _timeline_plot(with_dough, _timeline_rows(True), [])
    without_dough = tmp_path / "plain.png"
    _timeline_plot(without_dough, _timeline_rows(False), [])
    assert with_dough.is_file() and without_dough.is_file()
    assert plt.imread(str(with_dough)).shape[0] > 400  # two stacked subplots
    assert plt.imread(str(without_dough)).shape[0] > 400
