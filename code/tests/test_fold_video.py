"""Sequential live video analysis with synthetic frames."""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from maus_export.fold_plate import PlateEllipse
from maus_export.fold_video import analyse_video


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
        "logs": [],
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
    monkeypatch.setattr("maus_export.fold_video.CALIBRATION_PATH", path)
    monkeypatch.setattr(
        "maus_export.fold_video.detect_plate",
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
    result = analyse_video(tmp_path, 1)
    assert result["frames"] == 20
    out = tmp_path / "fold_segmentation" / "live" / "1"
    assert (out / "annotated.mp4").is_file()
    assert (out / "timeline.png").is_file()
    timeline = pd.read_csv(out / "timeline.csv")
    assert len(timeline) == 20
    assert set(timeline["state"]) == {"empty"}
    assert timeline.iloc[-1]["video_t"] == pytest.approx(1.9)
    with pytest.raises(FileExistsError):
        analyse_video(tmp_path, 1)
