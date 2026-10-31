"""Synthetic video extraction and output-safety checks."""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from maus_export.fold_dataset import extract_frames


def test_extract_check_frame_and_refuse_overwrite(tmp_path: Path) -> None:
    runs = tmp_path / "raw" / "runs"
    media = tmp_path / "raw" / "media" / "1"
    runs.mkdir(parents=True)
    media.mkdir(parents=True)
    run = {
        "tsStart": "2026-01-01T09:00:00Z",
        "tsEnd": "2026-01-01T09:00:05Z",
        "machineName": "synthetic",
        "phases": [{"timestamp": "2026-01-01T10:00:00", "elapsed_sec": 0}],
        "logs": [
            {"timestamp": "2026-01-01T10:00:02", "msg": "check_fold1 started"},
            {"timestamp": "2026-01-01T10:00:03", "msg": "check_fold1: failed"},
        ],
    }
    (runs / "1.json").write_text(json.dumps(run))
    calibration = {
        "schema_version": 1,
        "robot_corrections": {
            "synthetic": {
                "source_size_px": [100, 50],
                "target_size_px": [100, 50],
                "matrix_mean": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            }
        },
    }
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps(calibration))
    writer = cv2.VideoWriter(
        str(media / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (100, 50)
    )
    assert writer.isOpened()
    for index in range(50):
        writer.write(np.full((50, 100, 3), index * 4, dtype=np.uint8))
    writer.release()
    assert extract_frames(tmp_path, calibration_path) == {
        "runs": 1,
        "frames": 1,
    }
    table = pd.read_parquet(tmp_path / "fold_segmentation" / "frames.parquet")
    assert len(table) == 1
    assert table.iloc[0]["expected_state"] == "half"
    assert not table.iloc[0]["check_ok"]
    assert table.iloc[0]["video_t"] == pytest.approx(1.6)
    assert (tmp_path / "fold_segmentation" / table.iloc[0]["path"]).is_file()
    with pytest.raises(FileExistsError):
        extract_frames(tmp_path, calibration_path)
