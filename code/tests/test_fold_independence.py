"""Log expectations and verdicts cannot change image-based predictions."""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from maus_export.fold_plate import PlateEllipse
from maus_export.fold_reference import VisualFlatReference
from maus_export.fold_video import analyse_video


def test_flat_reference_is_causal_and_visual_only() -> None:
    reference = VisualFlatReference()
    reference.observe("quarter", 10)
    assert reference.area is None
    for _ in range(3):
        reference.observe("flat", 100)
    assert reference.area == 100
    reference.observe("half", 50)
    assert reference.area == 100


def test_check_labels_and_verdicts_do_not_change_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "raw"
    (raw / "runs").mkdir(parents=True)
    media = raw / "media/1"
    media.mkdir(parents=True)
    calibration = tmp_path / "calibration.json"
    calibration.write_text("{}")
    monkeypatch.setattr("maus_export.fold_video.CALIBRATION_PATH", calibration)
    monkeypatch.setattr(
        "maus_export.fold_video.rectify_video_frame", lambda f, *a: (f, None)
    )
    monkeypatch.setattr(
        "maus_export.fold_video.detect_plate",
        lambda f: PlateEllipse(100, 100, 180, 180, 0),
    )
    writer = cv2.VideoWriter(
        str(media / "video.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10,
        (200, 200),
    )
    assert writer.isOpened()
    for index in range(40):
        frame = np.full((200, 200, 3), 25, np.uint8)
        cv2.circle(frame, (100, 100), 70, (120, 185, 220), -1)
        if index >= 20:
            frame[:100] = 25
        writer.write(frame)
    writer.release()
    run = {
        "tsStart": "2026-01-01T09:00:00Z",
        "tsEnd": "2026-01-01T09:00:04Z",
        "machineName": "synthetic",
        "phases": [
            {
                "phase": "cook",
                "elapsed_sec": 0,
                "timestamp": "2026-01-01T10:00:00",
            }
        ],
        "logs": [],
    }
    timelines = []
    for name, result in (
        ("spread", "crepe spread successfully"),
        ("fold2", "failed synthetic check"),
    ):
        run["logs"] = [
            {
                "timestamp": "2026-01-01T10:00:01",
                "msg": f"check_{name} started",
            },
            {
                "timestamp": "2026-01-01T10:00:01.100",
                "msg": f"check_{name}: {result}",
            },
        ]
        (raw / "runs/1.json").write_text(json.dumps(run))
        analyse_video(tmp_path, 1, replace=bool(timelines))
        timelines.append(
            pd.read_csv(tmp_path / "fold_segmentation/live/1/timeline.csv")
        )
    columns = ["prediction", "plate_fraction", "status", "rect_angle"]
    pd.testing.assert_frame_equal(timelines[0][columns], timelines[1][columns])
    assert "flat" in set(timelines[0].prediction)
    assert "half" in set(timelines[0].prediction)
