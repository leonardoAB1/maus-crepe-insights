"""Synthetic dataset-to-mask and feature integration tests."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from maus_export.webserve_fold.fold_plate import PlateEllipse
from maus_export.webserve_fold.fold_segment import segment_dataset


def test_segment_dataset_and_no_self_flat_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "fold_segmentation"
    (root / "frames").mkdir(parents=True)
    empty = np.full((200, 200, 3), 25, np.uint8)
    flat = empty.copy()
    cv2.circle(flat, (100, 100), 72, (230, 230, 230), -1)
    for name, frame in (("empty", empty), ("flat", flat)):
        assert cv2.imwrite(str(root / "frames" / f"{name}.jpg"), frame)
    rows = [
        {
            "frame_id": state,
            "run_id": 1,
            "robot": "robot",
            "source_width": 200,
            "expected_state": state,
            "check_ok": True,
            "path": f"frames/{state}.jpg",
        }
        for state in ("empty", "flat")
    ]
    pd.DataFrame(rows).to_parquet(root / "frames.parquet", index=False)
    monkeypatch.setattr(
        "maus_export.webserve_fold.fold_segment.detect_plate",
        lambda image: PlateEllipse(100, 100, 180, 180, 0),
    )
    assert segment_dataset(tmp_path) == {"ok": 2}
    features = pd.read_parquet(root / "features.parquet").set_index("frame_id")
    assert features.loc["flat", "predicted_state"] == "flat"
    assert features.loc["flat", "reference_id"] == "empty"
    assert pd.isna(features.loc["flat", "flat_reference_id"])
    assert features.loc["empty", "predicted_state"] == "empty"
    with pytest.raises(FileExistsError):
        segment_dataset(tmp_path)
