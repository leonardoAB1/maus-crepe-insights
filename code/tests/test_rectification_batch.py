"""Batch export tests use only synthetic local media."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from maus_export.rectification_batch import (
    consistent_transforms,
    process_item,
    run_batch,
    video_alignment,
)


def identity() -> dict:
    return {
        "source_size": [320, 240],
        "target_size": [900, 900],
        "matrix": np.eye(3).tolist(),
    }


def test_consistency_checks_entire_field() -> None:
    first, second = identity(), identity()
    assert consistent_transforms(first, second)
    second["matrix"][0][2] = 4
    assert not consistent_transforms(first, second)
    second = identity()
    second["source_size"] = [640, 480]
    assert not consistent_transforms(first, second)


def test_missing_reference_is_reported_and_original_unchanged(
    tmp_path: Path,
) -> None:
    source = tmp_path / "01_spread_frame.jpg"
    cv2.imwrite(str(source), np.zeros((240, 320, 3), np.uint8))
    original = source.read_bytes()
    report = process_item(source, tmp_path / "output")
    assert report["status"] == "skipped"
    assert "Missing" in report["reason"]
    assert source.read_bytes() == original
    assert (tmp_path / "output" / f"{source.name}.json").exists()


def test_nonrectified_reference_is_not_exported(tmp_path: Path) -> None:
    source = tmp_path / "01_spread_frame.jpg"
    for name in (source, tmp_path / "01_spread_overlay.jpg"):
        cv2.imwrite(str(name), np.zeros((240, 320, 3), np.uint8))
    report = process_item(source, tmp_path / "output")
    assert report["status"] == "skipped"
    assert "not a square" in report["reason"]
    assert not list((tmp_path / "output").glob("*.png"))


def test_successful_synthetic_export_and_resume(tmp_path: Path) -> None:
    rng = np.random.default_rng(19)
    gray = rng.integers(20, 230, (480, 480), dtype=np.uint8)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    raw = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    source = tmp_path / "01_spread_frame.jpg"
    cv2.imwrite(str(source), raw)
    cv2.imwrite(str(tmp_path / "01_spread_overlay.jpg"), raw)
    destination = tmp_path / "output"
    report = process_item(source, destination)
    assert report["status"] == "exported"
    assert len(report["outputs"]) == 2
    before = (destination / report["outputs"][0]).stat().st_mtime_ns
    assert process_item(source, destination) == report
    assert (destination / report["outputs"][0]).stat().st_mtime_ns == before


def test_batch_all_skips_completes_with_private_report(tmp_path: Path) -> None:
    root = tmp_path / "data"
    folder = root / "raw" / "media" / "synthetic"
    folder.mkdir(parents=True)
    cv2.imwrite(
        str(folder / "01_spread_frame.jpg"),
        np.zeros((240, 320, 3), np.uint8),
    )
    (folder / "video.mp4").touch()
    output = root / "rectification" / "morph"
    run_batch(root, output, workers=1)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["complete"]
    assert summary["skipped"] == 2
    assert summary["exported"] == 0
    with pytest.raises(ValueError):
        run_batch(root, tmp_path / "public", workers=1)


def test_unreadable_video_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Cannot open"):
        video_alignment(tmp_path / "missing.mp4", [])
