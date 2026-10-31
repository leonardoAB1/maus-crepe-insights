from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from maus_export.robot_correction import (
    Candidate,
    aggregate_transforms,
    apply_correction_to_image,
    collect_candidates,
    image_inputs,
    output_path_for,
    transform_for_apply,
)


def estimate(matrix: np.ndarray, run_id: str = "1") -> dict:
    return {
        "run_id": run_id,
        "phase": "spread",
        "frame": f"{run_id}/frame.jpg",
        "overlay": f"{run_id}/overlay.jpg",
        "machine_id": 3,
        "machine_name": "maus-pixi",
        "source_size": [10, 8],
        "target_size": [12, 12],
        "matrix": matrix.tolist(),
        "validation": {"median_error_px": 0.1},
    }


def test_collect_candidates_for_robot_name(tmp_path: Path) -> None:
    data = tmp_path / "data"
    run_dir = data / "raw" / "runs"
    media = data / "raw" / "media" / "123"
    run_dir.mkdir(parents=True)
    media.mkdir(parents=True)
    (media / "00_spread_frame.jpg").write_bytes(b"x")
    (media / "00_spread_overlay.jpg").write_bytes(b"x")
    (run_dir / "123.json").write_text(
        json.dumps(
            {
                "machineId": 3,
                "machineName": "maus-pixi",
                "visionChecks": [
                    {
                        "phase": "spread",
                        "frame": "00_spread_frame.jpg",
                        "overlay": "00_spread_overlay.jpg",
                    },
                    {
                        "phase": "plate_loaded",
                        "frame": "01_plate_loaded_frame.jpg",
                        "overlay": "01_plate_loaded_overlay.jpg",
                    },
                ],
            }
        )
    )

    candidates = collect_candidates(data, "maus-pixi")

    assert candidates == [
        Candidate(
            run_id="123",
            phase="spread",
            frame=media / "00_spread_frame.jpg",
            overlay=media / "00_spread_overlay.jpg",
            machine_id=3,
            machine_name="maus-pixi",
        )
    ]


def test_collect_candidates_for_robot_id(tmp_path: Path) -> None:
    data = tmp_path / "data"
    run_dir = data / "raw" / "runs"
    media = data / "raw" / "media" / "123"
    run_dir.mkdir(parents=True)
    media.mkdir(parents=True)
    (media / "frame.jpg").write_bytes(b"x")
    (media / "overlay.jpg").write_bytes(b"x")
    (run_dir / "123.json").write_text(
        json.dumps(
            {
                "machineId": 3,
                "machineName": "maus-pixi",
                "visionChecks": [
                    {
                        "phase": "spread",
                        "frame": "frame.jpg",
                        "overlay": "overlay.jpg",
                    }
                ],
            }
        )
    )

    assert len(collect_candidates(data, "3")) == 1


def test_aggregate_transforms_reports_mean_and_stddev() -> None:
    first = np.array([[2.0, 0.0, 1.0], [0.0, 2.0, 2.0], [0.0, 0.0, 1.0]])
    second = np.array([[2.0, 0.0, 3.0], [0.0, 2.0, 4.0], [0.0, 0.0, 1.0]])

    report = aggregate_transforms(
        [estimate(first, "1"), estimate(second, "2")], "maus-pixi"
    )

    assert report["robot"] == "maus-pixi"
    assert report["machine_names"] == ["maus-pixi"]
    assert report["machine_ids"] == [3]
    assert report["accepted_estimates"] == 2
    np.testing.assert_allclose(
        report["matrix_mean"],
        [[2.0, 0.0, 2.0], [0.0, 2.0, 3.0], [0.0, 0.0, 1.0]],
    )
    assert np.array(report["matrix_stddev"]).shape == (3, 3)
    assert np.array(report["corner_stddev_px"]).shape == (4, 2)


def test_aggregate_rejects_mixed_sizes() -> None:
    item = estimate(np.eye(3))
    other = estimate(np.eye(3), "2")
    other["source_size"] = [11, 8]

    with pytest.raises(ValueError, match="mixed image dimensions"):
        aggregate_transforms([item, other], "maus-pixi")


def test_transform_for_apply_uses_matrix_mean() -> None:
    report = aggregate_transforms([estimate(np.eye(3))], "maus-pixi")

    transform = transform_for_apply(report)

    assert transform["matrix"] == report["matrix_mean"]
    assert transform["source_size"] == [10, 8]
    assert transform["target_size"] == [12, 12]


def test_image_inputs_and_output_paths(tmp_path: Path) -> None:
    source = tmp_path / "images"
    nested = source / "nested"
    nested.mkdir(parents=True)
    image = nested / "a.jpg"
    image.write_bytes(b"x")
    (nested / "note.txt").write_text("skip")

    root, images = image_inputs(source)
    out = output_path_for(image, root, tmp_path / "out")

    assert root == source
    assert images == [image]
    assert out == tmp_path / "out" / "nested" / "a.png"


def test_apply_correction_to_image(tmp_path: Path) -> None:
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    image[:, :] = (10, 20, 30)
    source = tmp_path / "source.png"
    cv2.imwrite(str(source), image)
    report = aggregate_transforms([estimate(np.eye(3))], "maus-pixi")

    corrected, valid = apply_correction_to_image(source, report)

    assert corrected.shape == (12, 12, 3)
    assert valid.shape == (12, 12)
    assert corrected[0, 0].tolist() == [10, 20, 30]
