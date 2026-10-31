"""Synthetic temporal statistics and ellipse geometry tests."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from maus_export.plate_normalization import (
    OUTPUT_RADIUS,
    aggregate_estimates,
    ellipse_covariance,
    fit_plate,
    opening_statistics,
    temporal_statistics,
)


def test_opening_video_statistics(tmp_path: Path) -> None:
    source = tmp_path / "synthetic.avi"
    writer = cv2.VideoWriter(
        str(source), cv2.VideoWriter_fourcc(*"MJPG"), 12, (160, 120)
    )
    assert writer.isOpened()
    for _ in range(72):
        writer.write(np.full((120, 160, 3), 80, np.uint8))
    writer.release()
    stats = opening_statistics(source)
    assert len(stats["sample_frames"]) == 24
    assert stats["source_size"] == [160, 120]
    assert stats["variance"].max() == 0
    with pytest.raises(ValueError):
        opening_statistics(tmp_path / "missing.avi")


def test_temporal_statistics_detect_moving_objects() -> None:
    frames = np.full((24, 120, 160, 3), 80, np.uint8)
    for index, frame in enumerate(frames):
        frame[20:40, index * 5 : index * 5 + 10] = 200
    stats = temporal_statistics(frames)
    assert stats["variance"][0, 0] == 0
    assert stats["variance"][25, 25] > 0
    assert stats["median"][25, 25, 0] == 80
    assert stats["mean"][25, 25, 0] > 80
    with pytest.raises(ValueError):
        temporal_statistics(frames[:2])


def test_plate_boundary_and_variance_rejection() -> None:
    image = np.zeros((380, 640, 3), np.uint8)
    ellipse = ((320, 190), (300, 260), 15)
    cv2.ellipse(image, ellipse, (180, 180, 180), -1)
    fit = fit_plate(image, np.zeros((380, 640), np.float32))
    np.testing.assert_allclose(fit["center"], [320, 190], atol=2)
    assert fit["support"] > 0.9
    with pytest.raises(ValueError, match="No supported"):
        fit_plate(image, np.full((380, 640), 10000, np.float32))
    with pytest.raises(ValueError):
        fit_plate(np.zeros_like(image), np.zeros((380, 640)))


def estimate(center: tuple = (320, 190)) -> dict:
    _, covariance = ellipse_covariance(((320, 190), (300, 260), 15))
    return {
        "source_size": [640, 380],
        "center": list(center),
        "covariance": covariance.tolist(),
    }


def test_robust_consensus_rejects_outlier_and_maps_rim_to_circle() -> None:
    estimates = [estimate(), estimate((321, 190)), estimate((319, 190))]
    estimates.append(estimate((500, 300)))
    transform = aggregate_estimates(estimates)
    assert transform["inlier_videos"] == 3
    assert not transform["approved_for_export"]
    matrix = np.array(transform["matrix"])
    covariance = np.array(transform["covariance"])
    result = matrix[:2, :2] @ covariance @ matrix[:2, :2].T
    np.testing.assert_allclose(
        result, np.eye(2) * (OUTPUT_RADIUS / 2) ** 2, atol=1e-8
    )
    np.testing.assert_allclose(matrix @ [320, 190, 1], [256, 256, 1])


def test_unstable_or_insufficient_groups_fail_closed() -> None:
    with pytest.raises(ValueError):
        aggregate_estimates([estimate()])
    with pytest.raises(ValueError, match="majority"):
        aggregate_estimates(
            [estimate((100, 100)), estimate(), estimate((500, 300))]
        )
    estimates = [estimate(), estimate(), estimate()]
    estimates[0]["source_size"] = [1280, 760]
    with pytest.raises(ValueError, match="Mixed"):
        aggregate_estimates(estimates)
