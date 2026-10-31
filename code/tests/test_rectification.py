"""Offline geometry checks using synthetic points, images and video."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from maus_export.rectification import (
    estimate_alignment,
    fit_points,
    private_output,
    rectify,
    write_video,
)

SIZE = (320, 240)


def synthetic_points() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(17)
    points = rng.uniform([10, 10], [280, 200], (200, 2)).astype(np.float32)
    return points, points + np.array([7, 5], dtype=np.float32)


def transform() -> dict:
    source, target = synthetic_points()
    return fit_points(source, target, SIZE, SIZE)


def test_known_transform_and_held_out_validation() -> None:
    fitted = transform()
    np.testing.assert_allclose(
        fitted["matrix"], [[1, 0, 7], [0, 1, 5], [0, 0, 1]], atol=1e-5
    )
    assert fitted["validation"]["held_out"] == 60
    assert fitted["validation"]["p95_error_px"] < 1e-4
    assert fitted["lens_distortion_corrected"] is False


def test_outliers_are_rejected_not_hidden_in_report() -> None:
    source, target = synthetic_points()
    target[::3] = np.random.default_rng(7).uniform([0, 0], SIZE, (67, 2))
    with pytest.raises(ValueError, match="rejected"):
        fit_points(source, target, SIZE, SIZE)


def test_clustered_and_degenerate_points_are_rejected() -> None:
    source, target = synthetic_points()
    with pytest.raises(ValueError, match="rejected"):
        fit_points(source / 10, target / 10, SIZE, SIZE)
    with pytest.raises(ValueError):
        fit_points(np.ones((50, 2)), np.ones((50, 2)), SIZE, SIZE)


@pytest.mark.parametrize("count", [0, 3, 39])
def test_too_few_points(count: int) -> None:
    with pytest.raises(ValueError, match="paired 2D"):
        fit_points(np.zeros((count, 2)), np.zeros((count, 2)), SIZE, SIZE)


def test_nonfinite_points() -> None:
    source, target = synthetic_points()
    source[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fit_points(source, target, SIZE, SIZE)


def test_warp_preserves_raw_pixels_and_reports_invalid_border() -> None:
    image = np.full((240, 320, 3), 127, np.uint8)
    output, valid = rectify(image, transform())
    assert output.shape == image.shape
    assert valid[0, 0] == 0
    assert valid[100, 100] == 255
    np.testing.assert_array_equal(output[100, 100], image[100, 100])


def test_horizon_crossing_is_rejected() -> None:
    fitted = transform()
    fitted["matrix"] = [[1, 0, 0], [0, 1, 0], [-0.01, 0, 1]]
    with pytest.raises(ValueError, match="horizon"):
        rectify(np.zeros((240, 320, 3), np.uint8), fitted)


def test_odd_video_dimensions_are_not_silently_cropped(tmp_path: Path) -> None:
    fitted = transform()
    fitted["target_size"] = [320, 239]
    with pytest.raises(ValueError, match="even target"):
        write_video(tmp_path / "source.mp4", tmp_path / "out.mp4", fitted)


def test_resolution_mismatch_is_not_silently_resized() -> None:
    with pytest.raises(ValueError, match="size mismatch"):
        rectify(np.zeros((120, 160, 3), np.uint8), transform())


def test_featureless_images_fail_safely() -> None:
    blank = np.zeros((240, 320, 3), np.uint8)
    with pytest.raises(ValueError, match="texture"):
        estimate_alignment(blank, blank)


def test_synthetic_feature_matching() -> None:
    rng = np.random.default_rng(19)
    gray = rng.integers(20, 230, (480, 640), dtype=np.uint8)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    raw = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    matrix = np.array([[1, 0.02, 5], [0.01, 1, 3], [0.00001, 0, 1.0]])
    overlay = cv2.warpPerspective(raw, matrix, (640, 480))
    cv2.putText(
        overlay,
        "annotation",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )
    fitted = estimate_alignment(raw, overlay)
    assert fitted["validation"]["median_error_px"] < 1


def test_private_output_guard(tmp_path: Path) -> None:
    root = tmp_path / "data"
    assert private_output(root / "rectified" / "x.png", root).parent.exists()
    for destination in (
        tmp_path / "public.png",
        root / "raw" / "x.png",
        root / ".." / "escape.png",
    ):
        with pytest.raises(ValueError):
            private_output(destination, root)
    existing = root / "existing.png"
    existing.touch()
    with pytest.raises(ValueError, match="overwrite"):
        private_output(existing, root)
    link = root / "link"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        private_output(link / "escape.png", root)


def test_video_stream_and_mismatched_resolution(tmp_path: Path) -> None:
    source = tmp_path / "source.avi"
    writer = cv2.VideoWriter(
        str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, SIZE
    )
    assert writer.isOpened()
    for _ in range(4):
        writer.write(np.full((240, 320, 3), 127, np.uint8))
    writer.release()
    destination = tmp_path / "out.mp4"
    assert write_video(source, destination, transform()) == 4
    capture = cv2.VideoCapture(str(destination))
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 4
    capture.release()
    fitted = transform()
    fitted["source_size"] = [640, 480]
    failed = tmp_path / "failed.mp4"
    with pytest.raises(ValueError, match="size mismatch"):
        write_video(source, failed, fitted)
    assert not failed.exists()
