"""Opening-window statistics and experimental ellipse normalization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .rectification import private_output

SAMPLES = 24
WINDOW_SECONDS = 12.0
MIN_SAMPLES = 12
MIN_SPAN_SECONDS = 5.0
ANALYSIS_WIDTH = 640
CANNY_LOW = 40
CANNY_HIGH = 100
MAX_VARIANCE = 400.0
MIN_CONTOUR = 30
MAX_CONTOURS = 80
ARC_CONTOURS = 20
ARC_LENGTHS = (60, 120, 240)
ARC_STARTS = 12
PERIMETER_SAMPLES = 360
SECTORS = 12
MIN_SECTORS = 8
MIN_SUPPORT = 0.60
EDGE_TOLERANCE = 3.0
MIN_VIDEOS = 3
MAX_CENTER_DRIFT = 0.025
MAX_SHAPE_DRIFT = 0.08
VIDEOS_PER_GROUP = 8
OUTPUT_SIZE = 512
OUTPUT_RADIUS = 224.0


def temporal_statistics(frames: np.ndarray) -> dict[str, np.ndarray]:
    """Compute intensity statistics, not geometric plate moments."""
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("Expected N x H x W x 3 frames")
    if len(frames) < MIN_SAMPLES:
        raise ValueError("Too few temporal samples")
    values = frames.astype(np.float32)
    gray = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames])
    return {
        "mean": values.mean(axis=0),
        "median": np.median(values, axis=0).astype(np.uint8),
        "variance": gray.astype(np.float32).var(axis=0),
    }


def opening_statistics(source: Path) -> dict[str, Any]:
    """Sample an opening time interval with bounded memory."""
    capture = cv2.VideoCapture(str(source))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not capture.isOpened() or fps <= 0 or count <= 0 or width <= 0:
            raise ValueError("Invalid or unreadable video metadata")
        last = min(count - 1, int(WINDOW_SECONDS * fps))
        if last / fps < MIN_SPAN_SECONDS:
            raise ValueError("Opening window is too short")
        scaled_width = min(width, ANALYSIS_WIDTH)
        scaled_height = round(height * scaled_width / width)
        frames = []
        indices = []
        for index in np.unique(np.linspace(0, last, SAMPLES, dtype=int)):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if ok:
                frames.append(cv2.resize(frame, (scaled_width, scaled_height)))
                indices.append(int(index))
        if (
            len(indices) < MIN_SAMPLES
            or (indices[-1] - indices[0]) / fps < MIN_SPAN_SECONDS
        ):
            raise ValueError("Insufficient decodable temporal coverage")
        result = temporal_statistics(np.stack(frames))
        result["source_size"] = [width, height]
        result["sample_frames"] = indices
        return result
    finally:
        capture.release()


def ellipse_covariance(ellipse: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Return centre and geometric covariance of a filled ellipse."""
    center, axes, angle = ellipse
    radians = np.deg2rad(angle)
    rotation = np.array(
        [
            [np.cos(radians), -np.sin(radians)],
            [np.sin(radians), np.cos(radians)],
        ]
    )
    covariance = rotation @ np.diag((np.asarray(axes) / 4) ** 2) @ rotation.T
    return np.asarray(center), covariance


def fit_plate(median: np.ndarray, variance: np.ndarray) -> dict[str, Any]:
    """Fit a central plate-rim candidate with stable perimeter support.

    Central-position and size priors are explicit experimental assumptions.
    A supported ellipse can still be a fixture or an inner ring; review it.
    """
    height, width = median.shape[:2]
    gray = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    stable = variance <= MAX_VARIANCE
    edges[~stable] = 0
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    distance = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
    theta = np.linspace(0, 2 * np.pi, PERIMETER_SAMPLES, endpoint=False)
    unit = np.column_stack([np.cos(theta), np.sin(theta)])
    candidates = []
    ordered = sorted(contours, key=len, reverse=True)[:MAX_CONTOURS]
    arcs = list(ordered)
    # Plate edges can join fixture edges; fit local arcs as well as contours.
    for contour in ordered[:ARC_CONTOURS]:
        for length in ARC_LENGTHS:
            if len(contour) <= length:
                continue
            for start in np.unique(
                np.linspace(0, len(contour) - length, ARC_STARTS, dtype=int)
            ):
                arcs.append(contour[start : start + length])
    for contour in arcs:
        if len(contour) < MIN_CONTOUR:
            continue
        ellipse = cv2.fitEllipse(contour)
        center, axes, angle = ellipse
        if not (
            0.3 * width < center[0] < 0.7 * width
            and 0.25 * height < center[1] < 0.75 * height
            and 0.3 * width < max(axes) < 0.95 * width
            and 0.35 * height < min(axes) < 1.1 * height
            and max(axes) / min(axes) < 2.5
        ):
            continue
        radians = np.deg2rad(angle)
        rotation = np.array(
            [
                [np.cos(radians), -np.sin(radians)],
                [np.sin(radians), np.cos(radians)],
            ]
        )
        points = (unit * (np.asarray(axes) / 2)) @ rotation.T + center
        pixels = np.rint(points).astype(int)
        inside = (
            (pixels[:, 0] >= 0)
            & (pixels[:, 0] < width)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < height)
        )
        supported = np.zeros(PERIMETER_SAMPLES, dtype=bool)
        x, y = pixels[inside].T
        supported[inside] = (distance[y, x] <= EDGE_TOLERANCE) & stable[y, x]
        support = float(supported.mean())
        sectors = sum(
            part.mean() >= MIN_SUPPORT
            for part in np.array_split(supported, SECTORS)
        )
        if support >= MIN_SUPPORT and sectors >= MIN_SECTORS:
            candidates.append((support, ellipse, int(sectors)))
    if not candidates:
        raise ValueError("No supported stable central plate ellipse")
    support, ellipse, sectors = max(candidates, key=lambda c: c[0])
    center, covariance = ellipse_covariance(ellipse)
    return {
        "center": center.tolist(),
        "covariance": covariance.tolist(),
        "ellipse": ellipse,
        "support": support,
        "supported_sectors": sectors,
        "warning": "Candidate rim requires visual review; not calibration",
    }


def aggregate_estimates(estimates: list[dict]) -> dict[str, Any]:
    """Require a cross-video majority with bounded geometric dispersion."""
    if len(estimates) < MIN_VIDEOS:
        raise ValueError("Need at least three independent video estimates")
    sizes = {tuple(e["source_size"]) for e in estimates}
    if len(sizes) != 1:
        raise ValueError("Mixed source dimensions")
    size = np.asarray(next(iter(sizes)), dtype=float)
    centers = np.array([e["center"] for e in estimates])
    covariances = np.array([e["covariance"] for e in estimates])
    center = np.median(centers, axis=0)
    covariance = np.median(covariances, axis=0)
    center_drift = np.linalg.norm((centers - center) / size, axis=1)
    shape_drift = np.linalg.norm(
        covariances - covariance, axis=(1, 2)
    ) / np.linalg.norm(covariance)
    accepted = (center_drift <= MAX_CENTER_DRIFT) & (
        shape_drift <= MAX_SHAPE_DRIFT
    )
    if accepted.sum() < max(MIN_VIDEOS, len(estimates) // 2 + 1):
        raise ValueError("No stable cross-video majority")
    center = np.median(centers[accepted], axis=0)
    covariance = np.median(covariances[accepted], axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if eigenvalues.min() <= 0:
        raise ValueError("Degenerate geometric moments")
    linear = (
        eigenvectors
        @ np.diag(OUTPUT_RADIUS / (2 * np.sqrt(eigenvalues)))
        @ eigenvectors.T
    )
    matrix = np.eye(3)
    matrix[:2, :2] = linear
    matrix[:2, 2] = OUTPUT_SIZE / 2 - linear @ center
    return {
        "schema_version": 1,
        "kind": "experimental_affine_ellipse_normalization",
        "approved_for_export": False,
        "lens_distortion_corrected": False,
        "source_size": size.astype(int).tolist(),
        "target_size": [OUTPUT_SIZE, OUTPUT_SIZE],
        "matrix": matrix.tolist(),
        "center": center.tolist(),
        "covariance": covariance.tolist(),
        "inlier_videos": int(accepted.sum()),
        "candidate_videos": len(estimates),
        "inlier_indices": np.flatnonzero(accepted).tolist(),
        "warning": "Affine approximation only; review rim and camera stability",
    }


def audit(data_dir: Path, output: Path) -> dict[str, Any]:
    """Audit grouped local videos; save private statistics and previews."""
    private_output(output / "summary.json", data_dir)
    cv2.setNumThreads(1)
    groups: dict[tuple, list[Path]] = {}
    for path in sorted((data_dir / "raw" / "runs").glob("*.json")):
        run = json.loads(path.read_text())
        video = data_dir / "raw" / "media" / path.stem / "video.mp4"
        if not video.exists() or run.get("machineId") is None:
            continue
        capture = cv2.VideoCapture(str(video))
        size = tuple(
            int(capture.get(key))
            for key in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)
        )
        capture.release()
        if min(size) > 0:
            groups.setdefault((run["machineId"], *size), []).append(video)
    summary = {"groups": len(groups), "accepted_groups": 0, "complete": False}
    for key, videos in groups.items():
        folder = output / ("robot-" + "-".join(map(str, key)))
        folder.mkdir(parents=True, exist_ok=True)
        indices = np.unique(
            np.linspace(
                0,
                len(videos) - 1,
                min(VIDEOS_PER_GROUP, len(videos)),
                dtype=int,
            )
        )
        estimates = []
        reports = []
        for index in indices:
            source = videos[index]
            report = {"source": str(source), "status": "rejected"}
            try:
                stats = opening_statistics(source)
                prefix = folder / source.parent.name
                np.savez_compressed(str(prefix) + "_statistics.npz", **stats)
                median = stats["median"]
                cv2.imwrite(str(prefix) + "_median.png", median)
                cv2.imwrite(
                    str(prefix) + "_mean.png",
                    np.clip(stats["mean"], 0, 255).astype(np.uint8),
                )
                cv2.imwrite(
                    str(prefix) + "_stddev.png",
                    np.clip(np.sqrt(stats["variance"]), 0, 255).astype(
                        np.uint8
                    ),
                )
                estimate = fit_plate(median, stats["variance"])
                diagnostic = median.copy()
                cv2.ellipse(diagnostic, estimate["ellipse"], (0, 255, 0), 2)
                cv2.imwrite(str(prefix) + "_candidate.png", diagnostic)
                scale = np.diag(
                    np.array(stats["source_size"])
                    / np.array([median.shape[1], median.shape[0]])
                )
                estimate["center"] = (
                    scale @ np.array(estimate["center"])
                ).tolist()
                estimate["covariance"] = (
                    scale @ np.array(estimate["covariance"]) @ scale.T
                ).tolist()
                estimate["source_size"] = stats["source_size"]
                estimate["source"] = str(source)
                estimates.append(estimate)
                report.update(status="candidate", estimate=estimate)
            except (ValueError, cv2.error) as error:
                report["reason"] = str(error)
            reports.append(report)
        result: dict[str, Any] = {"videos": reports}
        try:
            transform = aggregate_estimates(estimates)
            transform["camera_key"] = folder.name
            result["transform"] = transform
            summary["accepted_groups"] += 1
        except ValueError as error:
            result["reason"] = str(error)
        (folder / "report.json").write_text(json.dumps(result, indent=2))
        print(
            json.dumps(
                {
                    "group": folder.name,
                    "candidates": len(estimates),
                    "accepted": "transform" in result,
                }
            ),
            flush=True,
        )
    summary["complete"] = True
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    """Run a local video-statistics audit without changing original media."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.data_dir.resolve(), args.out.resolve())))


if __name__ == "__main__":
    main()
