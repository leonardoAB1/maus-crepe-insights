"""Per-robot image correction estimated from accepted homographies."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .rectification import (
    ANNOTATION_FLOOR,
    FEATURES,
    MASK_KERNEL,
    MIN_MATCHES,
    RANSAC_PX,
    RATIO,
    SEED,
    TRAIN_SHARE,
    _coverage,
    image_size,
    read_image,
    rectify,
)

SCHEMA_VERSION = 1
CHANNELS = 3
DEFAULT_PHASES = ("spread", "fold1", "fold2", "push_off")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_MEDIAN_PX = 3.0
MAX_INLIER_PX = 5.0
MIN_HELD_OUT_INLIER_SHARE = 0.60
MIN_TRAINING_INLIER_SHARE = 0.50
MIN_HELD_OUT_COVERAGE = 0.10


@dataclass(frozen=True)
class Candidate:
    """One raw image and its overlay from a robot run."""

    run_id: str
    phase: str
    frame: Path
    overlay: Path
    machine_id: int | None
    machine_name: str | None


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    """Normalize a homography so element-wise statistics are meaningful."""
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (CHANNELS, CHANNELS) or not np.isfinite(values).all():
        raise ValueError("Expected a finite 3x3 matrix")
    scale = values[2, 2]
    if abs(scale) < np.finfo(np.float64).eps:
        raise ValueError("Cannot normalize a matrix with zero projective scale")
    return values / scale


def robot_matches(run: dict[str, Any], robot: str) -> bool:
    """Match either a machine name or a numeric machine id."""
    name = run.get("machineName")
    machine_id = run.get("machineId")
    return robot == name or robot == str(machine_id)


def collect_candidates(
    data_dir: Path, robot: str, phases: tuple[str, ...] = DEFAULT_PHASES
) -> list[Candidate]:
    """Collect local raw/overlay pairs for one robot."""
    candidates: list[Candidate] = []
    phase_set = set(phases)
    for run_path in sorted((data_dir / "raw" / "runs").glob("*.json")):
        run = json.loads(run_path.read_text())
        if not robot_matches(run, robot):
            continue
        run_id = run_path.stem
        media_dir = data_dir / "raw" / "media" / run_id
        for check in run.get("visionChecks") or []:
            phase = check.get("phase")
            if phase not in phase_set:
                continue
            frame_name = check.get("frame")
            overlay_name = check.get("overlay")
            if not frame_name or not overlay_name:
                continue
            frame = media_dir / frame_name
            overlay = media_dir / overlay_name
            if frame.exists() and overlay.exists():
                candidates.append(
                    Candidate(
                        run_id=run_id,
                        phase=phase,
                        frame=frame,
                        overlay=overlay,
                        machine_id=run.get("machineId"),
                        machine_name=run.get("machineName"),
                    )
                )
    return candidates


def estimate_alignment_for_robot(
    raw: np.ndarray, overlay: np.ndarray
) -> dict[str, Any]:
    """Fit a homography for aggregation with robust warning metrics."""
    source_size, target_size = image_size(raw), image_size(overlay)
    hsv = cv2.cvtColor(overlay, cv2.COLOR_BGR2HSV)
    annotations = (
        (hsv[..., 1] > ANNOTATION_FLOOR) & (hsv[..., 2] > ANNOTATION_FLOOR)
    ).astype(np.uint8)
    mask = 255 - cv2.dilate(
        annotations * 255, np.ones((MASK_KERNEL, MASK_KERNEL), np.uint8)
    )
    sift = cv2.SIFT_create(nfeatures=FEATURES)
    keys_a, desc_a = sift.detectAndCompute(
        cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY), None
    )
    keys_b, desc_b = sift.detectAndCompute(
        cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY), mask
    )
    if desc_a is None or desc_b is None:
        raise ValueError("Not enough image texture for feature matching")
    pairs = cv2.BFMatcher().knnMatch(desc_a, desc_b, k=2)
    good = [
        m
        for pair in pairs
        if len(pair) == 2
        for m, n in [pair]
        if m.distance < RATIO * n.distance
    ]
    unique = {}
    for match in sorted(good, key=lambda m: m.distance):
        unique.setdefault(match.trainIdx, match)
    good = list(unique.values())
    source = np.array([keys_a[m.queryIdx].pt for m in good], dtype=np.float32)
    target = np.array([keys_b[m.trainIdx].pt for m in good], dtype=np.float32)
    if len(source) < MIN_MATCHES:
        raise ValueError("Need at least 40 matched features")
    order = np.random.default_rng(SEED).permutation(len(source))
    cut = int(len(order) * TRAIN_SHARE)
    train, test = order[:cut], order[cut:]
    cv2.setRNGSeed(SEED)
    matrix, inliers = cv2.findHomography(
        source[train], target[train], cv2.RANSAC, RANSAC_PX
    )
    if matrix is None or inliers is None or not np.isfinite(matrix).all():
        raise ValueError("Cannot estimate a finite homography")
    if np.linalg.matrix_rank(matrix) < CHANNELS:
        raise ValueError("Degenerate homography")
    predicted = cv2.perspectiveTransform(source[test, None, :], matrix)[:, 0]
    errors = np.linalg.norm(predicted - target[test], axis=1)
    accepted = errors <= MAX_INLIER_PX
    coverage = (
        min(
            _coverage(source[test][accepted], source_size),
            _coverage(target[test][accepted], target_size),
        )
        if accepted.sum() >= CHANNELS
        else 0.0
    )
    report = {
        "matches": len(source),
        "held_out": len(test),
        "training_inlier_share": float(inliers.mean()),
        "median_error_px": float(np.median(errors)),
        "p95_error_px": float(np.percentile(errors, 95)),
        "held_out_inlier_share": float(accepted.mean()),
        "held_out_coverage": float(coverage),
    }
    if (
        not np.isfinite(errors).all()
        or report["training_inlier_share"] < MIN_TRAINING_INLIER_SHARE
        or report["median_error_px"] > MAX_MEDIAN_PX
        or report["held_out_inlier_share"] < MIN_HELD_OUT_INLIER_SHARE
        or coverage < MIN_HELD_OUT_COVERAGE
    ):
        raise ValueError(f"Alignment rejected: {report}")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "experimental_robot_mean_alignment",
        "lens_distortion_corrected": False,
        "source_size": list(source_size),
        "target_size": list(target_size),
        "matrix": normalize_matrix(matrix).tolist(),
        "validation": report,
    }


def estimate_candidate(candidate: Candidate) -> dict[str, Any]:
    """Estimate one accepted homography for a raw/overlay pair."""
    raw = read_image(candidate.frame)
    overlay = read_image(candidate.overlay)
    result = estimate_alignment_for_robot(raw, overlay)
    result.update(
        {
            "run_id": candidate.run_id,
            "phase": candidate.phase,
            "frame": str(candidate.frame),
            "overlay": str(candidate.overlay),
            "machine_id": candidate.machine_id,
            "machine_name": candidate.machine_name,
        }
    )
    return result


def _corner_points(size: list[int]) -> np.ndarray:
    width, height = size
    return np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )


def aggregate_transforms(
    estimates: list[dict[str, Any]], robot: str
) -> dict[str, Any]:
    """Build a mean correction matrix with per-element standard deviation."""
    if not estimates:
        raise ValueError("No accepted correction estimates")
    source_sizes = {tuple(e["source_size"]) for e in estimates}
    target_sizes = {tuple(e["target_size"]) for e in estimates}
    if len(source_sizes) != 1 or len(target_sizes) != 1:
        raise ValueError("Accepted estimates have mixed image dimensions")
    matrices = np.array(
        [normalize_matrix(np.array(e["matrix"])) for e in estimates],
        dtype=np.float64,
    )
    mean = normalize_matrix(matrices.mean(axis=0))
    std = (
        matrices.std(axis=0, ddof=1)
        if len(matrices) > 1
        else np.zeros_like(mean)
    )
    corners = _corner_points(list(next(iter(source_sizes))))
    projected = np.array(
        [
            cv2.perspectiveTransform(corners[None], matrix)[0]
            for matrix in matrices
        ]
    )
    projected_std = (
        projected.std(axis=0, ddof=1)
        if len(matrices) > 1
        else np.zeros_like(projected[0])
    )
    displacement = np.linalg.norm(projected - projected.mean(axis=0), axis=2)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "mean_robot_image_correction",
        "robot": robot,
        "machine_names": sorted(
            {e["machine_name"] for e in estimates if e.get("machine_name")}
        ),
        "machine_ids": sorted(
            {
                e["machine_id"]
                for e in estimates
                if e.get("machine_id") is not None
            }
        ),
        "source_size": list(next(iter(source_sizes))),
        "target_size": list(next(iter(target_sizes))),
        "matrix_mean": mean.tolist(),
        "matrix_stddev": std.tolist(),
        "corner_stddev_px": projected_std.tolist(),
        "corner_displacement_mean_px": displacement.mean(axis=0).tolist(),
        "corner_displacement_max_px": displacement.max(axis=0).tolist(),
        "accepted_estimates": len(estimates),
        "estimates": estimates,
        "warning": (
            "Mean homography is experimental; inspect stddev for camera "
            "movement"
        ),
    }


def transform_for_apply(correction: dict[str, Any]) -> dict[str, Any]:
    """Convert a mean correction report to the rectification transform shape."""
    if correction.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported correction schema")
    matrix = correction.get("matrix_mean")
    if matrix is None:
        raise ValueError("Correction has no matrix_mean")
    return {
        "schema_version": SCHEMA_VERSION,
        "source_size": correction["source_size"],
        "target_size": correction["target_size"],
        "matrix": matrix,
    }


def apply_correction_to_image(
    image_path: Path, correction: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Read one image and apply the mean correction matrix."""
    image = read_image(image_path)
    return rectify(image, transform_for_apply(correction))


def output_path_for(source: Path, input_root: Path, output_root: Path) -> Path:
    """Map an input file to a PNG path below the output root."""
    relative = (
        source.relative_to(input_root) if input_root != source else source.name
    )
    if isinstance(relative, str):
        return output_root / Path(relative).with_suffix(".png")
    return output_root / relative.with_suffix(".png")


def image_inputs(path: Path) -> tuple[Path, list[Path]]:
    """Return an input root and sorted image files for a file or directory."""
    if path.is_file():
        return path, [path]
    if not path.is_dir():
        raise ValueError(f"Input does not exist: {path}")
    images = sorted(
        p
        for p in path.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError(f"No images found in {path}")
    return path, images
