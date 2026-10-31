"""Experimental photo-to-video transfer, not camera calibration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .phase_pilot import _hash_file
from .rectification import (
    FEATURES,
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
from .rectification_batch import consistent_transforms
from .robot_correction import (
    MAX_MEDIAN_PX,
    MIN_HELD_OUT_COVERAGE,
    MIN_HELD_OUT_INLIER_SHARE,
    estimate_alignment_for_robot,
)

SAMPLES = 5
MAX_REFERENCES = 6
REGISTRATION_INLIER_PX = 5.0
HOMOGENEOUS_ROW = [0.0, 0.0, 1.0]


def register_video_frame(source: np.ndarray, photo: np.ndarray) -> dict:
    """Measure a similarity map to an unpainted photo using held-out matches."""
    sift = cv2.SIFT_create(nfeatures=FEATURES)
    keys_a, desc_a = sift.detectAndCompute(
        cv2.cvtColor(source, cv2.COLOR_BGR2GRAY), None
    )
    keys_b, desc_b = sift.detectAndCompute(
        cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY), None
    )
    if desc_a is None or desc_b is None:
        raise ValueError("Not enough photo/video texture")
    pairs = cv2.BFMatcher().knnMatch(desc_a, desc_b, k=2)
    unique = {}
    good = [
        m
        for pair in pairs
        if len(pair) == 2
        for m, n in [pair]
        if m.distance < RATIO * n.distance
    ]
    for match in sorted(good, key=lambda m: m.distance):
        unique.setdefault(match.trainIdx, match)
    matches = list(unique.values())
    if len(matches) < MIN_MATCHES:
        raise ValueError("Not enough independent photo/video matches")
    a = np.array([keys_a[m.queryIdx].pt for m in matches], np.float32)
    b = np.array([keys_b[m.trainIdx].pt for m in matches], np.float32)
    order = np.random.default_rng(SEED).permutation(len(a))
    cut = int(len(order) * TRAIN_SHARE)
    train, test = order[:cut], order[cut:]
    cv2.setRNGSeed(SEED)
    matrix, _ = cv2.estimateAffinePartial2D(
        a[train],
        b[train],
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_PX,
    )
    if matrix is None or not np.isfinite(matrix).all():
        raise ValueError("Cannot register video to photo")
    full = np.vstack((matrix, HOMOGENEOUS_ROW))
    prediction = cv2.perspectiveTransform(a[test, None], full)[:, 0]
    errors = np.linalg.norm(prediction - b[test], axis=1)
    inliers = errors <= REGISTRATION_INLIER_PX
    coverage = (
        min(
            _coverage(a[test][inliers], image_size(source)),
            _coverage(b[test][inliers], image_size(photo)),
        )
        if inliers.sum() >= len(HOMOGENEOUS_ROW)
        else 0.0
    )
    report = {
        "matches": len(matches),
        "median_error_px": float(np.median(errors)),
        "p95_error_px": float(np.percentile(errors, 95)),
        "held_out_inlier_share": float(inliers.mean()),
        "held_out_coverage": float(coverage),
    }
    if (
        not np.isfinite(errors).all()
        or report["median_error_px"] > MAX_MEDIAN_PX
        or report["held_out_inlier_share"] < MIN_HELD_OUT_INLIER_SHARE
        or coverage < MIN_HELD_OUT_COVERAGE
    ):
        raise ValueError("Photo/video registration rejected")
    return {
        "schema_version": 1,
        "matrix": full.tolist(),
        "source_size": list(image_size(source)),
        "target_size": list(image_size(photo)),
        "validation": report,
    }


def experimental_alignment(
    frames: list[tuple[int, Path]], references: list[Path]
) -> dict[str, Any]:
    """Transfer a photo fit only after two consistent video registrations."""
    if not frames:
        raise ValueError("No extracted video frames")
    indices = sorted(set(np.linspace(0, len(frames) - 1, SAMPLES).astype(int)))
    samples = [(frames[i][0], read_image(frames[i][1])) for i in indices]
    for reference in references[:MAX_REFERENCES]:
        photo_path = reference.with_name(
            reference.name.replace("_overlay", "_frame")
        )
        if not photo_path.is_file() or photo_path.is_symlink():
            continue
        reference_hash, photo_hash = (
            _hash_file(reference),
            _hash_file(photo_path),
        )
        try:
            photo = read_image(photo_path)
            alignment = estimate_alignment_for_robot(
                photo, read_image(reference)
            )
            rectify(photo, alignment)
        except (ValueError, cv2.error):
            continue
        registrations = []
        for index, sample in samples:
            try:
                registration = register_video_frame(sample, photo)
            except (ValueError, cv2.error):
                continue
            for old_index, old in registrations:
                if old_index == index:
                    continue
                if not consistent_transforms(old, registration):
                    continue
                matrix = np.asarray(alignment["matrix"]) @ np.asarray(
                    old["matrix"]
                )
                result = {
                    **alignment,
                    "kind": "experimental_photo_video_transfer",
                    "matrix": matrix.tolist(),
                    "source_size": old["source_size"],
                    "video_validation_frames": [old_index, index],
                    "reference_path": str(reference),
                    "video_reference": reference.name,
                    "reference_sha256": reference_hash,
                    "reference_photo_sha256": photo_hash,
                    "photo_to_overlay": alignment,
                    "video_to_photo": [old, registration],
                    "warning": "Experimental transfer; visual review required",
                }
                rectify(sample, result)
                if (
                    _hash_file(reference) != reference_hash
                    or _hash_file(photo_path) != photo_hash
                ):
                    raise ValueError("Reference changed during fitting")
                return result
            registrations.append((index, registration))
    raise ValueError("No experimentally consistent photo/video transfer")
