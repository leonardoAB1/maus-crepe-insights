"""Experimental upstream-view alignment, not physical camera calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

FEATURES = 4000
RATIO = 0.7
ANNOTATION_FLOOR = 150
MASK_KERNEL = 7
MIN_MATCHES = 40
TRAIN_SHARE = 0.7
RANSAC_PX = 2.0
MAX_MEDIAN_PX = 1.5
MAX_P95_PX = 5.0
MIN_INLIER_SHARE = 0.8
MIN_COVERAGE = 0.15
SEED = 42
CHANNELS = 3
SCHEMA_VERSION = 1
VIDEO_ALIGNMENT = 2
REPO_ROOT = Path(__file__).resolve().parents[2]


def image_size(image: np.ndarray) -> tuple[int, int]:
    """Return width and height of a BGR image."""
    if image.ndim != CHANNELS or image.shape[2] != CHANNELS:
        raise ValueError("Expected a three-channel BGR image")
    return image.shape[1], image.shape[0]


def read_image(path: Path) -> np.ndarray:
    """Read an image, failing explicitly for missing or corrupt inputs."""
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def _coverage(points: np.ndarray, size: tuple[int, int]) -> float:
    return float(cv2.contourArea(cv2.convexHull(points))) / np.prod(size)


def fit_points(
    source: np.ndarray,
    target: np.ndarray,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> dict[str, Any]:
    """Fit a homography with held-out point checks; reject poor alignment.

    Point pairs must be independent correspondences in pixel coordinates.
    The validation subset is not used to estimate or refit the transform.
    """
    source = np.asarray(source, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if (
        source.shape != target.shape
        or source.ndim != 2
        or source.shape[1] != 2
        or len(source) < MIN_MATCHES
        or not np.isfinite(source).all()
        or not np.isfinite(target).all()
    ):
        raise ValueError("Need at least 40 finite paired 2D points")
    for points, size in ((source, source_size), (target, target_size)):
        if min(size) <= 0 or np.any(points < 0) or np.any(points >= size):
            raise ValueError("Points must lie within positive image dimensions")
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
    accepted = errors <= MAX_P95_PX
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
        or report["median_error_px"] > MAX_MEDIAN_PX
        or report["p95_error_px"] > MAX_P95_PX
        or report["held_out_inlier_share"] < MIN_INLIER_SHARE
        or coverage < MIN_COVERAGE
    ):
        raise ValueError(f"Alignment rejected: {report}")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "experimental_upstream_alignment",
        "lens_distortion_corrected": False,
        "source_size": list(source_size),
        "target_size": list(target_size),
        "matrix": matrix.tolist(),
        "validation": report,
    }


def estimate_alignment(raw: np.ndarray, overlay: np.ndarray) -> dict[str, Any]:
    """Match raw texture to an overlay, excluding saturated annotations.

    Saturated scene colours may also be excluded. Annotation masking is a
    heuristic, not a guarantee that every annotation pixel is removed.
    """
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
    # Multiple raw features must not share one overlay validation target.
    unique = {}
    for match in sorted(good, key=lambda m: m.distance):
        unique.setdefault(match.trainIdx, match)
    good = list(unique.values())
    source = np.array([keys_a[m.queryIdx].pt for m in good])
    target = np.array([keys_b[m.trainIdx].pt for m in good])
    return fit_points(source, target, source_size, target_size)


def rectify(
    image: np.ndarray, transform: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Warp raw pixels and return a validity mask; never rescale implicitly."""
    if transform.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported transform schema")
    if image_size(image) != tuple(transform["source_size"]):
        raise ValueError("Source size mismatch; estimate a separate transform")
    matrix = np.asarray(transform["matrix"], dtype=np.float64)
    size = tuple(transform["target_size"])
    if (
        matrix.shape != (CHANNELS, CHANNELS)
        or not np.isfinite(matrix).all()
        or np.linalg.matrix_rank(matrix) < CHANNELS
        or len(size) != 2
        or min(size) <= 0
    ):
        raise ValueError("Invalid transform matrix or target size")
    width, height = image_size(image)
    corners = np.array(
        [
            [0, 0, 1],
            [width - 1, 0, 1],
            [width - 1, height - 1, 1],
            [0, height - 1, 1],
        ],
        dtype=np.float64,
    )
    denominators = corners @ matrix[2]
    if not (np.all(denominators > 0) or np.all(denominators < 0)):
        raise ValueError("Projective horizon intersects the source image")
    output = cv2.warpPerspective(image, matrix, size)
    # Require a fully supported bilinear footprint, not just a nearest pixel.
    support = cv2.warpPerspective(
        np.ones(image.shape[:2], dtype=np.float32), matrix, size
    )
    valid = (support >= 1.0).astype(np.uint8) * 255
    return output, valid


def private_output(path: Path, data_dir: Path) -> Path:
    """Confine generated artifacts to a gitignored data/ directory."""
    root = data_dir.resolve()
    resolved = path.resolve()
    if root.name != "data" or not resolved.is_relative_to(root):
        raise ValueError("Outputs must be inside the selected data/ directory")
    if "raw" in resolved.relative_to(root).parts:
        raise ValueError("Do not write generated artifacts into raw/")
    if resolved.exists():
        raise ValueError(f"Refusing to overwrite: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def write_video(
    source: Path, destination: Path, transform: dict[str, Any]
) -> int:
    """Stream a silent rectified MP4, checking every frame's dimensions."""
    if any(side % VIDEO_ALIGNMENT for side in transform["target_size"]):
        raise ValueError("MP4 output requires even target dimensions")
    capture = cv2.VideoCapture(str(source))
    writer = None
    count = 0
    try:
        if not capture.isOpened():
            raise ValueError("Cannot open source video")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Invalid source frame rate")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            output, _ = rectify(frame, transform)
            if writer is None:
                writer = cv2.VideoWriter(
                    str(destination),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    image_size(output),
                )
                if not writer.isOpened():
                    raise ValueError("Cannot open video writer")
            writer.write(output)
            count += 1
        if not count:
            raise ValueError("Video has no decodable frames")
    except Exception:
        if writer is not None:
            writer.release()
            writer = None
        destination.unlink(missing_ok=True)
        raise
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    return count


def main() -> None:
    """Run the experimental local rectification commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit", help="estimate alignment from a check pair")
    fit.add_argument("--raw", type=Path, required=True)
    fit.add_argument("--overlay", type=Path, required=True)
    fit.add_argument("--camera-key", required=True)
    fit.add_argument("--out", type=Path, required=True)
    apply = sub.add_parser("apply", help="rectify raw images or silent video")
    apply.add_argument("--transform", type=Path, required=True)
    apply.add_argument("--input", type=Path, required=True)
    apply.add_argument("--out", type=Path, required=True)
    apply.add_argument("--camera-key", required=True)
    apply.add_argument("--video", action="store_true")
    args = parser.parse_args()
    out = private_output(args.out, args.data_dir)
    if args.command == "fit":
        transform = estimate_alignment(
            read_image(args.raw), read_image(args.overlay)
        )
        transform["camera_key"] = args.camera_key
        out.write_text(json.dumps(transform, indent=2), encoding="utf-8")
        print(json.dumps(transform["validation"], indent=2))
    else:
        transform = json.loads(args.transform.read_text(encoding="utf-8"))
        if transform.get("camera_key") != args.camera_key:
            raise ValueError("Camera configuration mismatch")
        if args.video:
            if out.suffix.lower() != ".mp4":
                raise ValueError("Video output must use .mp4")
            print(f"Frames written: {write_video(args.input, out, transform)}")
        else:
            if out.suffix.lower() != ".png":
                raise ValueError("Image output must use lossless .png")
            mask_path = private_output(
                out.with_name(out.stem + "_valid.png"), args.data_dir
            )
            image, valid = rectify(read_image(args.input), transform)
            if not cv2.imwrite(str(out), image):
                raise ValueError("Cannot write rectified image")
            if not cv2.imwrite(str(mask_path), valid):
                raise ValueError("Cannot write validity mask")


if __name__ == "__main__":
    main()
