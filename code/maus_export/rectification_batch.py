"""Resumable, fail-closed local rectification of images and silent videos."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .rectification import (
    estimate_alignment,
    private_output,
    read_image,
    rectify,
    write_video,
)

WORKERS = 2
VIDEO_SAMPLES = (0.1, 0.3, 0.5, 0.7, 0.9)
MAX_REFERENCES = 3
MAX_TRANSFER_PX = 3.0
GRID_POINTS = 5
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def consistent_transforms(first: dict, second: dict) -> bool:
    """Require agreement over the source field, not just feature clusters."""
    if (
        first["source_size"] != second["source_size"]
        or first["target_size"] != second["target_size"]
    ):
        return False
    width, height = first["source_size"]
    xs, ys = np.meshgrid(
        np.linspace(0, width - 1, GRID_POINTS),
        np.linspace(0, height - 1, GRID_POINTS),
    )
    points = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float32)
    projected = [
        cv2.perspectiveTransform(
            points[:, None, :], np.asarray(t["matrix"], dtype=np.float64)
        )[:, 0]
        for t in (first, second)
    ]
    distances = np.linalg.norm(projected[0] - projected[1], axis=1)
    return bool(
        np.isfinite(distances).all() and distances.max() <= MAX_TRANSFER_PX
    )


def video_alignment(source: Path, references: list[Path]) -> dict[str, Any]:
    """Require two temporal samples agreeing against the same reference."""
    capture = cv2.VideoCapture(str(source))
    try:
        if not capture.isOpened():
            raise ValueError("Cannot open video")
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if count <= 0:
            raise ValueError("Unknown or empty video frame count")
        frames = []
        for fraction in VIDEO_SAMPLES:
            index = int((count - 1) * fraction)
            if any(index == old for old, _ in frames):
                continue
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if ok:
                frames.append((index, frame))
        for reference in references[:MAX_REFERENCES]:
            overlay = read_image(reference)
            fits = []
            for index, frame in frames:
                try:
                    fitted = estimate_alignment(frame, overlay)
                    # Reject pathological projective domains before rendering.
                    rectify(frame, fitted)
                except ValueError:
                    continue
                for old_index, old in fits:
                    if consistent_transforms(old, fitted):
                        old["video_validation_frames"] = [old_index, index]
                        old["video_reference"] = reference.name
                        old["expected_frames"] = count
                        return old
                fits.append((index, fitted))
    finally:
        capture.release()
    raise ValueError("No two temporal samples passed consistent alignment")


def process_item(source: Path, destination: Path) -> dict[str, Any]:
    """Process one source, atomically publish its report, and resume safely."""
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / f"{source.name}.json"
    if report_path.exists():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        if previous.get("status") == "exported" and all(
            (destination / name).is_file() for name in previous["outputs"]
        ):
            return previous
    report: dict[str, Any] = {"source": str(source), "status": "skipped"}
    temporary: list[Path] = []
    try:
        if source.suffix.lower() == ".mp4":
            references = []
            for reference in sorted(source.parent.glob("*_overlay.jpg")):
                image = read_image(reference)
                if image.shape[0] == image.shape[1]:
                    references.append(reference)
            references.sort(key=lambda p: ("spread" not in p.name, p.name))
            if not references:
                raise ValueError("No square upstream reference for this run")
            fitted = video_alignment(source, references)
            output = destination / "video.mp4"
            staging = destination / "video.partial.mp4"
            temporary.append(staging)
            count = write_video(source, staging, fitted)
            if count != fitted["expected_frames"]:
                raise ValueError("Decoded count differs from source metadata")
            staging.replace(output)
            report["outputs"] = [output.name]
            report["frames"] = count
        else:
            reference = source.with_name(
                source.name.replace("_frame", "_overlay")
            )
            if not reference.exists():
                raise ValueError("Missing paired upstream overlay")
            overlay = read_image(reference)
            if overlay.shape[0] != overlay.shape[1]:
                raise ValueError("Reference is not a square rectified view")
            raw = read_image(source)
            fitted = estimate_alignment(raw, overlay)
            image, valid = rectify(raw, fitted)
            outputs = []
            for suffix, pixels in (("", image), ("_valid", valid)):
                output = destination / f"{source.stem}{suffix}.png"
                staging = output.with_name(output.stem + ".partial.png")
                temporary.append(staging)
                if not cv2.imwrite(str(staging), pixels):
                    raise ValueError("Cannot write output image")
                outputs.append((staging, output))
            for staging, output in outputs:
                staging.replace(output)
            report["outputs"] = [p.name for _, p in outputs]
        report["transform"] = fitted
        report["status"] = "exported"
    except (ValueError, cv2.error) as error:
        report["reason"] = str(error)
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)
    staging_report = report_path.with_suffix(".json.partial")
    staging_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    staging_report.replace(report_path)
    return report


def run_batch(data_dir: Path, output: Path, workers: int = WORKERS) -> None:
    """Attempt all raw images, then videos, without modifying originals."""
    if workers < 1:
        raise ValueError("Workers must be positive")
    root = data_dir.resolve()
    output = output.resolve()
    # Validate the output even on a resumed run without overwriting a file.
    private_output(output / ".output-check", root)
    source_dir = root / "raw" / "media"
    if not source_dir.is_dir():
        raise ValueError("Missing raw/media source directory")
    images = sorted(
        p
        for p in source_dir.glob("*/*_frame.*")
        if p.suffix.lower() in IMAGE_SUFFIXES
    )
    videos = sorted(source_dir.glob("*/*.mp4"))
    print(
        f"Queued {len(images)} raw images and {len(videos)} videos", flush=True
    )
    print(
        "Overlays are references only; no painted training images", flush=True
    )
    cv2.setNumThreads(1)
    counts = {"exported": 0, "skipped": 0}
    summary = {
        **counts,
        "processed": 0,
        "total": len(images) + len(videos),
        "complete": False,
    }

    def one(source: Path) -> dict[str, Any]:
        return process_item(source, output / source.parent.name)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for group in (images, videos):
            futures = [executor.submit(one, source) for source in group]
            for future in as_completed(futures):
                report = future.result()
                counts[report["status"]] += 1
                summary = {
                    **counts,
                    "processed": sum(counts.values()),
                    "total": len(images) + len(videos),
                    "complete": False,
                }
                staging = output / "summary.partial.json"
                staging.write_text(json.dumps(summary, indent=2))
                staging.replace(output / "summary.json")
                print(json.dumps(summary), flush=True)
    summary["complete"] = True
    (output / "summary.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    """Run the local batch export."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args()
    run_batch(args.data_dir, args.out, args.workers)


if __name__ == "__main__":
    main()
