"""Prepare immutable rectified labeling frames without re-encoding video."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import cv2

from .phase_pilot import MIN_FREE_BYTES, _hash_file
from .rectification import read_image, rectify
from .rectification_batch import video_alignment

PREPROCESSING_VERSION = "upstream-rectified-v1"
COORDINATE_SPACE = "rectified_upstream"
PILOT_SCHEMA_VERSION = 1


def _checked_source(root: Path, relative: str, expected_hash: str) -> Path:
    path = root / relative
    if (
        Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or path.absolute() != path.resolve(strict=True)
        or not path.is_file()
        or _hash_file(path) != expected_hash
    ):
        raise ValueError("Unsafe or changed source artifact")
    return path


def _write_json(path: Path, content: dict[str, Any]) -> None:
    path.write_text(json.dumps(content, indent=2, allow_nan=False))


def _rectify_run(
    data_dir: Path, pilot: Path, output: Path, run: dict[str, Any]
) -> dict[str, Any]:
    original_frames = run["frames"]
    if not original_frames:
        raise ValueError("No extracted frames for this run")
    video = _checked_source(data_dir, run["video_path"], run["video_sha256"])
    raw = _checked_source(data_dir, run["raw_path"], run["raw_sha256"])
    media = video.parent
    references = []
    for path in sorted(media.glob("*_overlay.jpg")):
        if path.absolute() != path.resolve():
            continue
        image = read_image(path)
        if image.shape[0] == image.shape[1]:
            references.append(path)
    references.sort(key=lambda p: ("spread" not in p.name, p.name))
    reference_hashes = {p.name: _hash_file(p) for p in references}
    if not references:
        raise ValueError("No square upstream reference for this run")
    transform = video_alignment(video, references)
    reference = media / transform["video_reference"]
    reference_hash = reference_hashes[reference.name]
    source_run = json.loads(raw.read_text())
    transform.update(
        {
            "video_sha256": run["video_sha256"],
            "reference_sha256": reference_hash,
            "machine_id": source_run.get("machineId"),
            "preprocessing_version": PREPROCESSING_VERSION,
        }
    )
    directory = Path(original_frames[0]["image"]).parent
    if len(directory.parts) != 1 or directory.name in {"", ".", ".."}:
        raise ValueError("Unsafe source image directory")
    transform_relative = Path("transforms") / f"{directory.name}.json"
    transform_path = output / transform_relative
    transform_path.parent.mkdir(mode=0o700, exist_ok=True)
    _write_json(transform_path, transform)
    transform_hash = _hash_file(transform_path)
    generated = []
    for frame in original_frames:
        relative = Path(frame["image"])
        if relative.parent != directory or relative.suffix != ".png":
            raise ValueError("Inconsistent source image directory")
        source = _checked_source(pilot, frame["image"], frame["image_sha256"])
        if frame["human_label"] is not None or frame["phase_suggestion"]:
            raise ValueError("Only unannotated source pilots are supported")
        destination = output / relative
        valid_relative = Path("valid") / relative
        valid_path = output / valid_relative
        if not destination.exists():
            pixels, valid = rectify(read_image(source), transform)
            for path, image in ((destination, pixels), (valid_path, valid)):
                path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                if not cv2.imwrite(str(path), image):
                    raise OSError("Cannot write rectified labeling artifact")
        _checked_source(pilot, frame["image"], frame["image_sha256"])
        generated.append(
            {
                **frame,
                "source_image_sha256": frame["image_sha256"],
                "image_sha256": _hash_file(destination),
                "coordinate_space": COORDINATE_SPACE,
                "preprocessing_version": PREPROCESSING_VERSION,
                "transform": transform_relative.as_posix(),
                "transform_sha256": transform_hash,
                "valid_mask": valid_relative.as_posix(),
                "valid_mask_sha256": _hash_file(valid_path),
                "image_size": transform["target_size"],
            }
        )
    _checked_source(data_dir, run["video_path"], run["video_sha256"])
    _checked_source(data_dir, run["raw_path"], run["raw_sha256"])
    _checked_source(media, reference.name, reference_hash)
    return {**run, "frames": generated}


def prepare_rectified(data_dir: Path, pilot: Path, output: Path) -> dict:
    """Publish a new pilot, recording rejected runs without raw fallback."""
    data_dir = data_dir.resolve(strict=True)
    pilot = pilot.absolute()
    output = output.absolute()
    parent = data_dir / "labeling"
    if (
        pilot.parent != parent
        or pilot != pilot.resolve(strict=True)
        or output.parent != parent
        or parent != parent.resolve(strict=True)
        or parent.stat().st_mode & 0o077
    ):
        raise ValueError("Use private direct children of data-dir/labeling")
    content = (pilot / "manifest.json").read_bytes()
    manifest = json.loads(content)
    if manifest["version"] != PILOT_SCHEMA_VERSION or manifest.get(
        "preprocessing"
    ):
        raise ValueError("Expected an original unrectified pilot")
    if shutil.disk_usage(parent).free < MIN_FREE_BYTES:
        raise ValueError("Insufficient free disk space")
    os.umask(0o077)
    output.mkdir(mode=0o700)
    summary = {"selected": len(manifest["runs"]), "ok": 0, "rejected": 0}
    try:
        result = copy.deepcopy(manifest)
        result["preprocessing"] = {
            "version": PREPROCESSING_VERSION,
            "coordinate_space": COORDINATE_SPACE,
            "source_manifest_sha256": hashlib.sha256(content).hexdigest(),
            "lens_distortion_corrected": False,
            "visual_review": "required",
        }
        runs = []
        for run in manifest["runs"]:
            candidate = {**run, "frames": []}
            if run["status"] == "ok":
                staging = output / ".pending"
                staging.mkdir(mode=0o700)
                try:
                    candidate = _rectify_run(data_dir, pilot, staging, run)
                    for artifact in staging.rglob("*"):
                        if artifact.is_file():
                            destination = output / artifact.relative_to(staging)
                            destination.parent.mkdir(
                                mode=0o700, parents=True, exist_ok=True
                            )
                            artifact.replace(destination)
                    summary["ok"] += 1
                except (ValueError, cv2.error) as error:
                    candidate["status"] = "alignment_rejected"
                    candidate["rectification_rejection"] = str(error)
                    summary["rejected"] += 1
                finally:
                    shutil.rmtree(staging)
            else:
                summary["rejected"] += 1
            runs.append(candidate)
        result["runs"] = runs
        for run in runs:
            for frame in run["frames"]:
                frame["source_manifest_sha256"] = result["preprocessing"][
                    "source_manifest_sha256"
                ]
        summary["frames"] = sum(len(r["frames"]) for r in runs)
        result["rectification_summary"] = summary
        if (pilot / "manifest.json").read_bytes() != content:
            raise ValueError("Source manifest changed during preparation")
        # The final manifest is the completion marker, written only at the end.
        _write_json(output / "manifest.json", result)
    except BaseException:
        shutil.rmtree(output)
        raise
    return summary


def main() -> int:
    """Prepare a shared rectified pilot and print aggregate counts only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    try:
        summary = prepare_rectified(args.data_dir, args.pilot, args.output)
    except (OSError, ValueError, KeyError, TypeError):
        print("Rectified preparation failed; inspect private source artifacts.")
        return 1
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["frames"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
