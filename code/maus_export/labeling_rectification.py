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

from .labeling_experimental import experimental_alignment
from .phase_pilot import MIN_FREE_BYTES, _hash_file
from .rectification import read_image, rectify
from .rectification_batch import video_alignment

PREPROCESSING_VERSION = "upstream-rectified-v1"
EXPERIMENTAL_VERSION = "upstream-rectified-experimental-v1"
COORDINATE_SPACE = "rectified_upstream"
PILOT_SCHEMA_VERSION = 1
PRIVATE_UMASK = 0o077  # new files readable by the owner only


class SourceChangedError(ValueError):
    """A source artifact moved, changed or became unsafe after extraction."""


def _checked_source(root: Path, relative: str, expected_hash: str) -> Path:
    path = root / relative
    if (
        Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or path.absolute() != path.resolve(strict=True)
        or not path.is_file()
        or _hash_file(path) != expected_hash
    ):
        raise SourceChangedError("Unsafe or changed source artifact")
    return path


def _write_json(path: Path, content: dict[str, Any]) -> None:
    path.write_text(json.dumps(content, indent=2, allow_nan=False))


def _rectify_run(
    data_dir: Path,
    pilot: Path,
    output: Path,
    run: dict[str, Any],
    *,
    experimental: bool = False,
    bank: dict[Any, list[Path]] | None = None,
) -> dict[str, Any]:
    original_frames = run["frames"]
    if not original_frames:
        raise ValueError("No extracted frames for this run")
    video = _checked_source(data_dir, run["video_path"], run["video_sha256"])
    raw = _checked_source(data_dir, run["raw_path"], run["raw_sha256"])
    media = video.parent
    source_run = json.loads(raw.read_text())
    version = EXPERIMENTAL_VERSION if experimental else PREPROCESSING_VERSION
    references = []
    for path in sorted(media.glob("*_overlay.jpg")):
        if path.absolute() != path.resolve():
            continue
        image = read_image(path)
        if image.shape[0] == image.shape[1]:
            references.append(path)
    references.sort(key=lambda p: ("spread" not in p.name, p.name))
    if experimental and bank:
        references = list(
            dict.fromkeys(
                references + bank.get(source_run.get("machineId"), [])
            )
        )
    reference_hashes = {p: _hash_file(p) for p in references}
    if not references:
        raise ValueError("No square upstream reference for this run")
    if experimental:
        samples = [
            (
                f["decoded_frame_index"],
                _checked_source(pilot, f["image"], f["image_sha256"]),
            )
            for f in original_frames
        ]
        transform = experimental_alignment(samples, references)
        reference = Path(transform["reference_path"])
    else:
        transform = video_alignment(video, references)
        reference = media / transform["video_reference"]
    reference_hash = reference_hashes[reference]
    transform.update(
        {
            "video_sha256": run["video_sha256"],
            "reference_sha256": reference_hash,
            "machine_id": source_run.get("machineId"),
            "preprocessing_version": version,
            "reference_path": str(reference.relative_to(data_dir)),
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
                "preprocessing_version": version,
                "transform": transform_relative.as_posix(),
                "transform_sha256": transform_hash,
                "valid_mask": valid_relative.as_posix(),
                "valid_mask_sha256": _hash_file(valid_path),
                "image_size": transform["target_size"],
            }
        )
    _checked_source(data_dir, run["video_path"], run["video_sha256"])
    _checked_source(data_dir, run["raw_path"], run["raw_sha256"])
    _checked_source(
        data_dir, str(reference.relative_to(data_dir)), reference_hash
    )
    return {**run, "frames": generated}


def prepare_rectified(
    data_dir: Path, pilot: Path, output: Path, *, experimental: bool = False
) -> dict:
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
    if any(
        frame["human_label"] is not None or frame["phase_suggestion"]
        for run in manifest["runs"]
        for frame in run.get("frames", [])
    ):
        raise ValueError("Only unannotated source pilots are supported")
    if shutil.disk_usage(parent).free < MIN_FREE_BYTES:
        raise ValueError("Insufficient free disk space")
    previous_umask = os.umask(PRIVATE_UMASK)
    try:
        return _publish(
            data_dir,
            pilot,
            output,
            content,
            manifest,
            experimental=experimental,
        )
    finally:
        os.umask(previous_umask)


def _publish(
    data_dir: Path,
    pilot: Path,
    output: Path,
    content: bytes,
    manifest: dict[str, Any],
    *,
    experimental: bool,
) -> dict:
    """Write the new pilot; the caller has validated inputs and umask."""
    output.mkdir(mode=0o700)
    summary = {
        "selected": len(manifest["runs"]),
        "ok": 0,
        "alignment_rejected": 0,
        "source_changed": 0,
        "unavailable": 0,
    }
    try:
        bank: dict[Any, list[Path]] = {}
        if experimental:
            for run in manifest["runs"]:
                raw = _checked_source(
                    data_dir, run["raw_path"], run["raw_sha256"]
                )
                machine_id = json.loads(raw.read_text()).get("machineId")
                if machine_id is None:
                    continue
                media = data_dir / "raw" / "media" / str(run["run_id"])
                for path in sorted(media.glob("*_overlay.jpg")):
                    if path.absolute() != path.resolve():
                        continue
                    image = read_image(path)
                    if image.shape[0] == image.shape[1]:
                        bank.setdefault(machine_id, []).append(path)
            for references in bank.values():
                references.sort(key=lambda p: ("spread" not in p.name, str(p)))
        result = copy.deepcopy(manifest)
        result["preprocessing"] = {
            "version": EXPERIMENTAL_VERSION
            if experimental
            else PREPROCESSING_VERSION,
            "experimental": experimental,
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
                    candidate = _rectify_run(
                        data_dir,
                        pilot,
                        staging,
                        run,
                        experimental=experimental,
                        bank=bank,
                    )
                    for artifact in staging.rglob("*"):
                        if artifact.is_file():
                            destination = output / artifact.relative_to(staging)
                            destination.parent.mkdir(
                                mode=0o700, parents=True, exist_ok=True
                            )
                            artifact.replace(destination)
                    summary["ok"] += 1
                except (ValueError, cv2.error) as error:
                    changed = isinstance(error, SourceChangedError)
                    candidate["status"] = "alignment_rejected"
                    candidate["rectification_rejection"] = str(error)
                    candidate["rectification_rejection_kind"] = (
                        "source_changed" if changed else "alignment"
                    )
                    summary[
                        "source_changed" if changed else "alignment_rejected"
                    ] += 1
                finally:
                    shutil.rmtree(staging)
            else:
                summary["unavailable"] += 1
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
    parser.add_argument(
        "--experimental-photo-transfer",
        action="store_true",
        help="use experimental photo fitting and measured video registration",
    )
    args = parser.parse_args()
    cv2.setNumThreads(1)
    try:
        summary = prepare_rectified(
            args.data_dir,
            args.pilot,
            args.output,
            experimental=args.experimental_photo_transfer,
        )
    except (OSError, ValueError, KeyError, TypeError):
        print("Rectified preparation failed; inspect private source artifacts.")
        return 1
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["frames"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
