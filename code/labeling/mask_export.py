"""Export reviewed human brush annotations in the original image coordinates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

if __package__:
    from .workflow import MASK_TAXONOMY_VERSION, build_plan
else:
    from workflow import MASK_TAXONOMY_VERSION, build_plan

RGBA_CHANNELS = 4
RLE_HEADER_BYTES = 4
BYTE_MAX = 255
ALPHA_THRESHOLD = 127
MAX_MASK_PIXELS = 16_000_000
MATERIAL_CLASS = 1
HOLE_CLASS = 2
IGNORE_CLASS = 255


def _key(data: dict) -> tuple:
    return (
        data["run_id"],
        data["decoded_frame_index"],
        data["requested_seconds"],
    )


def _decode_region(region: dict, size: tuple[int, int]) -> np.ndarray:
    from label_studio_sdk.converter.brush import decode_rle

    width, height = size
    if (
        region["original_width"] != width
        or region["original_height"] != height
        or region.get("image_rotation", 0) != 0
        or width * height > MAX_MASK_PIXELS
    ):
        raise ValueError(
            "Brush dimensions/rotation differ from the source image"
        )
    value = region["value"]
    rle = value["rle"]
    if (
        value.get("format") != "rle"
        or not isinstance(rle, list)
        or len(rle) < RLE_HEADER_BYTES
        or any(type(v) is not int or not 0 <= v <= BYTE_MAX for v in rle)
        or int.from_bytes(bytes(rle[:RLE_HEADER_BYTES]), "big")
        != width * height * RGBA_CHANNELS
    ):
        raise ValueError("Invalid brush RLE or unexpected decoded length")
    decoded = decode_rle(rle)
    return (
        decoded.reshape(height, width, RGBA_CHANNELS)[..., -1] > ALPHA_THRESHOLD
    )


def _masks(
    annotation: dict, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    height, width = valid.shape
    masks = {
        "crepe_material": np.zeros_like(valid),
        "hole": np.zeros_like(valid),
    }
    states = []
    for region in annotation["result"]:
        if region["from_name"] == "mask_state" and region["type"] == "choices":
            states.extend(region["value"]["choices"])
        elif (
            region["from_name"] == "segmentation"
            and region["type"] == "brushlabels"
        ):
            labels = region["value"]["brushlabels"]
            if len(labels) != 1 or labels[0] not in masks:
                raise ValueError("Unknown or ambiguous brush class")
            masks[labels[0]] |= _decode_region(region, (width, height)) & valid
        else:
            raise ValueError("Unexpected annotation control")
        if region.get("to_name") != "image":
            raise ValueError("Annotation is not attached to the source image")
    if len(states) != 1 or states[0] not in {"crepe_present", "no_crepe"}:
        raise ValueError("Exactly one reviewed mask state is required")
    material, holes = masks["crepe_material"], masks["hole"]
    if np.any(material & holes):
        raise ValueError("Material and hole masks overlap; erase the overlap")
    if states[0] == "crepe_present" and not material.any():
        raise ValueError("A visible crepe requires a material mask")
    if states[0] == "no_crepe" and (material.any() or holes.any()):
        raise ValueError(
            "A no-crepe annotation must not contain foreground masks"
        )
    return material, holes


def export_masks(
    pilot: Path, annotations: Path, output: Path
) -> dict[str, Any]:
    """Write reviewed PNG masks after checking immutable frame provenance."""
    pilot = pilot.resolve(strict=True)
    output = output.absolute()
    export_root = pilot.parent / "exports"
    source_root = Path(__file__).resolve().parents[2]
    if (
        output.parent != export_root
        or export_root != export_root.resolve()
        or output == pilot
        or pilot.stat().st_mode & 0o077
        or pilot.parent.stat().st_mode & 0o077
        or (
            pilot.is_relative_to(source_root)
            and not pilot.is_relative_to(source_root / "data")
        )
    ):
        raise ValueError(
            "Use a new direct child of the private labeling/exports"
        )
    plan = build_plan(pilot, kind="masks")
    source = annotations.read_bytes()
    tasks = json.loads(source)
    if not isinstance(tasks, list):
        raise ValueError("Expected a Label Studio JSON task export")
    expected = {_key(t["data"]): t["data"] for t in plan["tasks"]}
    os.umask(0o077)
    export_root.mkdir(mode=0o700, exist_ok=True)
    if export_root.stat().st_mode & 0o077:
        raise ValueError("Export directory must be private (0700)")
    output.mkdir(mode=0o700)
    records = []
    seen = set()
    seen_ids = set()
    try:
        for task in tasks:
            data = task["data"]
            key = _key(data)
            task_id = task["id"]
            if (
                key in seen
                or type(task_id) is not int
                or task_id < 1
                or task_id in seen_ids
                or data != expected.get(key)
            ):
                raise ValueError("Duplicate task or changed frame provenance")
            seen.add(key)
            seen_ids.add(task_id)
            submitted = [
                a
                for a in task["annotations"]
                if not a.get("was_cancelled", False)
            ]
            if not submitted:
                continue
            if len(submitted) != 1:
                raise ValueError("Resolve multiple submitted annotations first")
            annotation = submitted[0]
            with Image.open(pilot / data["valid_mask"]) as image:
                valid = np.asarray(image) == BYTE_MAX
            material, holes = _masks(annotation, valid)
            labels = np.zeros(valid.shape, dtype=np.uint8)
            labels[material] = MATERIAL_CLASS
            labels[holes] = HOLE_CLASS
            labels[~valid] = IGNORE_CLASS
            names = {}
            for name, pixels in (
                ("material", material.astype(np.uint8) * BYTE_MAX),
                ("holes", holes.astype(np.uint8) * BYTE_MAX),
                ("labels", labels),
            ):
                filename = f"task-{task_id}-{name}.png"
                Image.fromarray(pixels).save(output / filename)
                names[name] = filename
            records.append(
                {
                    "task_id": task_id,
                    "annotation_id": annotation["id"],
                    "run_id": data["run_id"],
                    "decoded_frame_index": data["decoded_frame_index"],
                    "decoded_pts": data["decoded_pts"],
                    "requested_seconds": data["requested_seconds"],
                    "time_base": data["time_base"],
                    "start_pts": data["start_pts"],
                    "image_sha256": data["image_sha256"],
                    "coordinate_space": data["coordinate_space"],
                    "preprocessing_version": data["preprocessing_version"],
                    "transform_sha256": data["transform_sha256"],
                    "valid_mask_sha256": data["valid_mask_sha256"],
                    "image_size": data["image_size"],
                    "outputs": names,
                }
            )
        if not records:
            raise ValueError("No submitted, reviewed masks in the export")
        result = {
            "mask_taxonomy_version": MASK_TAXONOMY_VERSION,
            "source_pilot_sha256": plan["digest"],
            "annotation_export_sha256": hashlib.sha256(source).hexdigest(),
            "classes": {
                "other": 0,
                "crepe_material": MATERIAL_CLASS,
                "hole": HOLE_CLASS,
                "unsupported": IGNORE_CLASS,
            },
            "records": records,
        }
        (output / "manifest.json").write_text(json.dumps(result, indent=2))
    except BaseException:
        shutil.rmtree(output)
        raise
    return {"exported": len(records), "tasks_in_json": len(tasks)}


def main() -> int:
    """Export reviewed human masks, never predictions or preparation fields."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = export_masks(args.pilot, args.annotations, args.output)
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        print(
            "Mask export rejected; check provenance, masks and new output path."
        )
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
