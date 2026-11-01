"""Hand-painted crepe masks from the private labels snapshots.

A snapshot record keeps the Label Studio result list unchanged, in original
video coordinates. Only `validation == "ok"` records are used, as the snapshots
require. Some set 1 masks may have started from a pixel suggestion, so these are
reviewed masks, not independent manual ground truth.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

RGBA_CHANNELS = 4
ALPHA_CHANNEL = 3
ALPHA_THRESHOLD = 127
BITS_PER_BYTE = 8
TOTAL_BITS = 32
WORD_SIZE_BITS = 5
RUN_SIZE_BITS = 4
RUN_SIZES = 4
RUN_SELECT_BITS = 2
OK = "ok"
PRESENT = "crepe_present"
ABSENT = "no_crepe"
MATERIAL = "crepe_material"
HOLE = "hole"


def _field(value: Any) -> Any:
    """Read a snapshot field that may be stored as its text form."""
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value
    return value


def decode_rle(rle: Sequence[int]) -> np.ndarray:
    """Decode a Label Studio brush RLE into its flat RGBA bytes."""
    bits = np.unpackbits(np.frombuffer(bytes(rle), np.uint8)).tolist()
    position = 0

    def read(count: int) -> int:
        nonlocal position
        value = 0
        for bit in bits[position : position + count]:
            value = value << 1 | bit
        position += count
        return value

    total = read(TOTAL_BITS)
    word_size = read(WORD_SIZE_BITS) + 1
    run_sizes = [read(RUN_SIZE_BITS) + 1 for _ in range(RUN_SIZES)]
    out = np.zeros(total, np.uint8)
    index = 0
    while index < total:
        repeated = read(1)
        end = index + 1 + read(run_sizes[read(RUN_SELECT_BITS)])
        if repeated:
            out[index:end] = read(word_size)
        else:
            for slot in range(index, end):
                out[slot] = read(word_size)
        index = end
    return out


def decode_region(
    region: Mapping[str, Any], size: tuple[int, int]
) -> np.ndarray:
    """One brush region as a boolean mask of `size` (width, height)."""
    width, height = size
    if (
        region["original_width"] != width
        or region["original_height"] != height
        or region.get("image_rotation", 0) != 0
    ):
        raise ValueError("brush dimensions or rotation differ from the frame")
    flat = decode_rle(region["value"]["rle"])
    if flat.size != width * height * RGBA_CHANNELS:
        raise ValueError("brush length differs from the frame")
    alpha = flat.reshape(height, width, RGBA_CHANNELS)[..., ALPHA_CHANNEL]
    return alpha > ALPHA_THRESHOLD


def record_masks(
    record: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Material and hole masks of a snapshot record, in video coordinates."""
    width, height = _field(record["image_size"])
    material = np.zeros((height, width), bool)
    holes = np.zeros((height, width), bool)
    for region in _field(record["result"]):
        if region.get("type") != "brushlabels":
            continue
        labels = region["value"]["brushlabels"]
        if labels == [MATERIAL]:
            material |= decode_region(region, (width, height))
        elif labels == [HOLE]:
            holes |= decode_region(region, (width, height))
        else:
            raise ValueError("unknown brush class")
    if record["state"] == PRESENT and not material.any():
        raise ValueError("a visible crepe needs a material mask")
    if record["state"] == ABSENT and (material.any() or holes.any()):
        raise ValueError("a no-crepe record must have no mask")
    return material, holes


def usable_records(snapshots: Iterable[str]) -> list[dict[str, Any]]:
    """Records that passed the contract with a reviewed state, one per image.

    The same image in two snapshots is refused rather than silently resolved.
    """
    seen: set[str] = set()
    records = []
    for text in snapshots:
        for record in json.loads(text)["frames"]:
            if record["validation"] != OK:
                continue
            if record["state"] not in (PRESENT, ABSENT):
                continue
            if record["image_sha256"] in seen:
                raise ValueError("the same image appears in two snapshots")
            seen.add(record["image_sha256"])
            records.append(record)
    return records


def frame_index(pilots: Iterable[Path]) -> dict[str, Path]:
    """Map image hashes to frame files, from the pilot manifests."""
    index: dict[str, Path] = {}
    for pilot in pilots:
        manifest = json.loads((pilot / "manifest.json").read_text())
        for run in manifest.get("runs", []):
            for frame in run.get("frames") or []:
                path = pilot / frame["image"]
                digest = frame.get("image_sha256") or frame.get("sha256")
                if digest and path.is_file():
                    index.setdefault(digest, path)
    return index


def frame_identity(record: Mapping[str, Any]) -> tuple[int, int, int]:
    """Run, decoded frame and video timestamp: the same video frame on any
    machine, even when another ffmpeg wrote different PNG bytes for it."""
    return (
        int(record["run_id"]),
        int(record["decoded_frame_index"]),
        int(record["decoded_pts"]),
    )


def identity_index(pilots: Iterable[Path]) -> dict[tuple[int, int, int], Path]:
    """Map frame identities to frame files, from the pilot manifests."""
    index: dict[tuple[int, int, int], Path] = {}
    for pilot in pilots:
        manifest = json.loads((pilot / "manifest.json").read_text())
        for run in manifest.get("runs", []):
            for frame in run.get("frames") or []:
                path = pilot / frame["image"]
                if path.is_file():
                    identity = frame_identity(frame | {"run_id": run["run_id"]})
                    index.setdefault(identity, path)
    return index
