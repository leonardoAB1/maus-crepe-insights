"""Mask suggestions from fixed colours and connected pixels, without models."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

BYTE_MAX = 255
CHANNELS = 3
CONNECTIVITY = 8
EDGE_KERNEL = 3
PREVIEW_ALPHA = 0.45
MATERIAL_COLOUR = (60, 190, 60)
HOLE_COLOUR = (40, 40, 230)
MATERIAL_LABEL = 1
HOLE_LABEL = 2
ALGORITHM_VERSION = "pixel-colour-components-v1"
DEFAULT_DATA = Path(__file__).resolve().parents[2] / "data"


@dataclass(frozen=True)
class PixelMaskConfig:
    """Fixed OpenCV byte-scale thresholds, not fitted colour statistics."""

    min_lightness: int = 90
    min_yellow: int = 135
    max_saturation: int = 200
    max_hole_lightness: int = 70
    opening: int = 3
    closing: int = 5
    min_area: int = 128
    min_hole_area: int = 32
    keep_largest: bool = True


DEFAULT_CONFIG = PixelMaskConfig()


def _regions(
    mask: np.ndarray,
    min_area: int,
    *,
    largest: bool = False,
    forbidden: np.ndarray | None = None,
) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=CONNECTIVITY
    )
    rejected = (
        set(np.unique(labels[forbidden])) if forbidden is not None else set()
    )
    selected = [
        index
        for index in range(1, count)
        if stats[index, cv2.CC_STAT_AREA] >= min_area and index not in rejected
    ]
    if largest and selected:
        selected = [max(selected, key=lambda i: stats[i, cv2.CC_STAT_AREA])]
    return np.isin(labels, selected)


def suggest_masks(
    image: np.ndarray,
    *,
    roi: tuple[int, int, int, int] | None = None,
    config: PixelMaskConfig = DEFAULT_CONFIG,
) -> tuple[np.ndarray, np.ndarray]:
    """Suggest material and enclosed dark gaps in original coordinates."""
    if (
        image.dtype != np.uint8
        or image.ndim != CHANNELS
        or image.shape[2] != CHANNELS
        or min(image.shape[:2]) < 1
    ):
        raise ValueError("Expected an 8-bit three-channel BGR image")
    for value in (
        config.min_lightness,
        config.min_yellow,
        config.max_saturation,
        config.max_hole_lightness,
    ):
        if not 0 <= value <= BYTE_MAX:
            raise ValueError("Colour thresholds must be between 0 and 255")
    if config.min_area < 1 or config.min_hole_area < 1:
        raise ValueError("Region areas must be positive")
    for size in (config.opening, config.closing):
        if size < 1 or size % 2 == 0:
            raise ValueError("Cleanup sizes must be positive odd integers")
    height, width = image.shape[:2]
    inside = np.ones((height, width), dtype=bool)
    if roi is not None:
        x, y, w, h = roi
        if min(x, y) < 0 or min(w, h) < 1 or x + w > width or y + h > height:
            raise ValueError("ROI must fit inside the original image")
        inside[:] = False
        inside[y : y + h, x : x + w] = True
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    warm = (
        (lab[..., 0] >= config.min_lightness)
        & (lab[..., 2] >= config.min_yellow)
        & (hsv[..., 1] <= config.max_saturation)
        & inside
    )
    candidate = warm.astype(np.uint8)
    for operation, size in (
        (cv2.MORPH_OPEN, config.opening),
        (cv2.MORPH_CLOSE, config.closing),
    ):
        candidate = cv2.morphologyEx(
            candidate, operation, np.ones((size, size), np.uint8)
        )
        candidate &= inside.astype(np.uint8)
    material = _regions(candidate, config.min_area, largest=config.keep_largest)
    if not material.any():
        return material, np.zeros_like(material)
    # Only background regions disconnected from every image edge are enclosed.
    background = (~material).astype(np.uint8)
    _, labels = cv2.connectedComponents(background, connectivity=CONNECTIVITY)
    exterior = np.unique(
        np.concatenate(
            (
                labels[0],
                labels[-1],
                labels[:, 0],
                labels[:, -1],
            )
        )
    )
    enclosed = ~material & ~np.isin(labels, exterior)
    silhouette = material | enclosed
    boundary = silhouette & ~cv2.erode(
        silhouette.astype(np.uint8),
        np.ones((EDGE_KERNEL, EDGE_KERNEL), np.uint8),
    ).astype(bool)
    dark = (lab[..., 0] <= config.max_hole_lightness) & silhouette & inside
    holes = _regions(dark, config.min_hole_area, forbidden=boundary)
    return material & ~holes, holes


def preview_masks(
    image: np.ndarray, material: np.ndarray, holes: np.ndarray
) -> np.ndarray:
    """Overlay suggestions on a copy; never paint into training input pixels."""
    preview = image.copy()
    for mask, colour in ((material, MATERIAL_COLOUR), (holes, HOLE_COLOUR)):
        preview[mask] = (
            (1 - PREVIEW_ALPHA) * preview[mask]
            + PREVIEW_ALPHA * np.asarray(colour)
        ).astype(np.uint8)
    return preview


def write_suggestions(
    source: Path,
    output: Path,
    *,
    data_dir: Path,
    roi: tuple[int, int, int, int] | None = None,
    config: PixelMaskConfig = DEFAULT_CONFIG,
) -> dict:
    """Write private suggestions without touching original images or labels."""
    source = source.resolve(strict=True)
    root = data_dir.resolve()
    output = output.absolute()
    if (
        root.name != "data"
        or output != output.resolve()
        or not output.is_relative_to(root)
        or output.is_relative_to(root / "raw")
        or (
            (source.parent.parent / "manifest.json").is_file()
            and output.is_relative_to(source.parent.parent)
        )
    ):
        raise ValueError(
            "Use a new data/ output outside raw/ and the image folder"
        )
    content = source.read_bytes()
    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Cannot decode the input image")
    material, holes = suggest_masks(image, roi=roi, config=config)
    os.umask(0o077)
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    output.mkdir(mode=0o700)
    try:
        labels = np.zeros(material.shape, np.uint8)
        labels[material] = MATERIAL_LABEL
        labels[holes] = HOLE_LABEL
        for name, pixels in (
            ("material", material.astype(np.uint8) * BYTE_MAX),
            ("holes", holes.astype(np.uint8) * BYTE_MAX),
            ("labels", labels),
            ("preview", preview_masks(image, material, holes)),
        ):
            if not cv2.imwrite(str(output / f"{name}.png"), pixels):
                raise OSError("Cannot write a suggestion PNG")
        report = {
            "algorithm": ALGORITHM_VERSION,
            "human_annotation": False,
            "requires_review": True,
            "coordinate_space": "original_video",
            "source_image_sha256": hashlib.sha256(content).hexdigest(),
            "image_size": [image.shape[1], image.shape[0]],
            "roi": list(roi) if roi is not None else None,
            "parameters": asdict(config),
            "material_pixels": int(material.sum()),
            "hole_pixels": int(holes.sum()),
            "warning": "Dark browning can look like holes; tools and glare can "
            "confuse colours. Empty output is not a reviewed negative.",
        }
        if source.read_bytes() != content:
            raise ValueError("Source changed during processing")
        (output / "report.json").write_text(json.dumps(report, indent=2))
    except BaseException:
        shutil.rmtree(output)
        raise
    return report


def main() -> int:
    """Run fixed colour thresholds and connected-pixel cleanup on one image."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H")
    )
    defaults = PixelMaskConfig()
    for name in (
        "min_lightness",
        "min_yellow",
        "max_saturation",
        "max_hole_lightness",
        "opening",
        "closing",
        "min_area",
        "min_hole_area",
    ):
        parser.add_argument(
            "--" + name.replace("_", "-"),
            type=int,
            default=getattr(defaults, name),
        )
    parser.add_argument("--keep-all", action="store_true")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    config = PixelMaskConfig(
        **{
            name: getattr(args, name)
            for name in asdict(defaults)
            if name != "keep_largest"
        },
        keep_largest=not args.keep_all,
    )
    try:
        result = write_suggestions(
            args.input,
            args.output,
            data_dir=args.data_dir,
            roi=tuple(args.roi) if args.roi else None,
            config=config,
        )
    except (OSError, ValueError, cv2.error) as error:
        parser.exit(1, f"Pixel-mask suggestion failed: {error}\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "algorithm",
                    "human_annotation",
                    "requires_review",
                    "material_pixels",
                    "hole_pixels",
                )
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
