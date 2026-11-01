"""Fixed-colour mask processing is tested with synthetic pixels only."""

import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from maus_export.pixel_masks import (
    PixelMaskConfig,
    preview_masks,
    suggest_masks,
    write_suggestions,
)

SIZE = 160
BACKGROUND = (25, 25, 25)
BATTER = (100, 190, 230)


def synthetic_crepe() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = np.full((SIZE, SIZE, 3), BACKGROUND, np.uint8)
    material = np.zeros((SIZE, SIZE), np.uint8)
    holes = np.zeros_like(material)
    cv2.circle(material, (80, 80), 45, 1, -1)
    cv2.circle(holes, (80, 80), 8, 1, -1)
    image[material > 0] = BATTER
    image[holes > 0] = BACKGROUND
    return image, (material > 0) & ~(holes > 0), holes > 0


def test_colour_and_connectivity_find_material_and_enclosed_hole() -> None:
    image, expected, expected_holes = synthetic_crepe()
    before = image.copy()
    material, holes = suggest_masks(image)
    assert np.array_equal(image, before)
    assert not (material & holes).any()
    assert (material & expected).sum() / (material | expected).sum() > 0.98
    assert np.array_equal(holes, expected_holes)
    again = suggest_masks(image)
    assert all(
        np.array_equal(a, b)
        for a, b in zip((material, holes), again, strict=True)
    )


def test_roi_and_small_component_filter_exclude_distractors() -> None:
    image, _, _ = synthetic_crepe()
    image[:20, :20] = BATTER
    image[150:152, 80:82] = BATTER
    material, holes = suggest_masks(image, roi=(30, 30, 100, 100))
    assert material.any()
    assert not material[:30].any()
    assert not material[130:].any()
    assert not holes[:30].any()


def test_keep_all_preserves_separate_large_fragments() -> None:
    image, _, _ = synthetic_crepe()
    image[5:25, 5:25] = BATTER
    largest, _ = suggest_masks(image)
    all_regions, _ = suggest_masks(
        image, config=PixelMaskConfig(keep_largest=False)
    )
    assert not largest[10, 10]
    assert all_regions[10, 10]


def test_open_outside_gaps_are_not_holes() -> None:
    image, _, _ = synthetic_crepe()
    image[70:91, 75:140] = BACKGROUND
    _, holes = suggest_masks(image, config=PixelMaskConfig(closing=1))
    assert not holes.any()


def test_dark_stain_is_a_known_false_hole_candidate() -> None:
    image, _, _ = synthetic_crepe()
    # Colour/topology alone cannot distinguish a dark stain from bare plate.
    cv2.circle(image, (80, 80), 10, BATTER, -1)
    cv2.circle(image, (80, 80), 8, (20, 30, 40), -1)
    _, holes = suggest_masks(image)
    assert holes[80, 80]


@pytest.mark.parametrize(
    "config",
    [
        replace(PixelMaskConfig(), opening=2),
        replace(PixelMaskConfig(), closing=0),
        replace(PixelMaskConfig(), min_lightness=-1),
        replace(PixelMaskConfig(), max_saturation=256),
        replace(PixelMaskConfig(), min_area=0),
    ],
)
def test_invalid_parameters_are_rejected(config: PixelMaskConfig) -> None:
    image, _, _ = synthetic_crepe()
    with pytest.raises(ValueError):
        suggest_masks(image, config=config)


def test_invalid_roi_or_image_is_rejected() -> None:
    image, _, _ = synthetic_crepe()
    with pytest.raises(ValueError):
        suggest_masks(image, roi=(0, 0, SIZE + 1, SIZE))
    with pytest.raises(ValueError):
        suggest_masks(image.astype(np.float32))
    with pytest.raises(ValueError):
        suggest_masks(image[..., 0])


def test_private_outputs_are_suggestions_and_keep_native_coordinates(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    source = data / "input.png"
    image, _, _ = synthetic_crepe()
    cv2.imwrite(str(source), image)
    before = source.read_bytes()
    output = data / "suggestions"
    report = write_suggestions(source, output, data_dir=data)
    assert report["human_annotation"] is False
    assert report["requires_review"] is True
    assert report["coordinate_space"] == "original_video"
    assert report["image_size"] == [SIZE, SIZE]
    assert source.read_bytes() == before
    assert json.loads((output / "report.json").read_text()) == report
    labels = cv2.imread(str(output / "labels.png"), cv2.IMREAD_UNCHANGED)
    assert labels.shape == image.shape[:2]
    assert set(np.unique(labels)) == {0, 1, 2}
    preview = preview_masks(image, labels == 1, labels == 2)
    assert np.array_equal(image, cv2.imread(str(source)))
    assert not np.array_equal(image, preview)
    with pytest.raises(FileExistsError):
        write_suggestions(source, output, data_dir=data)
    with pytest.raises(ValueError):
        write_suggestions(source, tmp_path / "public", data_dir=data)
    with pytest.raises(ValueError):
        write_suggestions(source, data / "raw/unsafe", data_dir=data)
