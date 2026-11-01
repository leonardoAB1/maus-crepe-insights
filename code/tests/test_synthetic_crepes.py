"""Synthetic crepe generator and pose metrics, on made-up images only."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from maus_export import synthetic_crepes as sc


def sources() -> sc.Sources:
    plate = np.full((sc.SIZE, sc.SIZE, 3), 30, np.float32)
    return sc.Sources([plate])


def disc_mask(x: int, y: int, radius: int) -> np.ndarray:
    mask = np.zeros((sc.SIZE, sc.SIZE), np.uint8)
    cv2.circle(mask, (x, y), radius, 1, -1)
    return mask > 0


def test_generated_samples_are_labelled_consistently() -> None:
    rng = np.random.default_rng(1)
    seen = set()
    for _ in range(60):
        sample = sc.generate_sample(rng, sources())
        seen.add(sample.shape)
        assert sample.image.shape == (sc.SIZE, sc.SIZE, 3)
        assert sample.image.dtype == np.uint8
        assert not (sample.mask & ~sc.zone_disc()).any()
        length = float(np.linalg.norm(sample.direction))
        if sample.shape in ("half", "quarter"):
            assert length == pytest.approx(1.0, abs=1e-5)
        else:
            assert length == 0
        if sample.shape == "empty":
            assert not sample.mask.any()
    assert seen == set(sc.SHAPES)


def test_generation_is_repeatable_for_a_seed() -> None:
    a = sc.generate_sample(np.random.default_rng(3), sources())
    b = sc.generate_sample(np.random.default_rng(3), sources())
    assert np.array_equal(a.image, b.image)
    assert np.array_equal(a.mask, b.mask)


def test_fold_direction_points_into_the_crepe() -> None:
    rng = np.random.default_rng(5)
    for shape in ("half", "quarter"):
        points, direction = sc.crepe_outline(rng, shape)
        apex = points[0]
        middle = points[1:].mean(0)
        toward = (middle - apex) / np.linalg.norm(middle - apex)
        assert sc.angle_error_deg(toward, direction) < 15


def test_centre_error_is_a_share_of_plate_radius() -> None:
    a = disc_mask(80, 80, 20)
    b = disc_mask(90, 80, 20)
    assert sc.centre_error_rel(a, b) == pytest.approx(10 / sc.ZONE_R, abs=0.01)
    assert sc.centre_error_rel(a, np.zeros_like(a)) is None


def test_iou_of_identical_and_of_two_empty_masks() -> None:
    a = disc_mask(80, 80, 20)
    assert sc.iou(a, a) == 1.0
    assert sc.iou(a, ~a) == 0.0
    empty = np.zeros_like(a)
    assert sc.iou(empty, empty) == 1.0


def test_angle_error_uses_the_full_circle() -> None:
    right = np.array([1.0, 0.0])
    assert sc.angle_error_deg(right, right) == pytest.approx(0)
    assert sc.angle_error_deg(right, -right) == pytest.approx(180)
    assert sc.angle_error_deg(right, np.array([0.0, 2.0])) == pytest.approx(90)


def test_clean_mask_keeps_the_largest_piece_and_fills_holes() -> None:
    mask = disc_mask(80, 80, 30) | disc_mask(10, 10, 4)
    mask[78:82, 78:82] = False
    cleaned = sc.clean_mask(mask)
    assert cleaned[80, 80]
    assert not cleaned[10, 10]
    assert not sc.clean_mask(disc_mask(10, 10, 2)).any()


def test_otsu_baseline_finds_a_bright_crepe_on_a_dark_plate() -> None:
    image = np.full((sc.SIZE, sc.SIZE, 3), 20, np.uint8)
    crepe = disc_mask(80, 80, 40)
    image[crepe] = (180, 210, 230)
    assert sc.iou(sc.otsu_baseline(image), crepe) > 0.9


def test_texture_from_frame_needs_a_big_enough_crepe() -> None:
    image = np.full((sc.SIZE, sc.SIZE, 3), 100, np.uint8)
    tile = sc.texture_from_frame(image, disc_mask(80, 80, 40))
    assert tile is not None and tile.shape == image.shape
    assert sc.texture_from_frame(image, disc_mask(80, 80, 3)) is None
    assert sc.texture_from_frame(image, np.zeros((160, 160), bool)) is None


def test_pasted_cutout_moves_its_mask_and_direction_together() -> None:
    # A half-plate crepe whose fold direction points at its own centroid
    mask = disc_mask(80, 80, 40)
    mask[:, :80] = False
    cutout = sc.Cutout(
        np.full((sc.SIZE, sc.SIZE, 3), 200, np.uint8),
        mask,
        np.array([1.0, 0.0], np.float32),
    )
    rng = np.random.default_rng(2)
    for _ in range(20):
        image = np.zeros((sc.SIZE, sc.SIZE, 3), np.float32)
        moved, direction = sc.paste_cutout(rng, image, cutout)
        ys, xs = np.nonzero(moved)
        # A half disc is narrowest along its fold direction, and its long
        # tail runs from the straight edge toward the arc
        points = np.column_stack([xs, ys]).astype(float)
        points -= points.mean(0)
        along = points @ direction
        across = points @ np.array([-direction[1], direction[0]])
        assert along.std() < 0.7 * across.std()
        assert abs(np.mean(along * across)) < 0.1 * along.std() * across.std()
        assert np.mean(along**3) > 0
        assert np.linalg.norm(direction) == pytest.approx(1.0, abs=1e-5)
        assert image[moved].mean() > 100


def test_rotate_direction_matches_opencv_points() -> None:
    turn = cv2.getRotationMatrix2D((0.0, 0.0), 90.0, 1.0)
    point = cv2.transform(np.array([[[1.0, 0.0]]]), turn)[0, 0]
    assert sc.rotate_direction(np.array([1.0, 0.0]), turn) == pytest.approx(
        point
    )


def test_rectify_for_model_matches_the_full_size_warp_closely() -> None:
    frame = np.full((509, 900, 3), 30, np.uint8)
    cv2.circle(frame, (450, 254), 150, (180, 210, 230), -1)
    shift = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 195.0], [0.0, 0.0, 1.0]])
    fast = sc.rectify_for_model(frame, shift)
    full = sc.shrink(sc.rectify_frame(frame, shift))
    assert fast.shape == (sc.SIZE, sc.SIZE, 3)
    assert np.abs(fast.astype(float) - full).mean() < 2
