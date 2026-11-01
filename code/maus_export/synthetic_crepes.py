"""Synthetic crepes on real plates, with exact mask and fold direction.

Experiment for challenge task 2. Every generated image knows its own crepe
mask and fold direction, so no labelling is needed. The reference for real
frames comes from the robot's overlays and is only a reference: the outline is
the robot's own, and the magenta fold is the fold it wanted, not the fold it
got.

Directions are unit vectors (x, y) in rectified image coordinates, pointing
from the fold apex along the bisector of the folded sector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from maus_export.vision_metrics import (
    filled_outline,
    fold_regions,
    outline_mask,
    prepare,
)

RECTIFIED_PX = 900
SIZE = 160
FAST_WARP_PX = 320
ZONE_CENTRE_SHARE = 0.5
ZONE_RADIUS_SHARE = 0.44
ZONE_R = SIZE * ZONE_RADIUS_SHARE
ZONE_C = SIZE * ZONE_CENTRE_SHARE

SHAPES = ("empty", "flat", "half", "quarter")
SHAPE_WEIGHTS = (0.12, 0.28, 0.32, 0.28)
SECTOR_DEGREES = {"half": 180.0, "quarter": 90.0}
REAL_TEXTURE_SHARE = 0.7
# Version 1 used only drawn shapes and learned their pasted look: perfect on
# synthetic images, blind on real ones. Most crepes are now real cut-outs.
CUTOUT_SHARE = 0.75
CUTOUT_SHIFT_MAX = 0.15
FEATHER_SIGMA = (0.5, 1.5)
BLUR_SHARE = 0.5
BLUR_SIGMA = (0.4, 1.2)
SHADOW_SHARE = 0.3

# Size and placement, as shares of the plate radius, from the real overlays
FLAT_RADIUS = (0.55, 0.98)
FOLD_RADIUS = (0.6, 1.0)
FLAT_OFFSET_MAX = 0.2
APEX_OFFSET_MAX = 0.35
EDGE_WOBBLE_MAX = 0.08
EDGE_WOBBLE_LOBES = 9
OUTLINE_POINTS = 180

HAND_SHARE = 0.12
STEAM_SHARE = 0.35
RESIDUE_SHARE = 0.25
GAIN_RANGE = (0.45, 1.5)
GAMMA_RANGE = (0.7, 1.4)
SHADOW_PX = 2
SHADOW_STRENGTH = 0.45
NOISE_SIGMA = 3.0
FULL_TURN = 360.0
MAX_U8 = 255
MIN_REGION_PX = 40


@dataclass
class Sample:
    image: np.ndarray
    mask: np.ndarray
    direction: np.ndarray
    shape: str


@dataclass
class Cutout:
    """A real crepe: its frame, the robot's outline and its fold direction."""

    image: np.ndarray
    mask: np.ndarray
    direction: np.ndarray


@dataclass
class Sources:
    """Plates and crepe textures taken from training frames only."""

    plates: list[np.ndarray]
    textures: dict[str, list[np.ndarray]] = field(default_factory=dict)
    cutouts: dict[str, list[Cutout]] = field(default_factory=dict)


def zone_disc(size: int = SIZE) -> np.ndarray:
    yy, xx = np.ogrid[:size, :size]
    centre = size * ZONE_CENTRE_SHARE
    radius = size * ZONE_RADIUS_SHARE
    return (xx - centre) ** 2 + (yy - centre) ** 2 <= radius**2


def rectify_frame(frame: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return cv2.warpPerspective(frame, matrix, (RECTIFIED_PX, RECTIFIED_PX))


def rectify_for_model(
    frame: np.ndarray, matrix: np.ndarray, target_px: int = RECTIFIED_PX
) -> np.ndarray:
    """A rectified frame at `SIZE`, about 25 times faster than warping to full
    size first. Warping straight to `SIZE` aliases and costs more accuracy."""
    scale = FAST_WARP_PX / target_px
    fast = np.diag([scale, scale, 1.0]) @ matrix
    return shrink(
        cv2.warpPerspective(frame, fast, (FAST_WARP_PX, FAST_WARP_PX))
    )


def shrink(image: np.ndarray) -> np.ndarray:
    return cv2.resize(image, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def overlay_reference(overlay: np.ndarray, check: str) -> dict[str, Any] | None:
    """The robot's crepe region and fold direction from one overlay, at
    `SIZE`. Direction is null on a spread, which has no fold."""
    prep = prepare(overlay)
    if prep is None:
        return None
    hsv, zone, inside = prep
    ideal = None
    if check == "spread":
        crepe = filled_outline(outline_mask(hsv) & inside, zone)
    else:
        crepe, ideal = fold_regions(hsv, zone, inside)
    if crepe is None:
        return None
    direction = None
    if ideal is not None:
        ys, xs = np.nonzero(ideal)
        vector = np.array([xs.mean() - zone[0], ys.mean() - zone[1]])
        length = float(np.linalg.norm(vector))
        if length > 0:
            direction = vector / length
    mask = cv2.resize(
        crepe.astype(np.uint8), (SIZE, SIZE), interpolation=cv2.INTER_NEAREST
    )
    return {"mask": mask.astype(bool), "direction": direction}


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """Largest connected region, with holes filled."""
    binary = mask.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if count <= 1:
        return np.zeros_like(mask, dtype=bool)
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[best, cv2.CC_STAT_AREA] < MIN_REGION_PX:
        return np.zeros_like(mask, dtype=bool)
    contours, _ = cv2.findContours(
        (labels == best).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    out = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(out, contours, -1, 1, thickness=cv2.FILLED)
    return out > 0


def mask_centre(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return np.array([xs.mean(), ys.mean()])


def centre_error_rel(found: np.ndarray, reference: np.ndarray) -> float | None:
    """Centre distance as a share of the plate radius, null if either mask
    is empty."""
    a, b = mask_centre(found), mask_centre(reference)
    if a is None or b is None:
        return None
    return float(np.linalg.norm(a - b) / ZONE_R)


def iou(found: np.ndarray, reference: np.ndarray) -> float:
    union = (found | reference).sum()
    return float((found & reference).sum() / union) if union else 1.0


def angle_error_deg(found: np.ndarray, reference: np.ndarray) -> float:
    """Smallest angle between two direction vectors, in degrees."""
    cosine = float(np.dot(found, reference))
    cosine /= float(np.linalg.norm(found) * np.linalg.norm(reference)) or 1.0
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def otsu_baseline(image: np.ndarray) -> np.ndarray:
    """Classical comparison: Otsu lightness threshold inside the plate."""
    disc = zone_disc()
    lightness = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[..., 0]
    threshold, _ = cv2.threshold(
        lightness[disc].reshape(-1, 1),
        0,
        MAX_U8,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return clean_mask((lightness > threshold) & disc)


def noise_field(rng: np.random.Generator, scale: int) -> np.ndarray:
    """Smooth noise in 0-1, `scale` cells across."""
    coarse = rng.random((scale, scale)).astype(np.float32)
    return cv2.resize(coarse, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)


def procedural_texture(rng: np.random.Generator, shape: str) -> np.ndarray:
    """Pale batter for a flat crepe, mottled brown for a folded one (BGR)."""
    low = noise_field(rng, 6)
    fine = noise_field(rng, 40)
    mix = np.clip(0.55 * low + 0.45 * fine, 0, 1)[..., None]
    if shape == "flat":
        dark = np.array([110, 150, 170], np.float32)
        light = np.array([180, 215, 230], np.float32)
    else:
        dark = np.array([35, 85, 140], np.float32)
        light = np.array([120, 185, 230], np.float32)
    return (dark + (light - dark) * mix).astype(np.float32)


def pick_texture(
    rng: np.random.Generator, sources: Sources, shape: str
) -> np.ndarray:
    pool = sources.textures.get(shape) or []
    if pool and rng.random() < REAL_TEXTURE_SHARE:
        chosen = pool[int(rng.integers(len(pool)))].astype(np.float32)
        turn = cv2.getRotationMatrix2D(
            (SIZE / 2, SIZE / 2), float(rng.uniform(0, FULL_TURN)), 1.0
        )
        return cv2.warpAffine(
            chosen, turn, (SIZE, SIZE), borderMode=cv2.BORDER_REFLECT
        )
    return procedural_texture(rng, shape)


def wobble(rng: np.random.Generator, angles: np.ndarray) -> np.ndarray:
    """Radius multiplier that makes an outline uneven like real batter."""
    out = np.ones_like(angles)
    amplitude = rng.uniform(0, EDGE_WOBBLE_MAX)
    for lobe in range(2, EDGE_WOBBLE_LOBES):
        out += (
            amplitude
            / lobe
            * rng.normal()
            * np.sin(lobe * angles + rng.uniform(0, 2 * np.pi))
        )
    return out


def crepe_outline(
    rng: np.random.Generator, shape: str
) -> tuple[np.ndarray, np.ndarray]:
    """Outline polygon in pixels and the fold direction (zero when flat)."""
    centre = np.array([ZONE_C, ZONE_C])
    heading = rng.uniform(0, 2 * np.pi)
    place = rng.uniform(0, 2 * np.pi)
    if shape == "flat":
        radius = rng.uniform(*FLAT_RADIUS) * ZONE_R
        origin = centre + rng.uniform(0, FLAT_OFFSET_MAX) * ZONE_R * np.array(
            [np.cos(place), np.sin(place)]
        )
        angles = np.linspace(0, 2 * np.pi, OUTLINE_POINTS, endpoint=False)
        reach = radius * wobble(rng, angles)
        points = origin + np.column_stack(
            [reach * np.cos(angles), reach * np.sin(angles)]
        )
        return points, np.zeros(2)
    half_span = np.radians(SECTOR_DEGREES[shape] / 2)
    radius = rng.uniform(*FOLD_RADIUS) * ZONE_R
    apex = centre + rng.uniform(0, APEX_OFFSET_MAX) * ZONE_R * np.array(
        [np.cos(place), np.sin(place)]
    )
    angles = np.linspace(-half_span, half_span, OUTLINE_POINTS) + heading
    reach = radius * wobble(rng, angles)
    arc = apex + np.column_stack(
        [reach * np.cos(angles), reach * np.sin(angles)]
    )
    direction = np.array([np.cos(heading), np.sin(heading)])
    return np.vstack([apex, arc]), direction


def rotated_plate(rng: np.random.Generator, plate: np.ndarray) -> np.ndarray:
    """Turn only the plate inside its circle, as on the robot. The room
    outside stays where it is, and so does the camera's glare roughly."""
    turn = cv2.getRotationMatrix2D(
        (ZONE_C, ZONE_C), float(rng.uniform(0, FULL_TURN)), 1.0
    )
    turned = cv2.warpAffine(plate, turn, (SIZE, SIZE))
    return np.where(zone_disc()[..., None], turned, plate).astype(np.float32)


def add_residue(rng: np.random.Generator, image: np.ndarray) -> None:
    """Small batter spots, which are not crepe and stay out of the mask."""
    for _ in range(int(rng.integers(1, 5))):
        place = rng.uniform(0, 2 * np.pi)
        distance = rng.uniform(0.4, 0.95) * ZONE_R
        spot = (
            int(ZONE_C + distance * np.cos(place)),
            int(ZONE_C + distance * np.sin(place)),
        )
        axes = (int(rng.integers(1, 5)), int(rng.integers(1, 5)))
        colour = tuple(float(v) for v in rng.uniform(90, 200, 3))
        cv2.ellipse(image, spot, axes, 0.0, 0.0, FULL_TURN, colour, -1)


def add_hand(rng: np.random.Generator, image: np.ndarray) -> np.ndarray:
    """A hand reaching in from the rim. Returns the pixels it covers."""
    place = rng.uniform(0, 2 * np.pi)
    start = np.array([ZONE_C, ZONE_C]) + 1.3 * ZONE_R * np.array(
        [np.cos(place), np.sin(place)]
    )
    end = np.array([ZONE_C, ZONE_C]) + rng.uniform(0.2, 0.8) * ZONE_R * (
        np.array([np.cos(place), np.sin(place)])
    )
    width = int(rng.integers(14, 28))
    covered = np.zeros((SIZE, SIZE), np.uint8)
    cv2.line(
        covered,
        tuple(int(v) for v in start),
        tuple(int(v) for v in end),
        1,
        width,
    )
    cv2.circle(covered, tuple(int(v) for v in end), width // 2 + 2, 1, -1)
    skin = np.array([130, 165, 215], np.float32) * rng.uniform(0.7, 1.15)
    shade = 0.8 + 0.3 * noise_field(rng, 8)
    image[covered > 0] = (skin * shade[..., None])[covered > 0]
    return covered > 0


def add_steam(rng: np.random.Generator, image: np.ndarray) -> np.ndarray:
    haze = np.clip(noise_field(rng, 5) - rng.uniform(0.3, 0.6), 0, 1)
    strength = rng.uniform(0.2, 0.7)
    alpha = (haze * strength)[..., None]
    return image * (1 - alpha) + np.float32(225) * alpha


def camera_effects(rng: np.random.Generator, image: np.ndarray) -> np.ndarray:
    scaled = np.clip(image / MAX_U8, 0, 1) ** rng.uniform(*GAMMA_RANGE)
    scaled = scaled * rng.uniform(*GAIN_RANGE) * MAX_U8
    scaled = scaled * rng.uniform(0.9, 1.1, 3)
    if rng.random() < BLUR_SHARE:
        sigma = float(rng.uniform(*BLUR_SIGMA))
        scaled = cv2.GaussianBlur(scaled, (0, 0), sigma)
    scaled = scaled + rng.normal(0, NOISE_SIGMA, scaled.shape)
    return np.clip(scaled, 0, MAX_U8).astype(np.uint8)


def rotate_direction(direction: np.ndarray, turn: np.ndarray) -> np.ndarray:
    """Apply the rotation part of an OpenCV affine matrix to a direction."""
    return turn[:, :2] @ direction


def feathered_paste(
    rng: np.random.Generator,
    image: np.ndarray,
    crepe: np.ndarray,
    mask: np.ndarray,
) -> None:
    """Blend a crepe in with a soft edge, so a hard seam gives nothing away."""
    sigma = float(rng.uniform(*FEATHER_SIGMA))
    alpha = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma)[..., None]
    image[:] = image * (1 - alpha) + crepe * alpha


def paste_cutout(
    rng: np.random.Generator, image: np.ndarray, cutout: Cutout
) -> tuple[np.ndarray, np.ndarray]:
    """Turn a real crepe about the plate centre, shift it a little and paste
    it. Returns the new mask and fold direction, both moved with the crepe."""
    turn = cv2.getRotationMatrix2D(
        (ZONE_C, ZONE_C), float(rng.uniform(0, FULL_TURN)), 1.0
    )
    turn[:, 2] += rng.uniform(-CUTOUT_SHIFT_MAX, CUTOUT_SHIFT_MAX, 2) * ZONE_R
    crepe = cv2.warpAffine(cutout.image.astype(np.float32), turn, (SIZE, SIZE))
    moved = cv2.warpAffine(
        cutout.mask.astype(np.uint8),
        turn,
        (SIZE, SIZE),
        flags=cv2.INTER_NEAREST,
    )
    mask = (moved > 0) & zone_disc()
    feathered_paste(rng, image, crepe, mask)
    return mask, rotate_direction(cutout.direction, turn).astype(np.float32)


def generate_sample(rng: np.random.Generator, sources: Sources) -> Sample:
    """One labelled image: real plate, generated crepe, disturbances."""
    plate = sources.plates[int(rng.integers(len(sources.plates)))]
    image = rotated_plate(rng, plate)
    shape = str(rng.choice(SHAPES, p=SHAPE_WEIGHTS))
    mask = np.zeros((SIZE, SIZE), bool)
    direction = np.zeros(2, np.float32)
    pool = sources.cutouts.get(shape) or []
    if pool and rng.random() < CUTOUT_SHARE:
        cutout = pool[int(rng.integers(len(pool)))]
        mask, direction = paste_cutout(rng, image, cutout)
    elif shape != "empty":
        points, vector = crepe_outline(rng, shape)
        filled = np.zeros((SIZE, SIZE), np.uint8)
        cv2.fillPoly(filled, [points.astype(np.int32)], 1)
        mask = (filled > 0) & zone_disc()
        direction = vector.astype(np.float32)
        if rng.random() < SHADOW_SHARE:
            shadow = cv2.dilate(
                filled,
                np.ones((SHADOW_PX * 2 + 1, SHADOW_PX * 2 + 1), np.uint8),
            )
            image[shadow > 0] *= SHADOW_STRENGTH
        texture = pick_texture(rng, sources, shape)
        feathered_paste(rng, image, texture, mask)
    if rng.random() < RESIDUE_SHARE:
        add_residue(rng, image)
    if rng.random() < HAND_SHARE:
        mask = mask & ~add_hand(rng, image)
    if rng.random() < STEAM_SHARE:
        image = add_steam(rng, image)
    return Sample(camera_effects(rng, image), mask, direction, shape)


def texture_from_frame(
    image: np.ndarray, mask: np.ndarray
) -> np.ndarray | None:
    """Crop the middle of a real crepe and stretch it into a full tile."""
    centre = mask_centre(mask)
    if centre is None:
        return None
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    half = int(distance.max() * np.sqrt(0.5))
    if half < MIN_REGION_PX // 4:
        return None
    y, x = np.unravel_index(int(np.argmax(distance)), distance.shape)
    crop = image[max(y - half, 0) : y + half, max(x - half, 0) : x + half]
    return cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
