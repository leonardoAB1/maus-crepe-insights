"""Pixel-level diffuse foreground evidence instead of bright components."""

from __future__ import annotations

import cv2
import numpy as np

MIN_FOOD_L = 100
MIN_FOOD_B = 134
MAX_FOOD_A = 145
DARK_L = 80
MIN_LIGHT_CHANGE = 22
MIN_COLOUR_CHANGE = 10
MIN_COLOUR_L = 125
SPECULAR_L = 245
SPECULAR_B = 140
MIN_PLATE_AREA = 0.03
MIN_COMPONENT_RATIO = 0.05
OPEN_SIZE = 3
CLOSE_SIZE = 7
MASK_VALUE = 255


def diffuse_mask(
    lab: np.ndarray,
    inside: np.ndarray,
    reference_lab: np.ndarray | None = None,
    highlight_l: np.ndarray | None = None,
) -> np.ndarray:
    """Retain changed warm diffuse pixels; no component can rescue glare.

    Colour changes allow food over a previously bright reflection. Neutral
    saturated pixels never qualify, even when attached to real food. This
    intentionally abstains from identifying colourless food from brightness.
    """
    light = lab[:, :, 0].astype(np.int16)
    warm = (
        (light > MIN_FOOD_L)
        & (lab[:, :, 2] >= MIN_FOOD_B)
        & (lab[:, :, 1] <= MAX_FOOD_A)
    )
    specular = (light >= SPECULAR_L) & (lab[:, :, 2] < SPECULAR_B)
    evidence = warm & ~specular & inside
    if reference_lab is not None:
        ref_light = reference_lab[:, :, 0].astype(np.int16)
        dark = inside & (light < DARK_L) & (ref_light < DARK_L)
        if not dark.any():
            raise ValueError("No dark pixels for diffuse exposure matching")
        shift = float(np.median((light - ref_light)[dark]))
        upper = (
            ref_light
            if highlight_l is None
            else np.maximum(ref_light, highlight_l)
        )
        light_change = light - upper - shift > MIN_LIGHT_CHANGE
        colour_change = (
            lab[:, :, 2].astype(np.int16)
            - reference_lab[:, :, 2].astype(np.int16)
        ) > MIN_COLOUR_CHANGE
        evidence &= light_change | (colour_change & (light > MIN_COLOUR_L))
    mask = evidence.astype(np.uint8)
    opening = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (OPEN_SIZE, OPEN_SIZE)
    )
    closing = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (CLOSE_SIZE, CLOSE_SIZE)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, opening)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closing)
    # Closing must never grow a known specular pixel back into foreground.
    mask &= (inside & ~specular).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    result = np.zeros_like(mask)
    if count > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        minimum = max(
            MIN_PLATE_AREA * inside.sum(), MIN_COMPONENT_RATIO * areas.max()
        )
        keep = np.flatnonzero(areas >= minimum) + 1
        lookup = np.zeros(count, np.uint8)
        lookup[keep] = MASK_VALUE
        result = lookup[labels]
    return result
