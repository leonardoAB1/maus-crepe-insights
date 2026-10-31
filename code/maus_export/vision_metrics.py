"""Continuous quality scores from the robot's vision-check overlays.

Each overlay is a top-down, rectified view of the plate on which the robot
draws its own analysis in fixed colours: the analysis zone (blue), the crepe
outline (green), a fitted circle (cyan), holes (red), batter residue
(orange) and, for folds, the ideal fold shape (magenta). We read those
annotations back as numbers. Sizes are relative to the analysis zone because
no millimetre scale is published for the plate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

PREFIX = "vision."
SPREAD_CHECK = "spread"
FOLD_CHECKS = ("fold1", "fold2")

# OpenCV hue runs 0-179. Annotation colours are fully saturated, batter and
# plate are not, so a high saturation and value floor isolates them.
HUE_TOL = 6
SAT_MIN = 170
VAL_MIN = 150
ORANGE_SAT_MIN = 200  # browned crepe is orange-ish but less saturated
# The cyan circle is drawn over the green outline and JPEG blends the two,
# so the outline is read as one green-to-cyan hue band.
OUTLINE_HUES = (54, 96)
OUTLINE_SAT_MIN = 100
OUTLINE_VAL_MIN = 100
HUE = {
    "red": 0,
    "orange": 18,
    "green": 60,
    "cyan": 90,
    "blue": 121,
    "magenta": 150,
}
# Annotations drawn just outside the zone (crepe outline touching the rim)
# still count; text and legend in the corners do not.
ZONE_MARGIN = 1.03
MIN_ZONE_POINTS = 50
WALL_KERNEL = 3
# Border pixels this close to the zone edge count as touching it
EDGE_BAND_PX = 4
# Pieces smaller than this share of the zone are noise, not a crepe
MIN_REGION_FRAC = 0.001
# The bare plate hugs the zone edge for about half its border; the crepe
# touches it far less, even when it runs off the zone
MAX_EDGE_FRAC = 0.42
BRIDGE_PX = 9
# Erode the crepe mask before reading pixel colour so the outline is skipped
COLOUR_ERODE = 7
# Bare plate is dark (CIE L* mostly below 33); raw batter stays above ~45
BARE_L_MAX = 35
# Opening kernel and minimum size that drop bubble specks from bare plate
SPECK_KERNEL = 5
MIN_GAP_FRAC = 0.0005  # of the zone area


def colour_mask(hsv: np.ndarray, name: str) -> np.ndarray:
    """Boolean mask of pixels drawn in one annotation colour."""
    hue = hsv[..., 0].astype(int)
    dist = np.abs(hue - HUE[name])
    dist = np.minimum(dist, 180 - dist)  # red wraps around 0/179
    sat_min = ORANGE_SAT_MIN if name == "orange" else SAT_MIN
    return (
        (dist <= HUE_TOL) & (hsv[..., 1] >= sat_min) & (hsv[..., 2] >= VAL_MIN)
    )


def fit_circle(mask: np.ndarray) -> tuple[float, float, float] | None:
    """Least-squares circle (cx, cy, r) through the pixels of a mask."""
    ys, xs = np.nonzero(mask)
    if len(xs) < MIN_ZONE_POINTS:
        return None
    a = np.column_stack([xs, ys, np.ones_like(xs)]).astype(float)
    b = (xs.astype(float) ** 2) + (ys.astype(float) ** 2)
    (c0, c1, c2), *_ = np.linalg.lstsq(a, b, rcond=None)
    cx, cy = c0 / 2, c1 / 2
    r = float(np.sqrt(c2 + cx**2 + cy**2))
    return float(cx), float(cy), r


def disc(shape: tuple[int, int], cx: float, cy: float, r: float) -> np.ndarray:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2


def zone_of(hsv: np.ndarray) -> tuple[float, float, float] | None:
    """The blue analysis-zone circle. The legend swatch sits in a corner,
    far from the circle, so it barely moves the fit; one refit on points
    near the first circle removes it."""
    blue = colour_mask(hsv, "blue")
    first = fit_circle(blue)
    if first is None:
        return None
    cx, cy, r = first
    ys, xs = np.nonzero(blue)
    near = np.abs(np.hypot(xs - cx, ys - cy) - r) < r * (ZONE_MARGIN - 1)
    ring = np.zeros_like(blue)
    ring[ys[near], xs[near]] = True
    return fit_circle(ring) or first


def outline_mask(hsv: np.ndarray) -> np.ndarray:
    """Crepe outline pixels (green, including where cyan overlaps it)."""
    hue = hsv[..., 0]
    return (
        (hue >= OUTLINE_HUES[0])
        & (hue <= OUTLINE_HUES[1])
        & (hsv[..., 1] >= OUTLINE_SAT_MIN)
        & (hsv[..., 2] >= OUTLINE_VAL_MIN)
    )


def filled_outline(
    outline: np.ndarray, zone: tuple[float, float, float]
) -> np.ndarray | None:
    """Region enclosed by the drawn crepe outline.

    The robot clips its outline to the analysis zone, so the zone edge is a
    wall too. That splits the zone into pieces: the crepe, the bare plate
    around it and small slivers. The crepe is the sizeable piece with the
    least of its border on the zone edge (the bare plate hugs the edge).
    """
    cx, cy, r = zone
    shape = outline.shape
    kernel = np.ones((WALL_KERNEL, WALL_KERNEL), np.uint8)
    wall = outline | ~disc(shape, cx, cy, r)
    thick = cv2.dilate(wall.astype(np.uint8) * 255, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        255 - thick, connectivity=4
    )
    near_edge = ~disc(shape, cx, cy, r - EDGE_BAND_PX)
    min_area = MIN_REGION_FRAC * np.pi * r**2
    scored = []  # (edge fraction, area, piece)
    for i in range(1, n):  # label 0 is the wall
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        piece = (labels == i).astype(np.uint8)
        border = piece.astype(bool) & ~(cv2.erode(piece, kernel) > 0)
        frac = (border & near_edge).sum() / max(border.sum(), 1)
        scored.append((frac, area, piece))
    # Slivers between a jagged outline and the fitted circle also stay off
    # the edge, so among the pieces that do, the crepe is the largest.
    inner = [s for s in scored if s[0] <= MAX_EDGE_FRAC]
    if not inner:
        return None
    best = max(inner, key=lambda s: s[1])[2]
    return cv2.dilate(best, kernel) > 0


def hull_region(mask: np.ndarray) -> np.ndarray | None:
    """Filled convex hull of a mask (the ideal fold is a convex sector)."""
    pts = cv2.findNonZero(mask.astype(np.uint8))
    if pts is None or len(pts) < MIN_ZONE_POINTS:
        return None
    out = np.zeros(mask.shape, np.uint8)
    cv2.fillConvexPoly(out, cv2.convexHull(pts), 1)
    return out > 0


def circularity(region: np.ndarray) -> float | None:
    contours, _ = cv2.findContours(
        region.astype(np.uint8) * 255,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        return None
    big = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(big, closed=True)
    if perimeter == 0:
        return None
    return float(4 * np.pi * cv2.contourArea(big) / perimeter**2)


def lightness(img: np.ndarray, region: np.ndarray) -> tuple[float, float]:
    """Mean and std of CIE L* (0-100) inside a region, outline excluded."""
    kernel = np.ones((COLOUR_ERODE, COLOUR_ERODE), np.uint8)
    inner = cv2.erode(region.astype(np.uint8), kernel) > 0
    if not inner.any():
        return float("nan"), float("nan")
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lum = lab[..., 0][inner].astype(float) * 100 / 255
    return float(lum.mean()), float(lum.std())


def bare_plate(
    img: np.ndarray, hsv: np.ndarray, crepe: np.ndarray, zone_r: float
) -> np.ndarray:
    """Plate showing through inside the crepe outline: dark pixels plus the
    robot's red hole marks (red paint hides how dark the plate is). The
    robot only marks some gaps, so this catches the rest."""
    kernel = np.ones((COLOUR_ERODE, COLOUR_ERODE), np.uint8)
    inner = cv2.erode(crepe.astype(np.uint8), kernel) > 0
    lum = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].astype(float)
    dark = lum * 100 / 255 < BARE_L_MAX
    bare = ((dark | colour_mask(hsv, "red")) & inner).astype(np.uint8)
    speck = np.ones((SPECK_KERNEL, SPECK_KERNEL), np.uint8)
    bare = cv2.morphologyEx(bare, cv2.MORPH_OPEN, speck)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bare)
    min_area = MIN_GAP_FRAC * np.pi * zone_r**2
    big = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    return np.isin(labels, big)


def prepare(img: np.ndarray) -> tuple[np.ndarray, Any, np.ndarray] | None:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    zone = zone_of(hsv)
    if zone is None:
        return None
    inside = disc(hsv.shape[:2], zone[0], zone[1], zone[2] * ZONE_MARGIN)
    return hsv, zone, inside


def spread_metrics(img: np.ndarray) -> dict[str, float | None]:
    """Scores for a spread-check overlay (BGR image)."""
    prep = prepare(img)
    if prep is None:
        return {}
    hsv, zone, inside = prep
    zx, zy, zr = zone
    zone_area = np.pi * zr**2
    out: dict[str, float | None] = {"zone_r_px": zr}

    crepe = filled_outline(outline_mask(hsv) & inside, zone)
    if crepe is not None:
        area = float(crepe.sum())
        ys, xs = np.nonzero(crepe)
        out["area_rel"] = area / zone_area
        out["diameter_rel"] = float(np.sqrt(area / zone_area))
        out["offset_rel"] = float(np.hypot(xs.mean() - zx, ys.mean() - zy) / zr)
        out["circularity"] = circularity(crepe)
        holes = colour_mask(hsv, "red") & crepe
        out["hole_frac"] = float(holes.sum()) / area if area else None
        bare = bare_plate(img, hsv, crepe, zr)
        out["uncovered_frac"] = float(bare.sum()) / area if area else None
        out["l_mean"], out["l_std"] = lightness(img, crepe & ~holes & ~bare)

    circle = fit_circle(colour_mask(hsv, "cyan") & inside)
    if circle is not None:
        cx, cy, r = circle
        out["circle_diameter_rel"] = r / zr
        out["circle_offset_rel"] = float(np.hypot(cx - zx, cy - zy) / zr)

    residue = colour_mask(hsv, "orange") & inside
    out["residue_frac"] = float(residue.sum()) / zone_area
    return out


def fold_regions(
    hsv: np.ndarray, zone: tuple[float, float, float], inside: np.ndarray
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Crepe region and the robot's ideal fold region on a fold overlay."""
    ideal_line = colour_mask(hsv, "magenta") & inside
    outline = outline_mask(hsv) & inside
    # Where the ideal fold line crosses the crepe outline it hides it; the
    # magenta pixels right next to the outline bridge those gaps.
    near = cv2.dilate(
        outline.astype(np.uint8), np.ones((BRIDGE_PX, BRIDGE_PX), np.uint8)
    )
    crepe = filled_outline(outline | (ideal_line & (near > 0)), zone)
    return crepe, hull_region(ideal_line)


def fold_metrics(img: np.ndarray) -> dict[str, float | None]:
    """Scores for a fold-check overlay: shape match against the robot's
    ideal fold, size, and browning."""
    prep = prepare(img)
    if prep is None:
        return {}
    hsv, zone, inside = prep
    zr = zone[2]
    out: dict[str, float | None] = {}
    crepe, ideal = fold_regions(hsv, zone, inside)
    if crepe is not None:
        out["area_rel"] = float(crepe.sum()) / (np.pi * zr**2)
        out["l_mean"], out["l_std"] = lightness(img, crepe)
    if crepe is not None and ideal is not None:
        union = float((crepe | ideal).sum())
        out["iou"] = float((crepe & ideal).sum()) / union if union else None
    return out


METRICS = {SPREAD_CHECK: spread_metrics} | {
    name: fold_metrics for name in FOLD_CHECKS
}


def run_metrics(run: dict[str, Any], media_dir: Path) -> dict[str, Any]:
    """`vision.<check>.<metric>` columns for one run, from the last attempt
    of each check (same rule as the run table)."""
    row: dict[str, Any] = {"id": run["id"]}
    last: dict[str, dict[str, Any]] = {}
    for vc in run.get("visionChecks") or []:
        if vc.get("phase") in METRICS:
            last[vc["phase"]] = vc
    for name, vc in last.items():
        path = media_dir / str(run["id"]) / (vc.get("overlay") or "")
        if not vc.get("overlay") or not path.is_file():
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue
        for key, value in METRICS[name](img).items():
            row[f"{PREFIX}{name}.{key}"] = value
    # A good second fold halves the folded area; about 1 means it did nothing
    first = row.get(f"{PREFIX}fold1.area_rel")
    second = row.get(f"{PREFIX}fold2.area_rel")
    if first and second is not None:
        row[f"{PREFIX}fold2.area_ratio"] = second / first
    return row


def build_vision_table(data_dir: Path) -> pd.DataFrame:
    """Score every run that has downloaded overlays; write vision.parquet."""
    media_dir = data_dir / "raw" / "media"
    rows = []
    for p in (data_dir / "raw" / "runs").glob("*.json"):
        run = json.loads(p.read_text(encoding="utf-8"))
        if not (media_dir / str(run["id"])).is_dir():
            continue
        row = run_metrics(run, media_dir)
        if len(row) > 1:
            rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("id").reset_index(drop=True)
    df.to_csv(data_dir / "vision.csv", index=False)
    df.to_parquet(data_dir / "vision.parquet", index=False)
    print(f"{len(df)} runs scored -> {data_dir / 'vision.parquet'}")
    return df
