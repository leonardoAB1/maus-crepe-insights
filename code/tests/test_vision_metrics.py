"""Offline tests for overlay scoring, on synthetic overlays drawn like the
robot draws them (no real images)."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from maus_export.vision_metrics import (
    build_vision_table,
    fold_metrics,
    spread_metrics,
)

SIZE = 900
CENTRE = (450, 450)
ZONE_R = 396
# BGR annotation colours as drawn on the robot overlays
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
CYAN = (255, 255, 0)
RED = (0, 0, 255)
ORANGE = (0, 165, 255)
MAGENTA = (255, 0, 255)
PLATE = (25, 25, 30)
BATTER = (190, 215, 225)
TOL = 0.03
# Robot lines sit side by side rather than exactly on top of each other
FIT_GAP = 4
IDEAL_GAP = 8
# The synthetic ideal fold is drawn slightly larger than the crepe
GOOD_IOU = 0.8


def blank() -> np.ndarray:
    img = np.zeros((SIZE, SIZE, 3), np.uint8)
    cv2.circle(img, CENTRE, ZONE_R + 20, PLATE, -1)
    # Legend swatches in the corner, like the real overlays
    for i, colour in enumerate((BLUE, GREEN, ORANGE, RED, CYAN)):
        cv2.rectangle(img, (8, 832 + 12 * i), (16, 840 + 12 * i), colour, -1)
    return img


def jpeg(img: np.ndarray) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def spread_overlay(
    r: int, offset: int = 0, hole_r: int = 0, residue: bool = False
) -> np.ndarray:
    img = blank()
    centre = (CENTRE[0] + offset, CENTRE[1])
    cv2.circle(img, centre, r, BATTER, -1)
    if hole_r:
        cv2.circle(img, centre, hole_r, RED, -1)
    if residue:
        cv2.rectangle(img, (700, 700), (730, 730), ORANGE, -1)
    cv2.circle(img, centre, r, GREEN, 2)
    # Fitted circle drawn just outside the outline, as on real overlays
    cv2.circle(img, centre, r + FIT_GAP, CYAN, 2)
    cv2.circle(img, CENTRE, ZONE_R, BLUE, 2)
    return jpeg(img)


def test_spread_size_and_centre() -> None:
    r = 280
    m = spread_metrics(spread_overlay(r))
    assert m["zone_r_px"] == pytest.approx(ZONE_R, rel=TOL)
    assert m["diameter_rel"] == pytest.approx(r / ZONE_R, rel=TOL)
    assert m["circle_diameter_rel"] == pytest.approx(r / ZONE_R, rel=TOL)
    assert m["offset_rel"] < TOL
    assert m["hole_frac"] == pytest.approx(0, abs=TOL)
    assert m["residue_frac"] == pytest.approx(0, abs=TOL)


def test_spread_hole_offset_and_residue() -> None:
    r, hole_r, offset = 250, 100, 60
    m = spread_metrics(spread_overlay(r, offset, hole_r, residue=True))
    assert m["hole_frac"] == pytest.approx((hole_r / r) ** 2, rel=0.1)
    assert m["offset_rel"] == pytest.approx(offset / ZONE_R, rel=0.1)
    assert m["residue_frac"] > 0


def test_spread_outline_clipped_by_zone() -> None:
    # Crepe running off the zone: the outline stops at the zone edge
    img = blank()
    centre = (CENTRE[0] + 250, CENTRE[1])
    cv2.circle(img, centre, 250, BATTER, -1)
    clip = np.zeros(img.shape[:2], np.uint8)
    cv2.circle(clip, CENTRE, ZONE_R - 3, 1, -1)
    line = np.zeros_like(clip)
    cv2.circle(line, centre, 250, 1, 2)
    img[(line & clip) > 0] = GREEN
    cv2.circle(img, CENTRE, ZONE_R, BLUE, 2)
    m = spread_metrics(jpeg(img))
    assert 0 < m["area_rel"] < (250 / ZONE_R) ** 2


def fold_overlay(shift: int) -> np.ndarray:
    """Half-disc ideal fold; the crepe is the same shape shifted."""
    img = blank()
    r = 250
    crepe = (CENTRE[0] + shift, CENTRE[1])
    cv2.ellipse(img, crepe, (r, r), 0, 180, 360, BATTER, -1)
    cv2.ellipse(img, crepe, (r, r), 0, 180, 360, GREEN, 2)
    cv2.line(img, (crepe[0] - r, crepe[1]), (crepe[0] + r, crepe[1]), GREEN, 2)
    ri = r + 2 * IDEAL_GAP
    top = CENTRE[1] + IDEAL_GAP
    cv2.ellipse(img, (CENTRE[0], top), (ri, ri), 0, 180, 360, MAGENTA, 2)
    cv2.line(img, (CENTRE[0] - ri, top), (CENTRE[0] + ri, top), MAGENTA, 2)
    cv2.circle(img, CENTRE, ZONE_R, BLUE, 2)
    return jpeg(img)


def test_fold_iou_drops_with_misalignment() -> None:
    good = fold_metrics(fold_overlay(0))
    bad = fold_metrics(fold_overlay(150))
    assert good["iou"] > GOOD_IOU
    assert bad["iou"] < good["iou"] - 0.2
    assert good["area_rel"] == pytest.approx(0.5 * (250 / ZONE_R) ** 2, rel=0.1)


def test_no_zone_gives_no_metrics() -> None:
    assert spread_metrics(np.zeros((SIZE, SIZE, 3), np.uint8)) == {}


def test_build_vision_table(tmp_path: Path) -> None:
    runs = tmp_path / "raw" / "runs"
    media = tmp_path / "raw" / "media" / "7"
    runs.mkdir(parents=True)
    media.mkdir(parents=True)
    cv2.imwrite(str(media / "a.jpg"), spread_overlay(300, hole_r=50))
    cv2.imwrite(str(media / "b.jpg"), spread_overlay(200))
    run = {
        "id": 7,
        "visionChecks": [
            {"phase": "spread", "ok": False, "overlay": "b.jpg"},
            {"phase": "spread", "ok": True, "overlay": "a.jpg"},
        ],
    }
    (runs / "7.json").write_text(json.dumps(run), encoding="utf-8")
    (runs / "8.json").write_text(json.dumps({"id": 8}), encoding="utf-8")
    df = build_vision_table(tmp_path)
    assert list(df["id"]) == [7]
    # Last attempt wins, like the run table
    assert df.at[0, "vision.spread.diameter_rel"] == pytest.approx(
        300 / ZONE_R, rel=TOL
    )
    assert (tmp_path / "vision.parquet").is_file()


def test_largest_enclosed_piece_wins_over_slivers() -> None:
    # A small closed loop off the crepe must not be taken as the crepe
    r = 250
    img = blank()
    cv2.circle(img, CENTRE, r, BATTER, -1)
    cv2.circle(img, CENTRE, r, GREEN, 2)
    cv2.circle(img, (CENTRE[0], CENTRE[1] - 320), 40, GREEN, 2)
    cv2.circle(img, CENTRE, ZONE_R, BLUE, 2)
    m = spread_metrics(jpeg(img))
    assert m["diameter_rel"] == pytest.approx(r / ZONE_R, rel=TOL)


def test_tiny_crepe_is_not_mistaken_for_the_plate() -> None:
    img = blank()
    cv2.circle(img, CENTRE, 30, BATTER, -1)
    cv2.circle(img, CENTRE, 30, GREEN, 2)
    cv2.circle(img, CENTRE, ZONE_R, BLUE, 2)
    m = fold_metrics(jpeg(img))
    assert m["area_rel"] < (40 / ZONE_R) ** 2


def disc_overlay(start_deg: int) -> np.ndarray:
    """Folded crepe as a disc sector from start_deg to 360 degrees."""
    img = blank()
    r = 250
    cv2.ellipse(img, CENTRE, (r, r), 0, start_deg, 360, BATTER, -1)
    cv2.ellipse(img, CENTRE, (r, r), 0, start_deg, 360, GREEN, 2)
    end = (CENTRE[0] + r, CENTRE[1])
    cv2.line(img, CENTRE, end, GREEN, 2)
    rad = np.deg2rad(start_deg)
    start = (int(CENTRE[0] + r * np.cos(rad)), int(CENTRE[1] + r * np.sin(rad)))
    cv2.line(img, CENTRE, start, GREEN, 2)
    cv2.circle(img, CENTRE, ZONE_R, BLUE, 2)
    return jpeg(img)


def test_fold2_area_ratio(tmp_path: Path) -> None:
    media = tmp_path / "raw" / "media" / "9"
    runs = tmp_path / "raw" / "runs"
    media.mkdir(parents=True)
    runs.mkdir(parents=True)
    cv2.imwrite(str(media / "f1.jpg"), disc_overlay(180))  # half
    cv2.imwrite(str(media / "f2.jpg"), disc_overlay(270))  # quarter
    run = {
        "id": 9,
        "visionChecks": [
            {"phase": "fold1", "ok": True, "overlay": "f1.jpg"},
            {"phase": "fold2", "ok": True, "overlay": "f2.jpg"},
        ],
    }
    (runs / "9.json").write_text(json.dumps(run), encoding="utf-8")
    df = build_vision_table(tmp_path)
    assert df.at[0, "vision.fold2.area_ratio"] == pytest.approx(0.5, rel=0.1)


def test_uncovered_counts_unmarked_gaps_but_not_bubbles() -> None:
    r, gap_r = 280, 90
    img = blank()
    cv2.circle(img, CENTRE, r, BATTER, -1)
    # Bare plate the robot did not paint red, plus bubble specks
    cv2.circle(img, (CENTRE[0] + 100, CENTRE[1]), gap_r, PLATE, -1)
    for dx in range(-200, -60, 25):
        cv2.circle(img, (CENTRE[0] + dx, CENTRE[1] - 120), 2, PLATE, -1)
    cv2.circle(img, CENTRE, r, GREEN, 2)
    cv2.circle(img, CENTRE, r + FIT_GAP, CYAN, 2)
    cv2.circle(img, CENTRE, ZONE_R, BLUE, 2)
    m = spread_metrics(jpeg(img))
    assert m["hole_frac"] == pytest.approx(0, abs=TOL)
    assert m["uncovered_frac"] == pytest.approx((gap_r / r) ** 2, rel=0.1)


def test_uncovered_includes_red_marked_holes() -> None:
    r, hole_r = 250, 100
    m = spread_metrics(spread_overlay(r, hole_r=hole_r))
    assert m["uncovered_frac"] == pytest.approx((hole_r / r) ** 2, rel=0.1)
