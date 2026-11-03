"""Offline tests for crepe position and angle read from synthetic overlays
drawn like the robot draws them (no real images)."""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from maus_export.overlay_pose import (
    OUTPUT,
    alignment_logs,
    build_pose_table,
    fold_pose,
    push_off_pose,
    spread_pose,
    wrap,
)

SIZE = 900
CENTRE = (450, 450)
ZONE_R = 396
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
MAGENTA = (255, 0, 255)
PLATE = (25, 25, 30)
BATTER = (190, 215, 225)
# Browned crepe edge: yellowish but far less saturated than the line
BROWNED = (60, 150, 190)
LINE_LEN = 330
CREPE_R = 240
ANGLE_TOL = 1.0
POS_TOL = 0.02
AXIS_TOL = 3.0
IDEAL_GAP = 8


def blank() -> np.ndarray:
    img = np.zeros((SIZE, SIZE, 3), np.uint8)
    cv2.circle(img, CENTRE, ZONE_R + 20, PLATE, -1)
    for i, colour in enumerate((BLUE, GREEN, YELLOW, MAGENTA)):
        cv2.rectangle(img, (8, 832 + 12 * i), (16, 840 + 12 * i), colour, -1)
    return img


def finish(img: np.ndarray) -> np.ndarray:
    cv2.circle(img, CENTRE, ZONE_R, BLUE, 2)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def end_point(angle_deg: float, length: float = LINE_LEN) -> tuple[int, int]:
    a = np.radians(angle_deg)
    return (
        int(round(CENTRE[0] + length * np.cos(a))),
        int(round(CENTRE[1] + length * np.sin(a))),
    )


def sector(
    img: np.ndarray, start: float, sweep: float, colour, thickness: int
) -> None:
    cv2.ellipse(
        img,
        CENTRE,
        (CREPE_R, CREPE_R),
        0,
        start,
        start + sweep,
        colour,
        thickness,
    )
    if thickness > 0:
        cv2.line(img, CENTRE, end_point(start, CREPE_R), colour, thickness)
        cv2.line(
            img, CENTRE, end_point(start + sweep, CREPE_R), colour, thickness
        )


def push_off_overlay(crepe_deg: float, plate_deg: float) -> np.ndarray:
    img = blank()
    sector(img, crepe_deg - 45, 90, BATTER, -1)
    sector(img, crepe_deg - 45, 90, GREEN, 2)
    # Browned specks around the crepe that a loose yellow mask would catch
    for k in range(0, 360, 20):
        cv2.circle(img, end_point(k, 150), 6, BROWNED, -1)
    cv2.line(img, CENTRE, end_point(plate_deg), YELLOW, 2)
    cv2.line(img, CENTRE, end_point(crepe_deg), MAGENTA, 2)
    return finish(img)


def test_push_off_reads_both_directions() -> None:
    crepe, plate = -161.0, -127.0
    pose = push_off_pose(push_off_overlay(crepe, plate))
    assert pose["lines_ok"]
    assert pose["crepe_deg"] == pytest.approx(crepe, abs=ANGLE_TOL)
    assert pose["plate_deg"] == pytest.approx(plate, abs=ANGLE_TOL)
    assert pose["align_deg"] == pytest.approx(plate - crepe, abs=ANGLE_TOL)


def test_push_off_wraps_across_the_seam() -> None:
    pose = push_off_pose(push_off_overlay(170.0, -170.0))
    assert pose["align_deg"] == pytest.approx(20.0, abs=ANGLE_TOL)


def test_push_off_without_lines_keeps_the_crepe_pose() -> None:
    img = blank()
    sector(img, 0, 90, BATTER, -1)
    sector(img, 0, 90, GREEN, 2)
    pose = push_off_pose(finish(img))
    assert "align_deg" not in pose
    assert pose["dx_rel"] > 0 and pose["dy_rel"] > 0


def test_spread_centre_is_signed() -> None:
    dx, dy = 40, -60
    img = blank()
    centre = (CENTRE[0] + dx, CENTRE[1] + dy)
    cv2.circle(img, centre, CREPE_R, BATTER, -1)
    cv2.circle(img, centre, CREPE_R, GREEN, 2)
    pose = spread_pose(finish(img))
    assert pose["dx_rel"] == pytest.approx(dx / ZONE_R, abs=POS_TOL)
    assert pose["dy_rel"] == pytest.approx(dy / ZONE_R, abs=POS_TOL)


def half_disc(
    img: np.ndarray,
    centre: tuple[float, float],
    r: int,
    start: float,
    colour,
    thickness: int,
) -> None:
    """Half disc from `start` sweeping 180 degrees, outlined or filled."""
    pts = cv2.ellipse2Poly(
        (int(round(centre[0])), int(round(centre[1]))),
        (r, r),
        0,
        int(start),
        int(start) + 180,
        1,
    )
    if thickness < 0:
        cv2.fillPoly(img, [pts], colour)
    else:
        cv2.polylines(img, [pts], True, colour, thickness)


def fold_overlay(crepe_start: float, ideal_start: float) -> np.ndarray:
    """Half-disc crepe and a slightly larger ideal fold drawn just outside
    it, as on real overlays (the two lines never sit on top of each other)."""
    img = blank()
    half_disc(img, CENTRE, CREPE_R, crepe_start, BATTER, -1)
    half_disc(img, CENTRE, CREPE_R, crepe_start, GREEN, 2)
    bulge = np.radians(ideal_start + 90)
    ideal_centre = (
        CENTRE[0] - IDEAL_GAP * np.cos(bulge),
        CENTRE[1] - IDEAL_GAP * np.sin(bulge),
    )
    ideal_r = CREPE_R + 2 * IDEAL_GAP
    half_disc(img, ideal_centre, ideal_r, ideal_start, MAGENTA, 2)
    return finish(img)


def test_fold_on_target_has_no_error() -> None:
    pose = fold_pose(fold_overlay(0, 0))
    assert pose["axis_err_deg"] == pytest.approx(0, abs=AXIS_TOL)
    assert pose["offset_err_rel"] < POS_TOL


def test_fold_rotated_off_target() -> None:
    turn = 30
    pose = fold_pose(fold_overlay(turn, 0))
    assert abs(pose["axis_err_deg"]) == pytest.approx(turn, abs=AXIS_TOL)
    assert pose["offset_err_rel"] > POS_TOL


def test_wrap() -> None:
    assert wrap(190) == pytest.approx(-170)
    assert wrap(100, 180) == pytest.approx(-80)


def test_alignment_logs_follow_their_attempt() -> None:
    from datetime import datetime

    run = {
        "logs": [
            {"msg": "push_off: not ready", "timestamp": "2026-10-30T17:56:01"},
            {
                "msg": "Aligning crepe with the serving plate: rotating 49 deg",
                "timestamp": "2026-10-30T17:56:01.443",
            },
            {
                "msg": "check_push_off: still not aligned after rotating 30 "
                "deg (residual 12.5 deg) - pushing off anyway",
                "timestamp": "2026-10-30T17:56:05.100",
            },
        ]
    }
    times = [
        datetime.fromisoformat("2026-10-30T17:56:01"),
        datetime.fromisoformat("2026-10-30T17:56:04"),
    ]
    logs = alignment_logs(run, times)
    assert logs[0] == {"log_rotate_deg": 49.0}
    assert logs[1] == {"log_residual_deg": 12.5}


def test_build_pose_table(tmp_path: Path) -> None:
    run_id = 7
    media = tmp_path / "raw" / "media" / str(run_id)
    media.mkdir(parents=True)
    (tmp_path / "raw" / "runs").mkdir(parents=True)
    cv2.imwrite(str(media / "01.jpg"), push_off_overlay(-150.0, -120.0))
    cv2.imwrite(str(media / "02.jpg"), push_off_overlay(-103.0, -120.0))
    run = {
        "id": run_id,
        "machineName": "maus-test",
        "logs": [
            {
                "msg": "Aligning crepe with the serving plate: rotating 45 deg",
                "timestamp": "2026-10-30T17:56:01.500",
            }
        ],
        "visionChecks": [
            {
                "phase": "push_off",
                "ok": False,
                "overlay": "01.jpg",
                "timestamp": "2026-10-30T17:56:01",
            },
            {
                "phase": "push_off",
                "ok": True,
                "overlay": "02.jpg",
                "timestamp": "2026-10-30T17:56:04",
            },
            {"phase": "plate_clean", "ok": True, "overlay": "missing.jpg"},
        ],
    }
    (tmp_path / "raw" / "runs" / f"{run_id}.json").write_text(
        json.dumps(run), encoding="utf-8"
    )
    df = build_pose_table(tmp_path)
    assert (tmp_path / OUTPUT).is_file()
    assert list(df["attempt"]) == [1, 2]
    assert df.loc[0, "log_rotate_deg"] == 45.0
    assert pd.isna(df.loc[1, "log_rotate_deg"])
    assert df.loc[0, "align_deg"] == pytest.approx(30.0, abs=ANGLE_TOL)
