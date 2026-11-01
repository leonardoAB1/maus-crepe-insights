"""Crepe position and angle read back from the robot's check overlays.

Task 2 asks for the crepe centre and angle. No pose ground truth is stored,
but the robot draws its own measurement on every overlay:

- push_off: a yellow line (plate direction) and a magenta line (crepe
  direction), both drawn from the zone centre. Their difference is what the
  robot rotates away, so it is the robot's own orientation reading. The
  crepe outline (green) gives the position.
- spread, fold1, fold2: the crepe outline (green) and, on folds, the ideal
  fold shape (magenta). We take centroids and principal axes from them.

All positions are relative to the blue analysis zone: dx and dy in zone
radii from its centre, x to the right and y down. Angles are degrees in
image coordinates, atan2(dy, dx), so -90 points up. No millimetre scale is
published for the plate.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from . import vision_metrics as vm

PUSH_OFF = "push_off"
CHECKS = (vm.SPREAD_CHECK, *vm.FOLD_CHECKS, PUSH_OFF)
OUTPUT = Path("training") / "overlay_pose.parquet"

YELLOW_HUE = 30
# The direction lines are thin and anti-aliased, so they are less
# saturated than the filled annotation colours.
LINE_SAT_MIN = 120
LINE_VAL_MIN = 120
MIN_LINE_POINTS = 30
# Both lines start at the zone centre (the plate axis). Pixels this close to
# the centre carry no direction.
RAY_MIN_RADIUS = 0.1
RAY_TOL_DEG = 2.0
# A clean line puts most of its pixels in the peak; less means another
# object of the same colour competes with it.
MIN_LINE_SHARE = 0.5
HALF_TURN = 180.0
FULL_TURN = 360.0

ROTATE_LOG = re.compile(
    r"Aligning crepe with the serving plate: rotating (-?\d+(?:\.\d+)?) deg"
)
RESIDUAL_LOG = re.compile(r"residual (-?\d+(?:\.\d+)?) deg")


def wrap(angle: float, period: float = FULL_TURN) -> float:
    """Wrap an angle into [-period/2, period/2)."""
    half = period / 2
    return float((angle + half) % period - half)


def line_mask(hsv: np.ndarray, hue: int) -> np.ndarray:
    """Pixels of a thin line drawn in one hue."""
    dist = np.abs(hsv[..., 0].astype(int) - hue)
    dist = np.minimum(dist, HALF_TURN - dist)
    return (
        (dist <= vm.HUE_TOL)
        & (hsv[..., 1] >= LINE_SAT_MIN)
        & (hsv[..., 2] >= LINE_VAL_MIN)
    )


def ray(
    mask: np.ndarray, zone: tuple[float, float, float]
) -> tuple[float, float] | None:
    """Direction of a line drawn outward from the zone centre.

    Returns the dominant pixel angle around the centre and the share of
    line pixels within RAY_TOL_DEG of it. Stray pixels of a similar colour
    (browned crepe reads as yellow) spread over all angles, the line does not.
    """
    zx, zy, zr = zone
    ys, xs = np.nonzero(mask)
    far = np.hypot(xs - zx, ys - zy) >= RAY_MIN_RADIUS * zr
    if far.sum() < MIN_LINE_POINTS:
        return None
    angles = np.degrees(np.arctan2(ys[far] - zy, xs[far] - zx))
    hist, edges = np.histogram(
        angles, bins=int(FULL_TURN), range=(-HALF_TURN, HALF_TURN)
    )
    peak = edges[int(hist.argmax())] + 0.5
    near = np.abs([wrap(a - peak) for a in angles]) <= RAY_TOL_DEG
    offsets = [wrap(a - peak) for a in angles[near]]
    return wrap(peak + float(np.mean(offsets))), float(near.mean())


def relative(
    x: float, y: float, zone: tuple[float, float, float]
) -> tuple[float, float]:
    zx, zy, zr = zone
    return (x - zx) / zr, (y - zy) / zr


def push_off_pose(img: np.ndarray) -> dict[str, Any]:
    """Robot's crepe and plate directions, plus the crepe outline pose."""
    prep = vm.prepare(img)
    if prep is None:
        return {}
    hsv, zone, inside = prep
    out: dict[str, Any] = {"zone_r_px": zone[2]}
    crepe = vm.filled_outline(vm.outline_mask(hsv) & inside, zone)
    pose = region_pose(crepe, zone) if crepe is not None else None
    if pose:
        out |= pose
    plate_ray = ray(line_mask(hsv, YELLOW_HUE) & inside, zone)
    crepe_ray = ray(line_mask(hsv, vm.HUE["magenta"]) & inside, zone)
    if plate_ray is None or crepe_ray is None:
        return out
    share = min(plate_ray[1], crepe_ray[1])
    return out | {
        "line_share": share,
        "lines_ok": share >= MIN_LINE_SHARE,
        "crepe_deg": crepe_ray[0],
        "plate_deg": plate_ray[0],
        "align_deg": wrap(plate_ray[0] - crepe_ray[0]),
    }


def region_pose(
    region: np.ndarray, zone: tuple[float, float, float]
) -> dict[str, float] | None:
    """Centroid (zone units) and principal axis (degrees, +-90) of a mask."""
    m = cv2.moments(region.astype(np.uint8), binaryImage=True)
    if m["m00"] == 0:
        return None
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    dx, dy = relative(cx, cy, zone)
    axis = 0.5 * np.degrees(np.arctan2(2 * m["mu11"], m["mu20"] - m["mu02"]))
    return {"dx_rel": dx, "dy_rel": dy, "axis_deg": float(axis)}


def spread_pose(img: np.ndarray) -> dict[str, Any]:
    """Centre and axis of the crepe outline on a spread overlay."""
    prep = vm.prepare(img)
    if prep is None:
        return {}
    hsv, zone, inside = prep
    out: dict[str, Any] = {"zone_r_px": zone[2]}
    crepe = vm.filled_outline(vm.outline_mask(hsv) & inside, zone)
    pose = region_pose(crepe, zone) if crepe is not None else None
    if pose:
        out |= pose
    return out


def fold_pose(img: np.ndarray) -> dict[str, Any]:
    """Crepe pose on a fold overlay and its error against the ideal fold."""
    prep = vm.prepare(img)
    if prep is None:
        return {}
    hsv, zone, inside = prep
    out: dict[str, Any] = {"zone_r_px": zone[2]}
    crepe, ideal = vm.fold_regions(hsv, zone, inside)
    pose = region_pose(crepe, zone) if crepe is not None else None
    if pose:
        out |= pose
    target = region_pose(ideal, zone) if ideal is not None else None
    if target:
        out |= {f"ideal_{k}": v for k, v in target.items()}
    if pose and target:
        out["offset_err_rel"] = float(
            np.hypot(
                pose["dx_rel"] - target["dx_rel"],
                pose["dy_rel"] - target["dy_rel"],
            )
        )
        out["axis_err_deg"] = wrap(
            pose["axis_deg"] - target["axis_deg"], HALF_TURN
        )
    return out


POSE = {vm.SPREAD_CHECK: spread_pose, PUSH_OFF: push_off_pose} | {
    name: fold_pose for name in vm.FOLD_CHECKS
}


def _local(ts: str | None) -> datetime | None:
    return datetime.fromisoformat(ts) if ts else None


def alignment_logs(
    run: dict[str, Any], check_times: list[datetime | None]
) -> list[dict[str, float]]:
    """Rotation and residual log values, assigned to the last push_off
    attempt that started before each log line (both are robot local time)."""
    out: list[dict[str, float]] = [{} for _ in check_times]
    for line in run.get("logs") or []:
        msg = line.get("msg") or ""
        rotate = ROTATE_LOG.search(msg)
        residual = RESIDUAL_LOG.search(msg)
        if not (rotate or residual):
            continue
        at = _local(line.get("timestamp"))
        before = [i for i, t in enumerate(check_times) if t and at and t <= at]
        if not before:
            continue
        slot = out[before[-1]]
        if rotate:
            slot.setdefault("log_rotate_deg", float(rotate.group(1)))
        if residual:
            slot.setdefault("log_residual_deg", float(residual.group(1)))
    return out


def run_pose_rows(run: dict[str, Any], media_dir: Path) -> list[dict[str, Any]]:
    """One row per check attempt with a readable overlay."""
    checks = run.get("visionChecks") or []
    pushes = [vc for vc in checks if vc.get("phase") == PUSH_OFF]
    logs = alignment_logs(run, [_local(vc.get("timestamp")) for vc in pushes])
    seen: dict[str, int] = {}
    rows = []
    for vc in checks:
        name = vc.get("phase")
        seen[name] = seen.get(name, 0) + 1
        push_index = seen[name] - 1
        if name not in POSE or not vc.get("overlay"):
            continue
        path = media_dir / str(run["id"]) / vc["overlay"]
        img = cv2.imread(str(path)) if path.is_file() else None
        if img is None:
            continue
        row = {
            "run_id": run["id"],
            "check": name,
            "attempt": seen[name],
            "robot": run.get("machineName"),
            "ok": vc.get("ok"),
        }
        row |= POSE[name](img)
        if name == PUSH_OFF:
            row |= logs[push_index]
        rows.append(row)
    return rows


def build_pose_table(data_dir: Path) -> pd.DataFrame:
    """Read every downloaded overlay; write training/overlay_pose.parquet."""
    media_dir = data_dir / "raw" / "media"
    rows: list[dict[str, Any]] = []
    for path in sorted((data_dir / "raw" / "runs").glob("*.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        if (media_dir / str(run["id"])).is_dir():
            rows.extend(run_pose_rows(run, media_dir))
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["run_id", "check", "attempt"])
        df = df.reset_index(drop=True)
    out = data_dir / OUTPUT
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"{len(df)} check attempts -> {out}")
    return df
