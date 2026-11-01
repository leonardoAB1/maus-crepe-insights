"""Crêpe position, axis and fold edge from the crêpe student mask.

The student mask is in the raw camera view. It is warped with the robot's
image correction into the corrected top view, the same view as the robot's
check overlays, and measured against the calibrated hotplate circle.

Conventions follow ``overlay_pose``: positions in zone (plate) radii from the
zone centre, x to the right and y down; angles in degrees in image
coordinates, ``atan2(dy, dx)``, so -90 points up.

- ``dx_rel``, ``dy_rel``: centroid of the crêpe
- ``axis_deg``: principal axis of the crêpe, +-90
- ``diameter_rel``: equivalent diameter in zone diameters
- ``edge_deg``: the longest straight edge (the fold line), 0 to 180;
  ``edge_mid_dx_rel``, ``edge_mid_dy_rel`` its middle, ``edge_len_rel`` its
  length in zone radii
- ``heading_deg``: from the middle of that edge to the centroid, +-180;
  the side of the fold line the crêpe lies on
- ``corner_dx_rel``, ``corner_dy_rel``, ``corner_deg``: where the two
  straight edges of a quarter meet, and the angle between them
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from .crepe_student import CrepeStudent, largest_part, read_frame
from .fold_frames import CALIBRATION_PATH, TARGET_SOURCE_SIZE, video_path
from .fold_state import QUARTER_STATE
from .overlay_pose import HALF_TURN, region_pose, wrap

DEFAULT_TOP_SIZE = (900, 900)
# A hull segment is a straight edge when it is at least this share of the
# crêpe's equivalent diameter: a half disc's chord is 1.41 of it, a
# quarter's two edges 1.0 each, the facets of a round outline far below.
EDGE_MIN_SHARE = 0.6
HULL_EPSILON_SHARE = 0.01
# A quarter's corner: two straight edges, each at least this share of the
# equivalent diameter (shorter than a fold line when the tip is rounded),
# meeting near their ends at an angle within CORNER_TOLERANCE_DEG of 90.
CORNER_EDGE_MIN_SHARE = 0.4
CORNER_CANDIDATES = 4
CORNER_TOLERANCE_DEG = 30.0
CORNER_GAP_SHARE = 0.25
RIGHT_ANGLE = 90.0
PARALLEL_EPS = 1e-9
OUTPUT = Path("fold_segmentation") / "student" / "pose"


@dataclass(frozen=True)
class Zone:
    """The calibrated hotplate circle in the corrected top view."""

    cx: float
    cy: float
    r: float

    def as_tuple(self) -> tuple[float, float, float]:
        return self.cx, self.cy, self.r


@dataclass
class CrepePose:
    """Geometry of one crêpe in the corrected top view."""

    dx_rel: float
    dy_rel: float
    axis_deg: float
    diameter_rel: float
    area_rel: float
    edge_count: int
    edge_deg: float | None = None
    edge_mid_dx_rel: float | None = None
    edge_mid_dy_rel: float | None = None
    edge_len_rel: float | None = None
    heading_deg: float | None = None
    corner_dx_rel: float | None = None
    corner_dy_rel: float | None = None
    corner_deg: float | None = None


POSE_FIELDS = tuple(f.name for f in fields(CrepePose))


def load_calibration(path: Path = CALIBRATION_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def robot_zone(machine: str | None, calibration: dict[str, Any]) -> Zone:
    """Hotplate circle of a robot, or a centred one for unknown robots."""
    roi = (calibration.get("hotplate_roi") or {}).get(machine or "")
    if roi:
        (cx, cy), r = roi["center_px"], roi["radius_px"]
        return Zone(float(cx), float(cy), float(r))
    width, height = calibration.get("image_size_px", DEFAULT_TOP_SIZE)
    share = 0.44
    return Zone(width / 2, height / 2, min(width, height) * share)


def top_view(
    mask: np.ndarray, machine: str | None, calibration: dict[str, Any]
) -> np.ndarray:
    """Warp a raw-view mask into the corrected top view of its robot."""
    source = cv2.resize(
        mask.astype(np.uint8),
        TARGET_SOURCE_SIZE,
        interpolation=cv2.INTER_NEAREST,
    )
    correction = (calibration.get("robot_corrections") or {}).get(machine or "")
    size = tuple(calibration.get("image_size_px", DEFAULT_TOP_SIZE))
    if not correction:
        return cv2.resize(source, size, interpolation=cv2.INTER_NEAREST) > 0
    matrix = np.asarray(correction["matrix_mean"], dtype=np.float64)
    size = tuple(correction.get("target_size_px", size))
    warped = cv2.warpPerspective(source, matrix, size, flags=cv2.INTER_NEAREST)
    return warped > 0


def _segments(contour: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    poly = cv2.approxPolyDP(hull, HULL_EPSILON_SHARE * perimeter, True)
    points = poly[:, 0, :].astype(float)
    return [
        (points[i], points[(i + 1) % len(points)]) for i in range(len(points))
    ]


def straight_edges(
    region: np.ndarray, diameter_px: float, min_share: float = EDGE_MIN_SHARE
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Long straight hull segments, longest first."""
    contours, _ = cv2.findContours(
        region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return []
    segments = _segments(max(contours, key=cv2.contourArea))
    long = [
        (a, b)
        for a, b in segments
        if np.hypot(*(b - a)) >= min_share * diameter_px
    ]
    return sorted(long, key=lambda s: -np.hypot(*(s[1] - s[0])))


def line_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Undirected angle of a segment, 0 to 180."""
    return float(np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0])) % HALF_TURN)


def intersection(
    first: tuple[np.ndarray, np.ndarray],
    second: tuple[np.ndarray, np.ndarray],
) -> np.ndarray | None:
    (a, b), (c, d) = first, second
    r, s = b - a, d - c
    denom = r[0] * s[1] - r[1] * s[0]
    if abs(denom) < PARALLEL_EPS:
        return None
    t = ((c - a)[0] * s[1] - (c - a)[1] * s[0]) / denom
    return a + t * r


def corner(
    edges: list[tuple[np.ndarray, np.ndarray]], diameter_px: float
) -> tuple[np.ndarray, float] | None:
    """The quarter corner: the edge pair closest to a right angle."""
    best: tuple[float, np.ndarray, float] | None = None
    candidates = edges[:CORNER_CANDIDATES]
    for i, first in enumerate(candidates):
        for second in candidates[i + 1 :]:
            angle = abs(
                wrap(line_angle(*first) - line_angle(*second), HALF_TURN)
            )
            miss = abs(angle - RIGHT_ANGLE)
            point = intersection(first, second)
            if point is None or miss > CORNER_TOLERANCE_DEG:
                continue
            gap = max(
                min(np.hypot(*(point - end)) for end in edge)
                for edge in (first, second)
            )
            if gap > CORNER_GAP_SHARE * diameter_px:
                continue
            if best is None or miss < best[0]:
                best = (miss, point, angle)
    return None if best is None else (best[1], best[2])


def crepe_pose(
    top_mask: np.ndarray, zone: Zone, quarter: bool | None = None
) -> CrepePose | None:
    """Pose of the largest crêpe part in a top-view mask.

    ``quarter`` False skips the corner search (a half disc's chord and arc
    can look like a corner); None searches whatever the state.
    """
    region = largest_part(top_mask)
    area = float(region.sum())
    if area == 0:
        return None
    base = region_pose(region, zone.as_tuple())
    if base is None:
        return None
    diameter_px = 2 * np.sqrt(area / np.pi)
    cx = zone.cx + base["dx_rel"] * zone.r
    cy = zone.cy + base["dy_rel"] * zone.r
    pose = CrepePose(
        dx_rel=base["dx_rel"],
        dy_rel=base["dy_rel"],
        axis_deg=base["axis_deg"],
        diameter_rel=diameter_px / (2 * zone.r),
        area_rel=area / (np.pi * zone.r**2),
        edge_count=0,
    )
    edges = straight_edges(region, diameter_px)
    pose.edge_count = len(edges)
    if not edges:
        return pose
    a, b = edges[0]
    middle = (a + b) / 2
    pose.edge_deg = line_angle(a, b)
    pose.edge_mid_dx_rel = float((middle[0] - zone.cx) / zone.r)
    pose.edge_mid_dy_rel = float((middle[1] - zone.cy) / zone.r)
    pose.edge_len_rel = float(np.hypot(*(b - a)) / zone.r)
    pose.heading_deg = float(
        np.degrees(np.arctan2(cy - middle[1], cx - middle[0]))
    )
    found = (
        None
        if quarter is False
        else corner(
            straight_edges(region, diameter_px, CORNER_EDGE_MIN_SHARE),
            diameter_px,
        )
    )
    if found is not None:
        point, angle = found
        pose.corner_dx_rel = float((point[0] - zone.cx) / zone.r)
        pose.corner_dy_rel = float((point[1] - zone.cy) / zone.r)
        pose.corner_deg = float(angle)
        pose.heading_deg = float(
            np.degrees(np.arctan2(cy - point[1], cx - point[0]))
        )
    return pose


def pose_from_mask(
    mask: np.ndarray,
    machine: str | None,
    calibration: dict[str, Any],
    quarter: bool | None = None,
) -> CrepePose | None:
    """Raw-view student mask -> pose in the robot's top view."""
    zone = robot_zone(machine, calibration)
    return crepe_pose(top_view(mask, machine, calibration), zone, quarter)


def pose_row(pose: CrepePose | None) -> dict[str, Any]:
    return asdict(pose) if pose is not None else {}


def check_frame_poses(
    data_dir: Path, student: CrepeStudent | None = None
) -> pd.DataFrame:
    """Student pose at every check frame of ``fold-frames``."""
    student = student or CrepeStudent.load()
    calibration = load_calibration()
    frames = pd.read_parquet(data_dir / "fold_segmentation" / "frames.parquet")
    rows = []
    for row in frames.itertuples(index=False):
        video = video_path(data_dir, int(row.run_id))
        if video is None:
            continue
        frame = read_frame(video, int(row.decoded_frame_index))
        pose = pose_from_mask(
            student.mask(frame),
            row.machine_name,
            calibration,
            quarter=row.expected_state == QUARTER_STATE,
        )
        rows.append(
            {
                "run_id": int(row.run_id),
                "check": row.check_name,
                "log_time": row.log_time,
                "robot": row.machine_name,
                "check_ok": bool(row.check_ok),
                **pose_row(pose),
            }
        )
    return pd.DataFrame(rows)


def compare_with_overlays(
    poses: pd.DataFrame, overlay: pd.DataFrame
) -> pd.DataFrame:
    """Join student poses with overlay pose labels, attempt by attempt.

    Attempts of one check are numbered from 1 in time order, as in
    ``overlay_pose``.
    """
    ours = poses.sort_values("log_time").copy()
    ours["attempt"] = ours.groupby(["run_id", "check"]).cumcount() + 1
    joined = ours.merge(
        overlay,
        on=["run_id", "check", "attempt"],
        suffixes=("", "_robot"),
        how="inner",
    ).dropna(subset=["dx_rel", "dx_rel_robot"])
    joined["centre_err_rel"] = np.hypot(
        joined["dx_rel"] - joined["dx_rel_robot"],
        joined["dy_rel"] - joined["dy_rel_robot"],
    )
    joined["axis_err_deg"] = [
        abs(wrap(a - b, HALF_TURN))
        for a, b in zip(
            joined["axis_deg"], joined["axis_deg_robot"], strict=True
        )
    ]
    return joined


def summary(joined: pd.DataFrame) -> pd.DataFrame:
    """Median and 90th percentile errors per check."""
    groups = joined.groupby("check")
    return pd.DataFrame(
        {
            "frames": groups.size(),
            "centre_err_median": groups["centre_err_rel"].median(),
            "centre_err_p90": groups["centre_err_rel"].quantile(0.9),
            "axis_err_median": groups["axis_err_deg"].median(),
            "axis_err_p90": groups["axis_err_deg"].quantile(0.9),
        }
    ).reset_index()


def evaluate(data_dir: Path) -> pd.DataFrame:
    """Student pose at the check frames against the overlay pose labels.

    Needs ``fold-frames`` and ``overlay-pose`` first. Writes
    ``fold_segmentation/student/pose/{poses,joined,summary}.csv``.
    """
    out = data_dir / OUTPUT
    out.mkdir(parents=True, exist_ok=True)
    poses = check_frame_poses(data_dir)
    poses.to_csv(out / "poses.csv", index=False)
    overlay = pd.read_parquet(data_dir / "training" / "overlay_pose.parquet")
    joined = compare_with_overlays(poses, overlay)
    joined.to_csv(out / "joined.csv", index=False)
    table = summary(joined)
    table.to_csv(out / "summary.csv", index=False)
    return table


POSE_COLOUR = (0, 200, 0)
ZONE_COLOUR = (235, 200, 70)
EDGE_COLOUR = (30, 210, 255)
HEADING_COLOUR = (240, 90, 240)
LINE_WIDTH = 4
DOT_RADIUS = 9
ARROW_SHARE = 0.5
MASK_ALPHA = 0.4


def draw_pose(
    top_image: np.ndarray,
    top_mask: np.ndarray,
    pose: CrepePose | None,
    zone: Zone,
) -> np.ndarray:
    """Top-view frame with the mask, zone, centre, fold edge and heading."""
    out = top_image.copy()
    tint = out.copy()
    tint[top_mask] = POSE_COLOUR
    out = cv2.addWeighted(out, 1 - MASK_ALPHA, tint, MASK_ALPHA, 0)
    centre = (round(zone.cx), round(zone.cy))
    cv2.circle(out, centre, round(zone.r), ZONE_COLOUR, LINE_WIDTH)
    if pose is None:
        return out
    cx = zone.cx + pose.dx_rel * zone.r
    cy = zone.cy + pose.dy_rel * zone.r
    cv2.circle(out, (round(cx), round(cy)), DOT_RADIUS, (255, 255, 255), -1)
    if pose.edge_deg is not None:
        mx = zone.cx + pose.edge_mid_dx_rel * zone.r
        my = zone.cy + pose.edge_mid_dy_rel * zone.r
        half = pose.edge_len_rel * zone.r / 2
        angle = np.radians(pose.edge_deg)
        dx, dy = np.cos(angle) * half, np.sin(angle) * half
        cv2.line(
            out,
            (round(mx - dx), round(my - dy)),
            (round(mx + dx), round(my + dy)),
            EDGE_COLOUR,
            LINE_WIDTH,
        )
    if pose.heading_deg is not None:
        length = ARROW_SHARE * pose.diameter_rel * 2 * zone.r
        angle = np.radians(pose.heading_deg)
        tip = (
            round(cx + np.cos(angle) * length),
            round(cy + np.sin(angle) * length),
        )
        cv2.arrowedLine(
            out, (round(cx), round(cy)), tip, HEADING_COLOUR, LINE_WIDTH
        )
    if pose.corner_dx_rel is not None:
        corner_px = (
            round(zone.cx + pose.corner_dx_rel * zone.r),
            round(zone.cy + pose.corner_dy_rel * zone.r),
        )
        cv2.circle(out, corner_px, DOT_RADIUS, HEADING_COLOUR, -1)
    return out


EXAMPLE_CHECKS = ("spread", "fold1", "fold2")
EXAMPLE_MARGIN = 1.15
GOOD_AXIS_ERR_DEG = 5.0


def _crop_zone(image: np.ndarray, zone: Zone) -> np.ndarray:
    half = round(zone.r * EXAMPLE_MARGIN)
    x0, y0 = max(0, round(zone.cx) - half), max(0, round(zone.cy) - half)
    return image[y0 : round(zone.cy) + half, x0 : round(zone.cx) + half]


def pose_example(data_dir: Path, out: Path) -> Path:
    """Grid of check frames in the top view with the student pose drawn.

    One typical frame per check (median centre error) and the fold frame
    with the largest axis error. Needs ``evaluate`` first.
    """
    from maus_classify.figures import Example, example_grid

    from .fold_frames import correct_perspective

    joined = pd.read_csv(data_dir / OUTPUT / "joined.csv")
    picks = []
    for check in EXAMPLE_CHECKS:
        part = joined[joined["check"] == check].sort_values("centre_err_rel")
        picks.append(part.iloc[len(part) // 2])
    folds = joined[
        (joined["check"] != "spread")
        & joined["check_ok"].astype(bool)
        & joined["edge_deg"].notna()
    ]
    picks.append(folds.loc[folds["axis_err_deg"].idxmax()])
    frames = pd.read_parquet(data_dir / "fold_segmentation" / "frames.parquet")
    student = CrepeStudent.load()
    calibration = load_calibration()
    examples = []
    for pick in picks:
        row = (
            frames[
                (frames["run_id"] == pick["run_id"])
                & (frames["check_name"] == pick["check"])
            ]
            .sort_values("log_time")
            .iloc[int(pick["attempt"]) - 1]
        )
        video = video_path(data_dir, int(row["run_id"]))
        frame = read_frame(video, int(row["decoded_frame_index"]))
        robot = row["machine_name"]
        mask = student.mask(frame)
        zone = robot_zone(robot, calibration)
        top_mask = top_view(mask, robot, calibration)
        pose = crepe_pose(
            top_mask, zone, quarter=row["expected_state"] == QUARTER_STATE
        )
        image = draw_pose(
            correct_perspective(frame, robot, calibration),
            top_mask,
            pose,
            zone,
        )
        fold = pick["check"] != "spread"
        examples.append(
            Example(
                image=_crop_zone(image, zone),
                truth=f"{pick['check']}, axis {pick['axis_deg_robot']:.0f} deg",
                predicted=f"axis {pick['axis_deg']:.0f} deg",
                confidence=float(pick["centre_err_rel"]),
                robot=str(robot),
                score_name="centre err",
                agree=not fold or pick["axis_err_deg"] <= GOOD_AXIS_ERR_DEG,
                note=(
                    f"axis error {pick['axis_err_deg']:.1f} deg"
                    if fold
                    else "round crepe: axis has no meaning"
                ),
            )
        )
    return example_grid(
        examples,
        out,
        "Crêpe student: centre, fold edge and heading in the top view",
        "green mask, cyan circle = plate zone, white dot = centre, yellow = "
        "fold edge, magenta arrow = heading, magenta dot = quarter corner",
    )
