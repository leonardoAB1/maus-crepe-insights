"""Fold-state rules and batch segmentation."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import pandas as pd

from .fold_geometry import GeometryFeatures, mask_geometry
from .fold_mask import crepe_mask
from .fold_plate import PlateEllipse, detect_plate_ellipse, median_ellipse

EMPTY_STATE = "empty"
FLAT_STATE = "flat"
HALF_STATE = "half"
QUARTER_STATE = "quarter"
STATE_ORDER = (EMPTY_STATE, FLAT_STATE, HALF_STATE, QUARTER_STATE)
FLAT_PLATE_AREA_SHARE = 0.24
FLAT_REFERENCE_SHARE = 0.75
FLAT_NO_EDGE_AREA_SHARE = 0.20
HALF_REFERENCE_SHARE = 0.38
HALF_PLATE_AREA_SHARE = 0.11
REFERENCE_RUN_ID_DISTANCE = 3
REFERENCE_STATES = {EMPTY_STATE, HALF_STATE, QUARTER_STATE}


def classify_state(
    features: GeometryFeatures, flat_area_plate_share: float | None = None
) -> str:
    """Classify fold state with the documented reference rules."""
    area = features.area_plate_share
    if area <= 0:
        return EMPTY_STATE
    if flat_area_plate_share and flat_area_plate_share > 0:
        relative = area / flat_area_plate_share
        if relative >= FLAT_REFERENCE_SHARE:
            return FLAT_STATE
    elif area >= FLAT_PLATE_AREA_SHARE:
        return FLAT_STATE
    if (
        flat_area_plate_share is None
        and not features.straight_edges
        and area >= FLAT_NO_EDGE_AREA_SHARE
    ):
        return FLAT_STATE
    if flat_area_plate_share and flat_area_plate_share > 0:
        relative = area / flat_area_plate_share
        return HALF_STATE if relative >= HALF_REFERENCE_SHARE else QUARTER_STATE
    return HALF_STATE if area >= HALF_PLATE_AREA_SHARE else QUARTER_STATE


def _read_frame(data_dir: Path, frame_path: str) -> Any:
    image = cv2.imread(str(data_dir / frame_path))
    if image is None:
        raise ValueError(f"Cannot read frame: {frame_path}")
    return image


def _plate_by_run(
    frames: pd.DataFrame,
    data_dir: Path,
) -> dict[int, PlateEllipse]:
    plates: dict[int, PlateEllipse] = {}
    for run_id, group in frames.groupby("run_id"):
        candidates: list[PlateEllipse] = []
        for row in group.itertuples(index=False):
            if row.expected_state in REFERENCE_STATES:
                image = _read_frame(data_dir, row.frame_path)
                candidates.append(detect_plate_ellipse(image))
        if not candidates:
            for row in group.itertuples(index=False):
                image = _read_frame(data_dir, row.frame_path)
                candidates.append(detect_plate_ellipse(image))
        plates[int(run_id)] = median_ellipse(candidates)
    return plates


def _empty_references(frames: pd.DataFrame, data_dir: Path) -> dict[int, Any]:
    references: dict[int, Any] = {}
    for row in frames.itertuples(index=False):
        if row.expected_state == EMPTY_STATE and bool(row.check_ok):
            references[int(row.run_id)] = _read_frame(data_dir, row.frame_path)
    return references


def _nearest_reference(
    run_id: int,
    machine_name: str | None,
    width: int,
    frames: pd.DataFrame,
    references: dict[int, Any],
) -> Any | None:
    if run_id in references:
        return references[run_id]
    best: tuple[int, int] | None = None
    for other_id in references:
        if abs(other_id - run_id) > REFERENCE_RUN_ID_DISTANCE:
            continue
        rows = frames[frames["run_id"] == other_id]
        if rows.empty:
            continue
        first = rows.iloc[0]
        robot_changed = first.get("machine_name") != machine_name
        width_changed = int(first.get("width")) != width
        if robot_changed or width_changed:
            continue
        distance = abs(other_id - run_id)
        if best is None or distance < best[0]:
            best = (distance, other_id)
    return references[best[1]] if best is not None else None


def segment_frames(data_dir: Path) -> pd.DataFrame:
    """Build features.parquet for extracted fold frames."""
    output = data_dir / "fold_segmentation"
    frames = pd.read_parquet(output / "frames.parquet")
    if frames.empty:
        table = pd.DataFrame()
        table.to_parquet(output / "features.parquet", index=False)
        return table
    plates = _plate_by_run(frames, data_dir)
    references = _empty_references(frames, data_dir)
    rows: list[dict[str, Any]] = []
    flat_area_by_run: dict[int, float] = {}
    for row in frames.itertuples(index=False):
        run_id = int(row.run_id)
        image = _read_frame(data_dir, row.frame_path)
        reference = _nearest_reference(
            run_id,
            row.machine_name,
            int(row.width),
            frames,
            references,
        )
        mask = crepe_mask(image, plates[run_id], reference)
        geometry = mask_geometry(mask.mask, plates[run_id])
        if row.expected_state == FLAT_STATE and bool(row.check_ok):
            previous = flat_area_by_run.get(run_id)
            if previous is None:
                flat_area_by_run[run_id] = geometry.area_plate_share
        flat_reference = None
        if row.expected_state != FLAT_STATE:
            flat_reference = flat_area_by_run.get(run_id)
        state = classify_state(geometry, flat_reference)
        edges = geometry.straight_edges
        item = {
            "run_id": run_id,
            "check_name": row.check_name,
            "expected_state": row.expected_state,
            "check_ok": bool(row.check_ok),
            "machine_name": row.machine_name,
            "predicted_state": state,
            "plate_cx": plates[run_id].cx,
            "plate_cy": plates[run_id].cy,
            "plate_major": plates[run_id].major,
            "plate_minor": plates[run_id].minor,
            "plate_source": plates[run_id].source,
            "component_count": mask.component_count,
            "has_reference": mask.has_reference,
            "edge_count": len(edges),
            "edge_angle_deg": edges[0].angle_deg if edges else None,
            **asdict(geometry),
        }
        item.pop("straight_edges")
        rows.append(item)
    table = pd.DataFrame(rows)
    table.to_parquet(output / "features.parquet", index=False)
    return table
