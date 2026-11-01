"""Build private segmentation masks and feature rows for check frames."""

from __future__ import annotations

import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from .fold_classifier import refine_fold_state
from .fold_geometry import measure_geometry
from .fold_mask import EmptyReference, segment_crepe, select_reference
from .fold_plate import detect_plate, ellipse_mask, median_plate
from .fold_state import classify_state


def _read_frame(root: Path, relative: str) -> np.ndarray:
    path = (root / relative).resolve()
    if not path.is_relative_to((root / "frames").resolve()):
        raise ValueError("Frame path escapes frames directory")
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError("Cannot read frame")
    return image


def segment_dataset(data_dir: Path) -> dict[str, int]:
    """Build masks and features from a previously extracted frame dataset."""
    root = data_dir / "fold_segmentation"
    frames = pd.read_parquet(root / "frames.parquet")
    masks_path, features_path = root / "masks", root / "features.parquet"
    if masks_path.exists() or features_path.exists():
        raise FileExistsError("Segmentation output exists; move it aside first")
    records = frames.to_dict("records")
    by_id = {row["frame_id"]: row for row in records}
    references: list[EmptyReference] = []
    plates: dict[int, list] = {}
    counts: Counter[str] = Counter()
    for row in records:
        try:
            plate = detect_plate(_read_frame(root, row["path"]))
            if plate is not None:
                plates.setdefault(int(row["run_id"]), []).append(plate)
        except (ValueError, cv2.error):
            counts["plate_detection_error"] += 1
    run_plates = {run: median_plate(values) for run, values in plates.items()}
    for row in records:
        plate = run_plates.get(int(row["run_id"]))
        if plate is None:
            continue
        try:
            image = _read_frame(root, row["path"])
            if not segment_crepe(image, plate).any():
                references.append(
                    EmptyReference(
                        row["frame_id"],
                        int(row["run_id"]),
                        row["robot"],
                        int(row["source_width"]),
                        True,
                    )
                )
        except (ValueError, cv2.error):
            continue
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix=".masks-", dir=root) as temp:
        stage = Path(temp)
        (stage / "masks").mkdir()
        for row in records:
            result = dict(row)
            result["status"] = "ok"
            plate = run_plates.get(int(row["run_id"]))
            if plate is None:
                result["status"] = "plate_not_found"
                rows.append(result)
                continue
            try:
                image = _read_frame(root, row["path"])
                candidate = select_reference(
                    references,
                    row["frame_id"],
                    int(row["run_id"]),
                    row["robot"],
                    int(row["source_width"]),
                )
                reference = (
                    _read_frame(root, by_id[candidate.frame_id]["path"])
                    if candidate is not None
                    else None
                )
                mask = segment_crepe(image, plate, reference)
                geometry = measure_geometry(mask, plate)
                mask_name = f"{row['frame_id']}.png"
                if not cv2.imwrite(str(stage / "masks" / mask_name), mask):
                    raise OSError("Cannot write segmentation mask")
                result.update(
                    {
                        "mask": f"masks/{mask_name}",
                        "reference_id": candidate.frame_id
                        if candidate
                        else None,
                        "plate_area": int(
                            np.count_nonzero(ellipse_mask(mask.shape, plate))
                        ),
                        "area": geometry.area,
                        "plate_fraction": geometry.plate_fraction,
                        "centre_x": geometry.centre[0]
                        if geometry.centre
                        else None,
                        "centre_y": geometry.centre[1]
                        if geometry.centre
                        else None,
                        "diameter": geometry.diameter,
                        "diameter_fraction": geometry.diameter_fraction,
                        "offset_fraction": geometry.offset_fraction,
                        "hole_fraction": geometry.hole_fraction,
                        "solidity": geometry.solidity,
                        "circularity": geometry.circularity,
                        "edge_count": len(geometry.edges),
                        "edge_angle": geometry.edges[0].angle
                        if geometry.edges
                        else None,
                        "corner_x": geometry.corner[0]
                        if geometry.corner
                        else None,
                        "corner_y": geometry.corner[1]
                        if geometry.corner
                        else None,
                        "corner_angle": geometry.corner_angle,
                        "rect_aspect": geometry.rect_aspect,
                        "rect_angle": geometry.rect_angle,
                    }
                )
            except (ValueError, cv2.error):
                result["status"] = "segmentation_failed"
            rows.append(result)
        flat_refs: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            if (
                row["status"] == "ok"
                and classify_state(
                    row["area"],
                    row["plate_area"],
                    row["edge_count"],
                    rect_aspect=row["rect_aspect"],
                )
                == "flat"
                and row["area"] > 0
            ):
                flat_refs.setdefault(int(row["run_id"]), []).append(row)
        for row in rows:
            if row["status"] != "ok":
                continue
            choices = [
                ref
                for ref in flat_refs.get(int(row["run_id"]), [])
                if ref["frame_id"] != row["frame_id"]
            ]
            best = None
            if choices:
                # The largest filled area is the most complete flat
                # crepe estimate; a check frame may be occluded or
                # captured while the crepe is still spreading.
                choices.sort(key=lambda ref: ref["frame_id"])
                best = max(choices, key=lambda ref: ref["area"])
            row["flat_reference_id"] = best["frame_id"] if best else None
            flat_area = best["area"] if best else None
            rule_state = classify_state(
                row["area"],
                row["plate_area"],
                row["edge_count"],
                flat_area,
                row["rect_aspect"] if pd.notna(row["rect_aspect"]) else None,
            )
            row["rule_state"] = rule_state
            row["predicted_state"] = refine_fold_state(
                rule_state,
                {
                    "relative": row["area"] / flat_area if flat_area else None,
                    "plate_fraction": row["plate_fraction"],
                    "rect_aspect": row["rect_aspect"],
                    "diameter_fraction": row["diameter_fraction"],
                    "offset_fraction": row["offset_fraction"],
                    "hole_fraction": row["hole_fraction"],
                    "solidity": row["solidity"],
                    "circularity": row["circularity"],
                    "edge_count": row["edge_count"],
                },
            )
        pd.DataFrame(rows).to_parquet(stage / "features.parquet", index=False)
        shutil.move(str(stage / "masks"), str(masks_path))
        try:
            shutil.move(str(stage / "features.parquet"), str(features_path))
        except Exception:
            shutil.rmtree(masks_path)
            raise
    counts.update(row["status"] for row in rows)
    return dict(counts)
