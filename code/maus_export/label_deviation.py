"""Deviation between reviewed human masks and model prediction masks.

Only the crepe material class is compared; the hole class is ignored for now.
Human masks come from the labeling mask export, predictions come from a model
run. Both are brought into one coordinate space, by default the rectified
upstream space the labels were prepared in, so a human mask and a prediction
can be overlaid pixel for pixel.

Nothing here reads the dashboard, trains anything or writes to a labeling
database. Frames, masks and predictions stay in the private data directory.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from .crepe_states import (
    STATE_TAXONOMY_VERSION,
    UNIFIED_STATES,
    UnknownState,
    normalise,
    state_table,
)

REPORT_VERSION = 1
LABEL_MANIFEST_NAME = "manifest.json"
PILOT_MANIFEST_NAME = "manifest.json"
MATERIAL_SUFFIX = "-material.png"
TRANSFORM_SUFFIX = ".json"

RECTIFIED_SPACE = "rectified_upstream"
NATIVE_SPACE = "native"
SPACES = (RECTIFIED_SPACE, NATIVE_SPACE)

DEFAULT_SCORE_THRESHOLD = 0.35
MASK_MAX = 255
BYTE_MAX = 255
TRANSFORM_SCHEMA_VERSION = 1
CHANNELS = 3
PERCENTILE = 95.0
BOUNDARY_SCALE_PX = 20.0
CENTROID_SCALE_PX = 60.0
AREA_TOLERANCE = 0.5
AREA_RATIO_CAP = 4.0

# Weights sum to one; overlap dominates because it is the pixel-level truth.
WEIGHTS: dict[str, float] = {
    "overlap": 0.5,
    "boundary": 0.25,
    "centroid": 0.15,
    "area": 0.1,
}

BOUNDARY_KERNEL = np.ones((3, 3), dtype=np.uint8)


class ComparisonError(ValueError):
    """Inputs cannot be compared without guessing provenance."""


@dataclass(frozen=True)
class FrameKey:
    """Frame identity that survives a move between machines."""

    run_id: int
    decoded_frame_index: int
    requested_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def text(self) -> str:
        seconds = f"{self.requested_seconds:g}"
        return f"{self.run_id}-{self.decoded_frame_index}-{seconds}"

    @classmethod
    def from_mapping(cls, record: Mapping[str, Any]) -> FrameKey:
        try:
            return cls(
                int(record["run_id"]),
                int(record["decoded_frame_index"]),
                float(record["requested_seconds"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ComparisonError(
                "frame record misses run_id, decoded_frame_index or "
                "requested_seconds"
            ) from error


def load_mask(path: Path) -> np.ndarray:
    """Read a binary mask as a boolean array; anything above zero is set."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ComparisonError(f"cannot read mask {path}")
    return image > 0


def save_mask(path: Path, mask: np.ndarray) -> None:
    """Write a binary mask as a 0/255 PNG, never overwriting a file."""
    if path.exists():
        raise ComparisonError(f"refusing to overwrite {path}")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask.astype(np.uint8) * MASK_MAX)


def _matrix_of(
    transform: Mapping[str, Any],
) -> tuple[np.ndarray, tuple[int, int]]:
    if transform.get("schema_version") != TRANSFORM_SCHEMA_VERSION:
        raise ComparisonError("unsupported rectification transform schema")
    matrix = np.asarray(transform["matrix"], dtype=np.float64)
    size = tuple(int(value) for value in transform["target_size"])
    if (
        matrix.shape != (CHANNELS, CHANNELS)
        or not np.isfinite(matrix).all()
        or np.linalg.matrix_rank(matrix) < CHANNELS
        or len(size) != 2
        or min(size) <= 0
    ):
        raise ComparisonError("invalid rectification matrix or target size")
    return matrix, size


def warp_mask(
    mask: np.ndarray, transform: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Rectify a mask and report which target pixels are fully supported.

    The frame rectification warps pixels bilinearly and keeps only pixels whose
    footprint lies inside the source. A mask cannot be resampled that way, so
    it is warped without interpolation and restricted to the same support the
    frame rectification would accept. Unsupported pixels never enter a metric.
    """
    matrix, size = _matrix_of(transform)
    source_height, source_width = mask.shape[:2]
    if [source_width, source_height] != [
        int(value) for value in transform["source_size"]
    ]:
        raise ComparisonError(
            "mask size differs from the transform source size"
        )
    warped = cv2.warpPerspective(
        mask.astype(np.uint8) * MASK_MAX,
        matrix,
        size,
        flags=cv2.INTER_NEAREST,
    )
    support = cv2.warpPerspective(
        np.ones((source_height, source_width), dtype=np.float32), matrix, size
    )
    valid = support >= 1.0
    return (warped > 0) & valid, valid


def _centroid(mask: np.ndarray) -> tuple[float, float] | None:
    moments = cv2.moments(mask.astype(np.uint8), binaryImage=True)
    if moments["m00"] == 0:
        return None
    return (moments["m01"] / moments["m00"], moments["m10"] / moments["m00"])


def _boundary(mask: np.ndarray) -> np.ndarray:
    eroded = cv2.erode(mask.astype(np.uint8), BOUNDARY_KERNEL, iterations=1)
    return (mask.astype(np.uint8) - eroded) > 0


def _directed_distances(edge: np.ndarray, other: np.ndarray) -> np.ndarray:
    if not edge.any():
        return np.zeros(0, dtype=np.float64)
    field = cv2.distanceTransform((1 - other.astype(np.uint8)), cv2.DIST_L2, 3)
    return field[edge].astype(np.float64)


def boundary_distances(
    reference: np.ndarray, prediction: np.ndarray
) -> np.ndarray:
    """Symmetric boundary distances in pixels between two masks."""
    return np.concatenate(
        [
            _directed_distances(_boundary(reference), prediction),
            _directed_distances(_boundary(prediction), reference),
        ]
    )


def mask_metrics(
    reference: np.ndarray, prediction: np.ndarray
) -> dict[str, float]:
    """Pixel and shape agreement of a prediction against a human mask.

    The human mask is the reference. Both masks are expected to share a shape
    and to be already restricted to supported pixels.
    """
    if reference.shape != prediction.shape:
        raise ComparisonError("masks do not share one coordinate space")
    reference_area = float(reference.sum())
    prediction_area = float(prediction.sum())
    intersection = float(np.logical_and(reference, prediction).sum())
    union = float(np.logical_or(reference, prediction).sum())
    dice = (
        1.0
        if union == 0
        else 2.0 * intersection / (reference_area + prediction_area)
    )
    iou = 1.0 if union == 0 else intersection / union
    precision = 1.0 if prediction_area == 0 else intersection / prediction_area
    recall = 1.0 if reference_area == 0 else intersection / reference_area
    if prediction_area == 0 and reference_area > 0:
        precision = 0.0
    if reference_area == 0 and prediction_area > 0:
        recall = 0.0
    area_ratio = (
        1.0
        if reference_area == 0
        else min(prediction_area / reference_area, AREA_RATIO_CAP)
    )
    reference_centroid = _centroid(reference)
    prediction_centroid = _centroid(prediction)
    if reference_centroid is None or prediction_centroid is None:
        centroid_offset = (
            0.0
            if reference_area == prediction_area
            else float(CENTROID_SCALE_PX)
        )
    else:
        centroid_offset = float(
            np.hypot(
                reference_centroid[0] - prediction_centroid[0],
                reference_centroid[1] - prediction_centroid[1],
            )
        )
    distances = boundary_distances(reference, prediction)
    boundary_mean = float(distances.mean()) if distances.size else 0.0
    boundary_p95 = (
        float(np.percentile(distances, PERCENTILE)) if distances.size else 0.0
    )
    return {
        "reference_area": reference_area,
        "prediction_area": prediction_area,
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "area_ratio": area_ratio,
        "centroid_offset_px": centroid_offset,
        "boundary_mean_px": boundary_mean,
        "boundary_p95_px": boundary_p95,
    }


def deviation_score(metrics: Mapping[str, float]) -> float:
    """One number in [0, 1] where 0 is a perfect match.

    Each term is capped at one before it is weighted, so a single extreme term
    cannot dominate the score. The weights are the module constant WEIGHTS.
    """
    overlap = 1.0 - float(metrics["dice"])
    boundary = min(float(metrics["boundary_p95_px"]) / BOUNDARY_SCALE_PX, 1.0)
    centroid = min(
        float(metrics["centroid_offset_px"]) / CENTROID_SCALE_PX, 1.0
    )
    area = min(abs(float(metrics["area_ratio"]) - 1.0) / AREA_TOLERANCE, 1.0)
    score = (
        WEIGHTS["overlap"] * overlap
        + WEIGHTS["boundary"] * boundary
        + WEIGHTS["centroid"] * centroid
        + WEIGHTS["area"] * area
    )
    return float(min(max(score, 0.0), 1.0))


def load_label_export(directory: Path) -> dict[FrameKey, dict[str, Any]]:
    """Index a labeling mask export by frame identity."""
    manifest_path = directory / LABEL_MANIFEST_NAME
    if not manifest_path.is_file():
        raise ComparisonError(f"missing label manifest in {directory}")
    manifest = json.loads(manifest_path.read_text())
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ComparisonError("label manifest has no records")
    labels: dict[FrameKey, dict[str, Any]] = {}
    for record in records:
        key = FrameKey.from_mapping(record)
        material = record.get("outputs", {}).get("material")
        if not material:
            raise ComparisonError("label record has no material mask")
        mask_path = directory / material
        if not mask_path.is_file():
            raise ComparisonError(f"missing material mask {mask_path}")
        if key in labels:
            raise ComparisonError(f"duplicate label for frame {key.text()}")
        labels[key] = {
            **record,
            "material_path": mask_path,
            "source_pilot_sha256": manifest.get("source_pilot_sha256"),
            "mask_taxonomy_version": manifest.get("mask_taxonomy_version"),
        }
    return labels


def load_prediction_index(path: Path) -> dict[FrameKey, dict[str, Any]]:
    """Index a prediction file by frame identity.

    The index is a JSON object with a `predictions` list; every entry names the
    frame, the mask file and, when the model also reports a state, the state in
    the unified vocabulary or in the model's own labels.
    """
    content = json.loads(path.read_text())
    entries = content.get("predictions")
    if not isinstance(entries, list):
        raise ComparisonError("prediction index has no predictions list")
    predictions: dict[FrameKey, dict[str, Any]] = {}
    for entry in entries:
        key = FrameKey.from_mapping(entry)
        if key in predictions:
            raise ComparisonError(
                f"duplicate prediction for frame {key.text()}"
            )
        mask_path = (path.parent / entry["mask"]).resolve()
        if not mask_path.is_file():
            raise ComparisonError(f"missing prediction mask {mask_path}")
        state = entry.get("state")
        if state is not None:
            try:
                state = normalise(str(state))
            except UnknownState as error:
                raise ComparisonError(str(error)) from error
        predictions[key] = {
            "mask_path": mask_path,
            "state": state,
            "confidence": entry.get("confidence"),
            "model_version": entry.get("model_version", content.get("model")),
            "source": entry.get("source", content.get("source")),
        }
    return predictions


def load_run_transforms(
    transforms: Path, run_directories: Mapping[int, str]
) -> dict[int, dict[str, Any]]:
    """Load one rectification transform per run.

    `run_directories` maps a run id to the pilot directory that holds its
    frames, because rectified preparations name a transform after that
    directory rather than after the run.
    """
    loaded: dict[int, dict[str, Any]] = {}
    for run_id, directory in run_directories.items():
        candidate = transforms / f"{directory}{TRANSFORM_SUFFIX}"
        if candidate.is_file():
            loaded[run_id] = json.loads(candidate.read_text())
    return loaded


def run_directories(pilot: Path) -> dict[int, str]:
    """Map run ids to their frame directory inside a pilot."""
    manifest_path = pilot / PILOT_MANIFEST_NAME
    if not manifest_path.is_file():
        raise ComparisonError(f"missing pilot manifest in {pilot}")
    manifest = json.loads(manifest_path.read_text())
    mapping: dict[int, str] = {}
    for run in manifest.get("runs", []):
        frames = run.get("frames") or []
        if not frames:
            continue
        mapping[int(run["run_id"])] = Path(frames[0]["image"]).parent.name
    return mapping


@dataclass
class FrameComparison:
    """One frame's agreement between a human mask and a prediction."""

    key: FrameKey
    label_state: str | None
    prediction_state: str | None
    metrics: dict[str, float]
    deviation: float
    status: str
    label_material_path: str | None = None
    prediction_mask_path: str | None = None
    label_mask_sha256: str | None = None
    prediction_mask_sha256: str | None = None


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_frame(
    key: FrameKey,
    label: Mapping[str, Any] | None,
    prediction: Mapping[str, Any] | None,
    *,
    space: str,
    transform: Mapping[str, Any] | None,
    prediction_space: str,
) -> FrameComparison:
    """Compare one frame, honouring the requested coordinate spaces."""
    if label is None:
        raise ComparisonError(f"no human label for frame {key.text()}")
    reference = load_mask(Path(label["material_path"]))
    valid: np.ndarray | None = None
    if space == RECTIFIED_SPACE:
        if transform is None:
            raise ComparisonError(f"no rectification for run {key.run_id}")
        reference, valid = warp_mask(reference, transform)
    label_state = label.get("state")
    if label_state is not None:
        label_state = normalise(str(label_state))
    if prediction is None:
        empty = np.zeros_like(reference)
        metrics = mask_metrics(reference, empty)
        return FrameComparison(
            key=key,
            label_state=label_state,
            prediction_state=None,
            metrics=metrics,
            deviation=1.0,
            status="label_without_prediction",
            label_material_path=str(label["material_path"]),
        )
    predicted = load_mask(Path(prediction["mask_path"]))
    if prediction_space == RECTIFIED_SPACE and space == RECTIFIED_SPACE:
        if transform is None:
            raise ComparisonError(f"no rectification for run {key.run_id}")
        predicted, _ = warp_mask(predicted, transform)
    elif prediction_space != space:
        raise ComparisonError(
            "prediction space and comparison space differ; pass masks in the "
            "comparison space or use the same space for both"
        )
    if predicted.shape != reference.shape:
        raise ComparisonError(
            f"prediction and label shapes differ for frame {key.text()}"
        )
    if valid is not None:
        reference = reference & valid
        predicted = predicted & valid
    metrics = mask_metrics(reference, predicted)
    return FrameComparison(
        key=key,
        label_state=label_state,
        prediction_state=prediction.get("state"),
        metrics=metrics,
        deviation=deviation_score(metrics),
        status="compared",
        label_material_path=str(label["material_path"]),
        prediction_mask_path=str(prediction["mask_path"]),
        label_mask_sha256=_sha256(Path(label["material_path"])),
        prediction_mask_sha256=_sha256(Path(prediction["mask_path"])),
    )


def _aggregate(
    rows: Sequence[FrameComparison],
) -> dict[str, Any]:
    compared = [row for row in rows if row.status == "compared"]
    if not compared:
        return {
            "frames": 0,
            "deviation_mean": None,
            "deviation_median": None,
        }
    deviations = np.asarray([row.deviation for row in compared])
    dice = np.asarray([row.metrics["dice"] for row in compared])
    iou = np.asarray([row.metrics["iou"] for row in compared])
    boundary = np.asarray([row.metrics["boundary_p95_px"] for row in compared])
    return {
        "frames": len(compared),
        "deviation_mean": float(deviations.mean()),
        "deviation_median": float(np.median(deviations)),
        "deviation_p90": float(np.percentile(deviations, 90)),
        "dice_mean": float(dice.mean()),
        "iou_mean": float(iou.mean()),
        "boundary_p95_mean_px": float(boundary.mean()),
    }


def _per_state(rows: Sequence[FrameComparison]) -> dict[str, Any]:
    grouped: dict[str, list[FrameComparison]] = defaultdict(list)
    for row in rows:
        if row.status == "compared" and row.label_state:
            grouped[row.label_state].append(row)
    summary = {}
    for state in UNIFIED_STATES:
        group = grouped.get(state, [])
        if not group:
            continue
        summary[state] = {
            "frames": len(group),
            "deviation_mean": float(np.mean([row.deviation for row in group])),
            "dice_mean": float(np.mean([row.metrics["dice"] for row in group])),
        }
    return summary


def _states(rows: Sequence[FrameComparison]) -> dict[str, Any]:
    labelled = Counter(
        row.label_state for row in rows if row.label_state is not None
    )
    predicted = Counter(
        row.prediction_state
        for row in rows
        if row.status == "compared" and row.prediction_state is not None
    )
    confusion: dict[str, dict[str, int]] = {}
    matched = 0
    both = 0
    for row in rows:
        if row.status != "compared":
            continue
        if row.label_state is None or row.prediction_state is None:
            continue
        both += 1
        matched += int(row.label_state == row.prediction_state)
        confusion.setdefault(row.label_state, {})
        confusion[row.label_state][row.prediction_state] = (
            confusion[row.label_state].get(row.prediction_state, 0) + 1
        )
    return {
        "label_counts": dict(labelled),
        "prediction_counts": dict(predicted),
        "agreement": {
            "compared": both,
            "matched": matched,
            "rate": None if both == 0 else matched / both,
        },
        "confusion": confusion,
    }


def _rows_frame(rows: Sequence[FrameComparison]) -> pd.DataFrame:
    records = []
    for row in rows:
        records.append(
            {
                "run_id": row.key.run_id,
                "decoded_frame_index": row.key.decoded_frame_index,
                "requested_seconds": row.key.requested_seconds,
                "status": row.status,
                "label_state": row.label_state,
                "prediction_state": row.prediction_state,
                "deviation": row.deviation,
                "label_material": row.label_material_path,
                "prediction_mask": row.prediction_mask_path,
                **row.metrics,
            }
        )
    frame = pd.DataFrame.from_records(records)
    if not frame.empty:
        frame = frame.sort_values(
            ["run_id", "decoded_frame_index"], ignore_index=True
        )
    return frame


def run_comparison(
    *,
    labels_directory: Path,
    prediction_index: Path,
    output: Path,
    space: str = RECTIFIED_SPACE,
    prediction_space: str = RECTIFIED_SPACE,
    pilot: Path | None = None,
    transforms: Path | None = None,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    worst_frames: int = 25,
) -> dict[str, Any]:
    """Compare every labelled frame against its prediction and write a report.

    Frames the model did not predict are kept and scored as worst case, so a
    model that silently skips hard frames cannot look good by omission.
    """
    if space not in SPACES or prediction_space not in SPACES:
        raise ComparisonError("unknown coordinate space")
    labels = load_label_export(labels_directory)
    predictions = load_prediction_index(prediction_index)
    transforms_by_run: dict[int, dict[str, Any]] = {}
    if space == RECTIFIED_SPACE:
        if pilot is None or transforms is None:
            raise ComparisonError(
                "rectified comparison needs a pilot and a transform directory"
            )
        transforms_by_run = load_run_transforms(
            transforms, run_directories(pilot)
        )
        if not transforms_by_run:
            raise ComparisonError("no rectification transform matched a run")
    rows: list[FrameComparison] = []
    for key in sorted(
        labels,
        key=lambda item: (item.run_id, item.decoded_frame_index),
    ):
        rows.append(
            compare_frame(
                key,
                labels.get(key),
                predictions.get(key),
                space=space,
                transform=transforms_by_run.get(key.run_id),
                prediction_space=prediction_space,
            )
        )
    unmatched = sorted(
        (key for key in predictions if key not in labels),
        key=lambda item: (item.run_id, item.decoded_frame_index),
    )
    frame = _rows_frame(rows)
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    frame.to_parquet(output / "deviation.parquet", index=False)
    compared = [row for row in rows if row.status == "compared"]
    ranked = sorted(compared, key=lambda row: -row.deviation)[:worst_frames]
    report = {
        "version": REPORT_VERSION,
        "space": space,
        "prediction_space": prediction_space,
        "score_threshold": score_threshold,
        "weights": dict(WEIGHTS),
        "state_taxonomy": state_table(),
        "state_taxonomy_version": STATE_TAXONOMY_VERSION,
        "inputs": {
            "labels": str(labels_directory),
            "predictions": str(prediction_index),
            "pilot": None if pilot is None else str(pilot),
            "transforms": None if transforms is None else str(transforms),
        },
        "counts": {
            "labelled_frames": len(labels),
            "predicted_frames": len(predictions),
            "compared": len(compared),
            "label_without_prediction": len(rows) - len(compared),
            "prediction_without_label": len(unmatched),
        },
        "aggregate": _aggregate(rows),
        "per_state": _per_state(rows),
        "states": _states(rows),
        "needs_finetune": sum(
            1 for row in compared if row.deviation >= score_threshold
        ),
        "worst_frames": [
            {
                "frame": row.key.as_dict(),
                "deviation": row.deviation,
                "dice": row.metrics["dice"],
                "label_state": row.label_state,
                "prediction_state": row.prediction_state,
            }
            for row in ranked
        ],
    }
    manifest = {
        "version": REPORT_VERSION,
        "score_threshold": score_threshold,
        "state_taxonomy_version": STATE_TAXONOMY_VERSION,
        "frames": [
            {
                **row.key.as_dict(),
                "label_state": row.label_state,
                "prediction_state": row.prediction_state,
                "deviation": row.deviation,
                "dice": row.metrics["dice"],
                "label_mask_sha256": row.label_mask_sha256,
                "prediction_mask_sha256": row.prediction_mask_sha256,
            }
            for row in compared
            if row.deviation >= score_threshold
        ],
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False)
    )
    (output / "finetune-manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False)
    )
    return report
