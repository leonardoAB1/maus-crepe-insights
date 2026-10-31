"""Turn a deviation report into a weighted, run-grouped fine-tuning dataset.

The dataset keeps whole runs together. Frames from one run are strongly
correlated, so a random frame split would put near-duplicates on both sides and
overstate accuracy; hashing runs instead keeps the evaluation honest.

Every sample carries the deviation score as a training weight, so the frames a
model currently gets wrong count more. Nothing here trains a model or copies
pixels into the public repository: the table holds private paths, and the
manifest holds counts, weights and provenance.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from .crepe_states import STATE_TAXONOMY_VERSION, state_table

DATASET_VERSION = 1
DEVATION_TABLE = "deviation.parquet"
DATASET_TABLE = "finetune.parquet"
MANIFEST_NAME = "dataset.json"

TRAIN_SPLIT = "train"
EVAL_SPLIT = "eval"
SPLITS = (TRAIN_SPLIT, EVAL_SPLIT)

HOLDOUT_RUN_SHARE = 0.2
SPLIT_SEED = "crepe-finetune-v1"
WEIGHT_BASE = 1.0
WEIGHT_GAIN = 2.0
WEIGHT_MAX = 3.0
REVIEW_THRESHOLD = 0.5
HASH_CHUNK = 1 << 20


class DatasetError(ValueError):
    """The deviation table cannot be turned into a dataset."""


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_order(runs: list[int]) -> list[int]:
    """Deterministic run order, independent of the input order."""
    return sorted(
        runs,
        key=lambda run: hashlib.sha256(
            f"{SPLIT_SEED}:{run}".encode()
        ).hexdigest(),
    )


def assign_splits(runs: list[int]) -> dict[int, str]:
    """Assign whole runs to train or eval by a fixed hash order."""
    ordered = run_order(sorted(set(runs)))
    if not ordered:
        raise DatasetError("no runs to split")
    holdout = max(1, round(len(ordered) * HOLDOUT_RUN_SHARE))
    if len(ordered) == 1:
        holdout = 0
    holdout_runs = set(ordered[:holdout])
    return {
        run: EVAL_SPLIT if run in holdout_runs else TRAIN_SPLIT
        for run in ordered
    }


def training_weight(deviation: float) -> float:
    """Sample weight that grows with the deviation, capped at WEIGHT_MAX."""
    weight = WEIGHT_BASE + WEIGHT_GAIN * float(deviation)
    return float(min(weight, WEIGHT_MAX))


def build_dataset(
    *,
    deviation_directory: Path,
    labels_directory: Path,
    predictions_directory: Path | None,
    output: Path,
) -> dict[str, Any]:
    """Write the weighted table and its manifest.

    Paths in the table stay private and absolute; the manifest records counts,
    weights, splits and the hash of every input it read.
    """
    table_path = deviation_directory / DEVATION_TABLE
    if not table_path.is_file():
        raise DatasetError(f"missing {DEVATION_TABLE} in {deviation_directory}")
    frame = pd.read_parquet(table_path)
    required = {
        "run_id",
        "decoded_frame_index",
        "requested_seconds",
        "deviation",
        "status",
        "label_state",
        "prediction_state",
        "label_material",
        "prediction_mask",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise DatasetError(f"deviation table misses {sorted(missing)}")
    if frame.empty:
        raise DatasetError("deviation table is empty")
    splits = assign_splits([int(value) for value in frame["run_id"]])
    frame = frame.copy()
    frame["split"] = frame["run_id"].map(splits).astype(str)
    frame["weight"] = frame["deviation"].map(training_weight).astype(float)
    frame["needs_review"] = frame["deviation"] >= REVIEW_THRESHOLD
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    destination = output / DATASET_TABLE
    if destination.exists():
        raise DatasetError(f"refusing to overwrite {destination}")
    frame.to_parquet(destination, index=False)
    counts = {split: int((frame["split"] == split).sum()) for split in SPLITS}
    runs = {
        split: sorted(
            int(value)
            for value in frame.loc[frame["split"] == split, "run_id"].unique()
        )
        for split in SPLITS
    }
    per_state = (
        frame.groupby("label_state")
        .agg(frames=("deviation", "size"), deviation_mean=("deviation", "mean"))
        .reset_index()
        .to_dict(orient="records")
    )
    manifest = {
        "version": DATASET_VERSION,
        "state_taxonomy": state_table(),
        "state_taxonomy_version": STATE_TAXONOMY_VERSION,
        "split": {
            "seed": SPLIT_SEED,
            "holdout_run_share": HOLDOUT_RUN_SHARE,
            "policy": "whole runs, hash order",
            "counts": counts,
            "runs": runs,
        },
        "weights": {
            "base": WEIGHT_BASE,
            "gain": WEIGHT_GAIN,
            "maximum": WEIGHT_MAX,
            "weight_sum": float(frame["weight"].sum()),
            "deviation_mean": float(frame["deviation"].mean()),
        },
        "review_threshold": REVIEW_THRESHOLD,
        "needs_review": int(frame["needs_review"].sum()),
        "per_state": per_state,
        "inputs": {
            "deviation_table_sha256": _file_hash(table_path),
            "labels_directory": str(labels_directory),
            "predictions_directory": (
                None
                if predictions_directory is None
                else str(predictions_directory)
            ),
        },
        "note": (
            "paths in the table are private and machine specific; the manifest "
            "is the portable summary"
        ),
    }
    (output / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, allow_nan=False, default=str)
    )
    return manifest


def usage_summary(manifest: Mapping[str, Any]) -> str:
    """One-line human summary of a built dataset."""
    counts = manifest["split"]["counts"]
    weights = manifest["weights"]
    return (
        f"{counts[TRAIN_SPLIT]} train frames, {counts[EVAL_SPLIT]} eval "
        f"frames, mean deviation {weights['deviation_mean']:.3f}, "
        f"{manifest['needs_review']} frames need review"
    )
