"""Offline checks of the weighted fine-tuning dataset builder."""

import json
from pathlib import Path

import pandas as pd
import pytest

from maus_export.finetune_dataset import (
    DATASET_TABLE,
    EVAL_SPLIT,
    HOLDOUT_RUN_SHARE,
    MANIFEST_NAME,
    REVIEW_THRESHOLD,
    SPLITS,
    TRAIN_SPLIT,
    WEIGHT_MAX,
    DatasetError,
    assign_splits,
    build_dataset,
    run_order,
    training_weight,
    usage_summary,
)


def deviation_frame(rows: list[tuple[int, float]]) -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "run_id": run_id,
                "decoded_frame_index": index,
                "requested_seconds": float(index * 5),
                "deviation": deviation,
                "status": "compared",
                "label_state": "flat",
                "prediction_state": "flat",
                "dice": 1.0 - deviation,
                "iou": 1.0 - deviation,
                "label_material": f"/private/task-{index}-material.png",
                "prediction_mask": f"/private/prediction-{index}.png",
            }
            for index, (run_id, deviation) in enumerate(rows)
        ]
    )


def write_deviation(path: Path, frame: pd.DataFrame) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path / "deviation.parquet", index=False)
    return path


def test_run_order_is_deterministic() -> None:
    assert run_order([3, 1, 2]) == run_order([1, 2, 3])


def test_splits_keep_whole_runs_and_hold_out_a_share() -> None:
    runs = list(range(1, 11))
    splits = assign_splits(runs)
    assert set(splits) == set(runs)
    assert set(splits.values()) == set(SPLITS)
    holdout = [run for run, split in splits.items() if split == EVAL_SPLIT]
    assert len(holdout) == round(len(runs) * HOLDOUT_RUN_SHARE)
    assert assign_splits(runs) == splits


def test_one_run_cannot_be_split() -> None:
    assert assign_splits([5]) == {5: TRAIN_SPLIT}


def test_weight_grows_with_deviation_and_is_capped() -> None:
    assert training_weight(0.0) < training_weight(0.5)
    assert training_weight(1.0) == pytest.approx(WEIGHT_MAX)
    assert training_weight(5.0) == pytest.approx(WEIGHT_MAX)


def test_dataset_carries_weights_and_splits(tmp_path: Path) -> None:
    frame = deviation_frame(
        [(run, 0.0) for run in range(1, 6)] + [(6, 1.0), (7, 0.6), (8, 0.9)]
    )
    deviation = write_deviation(tmp_path / "deviation", frame)
    manifest = build_dataset(
        deviation_directory=deviation,
        labels_directory=tmp_path / "labels",
        predictions_directory=tmp_path / "predictions",
        output=tmp_path / "dataset",
    )
    table = pd.read_parquet(tmp_path / "dataset" / DATASET_TABLE)
    assert set(table["split"]).issubset(set(SPLITS))
    assert table["weight"].max() == pytest.approx(WEIGHT_MAX)
    assert table["weight"].min() == pytest.approx(1.0)
    assert table.loc[
        table["deviation"] >= REVIEW_THRESHOLD, "needs_review"
    ].all()
    assert manifest["state_taxonomy_version"] == "crepe-state-v1"
    assert manifest["split"]["counts"][TRAIN_SPLIT] > 0
    assert sum(manifest["split"]["counts"].values()) == len(table)
    assert manifest["inputs"]["deviation_table_sha256"]
    assert (tmp_path / "dataset" / MANIFEST_NAME).is_file()
    assert "train frames" in usage_summary(manifest)


def test_dataset_refuses_to_overwrite(tmp_path: Path) -> None:
    deviation = write_deviation(
        tmp_path / "deviation", deviation_frame([(1, 0.2), (2, 0.3)])
    )
    build_dataset(
        deviation_directory=deviation,
        labels_directory=tmp_path / "labels",
        predictions_directory=None,
        output=tmp_path / "dataset",
    )
    with pytest.raises(DatasetError):
        build_dataset(
            deviation_directory=deviation,
            labels_directory=tmp_path / "labels",
            predictions_directory=None,
            output=tmp_path / "dataset",
        )


def test_dataset_rejects_a_table_without_the_deviation_column(
    tmp_path: Path,
) -> None:
    deviation = tmp_path / "deviation"
    deviation.mkdir()
    pd.DataFrame.from_records([{"run_id": 1}]).to_parquet(
        deviation / "deviation.parquet", index=False
    )
    with pytest.raises(DatasetError):
        build_dataset(
            deviation_directory=deviation,
            labels_directory=tmp_path / "labels",
            predictions_directory=None,
            output=tmp_path / "dataset",
        )


def test_dataset_rejects_an_empty_table(tmp_path: Path) -> None:
    deviation = write_deviation(
        tmp_path / "deviation", deviation_frame([]).dropna(axis=1, how="all")
    )
    with pytest.raises(DatasetError):
        build_dataset(
            deviation_directory=deviation,
            labels_directory=tmp_path / "labels",
            predictions_directory=None,
            output=tmp_path / "dataset",
        )


def test_manifest_is_json_serialisable(tmp_path: Path) -> None:
    deviation = write_deviation(
        tmp_path / "deviation", deviation_frame([(1, 0.4), (2, 0.1)])
    )
    manifest = build_dataset(
        deviation_directory=deviation,
        labels_directory=tmp_path / "labels",
        predictions_directory=None,
        output=tmp_path / "dataset",
    )
    text = (tmp_path / "dataset" / MANIFEST_NAME).read_text()
    assert json.loads(text) == manifest
