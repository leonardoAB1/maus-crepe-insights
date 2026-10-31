"""Offline deviation checks with synthetic masks and transforms."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from maus_export.label_deviation import (
    DEFAULT_SCORE_THRESHOLD,
    NATIVE_SPACE,
    RECTIFIED_SPACE,
    ComparisonError,
    FrameKey,
    compare_frame,
    deviation_score,
    load_label_export,
    load_prediction_index,
    mask_metrics,
    run_comparison,
    warp_mask,
)

SIZE = 64
MASK_MAX = 255


def square(size: int = SIZE, start: int = 8, extent: int = 20) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[start : start + extent, start : start + extent] = True
    return mask


def write_png(path: Path, mask: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask.astype(np.uint8) * MASK_MAX)
    return path


def identity_transform(size: int = SIZE) -> dict:
    return {
        "schema_version": 1,
        "matrix": np.eye(3).tolist(),
        "source_size": [size, size],
        "target_size": [size, size],
    }


def build_export(
    root: Path,
    frames: list[tuple[int, int, float, np.ndarray]],
    states: list[str | None] | None = None,
) -> Path:
    """Write a labeling mask export shaped like the real one."""
    records = []
    for index, (run_id, frame_index, seconds, mask) in enumerate(frames):
        name = f"task-{index + 1}-material.png"
        write_png(root / name, mask)
        record = {
            "run_id": run_id,
            "decoded_frame_index": frame_index,
            "requested_seconds": seconds,
            "image_sha256": "0" * 64,
            "image_size": [SIZE, SIZE],
            "coordinate_space": "original_video",
            "outputs": {"material": name, "holes": name},
        }
        if states is not None and states[index] is not None:
            record["state"] = states[index]
        records.append(record)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "mask_taxonomy_version": "crepe-material-holes-v1",
                "source_pilot_sha256": "1" * 64,
                "records": records,
            }
        )
    )
    return root


def build_predictions(
    root: Path, entries: list[tuple[FrameKey, np.ndarray, str | None]]
) -> Path:
    items = []
    for index, (key, mask, state) in enumerate(entries):
        name = f"prediction-{index + 1}.png"
        write_png(root / name, mask)
        item = {**key.as_dict(), "mask": name}
        if state is not None:
            item["state"] = state
        items.append(item)
    (root / "predictions.json").write_text(
        json.dumps({"model": "unit-test", "predictions": items})
    )
    return root / "predictions.json"


def test_identical_masks_are_a_perfect_match() -> None:
    mask = square()
    metrics = mask_metrics(mask, mask)
    assert metrics["dice"] == pytest.approx(1.0)
    assert metrics["iou"] == pytest.approx(1.0)
    assert metrics["boundary_p95_px"] == pytest.approx(0.0)
    assert deviation_score(metrics) == pytest.approx(0.0)


def test_disjoint_masks_score_near_the_worst() -> None:
    reference = square(start=4)
    prediction = square(start=40)
    metrics = mask_metrics(reference, prediction)
    assert metrics["dice"] == pytest.approx(0.0)
    assert metrics["precision"] == pytest.approx(0.0)
    assert metrics["recall"] == pytest.approx(0.0)
    # The score is a weighted mean of capped terms, so a total miss of a
    # same-area mask lands high but below the score of a missing prediction.
    assert deviation_score(metrics) > 0.7


def test_deviation_grows_with_the_shift() -> None:
    reference = square()
    scores = [
        deviation_score(mask_metrics(reference, square(start=8 + shift)))
        for shift in (0, 2, 6, 20)
    ]
    assert scores == sorted(scores)
    assert scores[0] < scores[-1]


def test_two_empty_masks_agree() -> None:
    empty = np.zeros((SIZE, SIZE), dtype=bool)
    metrics = mask_metrics(empty, empty)
    assert metrics["dice"] == pytest.approx(1.0)
    assert deviation_score(metrics) == pytest.approx(0.0)


def test_prediction_on_an_empty_reference_is_penalised() -> None:
    empty = np.zeros((SIZE, SIZE), dtype=bool)
    metrics = mask_metrics(empty, square())
    assert metrics["recall"] == pytest.approx(0.0)
    assert metrics["precision"] == pytest.approx(0.0)
    assert deviation_score(metrics) > DEFAULT_SCORE_THRESHOLD


def test_missing_prediction_on_a_visible_crepe_is_penalised() -> None:
    metrics = mask_metrics(square(), np.zeros((SIZE, SIZE), dtype=bool))
    assert metrics["recall"] == pytest.approx(0.0)
    assert deviation_score(metrics) == pytest.approx(1.0)


def test_metrics_reject_mismatched_shapes() -> None:
    with pytest.raises(ComparisonError):
        mask_metrics(square(), square(size=32))


def test_warp_keeps_a_mask_under_an_identity_transform() -> None:
    mask = square()
    warped, valid = warp_mask(mask, identity_transform())
    assert warped.shape == mask.shape
    assert np.array_equal(warped, mask)
    assert valid.all()


def test_warp_moves_a_mask_under_a_translation() -> None:
    transform = identity_transform()
    transform["matrix"] = [[1.0, 0.0, 5.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    warped, valid = warp_mask(square(), transform)
    centroid = np.argwhere(warped).mean(axis=0)
    original = np.argwhere(square()).mean(axis=0)
    assert centroid[1] - original[1] == pytest.approx(5.0, abs=1.0)
    # A target pixel is supported only where the warp found a source pixel.
    assert not valid[:, :5].any()
    assert valid[:, 5:].all()


def test_warp_rejects_a_size_mismatch() -> None:
    transform = identity_transform()
    transform["source_size"] = [SIZE * 2, SIZE]
    with pytest.raises(ComparisonError):
        warp_mask(square(), transform)


def test_label_export_is_indexed_by_frame_identity(tmp_path: Path) -> None:
    root = build_export(
        tmp_path / "export", [(7, 0, 0.0, square()), (7, 1, 5.0, square())]
    )
    labels = load_label_export(root)
    assert set(labels) == {FrameKey(7, 0, 0.0), FrameKey(7, 1, 5.0)}
    assert labels[FrameKey(7, 0, 0.0)]["material_path"].is_file()


def test_label_export_without_manifest_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ComparisonError):
        load_label_export(tmp_path / "missing")


def test_prediction_index_normalises_states(tmp_path: Path) -> None:
    key = FrameKey(7, 0, 0.0)
    index = build_predictions(tmp_path / "pred", [(key, square(), "spreading")])
    predictions = load_prediction_index(index)
    assert predictions[key]["state"] == "flat"


def test_prediction_index_rejects_an_unknown_state(tmp_path: Path) -> None:
    key = FrameKey(7, 0, 0.0)
    index = build_predictions(tmp_path / "pred", [(key, square(), "banana")])
    with pytest.raises(ComparisonError):
        load_prediction_index(index)


def test_compare_frame_flags_a_missing_prediction(tmp_path: Path) -> None:
    label = {"material_path": write_png(tmp_path / "x-material.png", square())}
    result = compare_frame(
        FrameKey(1, 0, 0.0),
        label,
        None,
        space=NATIVE_SPACE,
        transform=None,
        prediction_space=NATIVE_SPACE,
    )
    assert result.status == "label_without_prediction"
    assert result.deviation == pytest.approx(1.0)


def test_native_comparison_writes_a_report(tmp_path: Path) -> None:
    keys = [FrameKey(1, 0, 0.0), FrameKey(1, 1, 5.0), FrameKey(2, 0, 0.0)]
    labels = build_export(
        tmp_path / "labels",
        [
            (1, 0, 0.0, square()),
            (1, 1, 5.0, square()),
            (2, 0, 0.0, square()),
        ],
        states=["emptyplate", "flat", None],
    )
    index = build_predictions(
        tmp_path / "predictions",
        [
            (keys[0], square(), "emptyplate"),
            (keys[1], square(start=30), "flat"),
        ],
    )
    report = run_comparison(
        labels_directory=labels,
        prediction_index=index,
        output=tmp_path / "out",
        space=NATIVE_SPACE,
        prediction_space=NATIVE_SPACE,
        score_threshold=0.3,
    )
    assert report["counts"]["labelled_frames"] == 3
    assert report["counts"]["compared"] == 2
    assert report["counts"]["label_without_prediction"] == 1
    assert report["state_taxonomy_version"] == "crepe-state-v1"
    assert report["states"]["agreement"]["compared"] == 2
    assert (tmp_path / "out" / "report.json").is_file()
    assert (tmp_path / "out" / "deviation.parquet").is_file()
    assert report["needs_finetune"] >= 1
    manifest = json.loads(
        (tmp_path / "out" / "finetune-manifest.json").read_text()
    )
    assert manifest["frames"]
    assert all(entry["deviation"] >= 0.3 for entry in manifest["frames"])


def test_rectified_comparison_needs_a_pilot_and_a_transform(
    tmp_path: Path,
) -> None:
    labels = build_export(tmp_path / "labels", [(1, 0, 0.0, square())])
    index = build_predictions(
        tmp_path / "pred", [(FrameKey(1, 0, 0.0), square(), None)]
    )
    with pytest.raises(ComparisonError):
        run_comparison(
            labels_directory=labels,
            prediction_index=index,
            output=tmp_path / "out",
            space=RECTIFIED_SPACE,
            prediction_space=RECTIFIED_SPACE,
        )


def test_rectified_comparison_warps_both_sides(tmp_path: Path) -> None:
    key = FrameKey(1, 0, 0.0)
    labels = build_export(tmp_path / "labels", [(1, 0, 0.0, square())])
    index = build_predictions(tmp_path / "pred", [(key, square(), "flat")])
    pilot = tmp_path / "pilot"
    (pilot / "run-000").mkdir(parents=True)
    (pilot / "manifest.json").write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "run_id": 1,
                        "frames": [
                            {"image": "run-000/000000.png"},
                        ],
                    }
                ]
            }
        )
    )
    transforms = tmp_path / "transforms"
    transforms.mkdir()
    (transforms / "run-000.json").write_text(json.dumps(identity_transform()))
    report = run_comparison(
        labels_directory=labels,
        prediction_index=index,
        output=tmp_path / "out",
        space=RECTIFIED_SPACE,
        prediction_space=RECTIFIED_SPACE,
        pilot=pilot,
        transforms=transforms,
    )
    assert report["space"] == RECTIFIED_SPACE
    assert report["counts"]["compared"] == 1
    assert report["aggregate"]["deviation_mean"] == pytest.approx(0.0, abs=1e-6)
