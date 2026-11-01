"""Offline checks of the layer producers, on synthetic frames."""

import base64
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from maus_export.producers import (
    CheckVerdictProducer,
    CrepePoseProducer,
    FrameContext,
    LayerSet,
    PixelMaskProducer,
    ProducerError,
    RecordsProducer,
    as_dict,
    default_producers,
    describe,
    encode_mask,
    layer_fields,
)

SIZE = 96


def frame_with_blob() -> np.ndarray:
    """A dark frame with a warm bright disc, which the colour pass finds."""
    image = np.full((SIZE, SIZE, 3), 40, dtype=np.uint8)
    cv2.circle(image, (48, 48), 22, (170, 220, 245), -1)
    return image


def context(index: int = 0, record: dict | None = None) -> FrameContext:
    return FrameContext(frame_index=index, run_id=3350, record=record)


def write_mask(directory: Path, name: str, radius: int = 20) -> None:
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    cv2.circle(mask, (48, 48), radius, 255, -1)
    directory.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(directory / name), mask)


def test_pixel_producer_finds_material_and_names_no_state() -> None:
    layers = PixelMaskProducer().layers(frame_with_blob(), context())
    assert layers.material is not None and layers.material.any()
    assert layers.state is None
    joined = " ".join(layers.notes)
    assert "no fold state" in joined


def test_pixel_producer_carries_the_record_fields() -> None:
    record = {
        "state": "cooking",
        "state_source": "human",
        "prediction_state": "flat",
        "deviation": 0.2,
        "check": "spread",
        "verdict": "pass",
        "phase_context": "init -> oil",
    }
    layers = PixelMaskProducer(records={0: record}).layers(
        frame_with_blob(), context(0, record)
    )
    assert layers.state == "cooking"
    assert layers.prediction_state == "flat"
    assert layers.deviation == 0.2
    assert layers.check == "spread"


def test_pixel_producer_normalises_a_legacy_label() -> None:
    layers = PixelMaskProducer(records={0: {"state": "circle"}}).layers(
        frame_with_blob(), context(0)
    )
    payload = layers.to_json()
    assert payload["state"] == "cooking"


def test_unknown_state_is_refused() -> None:
    layers = LayerSet(state="banana")
    with pytest.raises(ProducerError):
        layers.to_json()


def test_records_producer_reads_fields_and_masks(tmp_path: Path) -> None:
    write_mask(tmp_path, "material.png")
    record = {
        "state": "flat",
        "material_mask": "material.png",
        "notes": ["from a record"],
    }
    producer = RecordsProducer(records={3: record}, masks_dir=tmp_path)
    layers = producer.layers(frame_with_blob(), context(3, record))
    assert layers.material is not None and layers.material.any()
    assert layers.notes == ["from a record"]


def test_records_producer_says_when_a_frame_has_no_record() -> None:
    producer = RecordsProducer(records={}, masks_dir=None)
    with pytest.raises(ProducerError) as error:
        producer.layers(frame_with_blob(), context(9))
    assert "no record" in str(error.value)


def test_records_producer_refuses_a_path_outside_the_masks_directory(
    tmp_path: Path,
) -> None:
    producer = RecordsProducer(
        records={0: {"material_mask": "../escape.png"}}, masks_dir=tmp_path
    )
    with pytest.raises(ProducerError):
        producer.layers(frame_with_blob(), context(0))


def test_records_producer_refuses_a_mask_of_the_wrong_size(
    tmp_path: Path,
) -> None:
    cv2.imwrite(str(tmp_path / "small.png"), np.zeros((8, 8), np.uint8))
    producer = RecordsProducer(
        records={0: {"material_mask": "small.png"}}, masks_dir=tmp_path
    )
    with pytest.raises(ProducerError):
        producer.layers(frame_with_blob(), context(0))


def test_encode_mask_round_trips() -> None:
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[10:20, 10:20] = True
    payload = encode_mask(mask)
    assert payload.startswith("data:image/png;base64,")
    raw = base64.b64decode(payload.split(",", 1)[1])
    decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    assert decoded.shape == mask.shape
    assert int((decoded > 0).sum()) == int(mask.sum())
    assert decoded[15, 15] > 0
    assert decoded[0, 0] == 0


def test_layer_json_carries_masks_as_data_urls() -> None:
    layers = LayerSet(
        state="flat", material=np.ones((8, 8), dtype=bool), producer="test"
    )
    payload = layers.to_json()
    assert payload["producer"] == "test"
    assert "material" in payload["masks"]
    assert payload["masks"]["material"].startswith("data:image/png;base64,")


def test_registry_lists_the_producers_it_has() -> None:
    registry = default_producers(records={0: {}}, masks_dir=None)
    names = [entry["name"] for entry in describe(registry)]
    assert names == ["records", "pixel_masks", "check_verdict"]
    assert all(entry["describes"] for entry in describe(registry))
    without_records = default_producers(records=None)
    assert "records" not in without_records


def test_check_verdict_producer_explains_what_is_missing() -> None:
    with pytest.raises(ProducerError) as error:
        CheckVerdictProducer().layers(frame_with_blob(), context())
    message = str(error.value)
    assert "onnxruntime" in message or "placeholder" in message


def test_as_dict_leaves_out_the_masks() -> None:
    layers = LayerSet(state="flat", material=np.ones((4, 4), dtype=bool))
    payload = as_dict(layers)
    assert "material" not in payload
    assert json.dumps(payload)  # serialisable without numpy
    assert "material" in layer_fields()


def test_layer_json_reports_mask_areas() -> None:
    mask = np.zeros((10, 20), dtype=bool)
    mask[0:2, 0:5] = True
    payload = LayerSet(material=mask).to_json()
    assert payload["areas"]["material_pixels"] == 10
    assert payload["areas"]["frame_pixels"] == 200
    assert "holes_pixels" not in payload["areas"]


def test_layer_json_without_masks_has_no_areas() -> None:
    payload = LayerSet(state="flat").to_json()
    assert payload["areas"] == {}


class StandInModel:
    """Marks a fixed square of the rectified view, like a tiny crepe model."""

    def run(self, _names: object, inputs: dict) -> list[np.ndarray]:
        assert inputs["x"].shape == (1, 3, 160, 160)
        chances = np.zeros((1, 1, 160, 160), np.float32)
        chances[0, 0, 40:120, 40:120] = 1.0
        return [chances]


def crepe_files(tmp_path: Path, matrix: list[list[float]]) -> tuple[Path, Path]:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"stand-in")
    transform = tmp_path / "transform.json"
    transform.write_text(
        json.dumps({"matrix": matrix, "target_size": [320, 320]})
    )
    return model, transform


def test_crepe_pose_maps_the_mask_back_to_the_frame(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    # The rectified view is the frame shifted 20 pixels to the right
    shift = [[1.0, 0.0, 20.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    model, transform = crepe_files(tmp_path, shift)
    producer = CrepePoseProducer(
        model=model, transform=transform, session=StandInModel()
    )
    frame = np.zeros((320, 400, 3), np.uint8)
    layers = producer.layers(frame, FrameContext(frame_index=0))
    assert layers.material.shape == frame.shape[:2]
    ys, xs = np.nonzero(layers.material)
    assert (ys.min(), ys.max()) == pytest.approx((80, 239), abs=2)
    assert (xs.min(), xs.max()) == pytest.approx((60, 219), abs=2)


def test_crepe_pose_names_a_missing_file(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    missing = tmp_path / "missing.onnx"
    producer = CrepePoseProducer(model=missing, transform=missing)
    with pytest.raises(ProducerError, match="crepe model"):
        producer.layers(np.zeros((8, 8, 3), np.uint8), FrameContext(0))


def test_crepe_pose_is_offered_only_with_a_model_and_transform(
    tmp_path: Path,
) -> None:
    assert "crepe_pose" not in default_producers()
    registry = default_producers(
        crepe_model=tmp_path / "m", crepe_transform=tmp_path / "t"
    )
    assert "crepe_pose" in registry


def test_crepe_pose_reports_how_long_the_model_took(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    same = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    model, transform = crepe_files(tmp_path, same)
    producer = CrepePoseProducer(
        model=model, transform=transform, session=StandInModel()
    )
    producer.layers(np.zeros((320, 320, 3), np.uint8), FrameContext(0))
    assert producer.last_ms > 0
