"""Crêpe student: preprocessing, plate-free state rule and live machine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from maus_export import crepe_student as cs
from maus_export.fold_live import EMPTY_HOLD_FRAMES

FRAME_SHAPE = (509, 900, 3)
FLAT_AREA = 0.30
HALF_AREA = FLAT_AREA * 0.39
QUARTER_AREA = FLAT_AREA * 0.21


class StubSession:
    """Returns a fixed mask and records the input it got."""

    def __init__(self, mask: np.ndarray) -> None:
        self.mask = mask
        self.inputs: list[np.ndarray] = []

    def run(self, output_names, feeds):
        self.inputs.append(feeds[cs.INPUT_NAME])
        return [self.mask[None, None].astype(np.uint8)]


def disc_mask(share: float) -> np.ndarray:
    """A filled disc covering about ``share`` of the model output."""
    height, width = cs.INPUT_SIZE[1], cs.INPUT_SIZE[0]
    radius = np.sqrt(share * height * width / np.pi)
    yy, xx = np.mgrid[:height, :width]
    return (yy - height / 2) ** 2 + (xx - width / 2) ** 2 <= radius**2


def feed(machine: cs.StudentStateMachine, area: float, frames: int) -> str:
    state = machine.state
    for _ in range(frames):
        _, state = machine.update(area)
    return state


def test_preprocess_shape_and_normalisation() -> None:
    frame = np.full(FRAME_SHAPE, 255, dtype=np.uint8)
    x = cs.preprocess(frame)
    assert x.shape == (1, 3, cs.INPUT_SIZE[1], cs.INPUT_SIZE[0])
    assert x.dtype == np.float32
    expected = (1.0 - cs.MEAN) / cs.STD
    assert np.allclose(x[0, :, 0, 0], expected)


def test_preprocess_converts_bgr_to_rgb() -> None:
    frame = np.zeros(FRAME_SHAPE, dtype=np.uint8)
    frame[..., 0] = 255  # blue in OpenCV order
    x = cs.preprocess(frame)
    assert x[0, 2].mean() > x[0, 0].mean()


def test_preprocess_rejects_grey_frames() -> None:
    with pytest.raises(ValueError):
        cs.preprocess(np.zeros(FRAME_SHAPE[:2], dtype=np.uint8))


def test_student_mask_uses_session_output() -> None:
    expected = disc_mask(FLAT_AREA)
    session = StubSession(expected)
    mask = cs.CrepeStudent(session).mask(np.zeros(FRAME_SHAPE, np.uint8))
    assert mask.dtype == bool
    assert np.array_equal(mask, expected)
    assert session.inputs[0].shape[1:] == (3, 216, 384)


def test_area_share_keeps_the_largest_part() -> None:
    mask = disc_mask(FLAT_AREA)
    mask[:3, :3] = True  # a separate speck
    assert cs.area_share(mask) == pytest.approx(FLAT_AREA, rel=0.02)


@pytest.mark.parametrize(
    ("area", "flat", "state"),
    [
        (0.0, None, "empty"),
        (cs.MIN_CREPE_SHARE / 2, FLAT_AREA, "empty"),
        (0.05, None, "flat"),
        (FLAT_AREA, FLAT_AREA, "flat"),
        (HALF_AREA, FLAT_AREA, "half"),
        (QUARTER_AREA, FLAT_AREA, "quarter"),
    ],
)
def test_classify_area(area: float, flat: float | None, state: str) -> None:
    assert cs.classify_area(area, flat) == state


def test_small_spread_still_counts_as_flat() -> None:
    machine = cs.StudentStateMachine()
    small = cs.MIN_CREPE_SHARE * 2
    assert feed(machine, small, 10) == "flat"


def test_live_sequence_through_both_folds() -> None:
    machine = cs.StudentStateMachine()
    assert feed(machine, 0.0, 20) == "empty"
    assert feed(machine, FLAT_AREA, cs.FLAT_STABLE_FRAMES + 5) == "flat"
    assert feed(machine, HALF_AREA, 20) == "half"
    assert feed(machine, QUARTER_AREA, 20) == "quarter"
    assert feed(machine, HALF_AREA, 20) == "quarter"  # never unfolds


def test_growing_spread_with_occlusion_is_not_a_fold() -> None:
    machine = cs.StudentStateMachine()
    for step in range(30):
        machine.update(0.05 + step * 0.008)
    assert feed(machine, 0.05, 15) == "flat"


def test_short_occlusion_does_not_empty_the_plate() -> None:
    machine = cs.StudentStateMachine()
    feed(machine, FLAT_AREA, cs.FLAT_STABLE_FRAMES + 5)
    assert feed(machine, 0.0, EMPTY_HOLD_FRAMES // 2) == "flat"


def test_empty_plate_resets_for_the_next_crepe() -> None:
    machine = cs.StudentStateMachine()
    feed(machine, FLAT_AREA, cs.FLAT_STABLE_FRAMES + 5)
    feed(machine, QUARTER_AREA, 20)
    assert feed(machine, 0.0, EMPTY_HOLD_FRAMES + 10) == "empty"
    assert machine.flat_area is None
    assert feed(machine, FLAT_AREA / 2, 10) == "flat"


def test_check_frame_states_use_the_run_spread() -> None:
    frames = pd.DataFrame(
        {
            "run_id": [1, 1, 1, 1],
            "expected_state": ["empty", "flat", "half", "quarter"],
        }
    )
    areas = pd.Series([0.0, FLAT_AREA, HALF_AREA, QUARTER_AREA])
    states = cs.check_frame_states(frames, areas)
    assert list(states) == ["empty", "flat", "half", "quarter"]


def test_agreement_groups() -> None:
    frames = pd.DataFrame(
        {
            "run_id": [1, 1, 2],
            "check_ok": [True, True, False],
            "expected_state": ["flat", "half", "flat"],
            "predicted_state": ["flat", "quarter", "empty"],
            "machine_name": ["a", "a", "b"],
        }
    )
    table = cs.agreement(frames).set_index("group")
    assert table.loc["all", "frames"] == 2
    assert table.loc["all", "agreement"] == pytest.approx(0.5)
    assert table.loc["expected_state=half", "agreement"] == 0.0


def test_evaluate_check_frames_offline(tmp_path: Path) -> None:
    root = tmp_path / "fold_segmentation"
    root.mkdir()
    pd.DataFrame(
        {
            "run_id": [7, 7],
            "check_name": ["spread", "fold1"],
            "expected_state": ["flat", "half"],
            "check_ok": [True, True],
            "machine_name": ["robot", "robot"],
            "decoded_frame_index": [0, 1],
        }
    ).to_parquet(root / "frames.parquet")
    video = tmp_path / "raw" / "media" / "7" / "video.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"")
    masks = {0: disc_mask(FLAT_AREA), 1: disc_mask(HALF_AREA)}

    class IndexSession:
        index = 0

        def run(self, output_names, feeds):
            return [masks[self.index][None, None].astype(np.uint8)]

    session = IndexSession()

    def reader(path: Path, index: int) -> np.ndarray:
        session.index = index
        return np.zeros(FRAME_SHAPE, np.uint8)

    metrics = cs.evaluate_check_frames(
        tmp_path, cs.CrepeStudent(session), reader
    )
    assert metrics.set_index("group").loc["all", "agreement"] == 1.0
    assert (root / "student" / "features.parquet").exists()


def test_shipped_model_runs() -> None:
    pytest.importorskip("onnxruntime")
    if not cs.MODEL_PATH.exists():
        pytest.skip("model.onnx not present")
    student = cs.CrepeStudent.load()
    mask = student.mask(np.zeros(FRAME_SHAPE, np.uint8))
    assert mask.shape == (cs.INPUT_SIZE[1], cs.INPUT_SIZE[0])
    assert cs.area_share(mask) < cs.MIN_CREPE_SHARE
