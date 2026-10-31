from __future__ import annotations

from datetime import datetime

import cv2
import numpy as np
import pandas as pd

from maus_export.fold_dashboard import build_dashboard
from maus_export.fold_eval import confusion_matrix, metric_tables
from maus_export.fold_frames import (
    check_events,
    correct_perspective,
    phase_clock_offset,
)
from maus_export.fold_geometry import mask_geometry
from maus_export.fold_live import FoldStateMachine
from maus_export.fold_mask import crepe_mask, exposure_shift
from maus_export.fold_plate import PlateEllipse, detect_plate_ellipse
from maus_export.fold_state import (
    EMPTY_STATE,
    FLAT_STATE,
    HALF_STATE,
    QUARTER_STATE,
    classify_state,
)


def test_phase_offset_intersects_whole_second_bounds() -> None:
    phases = [
        {"timestamp": "2026-10-31T12:00:10", "elapsed_sec": 10.0},
        {"timestamp": "2026-10-31T12:00:20", "elapsed_sec": 20.0},
    ]

    assert phase_clock_offset(phases) == datetime(
        2026,
        10,
        31,
        12,
        0,
        0,
        500000,
    )


def test_check_events_keep_failed_checks() -> None:
    run = {
        "logs": [
            {
                "timestamp": "2026-10-31 12:00:01.100",
                "msg": "check_spread started",
            },
            {
                "timestamp": "2026-10-31 12:00:01.300",
                "msg": "check_spread: crepe spread successfully",
            },
            {
                "timestamp": "2026-10-31 12:00:02.100",
                "msg": "check_fold1 started",
            },
            {
                "timestamp": "2026-10-31 12:00:02.300",
                "msg": "check_fold1: too small",
            },
        ]
    }

    assert [(name, ok) for name, _, ok in check_events(run)] == [
        ("spread", True),
        ("fold1", False),
    ]


def test_correct_perspective_uses_robot_calibration() -> None:
    frame = np.full((720, 1280, 3), 127, dtype=np.uint8)
    calibration = {
        "robot_corrections": {
            "robot": {
                "matrix_mean": np.eye(3).tolist(),
                "target_size_px": [900, 509],
            }
        }
    }

    corrected = correct_perspective(frame, "robot", calibration)

    assert corrected.shape == (509, 900, 3)


def test_plate_mask_geometry_and_state_rules() -> None:
    image = np.zeros((220, 220, 3), dtype=np.uint8)
    cv2.circle(image, (110, 110), 90, (40, 40, 40), -1)
    cv2.circle(image, (110, 110), 50, (210, 210, 210), -1)

    plate = detect_plate_ellipse(image)
    mask = crepe_mask(image, plate, None)
    features = mask_geometry(mask.mask, plate)

    assert mask.area_plate_share > 0.10
    assert features.center_x is not None
    assert classify_state(features) in {HALF_STATE, FLAT_STATE}


def test_exposure_shift_uses_signed_lightness_delta() -> None:
    frame = np.full((3, 3, 3), 10, dtype=np.uint8)
    reference = np.full((3, 3, 3), 40, dtype=np.uint8)

    assert exposure_shift(frame, reference) < 0


def test_reference_suppresses_static_glare() -> None:
    reference = np.zeros((180, 180, 3), dtype=np.uint8)
    frame = reference.copy()
    plate = PlateEllipse(90, 90, 160, 160, 0, "test")
    cv2.circle(reference, (45, 45), 15, (255, 255, 255), -1)
    cv2.circle(frame, (45, 45), 15, (255, 255, 255), -1)
    cv2.circle(frame, (90, 90), 35, (220, 220, 220), -1)

    mask = crepe_mask(frame, plate, reference)

    assert exposure_shift(frame, reference) == 0.0
    assert mask.area_px > 0
    assert mask.component_count == 1


def test_fold_state_reference_rules() -> None:
    empty = mask_geometry(
        np.zeros((50, 50), dtype=np.uint8),
        PlateEllipse(25, 25, 40, 40, 0, "test"),
    )
    assert classify_state(empty) == EMPTY_STATE

    half_mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(half_mask, (40, 40), (60, 60), 255, -1)
    features = mask_geometry(half_mask, PlateEllipse(50, 50, 80, 80, 0, "test"))

    assert classify_state(features, flat_area_plate_share=0.20) == HALF_STATE
    assert classify_state(features, flat_area_plate_share=0.90) == QUARTER_STATE


def test_state_machine_requires_flat_after_empty_and_empty_hold() -> None:
    machine = FoldStateMachine()

    assert machine.update(HALF_STATE) == EMPTY_STATE
    assert machine.update(FLAT_STATE) == FLAT_STATE
    assert machine.update(HALF_STATE) == HALF_STATE
    for _ in range(29):
        assert machine.update(EMPTY_STATE) == HALF_STATE
    assert machine.update(EMPTY_STATE) == EMPTY_STATE


def test_eval_tables_and_dashboard(tmp_path) -> None:
    features = pd.DataFrame(
        [
            {
                "run_id": 1,
                "check_name": "plate_clean",
                "check_ok": True,
                "expected_state": EMPTY_STATE,
                "predicted_state": EMPTY_STATE,
                "machine_name": "robot",
                "diameter_plate_share": 0.0,
                "offset_plate_share": 0.0,
                "hole_fraction": 0.0,
                "circularity": 0.0,
                "solidity": 0.0,
                "center_x": None,
                "edge_angle_deg": None,
                "corner_x": None,
            },
            {
                "run_id": 1,
                "check_name": "spread",
                "check_ok": True,
                "expected_state": FLAT_STATE,
                "predicted_state": FLAT_STATE,
                "machine_name": "robot",
                "diameter_plate_share": 0.7,
                "offset_plate_share": 0.1,
                "hole_fraction": 0.0,
                "circularity": 0.9,
                "solidity": 0.95,
                "center_x": 10.0,
                "edge_angle_deg": None,
                "corner_x": None,
            },
            {
                "run_id": 1,
                "check_name": "fold2",
                "check_ok": False,
                "expected_state": QUARTER_STATE,
                "predicted_state": HALF_STATE,
                "machine_name": "robot",
                "diameter_plate_share": 0.3,
                "offset_plate_share": 0.2,
                "hole_fraction": 0.1,
                "circularity": 0.5,
                "solidity": 0.8,
                "center_x": 11.0,
                "edge_angle_deg": 45.0,
                "corner_x": None,
            },
        ]
    )

    matrix = confusion_matrix(features)
    tables = metric_tables(features)
    dashboard = build_dashboard(tmp_path)

    assert matrix.loc[EMPTY_STATE, EMPTY_STATE] == 1
    assert tables["overall"].loc[0, "agreement"] == 1.0
    assert dashboard.exists()
