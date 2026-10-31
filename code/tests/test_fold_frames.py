"""Synthetic checks for phase-aligned video timestamps."""

from datetime import datetime

import pytest

from maus_export.fold_frames import check_events, phase_clock_origin, video_time


def test_intersection_uses_midpoint_not_first_phase() -> None:
    phases = [
        {"timestamp": "2026-01-01T10:00:01", "elapsed_sec": 1.4},
        {"timestamp": "2026-01-01T10:00:03", "elapsed_sec": 3.1},
    ]
    origin = phase_clock_origin(phases)
    assert origin == datetime.fromisoformat("2026-01-01T10:00:00.250000")
    logs = [
        {"timestamp": "2026-01-01T10:00:02.100", "msg": "check_fold1 started"},
        {"timestamp": "2026-01-01T10:00:02.200", "msg": "unrelated"},
        {"timestamp": "2026-01-01T10:00:02.300", "msg": "check_fold1: failed"},
    ]
    event = check_events(logs)[0]
    assert event.state == "half"
    assert not event.check_ok
    assert video_time(event, origin) == pytest.approx(1.95)


def test_all_checks_and_unfinished_results() -> None:
    timestamp = "2026-01-01T10:00:00"
    messages = [
        "check_plate_clean started",
        "check_plate_clean: hotplate clean enough",
        "check_spread started",
        "check_spread: crepe spread successfully",
        "check_fold1 started",
        "check_fold1: first fold completed",
        "check_fold2 started",
        "check_fold2: second fold completed",
        "check_fold2 started",
    ]
    events = check_events(
        [{"timestamp": timestamp, "msg": m} for m in messages]
    )
    assert [e.state for e in events] == ["empty", "flat", "half", "quarter"]
    assert all(e.check_ok for e in events)


def test_no_valid_clock_origin() -> None:
    with pytest.raises(ValueError):
        phase_clock_origin([])
    with pytest.raises(ValueError):
        phase_clock_origin(
            [
                {"timestamp": "2026-01-01T10:00:00", "elapsed_sec": 0},
                {"timestamp": "2026-01-01T10:00:02", "elapsed_sec": 0},
            ]
        )
    with pytest.raises(ValueError):
        phase_clock_origin(
            [{"timestamp": "2026-01-01T10:00:00+00:00", "elapsed_sec": 0}]
        )
