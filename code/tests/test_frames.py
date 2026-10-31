"""Offline tests for the frame-level check table, on synthetic runs."""

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from maus_classify.frames import (
    OUTPUT,
    SCHEMA,
    FramesError,
    build_frames,
    reason,
    runs_to_frames,
)


def check(phase: str, ok: bool, i: int, ts: str, errors=None, msg="ok"):
    return {
        "phase": phase,
        "ok": ok,
        "errors": errors or [],
        "message": msg,
        "frame": f"{i:02d}_{phase}_frame.jpg",
        "overlay": f"{i:02d}_{phase}_overlay.jpg",
        "timestamp": ts,
    }


def make_run(run_id: int, start: str, checks: list, logs=None) -> dict:
    return {
        "id": run_id,
        "machineName": "maus-a",
        "tsStart": start,
        "visionChecks": checks,
        "logs": logs or [],
        "parameters": {},
    }


# Robot local time is UTC+2 here (summer time), measured from the first log
SUMMER = make_run(
    1,
    "2026-10-13T11:25:00.000Z",
    [
        check("spread", True, 0, "2026-10-13T13:25:33"),
        check(
            "fold1",
            False,
            1,
            "2026-10-13T13:26:00",
            [
                "Not a half fold",
                " batter  left on the plate",
                "not a half fold",
            ],
        ),
        check("fold1", True, 2, "2026-10-13T13:26:30"),
        check("fold2", True, 3, "2026-10-13T13:27:00"),
        check("push_off", False, 4, "2026-10-13T13:28:00", msg="retry"),
    ],
    logs=[{"timestamp": "2026-10-13T13:25:00", "level": 20, "msg": "x"}],
)
# No logs: falls back to Europe/Zurich, which is UTC+1 after 2026-10-25
WINTER = make_run(
    2,
    "2026-10-30T16:00:00.000Z",
    [
        check(
            "spread",
            False,
            0,
            "2026-10-30T17:00:00",
            msg="No crepe in the plate",
        )
    ],
)
NO_CHECKS = make_run(3, "2026-08-01T10:00:00.000Z", [])


def values(series: pd.Series) -> list:
    """Column values with missing as None (pandas uses NaN for str)."""
    return [None if pd.isna(v) else v for v in series]


def write_media(data_dir: Path, run: dict, skip: set[int] = frozenset()):
    folder = data_dir / "raw" / "media" / str(run["id"])
    folder.mkdir(parents=True, exist_ok=True)
    for i, vc in enumerate(run["visionChecks"]):
        if i not in skip:
            (folder / vc["frame"]).write_bytes(b"jpg")


@pytest.fixture
def frames(tmp_path: Path) -> pd.DataFrame:
    write_media(tmp_path, SUMMER, skip={3})
    write_media(tmp_path, WINTER)
    return runs_to_frames([WINTER, NO_CHECKS, SUMMER], tmp_path)


def test_one_row_per_check_attempt(frames: pd.DataFrame) -> None:
    assert list(frames.columns) == SCHEMA.names
    assert len(frames) == 6
    assert frames["run_id"].tolist() == [1, 1, 1, 1, 1, 2]
    assert frames["check_index"].tolist() == [0, 1, 2, 3, 4, 0]


def test_attempts_and_last_attempt(frames: pd.DataFrame) -> None:
    run = frames[frames["run_id"] == 1]
    assert run["attempt"].tolist() == [1, 1, 2, 1, 1]
    assert run["is_last_attempt"].tolist() == [True, False, True, True, True]


def test_fold_class_only_for_passed_shape_checks(frames) -> None:
    assert values(frames["fold_class"]) == [
        "flat",
        None,
        "half",
        "quarter",
        None,
        None,
    ]


def test_reason_normalized_and_null_when_ok(frames: pd.DataFrame) -> None:
    assert values(frames["reason"]) == [
        None,
        "batter left on the plate; not a half fold",
        None,
        None,
        "retry",
        "no crepe in the plate",
    ]
    assert frames["errors"].iloc[1] == [
        "Not a half fold",
        " batter  left on the plate",
        "not a half fold",
    ]


def test_reason_null_when_failed_without_text() -> None:
    assert reason(False, [], None) is None
    assert reason(True, ["x"], "y") is None


def test_timestamps_converted_to_utc(frames: pd.DataFrame) -> None:
    assert str(frames["ts_utc"].dt.tz) == "UTC"
    first = frames["ts_utc"].iloc[0]
    assert first == pd.Timestamp("2026-10-13T11:25:33Z")
    assert frames["ts_utc"].iloc[-1] == pd.Timestamp("2026-10-30T16:00:00Z")


def test_frame_path_relative_and_null_when_missing(frames) -> None:
    assert frames["frame_path"].iloc[0] == "raw/media/1/00_spread_frame.jpg"
    assert frames["frame_path"].isna().tolist() == [
        False,
        False,
        False,
        True,
        False,
        False,
    ]


def test_split_is_per_run_and_versioned(frames) -> None:
    from maus_classify.split import SPLIT_VERSION

    assert frames.groupby("run_id")["split"].nunique().max() == 1
    split = frames.groupby("run_id")["split"].first().to_dict()
    # one robot with fewer than MIN_RUNS_FOR_TEST runs: it stays in train
    assert split == {1: "train", 2: "train"}
    assert set(frames["split_version"]) == {SPLIT_VERSION}


def test_build_writes_parquet(tmp_path: Path) -> None:
    runs_dir = tmp_path / "raw" / "runs"
    runs_dir.mkdir(parents=True)
    for run in (SUMMER, WINTER, NO_CHECKS):
        (runs_dir / f"{run['id']}.json").write_text(
            json.dumps(run), encoding="utf-8"
        )
    build_frames(tmp_path)
    table = pq.read_table(tmp_path / OUTPUT)
    assert table.schema.equals(SCHEMA)
    assert table.num_rows == 6


def test_build_without_runs_fails(tmp_path: Path) -> None:
    with pytest.raises(FramesError):
        build_frames(tmp_path)


def test_duplicate_run_ids_fail(tmp_path: Path) -> None:
    with pytest.raises(FramesError):
        runs_to_frames([SUMMER, SUMMER], tmp_path)
