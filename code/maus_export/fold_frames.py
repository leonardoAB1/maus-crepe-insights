"""Extract corrected hotplate frames at robot check moments."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_PATH = REPO_ROOT / "robot-image-calibration.json"
OUTPUT_DIR = "fold_segmentation"
FRAME_DIR = "frames"
LOCAL_FORMATS = (
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
)
UTC_FORMATS = (
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S.%f%z",
    "%Y-%m-%d %H:%M:%S%z",
)
CHECKS = ("plate_clean", "spread", "fold1", "fold2")
CHECK_STATE = {
    "plate_clean": "empty",
    "spread": "flat",
    "fold1": "half",
    "fold2": "quarter",
}
CHECK_OK_PREFIX = {
    "plate_clean": "hotplate clean",
    "spread": "crepe spread successfully",
    "fold1": "first fold completed",
    "fold2": "second fold completed",
}
FRAME_LEAD_SECONDS = 0.1
MAX_VIDEO_DURATION_ERROR_SECONDS = 2.0
TARGET_SOURCE_SIZE = (900, 509)
DEFAULT_TARGET_SIZE = (900, 900)
VIDEO_EXTENSIONS = (".mp4", ".MP4")


@dataclass(frozen=True)
class CheckFrame:
    """One extracted check frame row."""

    run_id: int
    check_name: str
    expected_state: str
    check_ok: bool
    log_time: datetime
    video_t: float
    frame_path: str
    machine_name: str | None
    terminal_status: str | None
    width: int
    height: int
    decoded_frame_index: int


def parse_local_time(value: str) -> datetime:
    """Parse a naive robot-local timestamp."""
    text = value.strip().replace("Z", "")
    for fmt in LOCAL_FORMATS:
        try:
            parsed = datetime.strptime(text, fmt)
            return parsed.replace(tzinfo=None)
        except ValueError:
            pass
    return datetime.fromisoformat(text).replace(tzinfo=None)


def parse_utc_time(value: str) -> datetime:
    """Parse an exported UTC timestamp, accepting a trailing Z."""
    text = value.strip().replace("Z", "+00:00")
    for fmt in UTC_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        raise ValueError("UTC timestamp must include an offset")
    return parsed


def phase_clock_offset(phases: Iterable[dict[str, Any]]) -> datetime:
    """Return the midpoint of the phase-derived run-start clock interval."""
    lower: datetime | None = None
    upper: datetime | None = None
    one_second = timedelta(seconds=1)
    for phase in phases:
        if phase.get("timestamp") is None or phase.get("elapsed_sec") is None:
            continue
        timestamp = parse_local_time(str(phase["timestamp"]))
        elapsed = timedelta(seconds=float(phase["elapsed_sec"]))
        start = timestamp - elapsed
        end = start + one_second
        lower = start if lower is None else max(lower, start)
        upper = end if upper is None else min(upper, end)
    if lower is None or upper is None:
        raise ValueError("Run has no phase timestamps")
    if lower > upper:
        raise ValueError("Phase timestamps have no common clock offset")
    return lower + (upper - lower) / 2


def check_ok(check_name: str, message: str) -> bool:
    """Return whether a result line is the expected passed check."""
    prefix = CHECK_OK_PREFIX[check_name]
    return message.strip().lower().startswith(prefix)


def check_events(run: dict[str, Any]) -> list[tuple[str, datetime, bool]]:
    """Parse check start/result pairs from logs."""
    events: list[tuple[str, datetime, bool]] = []
    pending: dict[str, datetime] = {}
    logs = sorted(
        run.get("logs") or [],
        key=lambda item: item["timestamp"],
    )
    for log in logs:
        message = str(log.get("msg") or "").strip()
        timestamp = parse_local_time(str(log["timestamp"]))
        for name in CHECKS:
            if message == f"check_{name} started":
                pending[name] = timestamp
                break
            prefix = f"check_{name}:"
            if message.startswith(prefix) and name in pending:
                result = message[len(prefix) :].strip()
                events.append((name, pending.pop(name), check_ok(name, result)))
                break
    return events


def video_duration_seconds(video: Path) -> float:
    """Read video duration from OpenCV metadata."""
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {video}")
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = capture.get(cv2.CAP_PROP_FPS)
        if frames <= 0 or fps <= 0:
            raise ValueError(f"Cannot read duration: {video}")
        return float(frames / fps)
    finally:
        capture.release()


def run_duration_seconds(run: dict[str, Any]) -> float:
    """Return exported run duration from UTC start and end."""
    start = parse_utc_time(str(run["tsStart"]))
    end = parse_utc_time(str(run["tsEnd"]))
    return float((end - start).total_seconds())


def video_path(data_dir: Path, run_id: int | str) -> Path | None:
    """Return the cached video path for a run if present."""
    media = data_dir / "raw" / "media" / str(run_id)
    for suffix in VIDEO_EXTENSIONS:
        candidate = media / f"video{suffix}"
        if candidate.exists():
            return candidate
    matches = sorted(media.glob("*.mp4"))
    return matches[0] if matches else None


def _load_calibration(path: Path = CALIBRATION_PATH) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def correct_perspective(
    frame: np.ndarray,
    machine_name: str | None,
    calibration: dict[str, Any] | None = None,
) -> np.ndarray:
    """Apply the shared robot hotplate perspective correction."""
    config = calibration if calibration is not None else _load_calibration()
    robots = config.get("robot_corrections") or {}
    correction = robots.get(machine_name or "")
    source = cv2.resize(frame, TARGET_SOURCE_SIZE, interpolation=cv2.INTER_AREA)
    if not correction:
        return cv2.resize(
            source,
            DEFAULT_TARGET_SIZE,
            interpolation=cv2.INTER_AREA,
        )
    matrix = np.asarray(correction["matrix_mean"], dtype=np.float64)
    target_size = tuple(correction.get("target_size_px", DEFAULT_TARGET_SIZE))
    return cv2.warpPerspective(source, matrix, target_size)


def read_video_frame(video: Path, second: float) -> tuple[np.ndarray, int]:
    """Read one frame by timestamp for dataset extraction."""
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {video}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        index = max(0, min(total - 1, int(round(second * fps))))
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"Cannot decode frame {index}: {video}")
        return frame, index
    finally:
        capture.release()


def frame_rows_for_run(
    run_path: Path,
    data_dir: Path,
    output_dir: Path,
    *,
    write_images: bool = True,
) -> list[CheckFrame]:
    """Extract all check frames for one run."""
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run_id = int(run.get("id") or run_path.stem)
    video = video_path(data_dir, run_id)
    if video is None:
        return []
    if abs(video_duration_seconds(video) - run_duration_seconds(run)) > (
        MAX_VIDEO_DURATION_ERROR_SECONDS
    ):
        return []
    offset = phase_clock_offset(run.get("phases") or [])
    calibration = _load_calibration()
    rows: list[CheckFrame] = []
    frame_dir = output_dir / FRAME_DIR
    frame_dir.mkdir(parents=True, exist_ok=True)
    for name, timestamp, ok in check_events(run):
        video_t = (timestamp - offset).total_seconds() + FRAME_LEAD_SECONDS
        raw, decoded = read_video_frame(video, video_t)
        corrected = correct_perspective(
            raw,
            run.get("machineName"),
            calibration,
        )
        height, width = corrected.shape[:2]
        relative = Path(OUTPUT_DIR) / FRAME_DIR / f"{run_id}_{name}.jpg"
        path = output_dir.parent / relative
        if write_images and not cv2.imwrite(str(path), corrected):
            raise OSError(f"Cannot write frame: {path}")
        rows.append(
            CheckFrame(
                run_id=run_id,
                check_name=name,
                expected_state=CHECK_STATE[name],
                check_ok=ok,
                log_time=timestamp,
                video_t=float(video_t),
                frame_path=relative.as_posix(),
                machine_name=run.get("machineName"),
                terminal_status=run.get("terminal_status"),
                width=width,
                height=height,
                decoded_frame_index=decoded,
            )
        )
    return rows


def build_frames(
    data_dir: Path,
    run_ids: set[int] | None = None,
) -> pd.DataFrame:
    """Build frames and frames.parquet under data/fold_segmentation/."""
    output_dir = data_dir / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for run_path in sorted((data_dir / "raw" / "runs").glob("*.json")):
        run_id = int(run_path.stem)
        if run_ids is not None and run_id not in run_ids:
            continue
        for row in frame_rows_for_run(run_path, data_dir, output_dir):
            item = asdict(row)
            item["log_time"] = row.log_time.isoformat()
            rows.append(item)
    table = pd.DataFrame(rows)
    table.to_parquet(output_dir / "frames.parquet", index=False)
    return table
