"""Extract private, rectified check frames from cached runs and videos."""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import pandas as pd

from .fold_frames import check_events, phase_clock_origin, video_time
from .fold_rectify import rectify_video_frame

MAX_DURATION_DIFFERENCE = 2.0
MILLISECONDS_PER_SECOND = 1000
CALIBRATION_PATH = (
    Path(__file__).resolve().parents[2] / "robot-image-calibration.json"
)
FRAME_COLUMNS = [
    "frame_id",
    "run_id",
    "robot",
    "check",
    "expected_state",
    "check_ok",
    "video_t",
    "source_width",
    "source_height",
    "path",
]


def run_duration(run: dict[str, Any]) -> float:
    """Get the elapsed UTC run duration, rejecting missing timezone data."""
    start = datetime.fromisoformat(run["tsStart"].replace("Z", "+00:00"))
    end = datetime.fromisoformat(run["tsEnd"].replace("Z", "+00:00"))
    if start.utcoffset() is None or end.utcoffset() is None:
        raise ValueError("Run timestamps must have timezone offsets")
    duration = (end - start).total_seconds()
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid run duration")
    return duration


def extract_frames(
    data_dir: Path, calibration_path: Path = CALIBRATION_PATH
) -> dict[str, int]:
    """Extract aligned frames and parquet, refusing to overwrite old outputs.

    Run-level input failures are counted, not printed with private metadata.
    Staging prevents partial frames from appearing as a completed dataset.
    """
    calibration = json.loads(calibration_path.read_text())
    output = data_dir / "fold_segmentation"
    output.mkdir(parents=True, exist_ok=True)
    frames_dir = output / "frames"
    table_path = output / "frames.parquet"
    if frames_dir.exists() or table_path.exists():
        raise FileExistsError(
            "Frame dataset already exists; move it aside first"
        )
    counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix=".frames-", dir=output) as temp:
        stage = Path(temp)
        (stage / "frames").mkdir()
        for path in sorted((data_dir / "raw" / "runs").glob("*.json")):
            video = data_dir / "raw" / "media" / path.stem / "video.mp4"
            if not video.is_file():
                continue
            try:
                run = json.loads(path.read_text())
                origin = phase_clock_origin(run["phases"])
                events = check_events(run["logs"])
                expected_duration = run_duration(run)
                robot = run["machineName"]
            except (ValueError, TypeError, KeyError):
                counts["invalid_metadata"] += 1
                continue
            capture = cv2.VideoCapture(str(video))
            run_files: list[Path] = []
            try:
                fps = capture.get(cv2.CAP_PROP_FPS)
                frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
                if not capture.isOpened() or not math.isfinite(fps) or fps <= 0:
                    counts["unreadable_video"] += 1
                    continue
                duration = frame_count / fps
                if (
                    not math.isfinite(duration)
                    or abs(duration - expected_duration)
                    > MAX_DURATION_DIFFERENCE
                ):
                    counts["duration_mismatch"] += 1
                    continue
                width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if robot not in calibration.get("robot_corrections", {}):
                    counts["missing_correction"] += 1
                    continue
                run_rows = []
                for index, event in enumerate(events):
                    time = video_time(event, origin)
                    if not 0 <= time < duration:
                        counts["check_outside_video"] += 1
                        continue
                    capture.set(
                        cv2.CAP_PROP_POS_MSEC, time * MILLISECONDS_PER_SECOND
                    )
                    success, frame = capture.read()
                    if not success:
                        counts["unreadable_frame"] += 1
                        continue
                    corrected, _ = rectify_video_frame(
                        frame, robot, calibration
                    )
                    frame_id = f"{path.stem}_{index:03d}_{event.name}"
                    destination = stage / "frames" / f"{frame_id}.jpg"
                    if not cv2.imwrite(str(destination), corrected):
                        raise OSError("Unable to write frame")
                    run_files.append(destination)
                    run_rows.append(
                        {
                            "frame_id": frame_id,
                            "run_id": int(path.stem),
                            "robot": robot,
                            "check": event.name,
                            "expected_state": event.state,
                            "check_ok": event.check_ok,
                            "video_t": time,
                            "source_width": width,
                            "source_height": height,
                            "path": f"frames/{frame_id}.jpg",
                        }
                    )
                rows.extend(run_rows)
                counts["runs"] += bool(run_rows)
            except (ValueError, KeyError, TypeError):
                for item in run_files:
                    item.unlink(missing_ok=True)
                counts["invalid_video_or_correction"] += 1
            finally:
                capture.release()
        if not rows:
            raise ValueError("No usable check frames found")
        pd.DataFrame(rows, columns=FRAME_COLUMNS).to_parquet(
            stage / "frames.parquet", index=False
        )
        shutil.move(str(stage / "frames"), str(frames_dir))
        try:
            shutil.move(str(stage / "frames.parquet"), str(table_path))
        except Exception:
            shutil.rmtree(frames_dir)
            raise
    counts["frames"] = len(rows)
    return dict(counts)
