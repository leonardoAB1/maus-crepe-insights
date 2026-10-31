"""Sequential live-style fold inference for full run videos."""

from __future__ import annotations

import csv
import json
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .fold_frames import correct_perspective, video_path
from .fold_geometry import mask_geometry
from .fold_mask import crepe_mask
from .fold_plate import detect_plate_ellipse
from .fold_state import (
    EMPTY_STATE,
    FLAT_STATE,
    HALF_STATE,
    QUARTER_STATE,
    classify_state,
)

ANALYSIS_FPS = 10.0
TARGET_LONG_SIDE = 900
VOTE_WINDOW = 9
EMPTY_HOLD_FRAMES = 30
STATE_INDEX = {
    EMPTY_STATE: 0,
    FLAT_STATE: 1,
    HALF_STATE: 2,
    QUARTER_STATE: 3,
}
STATE_COLOUR = {
    EMPTY_STATE: (120, 120, 120),
    FLAT_STATE: (0, 180, 0),
    HALF_STATE: (0, 160, 220),
    QUARTER_STATE: (0, 80, 255),
}


def majority_vote(states: deque[str]) -> str:
    """Return the most common state, breaking ties by recency."""
    counts = Counter(states)
    best_count = max(counts.values())
    for state in reversed(states):
        if counts[state] == best_count:
            return state
    return states[-1]


class FoldStateMachine:
    """Monotonic fold state machine with held empty reset."""

    def __init__(self) -> None:
        self.state = EMPTY_STATE
        self.empty_count = 0

    def update(self, observed: str) -> str:
        """Update and return the filtered state."""
        if observed == EMPTY_STATE:
            self.empty_count += 1
            if self.empty_count >= EMPTY_HOLD_FRAMES:
                self.state = EMPTY_STATE
            return self.state
        self.empty_count = 0
        if self.state == EMPTY_STATE and observed != FLAT_STATE:
            return self.state
        if STATE_INDEX[observed] >= STATE_INDEX[self.state]:
            self.state = observed
        return self.state


def _resize_for_live(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = TARGET_LONG_SIDE / max(height, width)
    if scale >= 1.0:
        return frame
    size = (round(width * scale), round(height * scale))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def _annotate(frame: np.ndarray, state: str, seconds: float) -> np.ndarray:
    output = frame.copy()
    cv2.putText(
        output,
        f"{seconds:7.1f}s  {state}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        STATE_COLOUR[state],
        2,
        cv2.LINE_AA,
    )
    return output


def analyze_video(
    video: Path,
    output_dir: Path,
    machine_name: str | None = None,
    run_id: int | str | None = None,
) -> list[dict[str, Any]]:
    """Analyse a video sequentially and write timeline artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    writer = None
    rows: list[dict[str, Any]] = []
    votes: deque[str] = deque(maxlen=VOTE_WINDOW)
    machine = FoldStateMachine()
    timings: list[float] = []
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {video}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        source_step = max(1, round(fps / ANALYSIS_FPS))
        frame_index = 0
        flat_reference: float | None = None
        while True:
            grabbed = capture.grab()
            if not grabbed:
                break
            if frame_index % source_step:
                frame_index += 1
                continue
            ok, frame = capture.retrieve()
            if not ok:
                break
            start = time.perf_counter()
            small = _resize_for_live(frame)
            corrected = correct_perspective(small, machine_name)
            plate = detect_plate_ellipse(corrected)
            mask = crepe_mask(corrected, plate, None)
            geometry = mask_geometry(mask.mask, plate)
            raw_state = classify_state(geometry, flat_reference)
            if raw_state == FLAT_STATE and flat_reference is None:
                flat_reference = geometry.area_plate_share
            votes.append(raw_state)
            voted = majority_vote(votes)
            state = machine.update(voted)
            seconds = frame_index / fps
            timings.append((time.perf_counter() - start) * 1000.0)
            annotated = _annotate(corrected, state, seconds)
            if writer is None:
                writer = cv2.VideoWriter(
                    str(output_dir / "annotated.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    ANALYSIS_FPS,
                    (annotated.shape[1], annotated.shape[0]),
                )
                if not writer.isOpened():
                    raise ValueError("Cannot open video writer")
            writer.write(annotated)
            rows.append(
                {
                    "run_id": run_id,
                    "second": seconds,
                    "raw_state": raw_state,
                    "voted_state": voted,
                    "state": state,
                    "area_plate_share": geometry.area_plate_share,
                    "edge_count": len(geometry.straight_edges),
                }
            )
            frame_index += 1
        if not rows:
            raise ValueError("Video has no analysed frames")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    _write_timeline(output_dir / "timeline.csv", rows)
    _write_timeline_plot(output_dir / "timeline.png", rows)
    median_ms = float(np.median(timings)) if timings else float("nan")
    (output_dir / "summary.json").write_text(
        json.dumps({"median_ms_per_frame": median_ms}, indent=2),
        encoding="utf-8",
    )
    return rows


def _write_timeline(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_timeline_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    seconds = [float(row["second"]) for row in rows]
    states = [STATE_INDEX[str(row["state"])] for row in rows]
    fig, axis = plt.subplots(figsize=(8, 2.5))
    axis.step(seconds, states, where="post")
    axis.set_yticks(list(STATE_INDEX.values()), list(STATE_INDEX))
    axis.set_xlabel("video second")
    axis.set_ylabel("state")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run_live(data_dir: Path, run_ids: set[int] | None = None) -> None:
    """Run live-style analysis for cached videos."""
    run_dir = data_dir / "raw" / "runs"
    output_root = data_dir / "fold_segmentation" / "live"
    for run_path in sorted(run_dir.glob("*.json")):
        run_id = int(run_path.stem)
        if run_ids is not None and run_id not in run_ids:
            continue
        run = json.loads(run_path.read_text(encoding="utf-8"))
        video = video_path(data_dir, run_id)
        if video is None:
            continue
        analyze_video(
            video,
            output_root / str(run_id),
            run.get("machineName"),
            run_id,
        )
