"""Crêpe masks from the distilled student model and a plate-free fold state.

The student (``models/crepe-student``) is a small segmentation network trained
on SAM 3 crêpe masks. It runs on the raw camera frame, not on the corrected
view, because that is what it was trained on.

The fold state needs no plate: it compares the largest mask part with the flat
crêpe of the same run. Half a crêpe covers about 0.39 of its flat area and a
quarter about 0.21 on all three robots, while absolute plate shares differ
between robots and camera setups.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
import pandas as pd

from .fold_live import (
    ANALYSIS_FPS,
    EMPTY_HOLD_FRAMES,
    STATE_COLOUR,
    STATE_INDEX,
    VOTE_WINDOW,
    majority_vote,
)
from .fold_state import EMPTY_STATE, FLAT_STATE, HALF_STATE, QUARTER_STATE

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "code" / "models" / "crepe-student" / "model.onnx"
INPUT_NAME = "image"
INPUT_SIZE = (384, 216)  # width, height
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
BYTE_MAX = 255.0
CPU_THREADS = 4

# Largest mask part as a share of the model output: below this the frame is
# empty. Empty check frames stay below 0.007, flat crêpes start near 0.11.
MIN_CREPE_SHARE = 0.02
# Area relative to the flat crêpe of the run (check frames: half 0.35 to
# 0.50, quarter 0.18 to 0.28 between the 2nd and 98th percentile).
FLAT_REFERENCE_SHARE = 0.60
HALF_REFERENCE_SHARE = 0.30
# While batter is poured and spread the flat area keeps growing, and the arm
# hiding part of it looks like a fold. A fold is only accepted once the flat
# area has not grown by more than FLAT_GROWTH for FLAT_STABLE_FRAMES frames
# (5 s at 10 analysed frames per second).
FLAT_GROWTH = 0.02
FLAT_STABLE_FRAMES = 50

OUTPUT_DIR = "fold_segmentation"
STUDENT_DIR = "student"
MASK_COLOUR = (0, 200, 0)
MASK_ALPHA = 0.45
TEXT_ORIGIN = (20, 40)
TEXT_SCALE = 1.0
TEXT_THICKNESS = 2
MS_PER_S = 1000.0
# Run split of the student training: sha1(run id) mod 5 == 0 is a test run.
TEST_MOD = 5


class Session(Protocol):
    """The part of an onnxruntime session the student uses."""

    def run(self, output_names: Any, feeds: dict[str, np.ndarray]) -> list:
        """Run the graph."""


def preprocess(frame: np.ndarray) -> np.ndarray:
    """Raw BGR frame -> normalised NCHW float32 tensor of the student."""
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("expected a BGR frame")
    small = cv2.resize(frame, INPUT_SIZE)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32)
    x = (rgb / BYTE_MAX - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def open_session(path: Path = MODEL_PATH, threads: int = CPU_THREADS) -> Any:
    """An onnxruntime CPU session for the student (needs the ml env)."""
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise ImportError(
            "the crepe student needs onnxruntime: use `pixi run -e ml`"
        ) from error
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    return ort.InferenceSession(
        str(path), options, providers=["CPUExecutionProvider"]
    )


@dataclass
class CrepeStudent:
    """Student mask for one raw frame, at the model resolution."""

    session: Session

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> CrepeStudent:
        """Load the shipped ONNX model."""
        return cls(open_session(path))

    def mask(self, frame: np.ndarray) -> np.ndarray:
        """Boolean crêpe mask of shape ``INPUT_SIZE[::-1]``."""
        out = self.session.run(None, {INPUT_NAME: preprocess(frame)})[0]
        return np.asarray(out)[0, 0] > 0


def largest_part(mask: np.ndarray) -> np.ndarray:
    """The largest connected part of a boolean mask."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8)
    )
    if count <= 1:
        return np.zeros(mask.shape, dtype=bool)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == biggest


def area_share(mask: np.ndarray) -> float:
    """Largest part of the mask as a share of all pixels."""
    if mask.size == 0:
        return 0.0
    return float(largest_part(mask).sum() / mask.size)


def classify_area(area: float, flat_area: float | None) -> str:
    """Fold state of one frame from its area and the run's flat area.

    Without a flat area any crêpe counts as flat: the first crêpe that
    appears on an empty plate is always a spread, however small.
    """
    if area < MIN_CREPE_SHARE:
        return EMPTY_STATE
    if not flat_area:
        return FLAT_STATE
    relative = area / flat_area
    if relative >= FLAT_REFERENCE_SHARE:
        return FLAT_STATE
    if relative >= HALF_REFERENCE_SHARE:
        return HALF_STATE
    return QUARTER_STATE


class StudentStateMachine:
    """Live fold state from per-frame areas.

    A majority vote smooths single frames. The state only moves forward
    (flat, half, quarter); empty must hold ``EMPTY_HOLD_FRAMES`` frames so
    a spatula hiding the crêpe does not reset it. The flat area is the
    largest area seen while the crêpe reads as flat, and is forgotten when
    the plate is empty again (several crêpes per video). A fold needs a
    flat area that has stopped growing (``FLAT_STABLE_FRAMES``).
    """

    def __init__(self) -> None:
        self.state = EMPTY_STATE
        self.flat_area: float | None = None
        self.votes: deque[str] = deque(maxlen=VOTE_WINDOW)
        self.empty_count = 0
        self.stable_count = 0

    def _track_flat_area(self, area: float) -> None:
        if self.flat_area is None or area > self.flat_area * (1 + FLAT_GROWTH):
            self.stable_count = 0
        else:
            self.stable_count += 1
        self.flat_area = max(self.flat_area or 0.0, area)

    def update(self, area: float) -> tuple[str, str]:
        """Feed one frame area; return (single-frame state, live state)."""
        raw = classify_area(area, self.flat_area)
        if raw == FLAT_STATE:
            self._track_flat_area(area)
        elif raw != EMPTY_STATE:
            self.stable_count += 1
        if raw in (HALF_STATE, QUARTER_STATE) and (
            self.stable_count < FLAT_STABLE_FRAMES
        ):
            raw = FLAT_STATE
        self.votes.append(raw)
        voted = majority_vote(self.votes)
        if voted == EMPTY_STATE:
            self.empty_count += 1
            if self.empty_count >= EMPTY_HOLD_FRAMES:
                self.state = EMPTY_STATE
                self.flat_area = None
                self.stable_count = 0
            return raw, self.state
        self.empty_count = 0
        if self.state == EMPTY_STATE:
            self.state = FLAT_STATE
        elif STATE_INDEX[voted] > STATE_INDEX[self.state]:
            self.state = voted
        return raw, self.state


def overlay(frame: np.ndarray, mask: np.ndarray, label: str) -> np.ndarray:
    """Frame with the mask painted in and a state label."""
    full = cv2.resize(
        mask.astype(np.uint8),
        (frame.shape[1], frame.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    output = frame.copy()
    colour = np.zeros_like(frame)
    colour[full] = MASK_COLOUR
    blended = cv2.addWeighted(frame, 1 - MASK_ALPHA, colour, MASK_ALPHA, 0)
    output[full] = blended[full]
    if not label:
        return output
    state = label.split()[-1]
    cv2.putText(
        output,
        label,
        TEXT_ORIGIN,
        cv2.FONT_HERSHEY_SIMPLEX,
        TEXT_SCALE,
        STATE_COLOUR.get(state, MASK_COLOUR),
        TEXT_THICKNESS,
        cv2.LINE_AA,
    )
    return output


def analyze_video(
    video: Path,
    output_dir: Path,
    student: CrepeStudent,
    run_id: int | None = None,
    robot: str | None = None,
) -> list[dict[str, Any]]:
    """Live-style student inference over a whole video.

    Writes ``annotated.mp4``, ``timeline.csv``, ``timeline.png`` and
    ``summary.json`` like ``fold_live.analyze_video``. Every frame with a
    crêpe also gets its pose in the robot's top view (``crepe_pose``).
    """
    from .crepe_pose import (
        POSE_FIELDS,
        load_calibration,
        pose_from_mask,
        pose_row,
    )
    from .fold_live import _write_timeline, _write_timeline_plot

    calibration = load_calibration()

    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    machine = StudentStateMachine()
    rows: list[dict[str, Any]] = []
    timings: list[float] = []
    writer = None
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {video}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        step = max(1, round(fps / ANALYSIS_FPS))
        index = -1
        while capture.grab():
            index += 1
            if index % step:
                continue
            ok, frame = capture.retrieve()
            if not ok:
                break
            start = time.perf_counter()
            mask = student.mask(frame)
            area = area_share(mask)
            raw, state = machine.update(area)
            pose = (
                pose_from_mask(
                    mask, robot, calibration, quarter=state == QUARTER_STATE
                )
                if state != EMPTY_STATE
                else None
            )
            timings.append((time.perf_counter() - start) * MS_PER_S)
            seconds = index / fps
            annotated = overlay(frame, mask, f"{seconds:7.1f}s  {state}")
            if writer is None:
                writer = cv2.VideoWriter(
                    str(output_dir / "annotated.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    ANALYSIS_FPS,
                    (annotated.shape[1], annotated.shape[0]),
                )
            writer.write(annotated)
            rows.append(
                {
                    "run_id": run_id,
                    "second": seconds,
                    "raw_state": raw,
                    "state": state,
                    "area_share": area,
                    "flat_area_share": machine.flat_area,
                    **dict.fromkeys(POSE_FIELDS),
                    **pose_row(pose),
                }
            )
        if not rows:
            raise ValueError("Video has no analysed frames")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    _write_timeline(output_dir / "timeline.csv", rows)
    _write_timeline_plot(output_dir / "timeline.png", rows)
    summary = {"median_ms_per_frame": float(np.median(timings))}
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return rows


def run_live(
    data_dir: Path,
    run_ids: set[int] | None = None,
    student: CrepeStudent | None = None,
) -> None:
    """Student live inference for cached run videos."""
    from .fold_frames import video_path

    student = student or CrepeStudent.load()
    output_root = data_dir / OUTPUT_DIR / STUDENT_DIR / "live"
    for run_path in sorted((data_dir / "raw" / "runs").glob("*.json")):
        run_id = int(run_path.stem)
        if run_ids is not None and run_id not in run_ids:
            continue
        video = video_path(data_dir, run_id)
        if video is None:
            continue
        run = json.loads(run_path.read_text(encoding="utf-8"))
        analyze_video(
            video,
            output_root / str(run_id),
            student,
            run_id,
            run.get("machineName"),
        )


def read_frame(video: Path, index: int) -> np.ndarray:
    """One decoded frame by index (the index ``fold-frames`` stored)."""
    capture = cv2.VideoCapture(str(video))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        raise ValueError(f"Cannot decode frame {index}: {video}")
    return frame


def check_frame_states(
    frames: pd.DataFrame,
    areas: pd.Series,
) -> pd.Series:
    """Offline state per check frame.

    The flat area of a run is the area at its spread check. A spread
    frame itself has no flat reference, as in the live state machine.
    """
    is_flat = frames["expected_state"] == FLAT_STATE
    with_crepe = areas >= MIN_CREPE_SHARE
    flat = areas[is_flat & with_crepe].groupby(frames["run_id"]).first()
    states = []
    for index, row in frames.iterrows():
        reference = None if is_flat[index] else flat.get(row["run_id"])
        states.append(classify_area(float(areas[index]), reference))
    return pd.Series(states, index=frames.index)


def evaluate_check_frames(
    data_dir: Path,
    student: CrepeStudent | None = None,
    frame_reader: Callable[[Path, int], np.ndarray] = read_frame,
) -> pd.DataFrame:
    """Student state at the robot check moments of ``fold-frames``.

    Reads ``fold_segmentation/frames.parquet``, decodes the raw frame of
    every check from the run video and writes
    ``fold_segmentation/student/features.parquet`` and ``metrics.csv``.
    """
    from .fold_frames import video_path

    student = student or CrepeStudent.load()
    root = data_dir / OUTPUT_DIR
    frames = pd.read_parquet(root / "frames.parquet")
    areas = []
    for row in frames.itertuples(index=False):
        video = video_path(data_dir, int(row.run_id))
        if video is None:
            areas.append(np.nan)
            continue
        frame = frame_reader(video, int(row.decoded_frame_index))
        areas.append(area_share(student.mask(frame)))
    frames = frames.assign(area_share=areas).dropna(subset=["area_share"])
    frames["predicted_state"] = check_frame_states(frames, frames["area_share"])
    out = root / STUDENT_DIR
    out.mkdir(parents=True, exist_ok=True)
    frames.to_parquet(out / "features.parquet", index=False)
    metrics = agreement(frames)
    metrics.to_csv(out / "metrics.csv", index=False)
    return metrics


def is_test_run(run_id: int) -> bool:
    """Whether a run was held out when the student was trained."""
    digest = hashlib.sha1(str(run_id).encode()).hexdigest()
    return int(digest, 16) % TEST_MOD == 0


def agreement(frames: pd.DataFrame) -> pd.DataFrame:
    """Agreement with the check moment on passed checks, per group."""
    passed = frames[frames["check_ok"].astype(bool)]
    passed = passed.assign(
        split=passed["run_id"].map(
            lambda run: "test" if is_test_run(int(run)) else "train"
        )
    )
    hit = passed["predicted_state"] == passed["expected_state"]
    rows = [{"group": "all", "frames": len(passed), "agreement": hit.mean()}]
    for column in ("split", "expected_state", "machine_name"):
        for value, part in hit.groupby(passed[column]):
            rows.append(
                {
                    "group": f"{column}={value}",
                    "frames": len(part),
                    "agreement": part.mean(),
                }
            )
    return pd.DataFrame(rows)
