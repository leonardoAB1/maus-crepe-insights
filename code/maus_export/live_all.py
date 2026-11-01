"""One run video with every vision model, laid out by the challenge tasks.

Per analysed frame (10 per second):

- crêpe student: mask, live state and vote support (``crepe_student``)
- crêpe pose in the top view: centre, diameter, axis, fold edge, heading,
  quarter corner (``crepe_pose``)
- frame heads on the shared embedding: ``fold-class`` (state and
  probability), ``check-verdict`` spread, fold1 and fold2 (probability
  that the robot would pass the check) and ``failure-reason`` spread
  ``too_small``
- the robot's logged checks and phases, aligned to the video clock

Writes ``annotated.mp4`` (H.264 when ffmpeg is present) and ``frames.csv``
under ``fold_segmentation/all/<run id>/``. The frame heads were trained on
check stills, so on video frames between checks they are unvalidated.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from .crepe_pose import (
    CrepePose,
    _crop_zone,
    draw_pose,
    ellipse_matrix,
    ellipse_top_view,
    ellipse_zone,
    load_calibration,
    pose_row,
    robot_zone,
    top_view,
)
from .crepe_pose import crepe_pose as measure_pose
from .crepe_student import (
    INPUT_SIZE,
    CrepeStudent,
    StudentStateMachine,
    area_share,
    open_session,
)
from .fold_frames import (
    TARGET_SOURCE_SIZE,
    correct_perspective,
    parse_local_time,
    phase_clock_offset,
    video_path,
)
from .fold_live import ANALYSIS_FPS
from .fold_state import EMPTY_STATE, FLAT_STATE, HALF_STATE, QUARTER_STATE
from .frame_preprocess import preprocess

MODELS = Path(__file__).resolve().parents[1] / "models"
EMBEDDER = MODELS / "frame-embedder" / "model.onnx"
HEAD_INPUT = "x"
HEAD_PROBA = "probabilities"
EMBED_OUTPUT = "y"
FOLD_CLASSES = ("empty", "flat", "half", "quarter")
VERDICT_CHECKS = ("spread", "fold1", "fold2")
LOGGED_CHECKS = ("plate_clean", "spread", "fold1", "fold2", "push_off")
OUTPUT = Path("fold_segmentation") / "all"

CANVAS = (1920, 1080)
MARGIN = 24
HEADER_H = 84
VIDEO_W = 1180
TOP_SIZE = 660
PANEL_Y = 760
BACKGROUND = (22, 18, 16)
PANEL = (38, 32, 28)
INK = (236, 240, 245)
MUTED = (160, 168, 176)
ACCENT = (79, 160, 232)
GOOD = (96, 200, 120)
BAD = (90, 90, 235)
FONT = cv2.FONT_HERSHEY_DUPLEX
TITLE_SCALE = 0.95
TEXT_SCALE = 0.72
LINE_H = 34
THICK = 1
BOLD = 2
EVENT_SHOW_S = 6.0
# A check head is only meaningful on frames of the state its check expects;
# its last such value is held and shown afterwards.
HELD_FROM = {
    "spread": ("p_pass_spread", FLAT_STATE),
    "too_small": ("p_too_small", FLAT_STATE),
    "fold1": ("p_pass_fold1", HALF_STATE),
    "fold2": ("p_pass_fold2", QUARTER_STATE),
}
PASS_LINE = 0.5


@dataclass
class Heads:
    """The frame embedder and every head that reads its embedding."""

    embedder: Any
    fold_class: Any
    verdict: dict[str, Any]
    too_small: Any

    @classmethod
    def load(cls) -> Heads:
        return cls(
            embedder=open_session(EMBEDDER),
            fold_class=open_session(MODELS / "fold-class" / "model.onnx"),
            verdict={
                check: open_session(
                    MODELS / "check-verdict" / check / "model.onnx"
                )
                for check in VERDICT_CHECKS
            },
            too_small=open_session(
                MODELS
                / "failure-reason"
                / "spread"
                / "too_small"
                / "model.onnx"
            ),
        )

    def predict(self, frame: np.ndarray) -> dict[str, float | str]:
        """Head outputs for one raw frame of any size."""
        still = cv2.resize(
            frame, TARGET_SOURCE_SIZE, interpolation=cv2.INTER_AREA
        )
        embedding = self.embedder.run(
            [EMBED_OUTPUT], {HEAD_INPUT: preprocess(still)}
        )[0].astype(np.float32)

        def proba(session: Any) -> np.ndarray:
            return session.run([HEAD_PROBA], {HEAD_INPUT: embedding})[0][0]

        fold = proba(self.fold_class)
        out: dict[str, float | str] = {
            "head_state": FOLD_CLASSES[int(fold.argmax())],
            "head_conf": float(fold.max()),
        }
        for check, session in self.verdict.items():
            out[f"p_pass_{check}"] = float(proba(session)[1])
        out["p_too_small"] = float(proba(self.too_small)[1])
        return out


@dataclass(frozen=True)
class Event:
    """One logged robot check with its result line."""

    name: str
    second: float
    ok: bool
    result: str


def robot_events(run: dict[str, Any]) -> list[Event]:
    """Logged check results on the video clock."""
    offset = phase_clock_offset(run.get("phases") or [])
    pending: dict[str, float] = {}
    events = []
    for log in sorted(run.get("logs") or [], key=lambda i: i["timestamp"]):
        message = str(log.get("msg") or "").strip()
        second = (
            parse_local_time(str(log["timestamp"])) - offset
        ).total_seconds()
        for name in LOGGED_CHECKS:
            if message == f"check_{name} started":
                pending[name] = second
                break
            prefix = f"check_{name}:"
            if message.startswith(prefix) and name in pending:
                result = message[len(prefix) :].strip()
                ok = not any(
                    word in result.lower()
                    for word in ("not ", "fail", "incomplete", "too ")
                )
                events.append(Event(name, pending.pop(name), ok, result))
                break
    return events


def phase_at(run: dict[str, Any], second: float) -> str:
    current = ""
    for phase in sorted(
        run.get("phases") or [], key=lambda p: p.get("elapsed_sec", 0)
    ):
        if float(phase.get("elapsed_sec", 0)) <= second:
            current = str(phase.get("phase") or "")
    return current


def text(
    canvas: np.ndarray,
    value: str,
    x: int,
    y: int,
    colour: tuple[int, int, int] = INK,
    scale: float = TEXT_SCALE,
    thick: int = THICK,
) -> None:
    cv2.putText(canvas, value, (x, y), FONT, scale, colour, thick, cv2.LINE_AA)


def fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    size = (round(image.shape[1] * scale), round(image.shape[0] * scale))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def paste(canvas: np.ndarray, image: np.ndarray, x: int, y: int) -> None:
    canvas[y : y + image.shape[0], x : x + image.shape[1]] = image


def fmt(value: float | None, pattern: str, missing: str = "-") -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return missing
    return pattern.format(value)


def verdict_colour(p_pass: float) -> tuple[int, int, int]:
    return GOOD if p_pass >= PASS_LINE else BAD


def panel(
    canvas: np.ndarray,
    x: int,
    width: int,
    title: str,
    lines: Sequence[tuple[str, tuple[int, int, int]]],
) -> None:
    bottom = CANVAS[1] - MARGIN
    cv2.rectangle(canvas, (x, PANEL_Y), (x + width, bottom), PANEL, -1)
    text(canvas, title, x + 20, PANEL_Y + 40, ACCENT, TITLE_SCALE, BOLD)
    for i, (value, colour) in enumerate(lines):
        text(canvas, value, x + 20, PANEL_Y + 84 + i * LINE_H, colour)


def compose(
    frame: np.ndarray,
    mask: np.ndarray,
    top: np.ndarray,
    row: dict[str, Any],
    header: str,
    events: list[Event],
) -> np.ndarray:
    canvas = np.full((CANVAS[1], CANVAS[0], 3), BACKGROUND, dtype=np.uint8)
    text(canvas, header, MARGIN, 52, INK, TITLE_SCALE, BOLD)
    view = frame.copy()
    full = cv2.resize(
        mask.astype(np.uint8),
        (frame.shape[1], frame.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    tint = view.copy()
    tint[full] = GOOD
    view = cv2.addWeighted(view, 0.6, tint, 0.4, 0)
    left = fit(view, VIDEO_W, PANEL_Y - HEADER_H - MARGIN)
    paste(canvas, left, MARGIN, HEADER_H)
    right = fit(top, TOP_SIZE, PANEL_Y - HEADER_H - MARGIN)
    paste(canvas, right, CANVAS[0] - MARGIN - right.shape[1], HEADER_H)
    second = row["second"]
    shown = [e for e in events if 0 <= second - e.second < EVENT_SHOW_S]
    if shown:
        event = shown[-1]
        label = f"robot {event.name}: {'ok' if event.ok else event.result}"
        cv2.rectangle(
            canvas,
            (MARGIN, HEADER_H),
            (MARGIN + VIDEO_W, HEADER_H + LINE_H + 22),
            PANEL,
            -1,
        )
        text(
            canvas,
            label[:80],
            MARGIN + 16,
            HEADER_H + LINE_H + 2,
            GOOD if event.ok else BAD,
            TEXT_SCALE,
            BOLD,
        )
    width = (CANVAS[0] - 4 * MARGIN) // 3
    spread = row["held_spread"]
    panel(
        canvas,
        MARGIN,
        width,
        "1  Spread quality",
        [
            (f"mask: {row['area_share'] * 100:.1f} % of frame", INK),
            (f"diameter: {fmt(row.get('diameter_rel'), '{:.2f}')} plate", INK),
            (
                f"quality P(pass): {fmt(spread, '{:.2f}')}",
                MUTED if spread is None else verdict_colour(spread),
            ),
            (
                f"P(too small): {fmt(row['held_too_small'], '{:.2f}')}",
                BAD if (row["held_too_small"] or 0) >= PASS_LINE else MUTED,
            ),
        ],
    )
    panel(
        canvas,
        2 * MARGIN + width,
        width,
        "2  Position & orientation",
        [
            (
                f"centre x {fmt(row.get('dx_rel'), '{:+.2f}')}  "
                f"y {fmt(row.get('dy_rel'), '{:+.2f}')} r",
                INK,
            ),
            (f"axis: {fmt(row.get('axis_deg'), '{:+.0f}')} deg", INK),
            (f"heading: {fmt(row.get('heading_deg'), '{:+.0f}')} deg", INK),
            ("units: plate radii, image angles", MUTED),
        ],
    )
    fold_check = "fold2" if row["state"] == QUARTER_STATE else "fold1"
    fold_pass = row[f"held_{fold_check}"]
    panel(
        canvas,
        3 * MARGIN + 2 * width,
        width,
        "3  Fold detection",
        [
            (
                f"state: {row['state'].upper()}  "
                f"(this frame: {row['raw_state']})",
                INK,
            ),
            (
                f"fold-class: {row['head_state']} {row['head_conf']:.2f}",
                INK,
            ),
            (
                f"edge {fmt(row.get('edge_deg'), '{:.0f}')} deg  "
                f"corner {fmt(row.get('corner_deg'), '{:.0f}')} deg",
                INK,
            ),
            (
                f"{fold_check} P(pass): {fmt(fold_pass, '{:.2f}')}",
                MUTED if fold_pass is None else verdict_colour(fold_pass),
            ),
        ],
    )
    return canvas


def to_h264(source: Path, target: Path) -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(target),
        ],
        check=False,
    )
    return result.returncode == 0


def render_run(
    data_dir: Path,
    run_id: int,
    student: CrepeStudent | None = None,
    heads: Heads | None = None,
    plates: dict[int, tuple] | None = None,
) -> Path:
    """Analyse one run video with every model; return the video path.

    ``plates`` maps run ids to a plate ellipse in raw frame pixels (for
    example SAM 3 plates); with one, the top view is an affine map of that
    ellipse to a circle. Without, the robot's mean image correction is used,
    which does not fit every camera setup.
    """
    student = student or CrepeStudent.load()
    heads = heads or Heads.load()
    calibration = load_calibration()
    run = json.loads(
        (data_dir / "raw" / "runs" / f"{run_id}.json").read_text(
            encoding="utf-8"
        )
    )
    robot = run.get("machineName")
    zone = robot_zone(robot, calibration)
    events = robot_events(run)
    video = video_path(data_dir, run_id)
    if video is None:
        raise FileNotFoundError(f"no video for run {run_id}")
    ellipse = (plates or {}).get(run_id)
    if ellipse is not None:
        zone = ellipse_zone()
    out = data_dir / OUTPUT / str(run_id)
    out.mkdir(parents=True, exist_ok=True)
    held: dict[str, float | None] = dict.fromkeys(HELD_FROM)
    raw_video = out / "annotated_mp4v.mp4"
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    step = max(1, round(fps / ANALYSIS_FPS))
    writer = cv2.VideoWriter(
        str(raw_video), cv2.VideoWriter_fourcc(*"mp4v"), ANALYSIS_FPS, CANVAS
    )
    machine = StudentStateMachine()
    rows = []
    index = -1
    try:
        while capture.grab():
            index += 1
            if index % step:
                continue
            ok, frame = capture.retrieve()
            if not ok:
                break
            second = index / fps
            mask = student.mask(frame)
            area = area_share(mask)
            raw_state, state = machine.update(area)
            support = machine.votes.count(machine.state) / len(machine.votes)
            if ellipse is not None:
                size = (frame.shape[1], frame.shape[0])
                top_mask = ellipse_top_view(
                    mask, ellipse_matrix(ellipse, size, INPUT_SIZE)
                )
                top_image = ellipse_top_view(
                    frame, ellipse_matrix(ellipse, size, size)
                )
            else:
                top_mask = top_view(mask, robot, calibration)
                top_image = correct_perspective(frame, robot, calibration)
            pose: CrepePose | None = None
            if state != EMPTY_STATE:
                pose = measure_pose(
                    top_mask, zone, quarter=state == QUARTER_STATE
                )
            row = {
                "run_id": run_id,
                "second": second,
                "phase": phase_at(run, second),
                "raw_state": raw_state,
                "state": state,
                "vote_support": support,
                "area_share": area,
                **pose_row(pose),
                **heads.predict(frame),
            }
            for key, (column, needed) in HELD_FROM.items():
                if state == needed:
                    held[key] = row[column]
                row[f"held_{key}"] = held[key]
            rows.append(row)
            top = _crop_zone(
                draw_pose(
                    top_image,
                    top_mask,
                    pose,
                    zone,
                ),
                zone,
            )
            header = (
                f"Run {run_id}  {robot}  t = {second:5.1f} s  "
                f"phase: {row['phase'] or '-'}"
            )
            writer.write(compose(frame, mask, top, row, header, events))
    finally:
        capture.release()
        writer.release()
    pd.DataFrame(rows).to_csv(out / "frames.csv", index=False)
    final = out / "annotated.mp4"
    if to_h264(raw_video, final):
        raw_video.unlink()
        return final
    raw_video.replace(final)
    return final
