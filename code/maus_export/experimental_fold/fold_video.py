"""Sequential, local video inference for fold state review."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path
from threading import Lock

import cv2
import numpy as np

from .fold_classifier import refine_fold_state
from .fold_config import DEFAULT_ANALYSIS_FPS
from .fold_dataset import (
    CALIBRATION_PATH,
    MAX_DURATION_DIFFERENCE,
    run_duration,
)
from .fold_dispense import summarise_dispense
from .fold_foreground import ForegroundOccluded, LiveForeground
from .fold_frames import check_starts, phase_clock_origin, video_time
from .fold_geometry import measure_geometry, prepare_geometry
from .fold_overlay import annotate_frame
from .fold_plate import detect_plate
from .fold_progress import VideoProgress
from .fold_rectify import rectify_video_frame
from .fold_reference import VisualFlatReference
from .fold_stages import INITIAL_CLEANING_LABEL, initial_cleaning_end
from .fold_state import STATE_ORDER, classify_state
from .fold_sync import CheckWindowEvaluator
from .fold_temporal import MaskWarming, TemporalMask

FRAME_BUDGET_MS = 200.0
MILLISECONDS_PER_SECOND = 1000
# Matplotlib's pyplot state is shared across worker threads.
_PLOT_LOCK = Lock()


def analyse_video(
    data_dir: Path,
    run_id: int,
    *,
    replace: bool = False,
    analysis_fps: float = DEFAULT_ANALYSIS_FPS,
) -> dict[str, float | int]:
    """Run the pipeline and publish progress for the local dashboard."""
    progress = VideoProgress(data_dir, run_id)
    try:
        result = _analyse_video(
            data_dir,
            run_id,
            replace=replace,
            analysis_fps=analysis_fps,
            progress=progress,
        )
    except BaseException as error:
        progress.update(
            force=True,
            state="failed",
            error=type(error).__name__,
            error_message=str(error),
        )
        raise
    progress.update(force=True, state="completed", fraction=1.0)
    return result


def _analyse_video(
    data_dir: Path,
    run_id: int,
    *,
    replace: bool = False,
    analysis_fps: float = DEFAULT_ANALYSIS_FPS,
    progress: VideoProgress,
) -> dict[str, float | int]:
    """Analyze one cached video sequentially and write private review outputs.

    Empty references are learned before dispensing, never from predicted
    states. Plate geometry and flat area use only already observed frames.
    """
    if run_id < 0:
        raise ValueError("Run id must be nonnegative")
    if not math.isfinite(analysis_fps) or analysis_fps <= 0:
        raise ValueError("Analysis rate must be positive and finite")
    raw = data_dir / "raw"
    run = json.loads((raw / "runs" / f"{run_id}.json").read_text())
    calibration = json.loads(CALIBRATION_PATH.read_text())
    video = raw / "media" / str(run_id) / "video.mp4"
    root = data_dir / "fold_segmentation" / "live"
    root.mkdir(parents=True, exist_ok=True)
    destination = root / str(run_id)
    if destination.exists() and not replace:
        raise FileExistsError("Live output exists; move it aside first")
    capture = cv2.VideoCapture(str(video))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if (
            not capture.isOpened()
            or not math.isfinite(fps)
            or fps < analysis_fps
        ):
            raise ValueError("Video cannot be sampled at the requested rate")
        if abs(count / fps - run_duration(run)) > MAX_DURATION_DIFFERENCE:
            raise ValueError("Video and run duration differ")
        progress.update(
            force=True,
            state="running",
            frames_total=math.ceil(count * analysis_fps / fps),
            analysis_fps=analysis_fps,
        )
        origin = phase_clock_origin(run["phases"])
        phases = sorted(
            (
                (float(phase["elapsed_sec"]), phase.get("phase", ""))
                for phase in run["phases"]
                if phase.get("elapsed_sec") is not None
            ),
            key=lambda item: item[0],
        )
        cleaning_end = initial_cleaning_end(phases)
        events = [
            (video_time(event, origin), event)
            for event in check_starts(run["logs"])
        ]
        evaluator = CheckWindowEvaluator(events)
        dispense_start = min(
            (t for t, name in phases if name.startswith("spread_batter")),
            default=0.0,
        )
        plate = None
        segmenter = None
        geometry_context = None
        flat_reference = VisualFlatReference()
        temporal = TemporalMask()
        times: list[float] = []
        render_times: list[float] = []
        processing_times: list[float] = []
        rows: list[dict[str, str | float]] = []
        index = 0
        sample = 0
        writer = None
        with tempfile.TemporaryDirectory(prefix=".live-", dir=root) as temp:
            stage = Path(temp)
            try:
                while capture.grab():
                    if index / fps + 0.5 / fps < sample / analysis_fps:
                        index += 1
                        continue
                    processing_started = time.perf_counter()
                    success, raw_frame = capture.retrieve()
                    if not success:
                        raise ValueError("Video frame could not be decoded")
                    started = time.perf_counter()
                    image, _ = rectify_video_frame(
                        raw_frame, run["machineName"], calibration
                    )
                    if plate is None:
                        plate = detect_plate(image)
                    prediction = None
                    structural_prediction = None
                    raw_fraction: float | str = ""
                    mask = None
                    geometry = None
                    status = "plate_not_found"
                    if plate is not None:
                        try:
                            if segmenter is None:
                                segmenter = LiveForeground(
                                    image.shape[:2], plate
                                )
                                geometry_context = prepare_geometry(
                                    image.shape[:2], plate
                                )
                            raw_mask = segmenter.segment(
                                image,
                                learn_empty=index / fps < dispense_start,
                            )
                            raw_fraction = float(
                                np.count_nonzero(raw_mask)
                                / segmenter.segmenter.plate_area
                            )
                            mask = temporal.update(raw_mask)
                            if mask is None:
                                raise MaskWarming("Collecting causal masks")
                            geometry = measure_geometry(
                                mask, plate, context=geometry_context
                            )
                            prediction = refine_fold_state(
                                classify_state(
                                    geometry.area,
                                    segmenter.segmenter.plate_area,
                                    len(geometry.edges),
                                    flat_reference.area,
                                    geometry.rect_aspect,
                                ),
                                {
                                    "relative": (
                                        geometry.area / flat_reference.area
                                        if flat_reference.area
                                        else None
                                    ),
                                    "plate_fraction": geometry.plate_fraction,
                                    "rect_aspect": geometry.rect_aspect,
                                    "diameter_fraction": (
                                        geometry.diameter_fraction
                                    ),
                                    "offset_fraction": geometry.offset_fraction,
                                    "hole_fraction": geometry.hole_fraction,
                                    "solidity": geometry.solidity,
                                    "circularity": geometry.circularity,
                                    "edge_count": len(geometry.edges),
                                },
                            )
                            structural_prediction = prediction
                            status = "ok"
                        except MaskWarming:
                            status = "temporal_warming"
                        except ForegroundOccluded:
                            temporal.reset()
                            mask = None
                            geometry = None
                            prediction = None
                            status = "occluded"
                        except (ValueError, cv2.error):
                            temporal.reset()
                            mask = None
                            geometry = None
                            prediction = None
                            status = "segmentation_failed"
                    completed = evaluator.update(
                        index / fps,
                        prediction,
                        geometry.area if geometry is not None else None,
                        flat_reference_available=flat_reference.area
                        is not None,
                    )
                    flat_reference.observe(
                        structural_prediction,
                        geometry.area if geometry is not None else None,
                    )
                    latest = completed[-1] if completed else None
                    elapsed = (
                        time.perf_counter() - started
                    ) * MILLISECONDS_PER_SECOND
                    times.append(elapsed)
                    render_started = time.perf_counter()
                    image = annotate_frame(
                        image,
                        plate,
                        mask,
                        geometry,
                        latest.observation if latest else prediction,
                        status,
                        check=latest.event.name if latest else None,
                        expected=latest.event.state if latest else None,
                        phase=(
                            INITIAL_CLEANING_LABEL
                            if index / fps < cleaning_end
                            else _phase_at(index / fps, phases)
                        ),
                    )
                    render_times.append(
                        (time.perf_counter() - render_started)
                        * MILLISECONDS_PER_SECOND
                    )
                    if writer is None:
                        height, width = image.shape[:2]
                        writer = cv2.VideoWriter(
                            str(stage / "annotated.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            analysis_fps,
                            (width, height),
                        )
                        if not writer.isOpened():
                            raise OSError("Cannot write annotated video")
                    writer.write(image)
                    processing_times.append(
                        (time.perf_counter() - processing_started)
                        * MILLISECONDS_PER_SECOND
                    )
                    rows.append(
                        {
                            "video_t": index / fps,
                            "phase": _phase_at(index / fps, phases),
                            "initial_cleaning_prep": index / fps < cleaning_end,
                            "prediction": prediction or "",
                            "structural_prediction": structural_prediction
                            or "",
                            "raw_plate_fraction": raw_fraction,
                            "state": latest.observation
                            if latest and latest.observation
                            else "",
                            "check": latest.event.name if latest else "",
                            "expected_state": latest.event.state
                            if latest
                            else "",
                            "check_status": latest.status if latest else "",
                            "verdict": latest.verdict if latest else "unknown",
                            "confidence": latest.confidence if latest else 0.0,
                            "status": status,
                            "inference_ms": elapsed,
                            "render_ms": render_times[-1],
                            "processing_ms": processing_times[-1],
                            "plate_fraction": geometry.plate_fraction
                            if geometry
                            else "",
                            "diameter_px": geometry.diameter
                            if geometry
                            else "",
                            "offset_fraction": geometry.offset_fraction
                            if geometry
                            else "",
                            "hole_fraction": geometry.hole_fraction
                            if geometry
                            else "",
                            "edge_angle": geometry.edges[0].angle
                            if geometry and geometry.edges
                            else "",
                            "corner_angle": geometry.corner_angle
                            if geometry
                            else "",
                            "rect_angle": geometry.rect_angle
                            if geometry and geometry.rect_angle is not None
                            else "",
                        }
                    )
                    index += 1
                    sample += 1
                    progress.update(
                        frames_done=sample,
                        fraction=min(index / count, 1.0) if count else 0.0,
                    )
            finally:
                if writer is not None:
                    writer.release()
            if not rows:
                raise ValueError("No video frames analyzed")
            progress.update(force=True, state="finalizing")
            if segmenter is not None:
                segmenter.save_reference(stage / "reference")
            with (stage / "timeline.csv").open("w", newline="") as file:
                out = csv.DictWriter(file, fieldnames=list(rows[0]))
                out.writeheader()
                out.writerows(rows)
            evaluator.finish()
            observation_rows = [
                {
                    "check": check.event.name,
                    "start": check.start,
                    "expected_state": check.event.state,
                    "robot_verdict": (
                        "reported_pass"
                        if check.event.check_ok is True
                        else "reported_fail"
                        if check.event.check_ok is False
                        else "unknown"
                    ),
                    "target": check.target,
                    "observed_at": check.observed_at
                    if check.observed_at is not None
                    else "",
                    "observation": check.observation or "",
                    "status": check.status,
                    "valid_frames": len(check.votes),
                    "samples": check.samples,
                    "verdict": check.verdict,
                    "confidence": check.confidence,
                }
                for check in evaluator.checks
            ]
            columns = [
                "check",
                "start",
                "expected_state",
                "robot_verdict",
                "target",
                "observed_at",
                "observation",
                "status",
                "valid_frames",
                "samples",
                "verdict",
                "confidence",
            ]
            with (stage / "observations.csv").open("w", newline="") as file:
                out = csv.DictWriter(file, fieldnames=columns)
                out.writeheader()
                out.writerows(observation_rows)
            with _PLOT_LOCK:
                _timeline_plot(
                    stage / "timeline.png",
                    rows,
                    events,
                    [
                        (float(phase["elapsed_sec"]), phase.get("phase", ""))
                        for phase in run["phases"]
                        if phase.get("elapsed_sec") is not None
                    ],
                )
            dispense = summarise_dispense(
                (row["video_t"], row["plate_fraction"])
                for row in rows
                if row["phase"].startswith("spread_batter")
                and row["plate_fraction"] != ""
            )
            (stage / "summary.json").write_text(
                json.dumps(
                    {
                        "frames": len(rows),
                        "median_ms": float(np.median(times)),
                        "median_render_ms": float(np.median(render_times)),
                        "median_processing_ms": float(
                            np.median(processing_times)
                        ),
                        "over_budget_frames": sum(
                            t > FRAME_BUDGET_MS for t in processing_times
                        ),
                        "analysis_fps": analysis_fps,
                        "foreground_method": "diffuse_signed_reference_v2",
                        "initial_cleaning_prep_end": cleaning_end,
                        "mask_window_frames": temporal.history.maxlen,
                        "toppings_stage_logged": any(
                            str(row["phase"]).startswith("add_topping")
                            for row in rows
                        ),
                        "dispense": {
                            "frames": dispense.frames,
                            "first_fraction": dispense.first_fraction,
                            "peak_fraction": dispense.peak_fraction,
                            "final_fraction": dispense.final_fraction,
                            "ok": dispense.ok,
                        },
                    }
                )
            )
            if destination.exists():
                backup = Path(tempfile.mkdtemp(prefix=".backup-", dir=root))
                backup.rmdir()
                os.replace(destination, backup)
                try:
                    os.replace(stage, destination)
                except Exception:
                    os.replace(backup, destination)
                    raise
                shutil.rmtree(backup)
            else:
                os.replace(stage, destination)
        return {
            "frames": len(rows),
            "median_ms": float(np.median(times)),
            "median_render_ms": float(np.median(render_times)),
            "median_processing_ms": float(np.median(processing_times)),
            "over_budget_frames": sum(
                t > FRAME_BUDGET_MS for t in processing_times
            ),
        }
    finally:
        capture.release()


def _phase_at(time: float, phases: list[tuple[float, str]]) -> str:
    """Return the phase active at a video time; empty before the first."""
    current = ""
    for start, name in phases:
        if start <= time:
            current = name
        else:
            break
    return current


PHASE_COLORS = {
    "spread_batter": "#e8a35d",
    "cook": "#f2cf5b",
    "fold_carefully": "#9d8cff",
    "wipe_spatula": "#59c1e0",
    "plate_handover": "#61d8a2",
    "add_topping": "#ec86cb",
    "discard_crepe": "#8295ac",
}


def _phase_color(name: str) -> str | None:
    """Color a logged phase; all ``spread_batter_*`` variants are dough."""
    if name in PHASE_COLORS:
        return PHASE_COLORS[name]
    if name.startswith("spread_batter"):
        return PHASE_COLORS["spread_batter"]
    if name.startswith("add_topping"):
        return PHASE_COLORS["add_topping"]
    return None


def _timeline_plot(
    path: Path,
    rows: list[dict],
    events: list[tuple],
    phases: list[tuple[float, str]] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dough = [
        (
            float(row["video_t"]),
            float(row["plate_fraction"])
            if row["plate_fraction"] != ""
            else float("nan"),
        )
        for row in rows
    ]
    figure, axis = plt.subplots(
        2 if dough else 1,
        1,
        figsize=(12, 4.6 if dough else 3.4),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]} if dough else None,
    )
    axis = figure.axes[0]
    if dough:
        dough_axis = figure.axes[1]
    observations = [row for row in rows if row["state"] in STATE_ORDER]
    axis.scatter(
        [row["video_t"] for row in observations],
        [STATE_ORDER.index(row["state"]) for row in observations],
        label="Visual estimate at check",
        c=[
            "green" if row.get("verdict") == "success" else "red"
            for row in observations
        ],
    )
    for row in observations:
        axis.annotate(
            f"{row.get('verdict', 'unknown')} "
            f"{float(row.get('confidence', 0)):.0%}",
            (row["video_t"], STATE_ORDER.index(row["state"])),
            xytext=(4, 8),
            textcoords="offset points",
            fontsize=7,
        )
    for timestamp, event in events:
        axis.axvline(
            timestamp,
            color="green"
            if event.check_ok is True
            else "red"
            if event.check_ok is False
            else "grey",
            alpha=0.5,
        )
    if phases:
        ordered = sorted(phases)
        for index, (start, name) in enumerate(ordered):
            end = (
                ordered[index + 1][0]
                if index + 1 < len(ordered)
                else max((row["video_t"] for row in rows), default=start)
            )
            axis.fill_between(
                [start, end],
                len(STATE_ORDER) - 0.35,
                len(STATE_ORDER) + 0.35,
                color=_phase_color(name) or "#243349",
                alpha=0.85,
            )
            if _phase_color(name) is not None and end - start > 12:
                axis.text(
                    (start + end) / 2,
                    len(STATE_ORDER) + 0.35,
                    name,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
        cleaning_end = initial_cleaning_end(phases)
        if cleaning_end > 0:
            axis.fill_between(
                [0, cleaning_end],
                len(STATE_ORDER) + 0.6,
                len(STATE_ORDER) + 0.9,
                color="#59c1e0",
            )
            axis.text(
                0,
                len(STATE_ORDER) + 1.0,
                INITIAL_CLEANING_LABEL,
                fontsize=7,
                va="bottom",
            )
        axis.set_ylim(-0.6, len(STATE_ORDER) + 1.6)
    axis.set_yticks(range(len(STATE_ORDER)), STATE_ORDER)
    if dough:
        times = [value for value, _ in dough]
        values = [value for _, value in dough]
        dough_axis.fill_between(times, 0, values, color="#e8a35d", alpha=0.3)
        dough_axis.plot(times, values, color="#e8a35d", linewidth=1)
        dough_axis.axhline(0.4, color="grey", linestyle="--", linewidth=0.8)
        dough_axis.set_ylim(0, 1)
        dough_axis.set_ylabel("Plate coverage")
    axis.set_xlabel("Video time (s); lines = logged check starts")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
