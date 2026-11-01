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

import cv2
import numpy as np

from .fold_dataset import (
    CALIBRATION_PATH,
    MAX_DURATION_DIFFERENCE,
    run_duration,
)
from .fold_dispense import summarise_dispense
from .fold_frames import check_starts, phase_clock_origin, video_time
from .fold_geometry import measure_geometry, prepare_geometry
from .fold_mask import CrepeSegmenter
from .fold_overlay import annotate_frame
from .fold_plate import detect_plate
from .fold_rectify import rectify_video_frame
from .fold_state import STATE_ORDER, classify_state
from .fold_sync import CheckWindowEvaluator

DEFAULT_ANALYSIS_FPS = 10.0
FRAME_BUDGET_MS = 200.0
MILLISECONDS_PER_SECOND = 1000


def analyse_video(
    data_dir: Path,
    run_id: int,
    *,
    replace: bool = False,
    analysis_fps: float = DEFAULT_ANALYSIS_FPS,
) -> dict[str, float | int]:
    """Analyze one cached video sequentially and write private review outputs.

    The first detected empty image serves as reference. Plate geometry and
    flat area are learned online, without consulting future check frames.
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
        origin = phase_clock_origin(run["phases"])
        phases = sorted(
            (
                (float(phase["elapsed_sec"]), phase.get("phase", ""))
                for phase in run["phases"]
                if phase.get("elapsed_sec") is not None
            ),
            key=lambda item: item[0],
        )
        events = [
            (video_time(event, origin), event)
            for event in check_starts(run["logs"])
        ]
        evaluator = CheckWindowEvaluator(events)
        reference = None
        plate = None
        segmenter = None
        geometry_context = None
        flat_area: float | None = None
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
                    mask = None
                    geometry = None
                    status = "plate_not_found"
                    if plate is not None:
                        try:
                            if segmenter is None:
                                segmenter = CrepeSegmenter(
                                    image.shape[:2], plate
                                )
                                geometry_context = prepare_geometry(
                                    image.shape[:2], plate
                                )
                            mask = segmenter.segment(image)
                            geometry = measure_geometry(
                                mask, plate, context=geometry_context
                            )
                            prediction = classify_state(
                                geometry.area,
                                segmenter.plate_area,
                                len(geometry.edges),
                                flat_area,
                                geometry.rect_aspect,
                            )
                            if reference is None and prediction == "empty":
                                reference = image.copy()
                                segmenter = CrepeSegmenter(
                                    image.shape[:2], plate, reference
                                )
                            status = "ok"
                        except (ValueError, cv2.error):
                            mask = None
                            geometry = None
                            status = "segmentation_failed"
                    completed = evaluator.update(
                        index / fps,
                        prediction,
                        geometry.area if geometry is not None else None,
                        flat_reference_available=flat_area is not None,
                    )
                    for check in completed:
                        if (
                            check.event.name == "spread"
                            and check.observation == "flat"
                            and check.flat_areas
                        ):
                            flat_area = float(np.median(check.flat_areas))
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
                        latest.observation if latest else None,
                        status,
                        check=latest.event.name if latest else None,
                        expected=latest.event.state if latest else None,
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
                            "prediction": prediction or "",
                            "state": latest.observation
                            if latest and latest.observation
                            else "",
                            "check": latest.event.name if latest else "",
                            "expected_state": latest.event.state
                            if latest
                            else "",
                            "check_status": latest.status if latest else "",
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
            finally:
                if writer is not None:
                    writer.release()
            if not rows:
                raise ValueError("No video frames analyzed")
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
            ]
            with (stage / "observations.csv").open("w", newline="") as file:
                out = csv.DictWriter(file, fieldnames=columns)
                out.writeheader()
                out.writerows(observation_rows)
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
                    if row["phase"] == "spread_batter"
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
    "discard_crepe": "#8295ac",
}


def _timeline_plot(
    path: Path,
    rows: list[dict],
    events: list[tuple],
    phases: list[tuple[float, str]] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(12, 3.4))
    observations = [row for row in rows if row["state"] in STATE_ORDER]
    axis.scatter(
        [row["video_t"] for row in observations],
        [STATE_ORDER.index(row["state"]) for row in observations],
        label="Visual estimate at check",
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
                color=PHASE_COLORS.get(name, "#243349"),
                alpha=0.85,
            )
            if name in PHASE_COLORS and end - start > 12:
                axis.text(
                    (start + end) / 2,
                    len(STATE_ORDER) + 0.35,
                    name,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
        axis.set_ylim(-0.6, len(STATE_ORDER) + 1.0)
    axis.set_yticks(range(len(STATE_ORDER)), STATE_ORDER)
    axis.set_xlabel("Video time (s); lines = logged check starts")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
