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
from .fold_frames import check_events, phase_clock_origin, video_time
from .fold_geometry import measure_geometry, prepare_geometry
from .fold_live import STATE_ORDER, FoldStateMachine
from .fold_mask import CrepeSegmenter
from .fold_overlay import annotate_frame
from .fold_plate import detect_plate
from .fold_rectify import rectify_video_frame
from .fold_state import classify_state

ANALYSIS_FPS = 10.0
MILLISECONDS_PER_SECOND = 1000


def analyse_video(
    data_dir: Path, run_id: int, *, replace: bool = False
) -> dict[str, float | int]:
    """Analyze one cached video sequentially and write private review outputs.

    The first detected empty image serves as reference. Plate geometry and
    flat area are learned online, without consulting future check frames.
    """
    if run_id < 0:
        raise ValueError("Run id must be nonnegative")
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
            or fps < ANALYSIS_FPS
        ):
            raise ValueError("Video cannot be sampled at the requested rate")
        if abs(count / fps - run_duration(run)) > MAX_DURATION_DIFFERENCE:
            raise ValueError("Video and run duration differ")
        origin = phase_clock_origin(run["phases"])
        events = [
            (video_time(event, origin), event)
            for event in check_events(run["logs"])
        ]
        machine = FoldStateMachine()
        reference = None
        plate = None
        segmenter = None
        geometry_context = None
        flat_area: float | None = None
        times: list[float] = []
        render_times: list[float] = []
        rows: list[dict[str, str | float]] = []
        index = 0
        sample = 0
        writer = None
        with tempfile.TemporaryDirectory(prefix=".live-", dir=root) as temp:
            stage = Path(temp)
            try:
                while capture.grab():
                    if index / fps + 0.5 / fps < sample / ANALYSIS_FPS:
                        index += 1
                        continue
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
                            )
                            if reference is None and prediction == "empty":
                                reference = image.copy()
                                segmenter = CrepeSegmenter(
                                    image.shape[:2], plate, reference
                                )
                            if prediction == "flat":
                                flat_area = max(flat_area or 0.0, geometry.area)
                            machine.update(prediction)
                            status = "ok"
                        except (ValueError, cv2.error):
                            mask = None
                            geometry = None
                            status = "segmentation_failed"
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
                        machine.state,
                        prediction,
                        status,
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
                            ANALYSIS_FPS,
                            (width, height),
                        )
                        if not writer.isOpened():
                            raise OSError("Cannot write annotated video")
                    writer.write(image)
                    rows.append(
                        {
                            "video_t": index / fps,
                            "prediction": prediction or "",
                            "state": machine.state,
                            "status": status,
                            "inference_ms": elapsed,
                            "render_ms": render_times[-1],
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
            _timeline_plot(stage / "timeline.png", rows, events)
            (stage / "summary.json").write_text(
                json.dumps(
                    {
                        "frames": len(rows),
                        "median_ms": float(np.median(times)),
                        "median_render_ms": float(np.median(render_times)),
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
        }
    finally:
        capture.release()


def _timeline_plot(path: Path, rows: list[dict], events: list[tuple]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(12, 3))
    axis.step(
        [row["video_t"] for row in rows],
        [STATE_ORDER.index(row["state"]) for row in rows],
        where="post",
    )
    for timestamp, event in events:
        axis.axvline(
            timestamp, color="green" if event.check_ok else "red", alpha=0.5
        )
    axis.set_yticks(range(len(STATE_ORDER)), STATE_ORDER)
    axis.set_xlabel("Video time (s)")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
