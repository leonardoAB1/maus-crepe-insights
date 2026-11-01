"""Render crepe-pose results for the experimental fold dashboard.

Writes its own dashboard folder, so nothing from the fold pipeline is touched:

    python models/crepe-pose/dashboard_render.py --data-dir DATA --runs RUN_ID
    maus-export fold-experimental-dashboard --data-dir DATA/synthetic/dashboard
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from maus_export import synthetic_crepes as sc
from maus_export.producers import CrepePoseProducer, FrameContext

HERE = Path(__file__).resolve().parent
TRANSFORM_PILOT = "phase-snapshot/labeling/rectified-experimental-v1"
VIDEO = "phase-snapshot/raw/media/{run}/video.mp4"
# H.264, because browsers cannot play the mp4v the OpenCV default writes
FOURCC = "avc1"
FILL_SHARE = 0.4
COLOUR = np.array([60, 220, 60], np.float32)
OUTLINE_PX = 2
FIELDS = ("frame", "video_t", "plate_fraction", "inference_ms")


def plate_mask(transform: dict, shape: tuple[int, int]) -> np.ndarray:
    """The plate circle, moved from the rectified view back to the video."""
    size = tuple(int(v) for v in transform["target_size"])
    centre = round(size[0] * sc.ZONE_CENTRE_SHARE)
    radius = round(size[0] * sc.ZONE_RADIUS_SHARE)
    disc = np.zeros((size[1], size[0]), np.uint8)
    cv2.circle(disc, (centre, centre), radius, 1, -1)
    warped = cv2.warpPerspective(
        disc,
        np.array(transform["matrix"], np.float64),
        (shape[1], shape[0]),
        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
    )
    return warped > 0


def paint(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Draw the mask as a translucent fill with a solid outline."""
    image = frame.copy()
    image[mask] = image[mask] * (1 - FILL_SHARE) + COLOUR * FILL_SHARE
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    colour = tuple(int(v) for v in COLOUR)
    cv2.drawContours(image, contours, -1, colour, OUTLINE_PX)
    return image


def render_run(
    video: Path, producer: CrepePoseProducer, plate: dict, out: Path
) -> dict[str, float]:
    """Write `annotated.mp4` and `timeline.csv` for one run."""
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise OSError(f"cannot read {video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    out.mkdir(parents=True, exist_ok=True)
    writer = None
    timings = []
    disc = None
    with (out / "timeline.csv").open("w", newline="") as file:
        table = csv.DictWriter(file, fieldnames=FIELDS)
        table.writeheader()
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if writer is None:
                height, width = frame.shape[:2]
                disc = plate_mask(plate, (height, width))
                writer = cv2.VideoWriter(
                    str(out / "annotated.mp4"),
                    cv2.VideoWriter_fourcc(*FOURCC),
                    fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise OSError("cannot write the annotated video")
            layers = producer.layers(frame, FrameContext(frame_index=index))
            timings.append(producer.last_ms)
            writer.write(paint(frame, layers.material))
            table.writerow(
                {
                    "frame": index,
                    "video_t": round(index / fps, 3),
                    "plate_fraction": round(
                        float((layers.material & disc).sum() / disc.sum()), 4
                    ),
                    "inference_ms": round(producer.last_ms, 2),
                }
            )
            index += 1
    capture.release()
    if writer is not None:
        writer.release()
    return {
        "frames": index,
        "median_ms": round(float(np.median(timings)), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--runs", type=int, nargs="+", required=True)
    args = parser.parse_args()
    data = args.data_dir
    pilot = data / TRANSFORM_PILOT
    manifest = json.loads((pilot / "manifest.json").read_text())
    folders = {
        run["run_id"]: Path(run["frames"][0]["image"]).parent.name
        for run in manifest["runs"]
        if run.get("frames")
    }
    root = data / "synthetic" / "dashboard"
    (root / "raw" / "runs").mkdir(parents=True, exist_ok=True)
    for run in args.runs:
        transform = pilot / "transforms" / f"{folders[run]}.json"
        producer = CrepePoseProducer(
            model=HERE / "model.onnx", transform=transform
        )
        result = render_run(
            data / VIDEO.format(run=run),
            producer,
            json.loads(transform.read_text()),
            root / "fold_segmentation" / "live" / str(run),
        )
        shutil.copyfile(
            data / "raw" / "runs" / f"{run}.json",
            root / "raw" / "runs" / f"{run}.json",
        )
        print(run, result, flush=True)


if __name__ == "__main__":
    main()
