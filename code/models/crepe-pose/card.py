"""Rebuild the crepe-pose model card and example image from local results.

From code/ in the ml environment:

    python models/crepe-pose/card.py --data-dir DATA --video RUN.mp4 \
        --transform TRANSFORM.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd

from maus_export import synthetic_crepes as sc

HERE = Path(__file__).resolve().parent
EXAMPLE = HERE.parents[2] / "assets" / "crepe-pose-example.jpg"
THRESHOLD = 0.5
OPSET = 17
PARAMETERS = 619_267
PARITY_GAP = 2.59e-06
TARGET_MS = 300.0
STATUS = "experiment, not used by any pipeline, the robot or the team's results"
PI5_FACTORS = (4, 8)
THREADS = (1, 4)
WARM_UP_RUNS = 10
RUNS = 50
LATENCY_FRAME = 2000
P95 = 95
TILE = 260
FILL_SHARE = 0.35
EXAMPLE_TYPICAL = 3
EXAMPLE_WORST = 3
LABEL_HEIGHT = 20


def to_batch(image: np.ndarray) -> np.ndarray:
    pixels = image.astype(np.float32) / sc.MAX_U8
    return pixels.transpose(2, 0, 1)[None]


def latency(model: Path, video: Path, transform: Path) -> dict[str, Any]:
    """Time the whole path from a video frame to a cleaned mask."""
    spec = json.loads(transform.read_text())
    matrix = np.array(spec["matrix"], np.float64)
    size = tuple(int(v) for v in spec["target_size"])
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, LATENCY_FRAME)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise ValueError(f"cannot read a frame from {video}")
    out: dict[str, Any] = {
        "host": {
            "machine": platform.machine(),
            "system": platform.platform(),
            "note": "Apple M4 Pro laptop, not the robot",
        },
        "batch": 1,
        "runs": RUNS,
    }
    for threads in THREADS:
        cv2.setNumThreads(threads)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(model), options)
        prepare, infer, total = [], [], []
        for run in range(WARM_UP_RUNS + RUNS):
            started = time.perf_counter()
            batch = to_batch(sc.rectify_for_model(frame, matrix, size[0]))
            prepared = time.perf_counter()
            chances = session.run(None, {"x": batch})[0][0, 0]
            inferred = time.perf_counter()
            sc.clean_mask(chances > THRESHOLD)
            ended = time.perf_counter()
            if run >= WARM_UP_RUNS:
                prepare.append(prepared - started)
                infer.append(inferred - prepared)
                total.append(ended - started)
        out[f"threads_{threads}"] = {
            "prepare_median_ms": round(float(np.median(prepare)) * 1000, 1),
            "model_median_ms": round(float(np.median(infer)) * 1000, 1),
            "total_median_ms": round(float(np.median(total)) * 1000, 1),
            "total_p95_ms": round(float(np.percentile(total, P95)) * 1000, 1),
        }
    single = out[f"threads_{THREADS[0]}"]["total_median_ms"]
    out["pi5_estimate_ms"] = [round(single * f) for f in PI5_FACTORS]
    out["pi5_estimate_method"] = (
        "single-thread total x 4 to x 8. The repo's rough x 4 is for an x86 "
        "laptop core and an M4 core is likely faster, so the top end is a "
        "judgement. Not measured on a Raspberry Pi 5."
    )
    out["target_ms"] = TARGET_MS
    return out


def unseen_frames(
    data: Path, model: Path
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Test check frames from runs the model never trained on."""
    out = data / "synthetic"
    table = pd.read_parquet(out / "real.parquet")
    arrays = np.load(out / "real.npz")
    hand_runs = set(pd.read_parquet(out / "hand.parquet")["run_id"])
    keep = (
        (table["split"] == "test")
        & table["check"].isin(("spread", "fold1", "fold2"))
        & ~table["run_id"].isin(hand_runs)
    )
    part = table[keep]
    images = arrays["images"][part.index]
    truth = arrays["masks"][part.index]
    session = ort.InferenceSession(str(model))
    found = np.stack(
        [
            sc.clean_mask(
                session.run(None, {"x": to_batch(image)})[0][0, 0] > THRESHOLD
            )
            for image in images
        ]
    )
    return part, images, truth, found


def unseen_metrics(
    part: pd.DataFrame, truth: np.ndarray, found: np.ndarray
) -> dict[str, Any]:
    ious = np.array([sc.iou(f, t) for f, t in zip(found, truth, strict=True)])
    centres = [
        sc.centre_error_rel(f, t) for f, t in zip(found, truth, strict=True)
    ]
    centre = np.array([np.nan if c is None else c for c in centres])
    out: dict[str, Any] = {
        "reference": "robot overlay outline, not ground truth",
        "frames": len(part),
        "runs": int(part["run_id"].nunique()),
        "iou_median": round(float(np.median(ious)), 3),
        "centre_error_median_share_of_plate_radius": round(
            float(np.nanmedian(centre)), 4
        ),
        "per_robot": {},
    }
    for robot in sorted(part["robot"].unique()):
        chosen = (part["robot"] == robot).to_numpy()
        out["per_robot"][robot] = {
            "frames": int(chosen.sum()),
            "iou_median": round(float(np.median(ious[chosen])), 3),
        }
    return out


def example_image(
    part: pd.DataFrame,
    images: np.ndarray,
    truth: np.ndarray,
    found: np.ndarray,
) -> None:
    """Typical frames beside the worst ones, including the known failure."""
    ious = np.array([sc.iou(f, t) for f, t in zip(found, truth, strict=True)])
    order = np.argsort(ious)
    middle = len(order) // 2
    picks = [
        *order[middle : middle + EXAMPLE_TYPICAL],
        *order[:EXAMPLE_WORST],
    ]
    tiles = []
    for index in picks:
        tile = cv2.resize(images[index], (TILE, TILE))
        mask = cv2.resize(
            found[index].astype(np.uint8),
            (TILE, TILE),
            interpolation=cv2.INTER_NEAREST,
        )
        painted = tile[mask > 0] * (1 - FILL_SHARE)
        tile[mask > 0] = painted + np.array([40, 40, 255]) * FILL_SHARE
        reference = cv2.resize(
            truth[index].astype(np.uint8),
            (TILE, TILE),
            interpolation=cv2.INTER_NEAREST,
        )
        contours, _ = cv2.findContours(
            reference, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        cv2.drawContours(tile, contours, -1, (0, 255, 0), 1)
        row = part.iloc[index]
        cv2.rectangle(tile, (0, 0), (TILE, LABEL_HEIGHT), (0, 0, 0), -1)
        cv2.putText(
            tile,
            f"{row['check']} {row['robot'][5:]} IoU {ious[index]:.2f}",
            (4, LABEL_HEIGHT - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    half = len(tiles) // 2
    cv2.imwrite(
        str(EXAMPLE),
        np.vstack([np.hstack(tiles[:half]), np.hstack(tiles[half:])]),
    )


def training_data(data: Path) -> dict[str, Any]:
    out = data / "synthetic"
    table = pd.read_parquet(out / "real.parquet")
    frames = pd.read_parquet(data / "training" / "frames.parquet")
    hand = pd.read_parquet(out / "hand.parquet")
    train = table[table["split"] == "train"]
    times = frames[frames["run_id"].isin(train["run_id"])]["ts_utc"]
    return {
        "chain": [
            "6,000 synthetic images: real plates, 75% real cut-outs",
            "fine-tuned on robot overlay outlines, training split",
            "fine-tuned on reviewed hand masks, all five labelled runs",
        ],
        "robot_outlines": {
            "frames": len(train),
            "runs": int(train["run_id"].nunique()),
            "robots": sorted(train["robot"].unique()),
            "first_utc": str(times.min()),
            "last_utc": str(times.max()),
        },
        "hand_masks": {
            "frames": len(hand),
            "with_crepe": int((hand["state"] == "crepe_present").sum()),
            "runs": int(hand["run_id"].nunique()),
            "robots": ["maus-pixi"],
            "source": "private labels repo, with an update not merged there",
            "only": "validation ok",
        },
    }


def build(args: argparse.Namespace) -> None:
    source = args.data_dir / "synthetic" / "hand.onnx"
    target = HERE / "model.onnx"
    shutil.copyfile(source, target)
    graph = onnx.load(str(target))
    part, images, truth, found = unseen_frames(args.data_dir, target)
    folds = json.loads(
        (args.data_dir / "synthetic" / "scores-hand-finetune.json").read_text()
    )
    card = {
        "name": "crepe-pose",
        "status": STATUS,
        "description": (
            "Crepe mask and position from one rectified frame. Challenge task "
            "2. A local experiment, not a shipped model."
        ),
        "estimator": "small U-Net with a direction head, in PyTorch",
        "inputs": {
            "x": {
                "dtype": "float32",
                "shape": [1, 3, sc.SIZE, sc.SIZE],
                "meaning": "rectified frame, BGR, scaled 0 to 1",
            }
        },
        "outputs": {
            "mask": {
                "shape": [1, 1, sc.SIZE, sc.SIZE],
                "meaning": (
                    f"crepe probability per pixel. Threshold {THRESHOLD}, then "
                    "keep the largest region with holes filled. The centre is "
                    "the mask centroid."
                ),
            },
            "direction": {
                "shape": [1, 2],
                "meaning": "unproven fold direction. Do not use.",
            },
        },
        "size": {
            "parameters": PARAMETERS,
            "bytes": target.stat().st_size,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        },
        "onnx": {
            "opset": graph.opset_import[0].version,
            "nodes": len(graph.graph.node),
            "parity_max_abs_diff": PARITY_GAP,
        },
        "latency": latency(target, args.video, args.transform),
        "data": training_data(args.data_dir),
        "metrics": {
            "hand_masks_held_out": {
                "reference": "reviewed hand masks, may be pixel-assisted",
                "method": "each fold trains on four runs, scores the fifth",
                "before_hand_tuning": folds["all_held_out"]["before"],
                "after_hand_tuning": folds["all_held_out"]["after"],
                "per_run": list(folds["folds"].values()),
                "note": "folds, not the final model: it trained on all five",
            },
            "unseen_check_frames": unseen_metrics(part, truth, found),
        },
        "deployable": False,
        "generalization": "not established",
        "generalization_reason": (
            "Hand masks cover one robot, there is no leave-one-robot-out check "
            "and maus-otto has only 6 unseen frames."
        ),
        "missing": [
            "bootstrap intervals for the final model",
            "leave-one-robot-out check, required by models/README.md",
            "any measurement on a Raspberry Pi 5",
            "robustness tests for real steam and light changes",
        ],
        "caveats": [
            "Can mark a hand or sleeve as crepe.",
            "False crepes on 4 of 80 empty plates in the held-out folds.",
            "Hand masks come from one robot and may start from suggestions.",
            "Scores used the slower full-size warp. The fast warp scored "
            "median IoU 0.962 vs 0.964 on 79 unseen frames, chosen on them.",
            "Fixed batch of 1: frames must be run one at a time.",
            "Check-frame scores use the robot's outline as the reference.",
            "All hand-labelled runs are in the shared test split.",
            "Needs a rectification transform for the run to read a frame.",
        ],
        "out_of_scope": [
            "fold state",
            "fold angle: direction never beat the training average, and hand "
            "fine-tuning did not train it",
            "holes: they are filled in the mask",
        ],
        "commit": args.train_commit,
        "write_up": "documentation/analysis/synthetic-crepes.md",
    }
    (HERE / "model.json").write_text(json.dumps(card, indent=2) + "\n")
    example_image(part, images, truth, found)
    print(json.dumps({k: card[k] for k in ("size", "latency")}, indent=1))
    print(json.dumps(card["metrics"]["unseen_check_frames"], indent=1))
    print(json.dumps(card["data"], indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--transform", type=Path, required=True)
    parser.add_argument("--train-commit", default=None)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
