"""Compare crepe-pose with the distilled crepe-student on the same frames.

Local comparison. Both masks are scored in the rectified view at 160 pixels, the
same way as the other crepe-pose results, against reviewed hand masks and robot
outlines. The final hand-tuned crepe-pose trained on the hand masks, so on those
it is shown only for reference and `tuned` is the fair one. The student split is
by run hash, so each table separates the runs it held out.

    python models/crepe-pose/compare_student.py --data-dir DATA --student ONNX
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd

from maus_export import human_masks as hm
from maus_export import synthetic_crepes as sc
from maus_export.label_deviation import run_directories, warp_mask

HERE = Path(__file__).resolve().parent
CALIBRATION = HERE.parents[2] / "robot-image-calibration.json"
# The student contract, from maus_export.crepe_student
STUDENT_SIZE = (384, 216)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
MIN_CREPE_SHARE = 0.02
STUDENT_TEST_MOD = 5
THRESHOLD = 0.5
LOW_PERCENTILE = 10
PRESENT = "crepe_present"
RUNS = 40
WARM_UP = 10
BOOTSTRAP = 1000
PERCENTILES = (2.5, 97.5)
SEED = 7
TILE = 200
SHEET_COLUMNS = 6
FILL_SHARE = 0.35


def load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def student_held_out(run: int) -> bool:
    """Whether the student's run split kept this run out of training."""
    digest = hashlib.sha1(str(run).encode()).hexdigest()
    return int(digest, 16) % STUDENT_TEST_MOD == 0


def session(path: Path, threads: int = 0) -> ort.InferenceSession:
    options = ort.SessionOptions()
    if threads:
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(path), options, providers=["CPUExecutionProvider"]
    )


def student_input(frame: np.ndarray) -> np.ndarray:
    small = cv2.resize(frame, STUDENT_SIZE)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32)
    return ((rgb / sc.MAX_U8 - MEAN) / STD).transpose(2, 0, 1)[None]


def student_mask(
    model: ort.InferenceSession,
    frame: np.ndarray,
    matrix: np.ndarray,
    size: int,
) -> np.ndarray:
    """The largest student region, in the rectified view at 160 pixels. An
    area below the student's own empty threshold counts as no crepe."""
    raw = model.run(None, {"image": student_input(frame)})[0][0, 0] > 0
    largest = sc.clean_mask(raw)
    if largest.mean() < MIN_CREPE_SHARE:
        largest = np.zeros_like(largest)
    height, width = frame.shape[:2]
    full = cv2.resize(
        largest.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    scale = sc.SIZE / size
    warp = np.diag([scale, scale, 1.0]) @ matrix
    return (
        cv2.warpPerspective(
            full, warp, (sc.SIZE, sc.SIZE), flags=cv2.INTER_NEAREST
        )
        > 0
    )


def pose_mask(
    model: ort.InferenceSession,
    frame: np.ndarray,
    matrix: np.ndarray,
    size: int,
) -> np.ndarray:
    small = sc.rectify_for_model(frame, matrix, size)
    batch = (small.astype(np.float32) / sc.MAX_U8).transpose(2, 0, 1)[None]
    return sc.clean_mask(model.run(None, {"x": batch})[0][0, 0] > THRESHOLD)


def interval(values: np.ndarray, runs: np.ndarray) -> list[float]:
    """95% interval of the median, resampling whole runs."""
    groups = [values[runs == run] for run in np.unique(runs)]
    rng = np.random.default_rng(SEED)
    draws = [
        np.median(
            np.concatenate(
                [groups[i] for i in rng.integers(len(groups), size=len(groups))]
            )
        )
        for _ in range(BOOTSTRAP)
    ]
    return [round(float(v), 3) for v in np.percentile(draws, PERCENTILES)]


def scores(
    found: np.ndarray,
    truth: np.ndarray,
    valid: np.ndarray,
    present: np.ndarray,
    runs: np.ndarray,
) -> dict[str, object]:
    out: dict[str, object] = {
        "frames": len(found),
        "runs": len(set(runs.tolist())),
        "crepes": int(present.sum()),
    }
    picks = np.flatnonzero(present)
    if len(picks):
        ious = np.array(
            [sc.iou(found[i] & valid[i], truth[i] & valid[i]) for i in picks]
        )
        centres = [
            sc.centre_error_rel(found[i] & valid[i], truth[i] & valid[i])
            for i in picks
        ]
        centre = np.array([np.nan if c is None else c for c in centres])
        out["iou"] = round(float(np.median(ious)), 3)
        out["iou_ci"] = interval(ious, runs[picks])
        out["iou_low"] = round(float(np.percentile(ious, LOW_PERCENTILE)), 3)
        if np.isfinite(centre).any():
            out["centre"] = round(float(np.nanmedian(centre)), 4)
        out["missed"] = int(sum(not found[i].any() for i in picks))
    empty = np.flatnonzero(~present)
    if len(empty):
        out["empty"] = len(empty)
        out["false_on_empty"] = int(sum(found[i].any() for i in empty))
    return out


def hand_frames(data: Path, hand_masks) -> dict[str, object]:
    """Raw frames, rectified truth and transforms for the usable hand masks."""
    labeling = data / hand_masks.LABELING
    repo = data / "maus-crepe-labels"
    records = hm.usable_records(
        hand_masks.snapshot_text(repo, p) for p in hand_masks.SNAPSHOTS
    )
    frames = hm.frame_index(labeling / p for p in hand_masks.PILOTS)
    identities = hm.identity_index(labeling / p for p in hand_masks.PILOTS)
    pilot = labeling / hand_masks.TRANSFORM_PILOT
    directories = run_directories(pilot)
    raw, matrices, truth, valid, runs, present = [], [], [], [], [], []
    for record in records:
        run = int(record["run_id"])
        path = frames.get(record["image_sha256"]) or identities.get(
            hm.frame_identity(record)
        )
        name = pilot / "transforms" / f"{directories.get(run, '')}.json"
        if path is None or not name.is_file():
            raise ValueError(f"no frame or transform for run {run}")
        transform = json.loads(name.read_text())
        material, holes = hm.record_masks(record)
        warped, support = warp_mask(material | holes, transform)
        raw.append(cv2.imread(str(path)))
        matrices.append(np.array(transform["matrix"], np.float64))
        truth.append(hand_masks.small(warped))
        valid.append(hand_masks.small(support))
        runs.append(run)
        present.append(record["state"] == PRESENT)
    return {
        "raw": raw,
        "matrices": matrices,
        "truth": np.stack(truth),
        "valid": np.stack(valid),
        "runs": np.array(runs),
        "present": np.array(present),
    }


def check_frames(data: Path) -> dict[str, object]:
    """Test check frames from runs without hand masks, with robot outlines."""
    out = data / "synthetic"
    table = pd.read_parquet(out / "real.parquet")
    masks = np.load(out / "real.npz")["masks"]
    hand = set(pd.read_parquet(out / "hand.parquet")["run_id"])
    frames = pd.read_parquet(data / "training" / "frames.parquet")
    corrections = json.loads(CALIBRATION.read_text())["robot_corrections"]
    keep = (
        (table["split"] == "test")
        & table["check"].isin(("spread", "fold1", "fold2"))
        & ~table["run_id"].isin(hand)
    )
    raw, matrices, truth, runs, robots = [], [], [], [], []
    seen: set[tuple[int, str]] = set()
    for index, row in table[keep].iterrows():
        pair = (row["run_id"], row["check"])
        if pair in seen:
            continue
        chosen = frames[
            (frames["run_id"] == row["run_id"])
            & (frames["check"] == row["check"])
            & frames["frame_path"].notna()
            & frames["is_last_attempt"]
        ]
        if chosen.empty:
            continue
        seen.add(pair)
        raw.append(cv2.imread(str(data / chosen.iloc[0]["frame_path"])))
        matrices.append(np.array(corrections[row["robot"]]["matrix_mean"]))
        truth.append(masks[index])
        runs.append(row["run_id"])
        robots.append(row["robot"])
    return {
        "raw": raw,
        "matrices": matrices,
        "truth": np.stack(truth),
        "valid": np.ones_like(np.stack(truth)),
        "runs": np.array(runs),
        "robots": np.array(robots),
        "present": np.ones(len(raw), bool),
    }


def speed(
    student: Path, pose: Path, frame: np.ndarray, matrix: np.ndarray
) -> dict[str, dict[str, float]]:
    """Frame to cleaned mask on this machine. Single process, same frame."""
    out: dict[str, dict[str, float]] = {}
    for threads in (1, 4):
        cv2.setNumThreads(threads)
        models = {
            "student": (session(student, threads), student_mask),
            "pose": (session(pose, threads), pose_mask),
        }
        for name, (model, method) in models.items():
            times = []
            for run in range(WARM_UP + RUNS):
                started = time.perf_counter()
                method(model, frame, matrix, sc.RECTIFIED_PX)
                if run >= WARM_UP:
                    times.append(time.perf_counter() - started)
            out.setdefault(name, {})[f"threads_{threads}_ms"] = round(
                float(np.median(times)) * 1000, 1
            )
    cv2.setNumThreads(-1)
    return out


def sheet(
    path: Path,
    data: dict[str, object],
    pose: np.ndarray,
    student: np.ndarray,
) -> None:
    """Largest disagreements: pose in red, student in blue, truth green."""
    truth, valid = data["truth"], data["valid"]
    picks = np.flatnonzero(data["present"])
    gaps = [sc.iou(pose[i] & valid[i], student[i] & valid[i]) for i in picks]
    order = picks[np.argsort(gaps)][: SHEET_COLUMNS * 2]
    tiles = []
    for index in order:
        image = sc.rectify_for_model(
            data["raw"][index], data["matrices"][index], sc.RECTIFIED_PX
        )
        tile = cv2.resize(image, (TILE, TILE))
        for mask, colour in (
            (pose[index], (40, 40, 255)),
            (student[index], (255, 120, 40)),
            (truth[index], (0, 255, 0)),
        ):
            big = cv2.resize(
                mask.astype(np.uint8),
                (TILE, TILE),
                interpolation=cv2.INTER_NEAREST,
            )
            contours, _ = cv2.findContours(
                big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            cv2.drawContours(tile, contours, -1, colour, 1)
        tiles.append(tile)
    while len(tiles) % SHEET_COLUMNS:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [
        np.hstack(tiles[i : i + SHEET_COLUMNS])
        for i in range(0, len(tiles), SHEET_COLUMNS)
    ]
    cv2.imwrite(str(path), np.vstack(rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--student-commit", default="not merged")
    args = parser.parse_args()
    data = args.data_dir
    out = data / "synthetic"
    train = load_module("train")
    student_model = session(args.student)
    tuned_path = out / "tuned-fixed.onnx"
    train_model = train.PoseNet()
    train_model.load_state_dict(
        train.torch.load(train.weights_path(data, "tuned"), weights_only=True)
    )
    train_model.eval()
    train.torch.onnx.export(
        train_model,
        (train.torch.rand(1, 3, sc.SIZE, sc.SIZE),),
        str(tuned_path),
        input_names=["x"],
        output_names=["mask", "direction"],
        opset_version=17,
        dynamo=False,
    )
    pose_models = {
        "tuned": session(tuned_path),
        "final": session(HERE / "model.onnx"),
    }

    hand = hand_frames(data, load_module("hand_masks"))
    checks = check_frames(data)
    results: dict[str, object] = {"student_commit": args.student_commit}
    for name, part in (("hand", hand), ("check", checks)):
        found = {
            "student": np.stack(
                [
                    student_mask(student_model, f, m, sc.RECTIFIED_PX)
                    for f, m in zip(part["raw"], part["matrices"], strict=True)
                ]
            )
        }
        for pose_name, model in pose_models.items():
            found[pose_name] = np.stack(
                [
                    pose_mask(model, f, m, sc.RECTIFIED_PX)
                    for f, m in zip(part["raw"], part["matrices"], strict=True)
                ]
            )
        held = np.array([student_held_out(int(r)) for r in part["runs"]])
        section: dict[str, object] = {}
        for group, chosen in (
            ("all", np.ones(len(held), bool)),
            ("student_held_out", held),
        ):
            if not chosen.any():
                continue
            section[group] = {
                method: scores(
                    masks[chosen],
                    part["truth"][chosen],
                    part["valid"][chosen],
                    part["present"][chosen],
                    part["runs"][chosen],
                )
                for method, masks in found.items()
            }
        if "robots" in part:
            for robot in sorted(set(part["robots"].tolist())):
                chosen = part["robots"] == robot
                section[robot] = {
                    method: scores(
                        masks[chosen],
                        part["truth"][chosen],
                        part["valid"][chosen],
                        part["present"][chosen],
                        part["runs"][chosen],
                    )
                    for method, masks in found.items()
                }
        results[name] = section
        sheet(
            out / f"compare-student-{name}.jpg",
            part,
            found["tuned"],
            found["student"],
        )
    results["speed"] = speed(
        args.student, HERE / "model.onnx", hand["raw"][0], hand["matrices"][0]
    )
    results["size_mb"] = {
        "student": round(args.student.stat().st_size / 1e6, 2),
        "pose": round((HERE / "model.onnx").stat().st_size / 1e6, 2),
    }
    (out / "compare-student.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
