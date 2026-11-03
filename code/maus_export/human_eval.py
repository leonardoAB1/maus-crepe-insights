"""The crêpe student against human Label Studio labels.

Reads the reviewed exports of the labeling project (phase labels in the
``crepe-visual-v2`` taxonomy, brush masks in ``crepe-material-holes-v1``),
one folder per set with ``phase-v1.json`` and ``masks-v1.json``. The labels
are private: they stay in a local folder and only aggregate numbers leave
it.

A label is tied to a video frame by ``run_id`` and ``decoded_frame_index``.
The frames are decoded with the same ffmpeg selection as
``phase_pilot.py`` and checked against ``image_sha256``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from .crepe_student import CrepeStudent, analyze_video, is_test_run
from .fold_frames import video_path
from .fold_state import EMPTY_STATE, FLAT_STATE, HALF_STATE, QUARTER_STATE

RGBA_CHANNELS = 4
HEADER_BITS = 32
WORD_BITS = 5
SIZE_BITS = 4
SIZE_COUNT = 4
SIZE_INDEX_BITS = 2
BITS_PER_BYTE = 8
MATERIAL = "crepe_material"
HOLE = "hole"
CREPE_PRESENT = "crepe_present"
NO_CREPE = "no_crepe"
VALID = "ok"
# Human phase (crepe-visual-v2) -> fold state; "spreading" and "ugly"
# have no state and only show up in the confusion table.
PHASE_STATE = {
    "emptyplate": EMPTY_STATE,
    "circle": FLAT_STATE,
    "halffold": HALF_STATE,
    "quarter fold": QUARTER_STATE,
}
FRAME_KEY = ["run_id", "decoded_frame_index"]
OUTPUT = Path("fold_segmentation") / "student" / "human_eval"


def decode_rle(rle: list[int], height: int, width: int) -> np.ndarray:
    """Label Studio brush RLE -> boolean mask (alpha channel above 0).

    Same bit layout as ``label_studio_sdk.converter.brush.decode_rle``:
    a 32-bit pixel-byte count, a 5-bit word size, four 4-bit run-length
    sizes, then blocks of either one repeated word or literal words.
    """
    bits = "".join(f"{byte:0{BITS_PER_BYTE}b}" for byte in rle)
    position = 0

    def read(count: int) -> int:
        nonlocal position
        value = int(bits[position : position + count], 2)
        position += count
        return value

    total = read(HEADER_BITS)
    if total != height * width * RGBA_CHANNELS:
        raise ValueError("RLE length does not match the image size")
    word = read(WORD_BITS) + 1
    sizes = [read(SIZE_BITS) + 1 for _ in range(SIZE_COUNT)]
    out = np.zeros(total, dtype=np.uint8)
    index = 0
    while index < total:
        repeated = read(1)
        end = index + 1 + read(sizes[read(SIZE_INDEX_BITS)])
        if repeated:
            out[index:end] = read(word)
            index = end
        else:
            while index < end:
                out[index] = read(word)
                index += 1
    return out.reshape(height, width, RGBA_CHANNELS)[..., -1] > 0


def brush_masks(
    result: list[dict[str, Any]], height: int, width: int
) -> dict[str, np.ndarray]:
    """Union of the brush regions per class (material, hole)."""
    masks = {
        MATERIAL: np.zeros((height, width), dtype=bool),
        HOLE: np.zeros((height, width), dtype=bool),
    }
    for region in result:
        if region.get("type") != "brushlabels":
            continue
        value = region["value"]
        mask = decode_rle(value["rle"], height, width)
        for label in value["brushlabels"]:
            if label in masks:
                masks[label] |= mask
    return masks


def _records(labels: Path, kind: str) -> pd.DataFrame:
    rows = []
    for path in sorted(labels.glob(f"*/{kind}-v1.json")):
        for record in json.loads(path.read_text())["frames"]:
            rows.append({"set": path.parent.name, **record})
    return pd.DataFrame(rows)


def load_phase(labels: Path) -> pd.DataFrame:
    """Phase labels of all sets; the label sits in ``phase`` or ``label``."""
    table = _records(labels, "phase")
    if table.empty:
        return table
    if "label" in table:
        phase = table["phase"] if "phase" in table else None
        table["phase"] = (
            table["label"] if phase is None else phase.fillna(table["label"])
        )
    return table


def load_masks(labels: Path) -> pd.DataFrame:
    """Mask labels of all sets."""
    return _records(labels, "masks")


def frame_file(frames_dir: Path, run_id: int, index: int) -> Path:
    return frames_dir / str(run_id) / f"{index:06d}.png"


def extract_frames(video: Path, indices: list[int], out_dir: Path) -> None:
    """Decode frames by index with ffmpeg, named by their index."""
    out_dir.mkdir(parents=True, exist_ok=True)
    select = "+".join(f"eq(n,{index})" for index in indices)
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostats",
                "-xerror",
                "-copyts",
                "-i",
                str(video),
                "-map",
                "0:v:0",
                "-vf",
                f"select='{select}'",
                "-fps_mode",
                "passthrough",
                "-start_number",
                "0",
                f"{tmp}/%06d.png",
            ],
            capture_output=True,
            check=True,
        )
        for number, index in enumerate(indices):
            shutil.move(f"{tmp}/{number:06d}.png", out_dir / f"{index:06d}.png")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decode_label_frames(
    data_dir: Path, labels: pd.DataFrame, frames_dir: Path
) -> pd.DataFrame:
    """Decode every labelled frame once; report the PNG hash match."""
    keys = labels[[*FRAME_KEY, "image_sha256"]].drop_duplicates()
    rows = []
    for run_id, group in keys.groupby("run_id"):
        indices = sorted(int(i) for i in group["decoded_frame_index"].unique())
        missing = [
            index
            for index in indices
            if not frame_file(frames_dir, int(run_id), index).exists()
        ]
        video = video_path(data_dir, int(run_id))
        if missing and video is not None:
            extract_frames(video, missing, frames_dir / str(run_id))
        for row in group.itertuples(index=False):
            path = frame_file(
                frames_dir, int(run_id), int(row.decoded_frame_index)
            )
            rows.append(
                {
                    "run_id": int(run_id),
                    "decoded_frame_index": int(row.decoded_frame_index),
                    "image_sha256": row.image_sha256,
                    "decoded": path.exists(),
                    "sha_match": path.exists()
                    and sha256(path) == row.image_sha256,
                }
            )
    return pd.DataFrame(rows)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else np.nan


def full_size(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    resized = cv2.resize(
        mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
    )
    return resized.astype(bool)


def mask_metrics(
    masks: pd.DataFrame, frames_dir: Path, student: CrepeStudent
) -> pd.DataFrame:
    """Per labelled frame: student vs human material, holes, area."""
    rows = []
    for row in masks.itertuples(index=False):
        image = cv2.imread(
            str(frame_file(frames_dir, row.run_id, row.decoded_frame_index))
        )
        if image is None:
            continue
        height, width = image.shape[:2]
        human = brush_masks(row.result, height, width)
        predicted = full_size(student.mask(image), height, width)
        holes = human[HOLE]
        rows.append(
            {
                "set": row.set,
                "run_id": row.run_id,
                "decoded_frame_index": row.decoded_frame_index,
                "state": row.state,
                "validation": row.validation,
                "test_run": is_test_run(int(row.run_id)),
                "iou": iou(predicted, human[MATERIAL]),
                "human_share": human[MATERIAL].mean(),
                "student_share": predicted.mean(),
                "hole_pixels": int(holes.sum()),
                "hole_as_crepe": (
                    float(predicted[holes].mean()) if holes.any() else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def annotator_iou(masks: pd.DataFrame) -> pd.Series:
    """Material IoU between sets on frames both labelled as crêpe."""
    present = masks[masks["state"] == CREPE_PRESENT]
    values = []
    for _, group in present.groupby(FRAME_KEY):
        if group["set"].nunique() < 2:
            continue
        width, height = group["image_size"].iloc[0]
        first, second = (
            brush_masks(r.result, height, width)[MATERIAL]
            for r in group.head(2).itertuples(index=False)
        )
        values.append(iou(first, second))
    return pd.Series(values, dtype=float)


def label_seconds(phase: pd.DataFrame) -> pd.Series:
    """Video time of each label from its pts and time base."""
    return phase.apply(
        lambda r: float(
            (int(r["decoded_pts"]) - int(r["start_pts"]))
            * Fraction(str(r["time_base"]))
        ),
        axis=1,
    )


def phase_metrics(
    phase: pd.DataFrame, timelines: dict[int, pd.DataFrame]
) -> pd.DataFrame:
    """Live and single-frame state at each human phase label.

    A label is compared with the last analysed frame at or before it.
    """
    out = []
    labelled = phase.assign(second=label_seconds(phase))
    for run_id, group in labelled.groupby("run_id"):
        timeline = timelines.get(int(run_id))
        if timeline is None:
            continue
        joined = pd.merge_asof(
            group.sort_values("second"),
            timeline[["second", "state", "raw_state"]].sort_values("second"),
            on="second",
            direction="backward",
        )
        out.append(joined)
    if not out:
        return pd.DataFrame()
    table = pd.concat(out, ignore_index=True)
    table["human_state"] = table["phase"].map(PHASE_STATE)
    return table


def summary(
    masks: pd.DataFrame, phases: pd.DataFrame, annotators: pd.Series
) -> pd.DataFrame:
    """Aggregate numbers only, safe to copy into docs."""
    rows: list[dict[str, Any]] = []
    crepe = masks[masks["state"] == CREPE_PRESENT]
    for valid, group in crepe.groupby(crepe["validation"] == VALID):
        rows.append(
            {
                "metric": "mask iou, crepe frames",
                "group": "valid" if valid else "needs review",
                "frames": len(group),
                "median": group["iou"].median(),
                "mean": group["iou"].mean(),
            }
        )
    empty = masks[masks["state"] == NO_CREPE]
    rows.append(
        {
            "metric": "student share on empty plates",
            "group": "all",
            "frames": len(empty),
            "median": empty["student_share"].median(),
            "mean": empty["student_share"].mean(),
        }
    )
    holes = masks[masks["hole_pixels"] > 0]
    rows.append(
        {
            "metric": "human hole pixels the student calls crepe",
            "group": "all",
            "frames": len(holes),
            "median": holes["hole_as_crepe"].median(),
            "mean": holes["hole_as_crepe"].mean(),
        }
    )
    rows.append(
        {
            "metric": "mask iou, human vs human",
            "group": "shared frames",
            "frames": len(annotators),
            "median": annotators.median(),
            "mean": annotators.mean(),
        }
    )
    known = phases.dropna(subset=["human_state"]) if len(phases) else phases
    for name, column in (("live", "state"), ("single frame", "raw_state")):
        if known.empty:
            break
        hit = known[column] == known["human_state"]
        rows.append(
            {
                "metric": f"state agreement, {name}",
                "group": "all",
                "frames": len(known),
                "median": np.nan,
                "mean": hit.mean(),
            }
        )
        for state, part in hit.groupby(known["human_state"]):
            rows.append(
                {
                    "metric": f"state agreement, {name}",
                    "group": state,
                    "frames": len(part),
                    "median": np.nan,
                    "mean": part.mean(),
                }
            )
    return pd.DataFrame(rows)


def evaluate(
    data_dir: Path, labels: Path, student: CrepeStudent | None = None
) -> pd.DataFrame:
    """Decode the labelled frames, run the student, write the tables.

    Writes ``fold_segmentation/student/human_eval/`` under ``data_dir``:
    ``frames.csv`` (hash check), ``masks.csv``, ``phases.csv``,
    ``confusion.csv`` and ``summary.csv``.
    """
    student = student or CrepeStudent.load()
    out = data_dir / OUTPUT
    out.mkdir(parents=True, exist_ok=True)
    frames_dir = out / "frames"
    phase = load_phase(labels)
    masks = load_masks(labels)
    keys = pd.concat(
        [t[[*FRAME_KEY, "image_sha256"]] for t in (phase, masks) if len(t)]
    )
    decode_label_frames(data_dir, keys, frames_dir).to_csv(
        out / "frames.csv", index=False
    )
    mask_table = mask_metrics(masks, frames_dir, student)
    mask_table.to_csv(out / "masks.csv", index=False)
    timelines = {}
    for run_id in sorted(phase["run_id"].unique()) if len(phase) else []:
        video = video_path(data_dir, int(run_id))
        if video is None:
            continue
        rows = analyze_video(video, out / "live" / str(run_id), student)
        timelines[int(run_id)] = pd.DataFrame(rows)
    phase_table = phase_metrics(phase, timelines)
    phase_table.drop(columns=["result"], errors="ignore").to_csv(
        out / "phases.csv", index=False
    )
    if len(phase_table):
        pd.crosstab(phase_table["phase"], phase_table["state"]).to_csv(
            out / "confusion.csv"
        )
    table = summary(mask_table, phase_table, annotator_iou(masks))
    table.to_csv(out / "summary.csv", index=False)
    return table
