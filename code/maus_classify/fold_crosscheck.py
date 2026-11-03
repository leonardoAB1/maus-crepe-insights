"""Leakage check for the fold-class head against an independent method.

The fold-class head learned its labels from which check took the frame, so
it could read check timing instead of the fold. The classical segmentation
pipeline (maus_export.fold_*, #133) decides the fold state from the crepe
mask area and edges, without knowing the check. Here both run on the same
raw check frames: theirs after their own perspective correction, unchanged.

Outputs stay private under data/robustness/. Only aggregates leave it.
Needs the ml environment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd

from maus_export.fold_frames import CHECK_STATE, correct_perspective
from maus_export.fold_state import segment_frames

from .fold_class import (
    GEOMETRY_CASES,
    VISION,
    embedding_columns,
    load_frames,
    no_change_frames,
    predict,
)
from .split import TEST

OUT_DIR = Path("robustness")
SEG_ROOT = OUT_DIR / "segmentation"  # their pipeline's data_dir
SEG_DIR = "fold_segmentation"  # fixed name inside their data_dir
REPORT = OUT_DIR / "fold_crosscheck.json"
STATES = ("empty", "flat", "half", "quarter")
JPEG_QUALITY = 95
OURS, THEIRS = "ours", "theirs"


def prepare_segmentation_input(df: pd.DataFrame, data_dir: Path) -> Path:
    """Write our check frames, perspective-corrected, in the layout their
    segment_frames reads. Returns their data_dir."""
    root = data_dir / SEG_ROOT
    frame_dir = root / SEG_DIR / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in df.itertuples(index=False):
        image = cv2.imread(str(data_dir / row.frame_path))
        if image is None:
            continue
        corrected = correct_perspective(image, row.robot)
        name = f"{row.run_id}_{row.check_index:02d}_{row.check}.jpg"
        cv2.imwrite(
            str(frame_dir / name),
            corrected,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
        )
        rows.append(
            {
                "run_id": int(row.run_id),
                "check_name": row.check,
                "expected_state": CHECK_STATE[row.check],
                "check_ok": bool(row.ok),
                "machine_name": row.robot,
                "width": corrected.shape[1],
                "height": corrected.shape[0],
                "frame_path": f"{SEG_DIR}/frames/{name}",
                "attempt": int(row.attempt),
            }
        )
    pd.DataFrame(rows).to_parquet(root / SEG_DIR / "frames.parquet")
    return root


def run_both(data_dir: Path, model_dir: Path) -> pd.DataFrame:
    """One row per check frame with our and their fold state."""
    df = load_frames(data_dir)
    df = df[df["check"].isin(CHECK_STATE)].reset_index(drop=True)
    card = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
    classes = card["outputs"][1]["classes"]
    sess = ort.InferenceSession(
        str(model_dir / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    proba = predict(sess, df[embedding_columns(df)].to_numpy())
    df[OURS] = np.asarray(classes)[proba.argmax(axis=1)]
    df["ours_confidence"] = proba.max(axis=1)

    root = prepare_segmentation_input(df, data_dir)
    seg_frames = pd.read_parquet(root / SEG_DIR / "frames.parquet")
    features = segment_frames(root)  # same row order as seg_frames
    seg = seg_frames[["run_id", "check_name", "attempt"]].assign(
        **{THEIRS: features["predicted_state"].to_numpy()}
    )
    seg = seg.rename(columns={"check_name": "check"})
    keep = ["run_id", "check", "attempt", "check_index", "robot", "ok"]
    keep += ["split", "is_last_attempt", OURS, "ours_confidence", "label"]
    return df[keep].merge(seg, on=["run_id", "check", "attempt"])


def _confusion(a: pd.Series, b: pd.Series) -> dict[str, Any]:
    table = pd.crosstab(a, b).reindex(
        index=STATES, columns=STATES, fill_value=0
    )
    return {
        "rows": a.name,
        "cols": b.name,
        "labels": list(STATES),
        "matrix": table.to_numpy().tolist(),
    }


def _agreement(rows: pd.DataFrame) -> dict[str, Any]:
    if rows.empty:
        return {"n": 0}
    return {
        "n": len(rows),
        "agreement": float((rows[OURS] == rows[THEIRS]).mean()),
    }


def _shares(values: pd.Series) -> dict[str, float]:
    return {s: float((values == s).mean()) for s in STATES}


def summarize(both: pd.DataFrame, vision: pd.DataFrame | None) -> dict:
    """Aggregates only: no run ids."""
    passed = both[both["ok"]]
    expected = passed["check"].map(CHECK_STATE)
    out: dict[str, Any] = {
        "n_frames": len(both),
        "their_validity_on_our_frames": {
            "rule": "theirs == state implied by a passed check",
            "n": len(passed),
            "agreement": float((passed[THEIRS] == expected).mean()),
            "per_state": {
                s: float((passed[THEIRS] == s)[expected == s].mean())
                for s in STATES
                if (expected == s).any()
            },
            "published_on_video_frames": 0.885,
        },
    }
    sides = {"holdout": both[both["split"] == TEST], "all": both}
    out["agreement"] = {}
    for name, rows in sides.items():
        ok = rows[rows["ok"]]
        out["agreement"][name] = {
            "passed": _agreement(ok),
            "per_check_passed": {
                c: _agreement(g) for c, g in ok.groupby("check")
            },
            "per_robot_passed": {
                r: _agreement(g) for r, g in ok.groupby("robot")
            },
            "confusion_passed": _confusion(
                ok[OURS].rename(OURS), ok[THEIRS].rename(THEIRS)
            ),
        }
    failed = both[~both["ok"] & both["check"].isin(["fold1", "fold2"])]
    leak: dict[str, Any] = {}
    for check, rows in failed.groupby("check"):
        leak[check] = {
            "n": len(rows),
            "ours": _shares(rows[OURS]),
            "theirs": _shares(rows[THEIRS]),
            "agreement": float((rows[OURS] == rows[THEIRS]).mean()),
            "confusion": _confusion(
                rows[OURS].rename(OURS), rows[THEIRS].rename(THEIRS)
            ),
        }
    # Failed fold2 frames that the check-blind method still calls half:
    # a head reading shape says half, a head reading the check says quarter
    still_half = failed[
        (failed["check"] == "fold2") & (failed[THEIRS] == "half")
    ]
    leak["fold2_failed_theirs_half"] = {
        "n": len(still_half),
        "ours": _shares(still_half[OURS]) if len(still_half) else {},
    }
    if vision is not None:
        case = GEOMETRY_CASES["fold2_no_change"]
        sideways = no_change_frames(both, vision, case)
        leak["fold2_no_change"] = {
            "n": len(sideways),
            "ours": _shares(sideways[OURS]) if len(sideways) else {},
            "theirs": _shares(sideways[THEIRS]) if len(sideways) else {},
        }
    out["leakage"] = leak
    return out


def build_crosscheck(data_dir: Path, model_dir: Path) -> dict[str, Any]:
    both = run_both(data_dir, model_dir)
    vision_path = data_dir / VISION
    vision = pd.read_parquet(vision_path) if vision_path.is_file() else None
    report = summarize(both, vision)
    out = data_dir / REPORT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    both.to_parquet(data_dir / OUT_DIR / "fold_crosscheck_rows.parquet")
    return report
