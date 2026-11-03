"""Robustness of the frame classifiers to light, steam, rotation and robot.

Each holdout frame (robot-chrono-v2 split) is perturbed, re-embedded with
the frame embedder and re-scored by the fold-class head and the four
check-verdict heads. Reported per perturbation strength, per check and per
robot: the metric against the robot label, and the flip rate (share of
answers that change from the unperturbed frame, which needs no label).

Perturbations:
- gamma and brightness: exposure and light changes
- haze: blend toward a light grey plus blur, a stand-in for steam
- rotation: the plate area turns about its centre. The frame is warped to
  the top-down plate view with the robot's calibration, the plate disc is
  rotated, and it is warped back over the original frame.

Outputs stay private under data/robustness/. Needs the ml environment.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd

from maus_export.fold_frames import CALIBRATION_PATH, TARGET_SOURCE_SIZE
from maus_export.frame_preprocess import load_frame, preprocess

from .check_verdict import CHECKS as VERDICT_CHECKS
from .check_verdict import FAIL, PASS
from .harness import majority_baseline
from .split import TEST

OUT = Path("robustness") / "perturbations.json"
FRAMES = Path("training") / "frames.parquet"
PIXEL_MAX = 255.0
HAZE_GREY = 225.0  # steam is a light, low-contrast veil
HAZE_BLUR_PER_ALPHA = 6.0  # blur sigma grows with haze strength
BATCH = 32
GAMMAS = (0.5, 0.7, 1.5, 2.0)
BRIGHTNESS = (0.6, 0.8, 1.25, 1.5)
HAZE = (0.2, 0.4, 0.6)
ROTATIONS = (30, 90, 180)
FOLD_CHECKS = {"plate_clean": "empty", "spread": "flat"}
FOLD_CHECKS |= {"fold1": "half", "fold2": "quarter"}
IDENTITY = "none"


@dataclass(frozen=True)
class Perturbation:
    kind: str
    strength: float

    @property
    def name(self) -> str:
        return (
            IDENTITY
            if self.kind == IDENTITY
            else f"{self.kind}={self.strength:g}"
        )


def perturbations() -> list[Perturbation]:
    out = [Perturbation(IDENTITY, 0.0)]
    out += [Perturbation("gamma", g) for g in GAMMAS]
    out += [Perturbation("brightness", b) for b in BRIGHTNESS]
    out += [Perturbation("haze", h) for h in HAZE]
    out += [Perturbation("rotation", r) for r in ROTATIONS]
    return out


def gamma(img: np.ndarray, g: float) -> np.ndarray:
    lut = (np.linspace(0, 1, 256) ** g * PIXEL_MAX).astype(np.uint8)
    return cv2.LUT(img, lut)


def brightness(img: np.ndarray, scale: float) -> np.ndarray:
    return cv2.convertScaleAbs(img, alpha=scale, beta=0)


def haze(img: np.ndarray, alpha: float) -> np.ndarray:
    sigma = alpha * HAZE_BLUR_PER_ALPHA
    blurred = cv2.GaussianBlur(img, (0, 0), sigma) if sigma > 0 else img
    veiled = (1 - alpha) * blurred.astype(np.float32) + alpha * HAZE_GREY
    return np.clip(veiled, 0, PIXEL_MAX).astype(np.uint8)


def load_calibration(path: Path = CALIBRATION_PATH) -> dict[str, Any]:
    return (
        json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    )


def rotate_plate(
    img: np.ndarray, degrees: float, robot: str, calibration: dict[str, Any]
) -> np.ndarray | None:
    """Turn the plate disc about its centre in the top-down view and warp
    it back. None when the robot has no calibration."""
    corr = (calibration.get("robot_corrections") or {}).get(robot)
    roi = (calibration.get("hotplate_roi") or {}).get(robot)
    if not corr or not roi:
        return None
    src = cv2.resize(img, TARGET_SOURCE_SIZE, interpolation=cv2.INTER_AREA)
    h = np.asarray(corr["matrix_mean"], dtype=np.float64)
    size = tuple(corr.get("target_size_px", calibration["image_size_px"]))
    top = cv2.warpPerspective(src, h, size)
    centre = tuple(float(c) for c in roi["center_px"])
    turn = cv2.getRotationMatrix2D(centre, degrees, 1.0)
    turned = cv2.warpAffine(top, turn, size)
    disc = np.zeros(size[::-1], np.uint8)
    cv2.circle(
        disc, tuple(int(c) for c in centre), int(roi["radius_px"]), 255, -1
    )
    back_size = (src.shape[1], src.shape[0])
    back = cv2.warpPerspective(turned, np.linalg.inv(h), back_size)
    mask = cv2.warpPerspective(disc, np.linalg.inv(h), back_size) > 0
    out = src.copy()
    out[mask] = back[mask]
    return out


def apply(
    p: Perturbation, img: np.ndarray, robot: str, calibration: dict
) -> np.ndarray | None:
    ops: dict[str, Callable[[], np.ndarray | None]] = {
        IDENTITY: lambda: img,
        "gamma": lambda: gamma(img, p.strength),
        "brightness": lambda: brightness(img, p.strength),
        "haze": lambda: haze(img, p.strength),
        "rotation": lambda: rotate_plate(img, p.strength, robot, calibration),
    }
    return ops[p.kind]()


class Heads:
    """Embedder plus the fold-class and check-verdict heads (ONNX)."""

    def __init__(self, models: Path) -> None:
        def session(path: Path) -> ort.InferenceSession:
            return ort.InferenceSession(
                str(path), providers=["CPUExecutionProvider"]
            )

        def classes(path: Path) -> list[str]:
            card = json.loads(path.read_text(encoding="utf-8"))
            return card["outputs"][1]["classes"]

        self.embedder = session(models / "frame-embedder" / "model.onnx")
        self.fold = session(models / "fold-class" / "model.onnx")
        self.fold_classes = classes(models / "fold-class" / "model.json")
        self.verdict = {
            c: session(models / "check-verdict" / c / "model.onnx")
            for c in VERDICT_CHECKS
        }
        self.verdict_classes = {
            c: classes(models / "check-verdict" / c / "model.json")
            for c in VERDICT_CHECKS
        }

    def embed(self, images: list[np.ndarray]) -> np.ndarray:
        name = self.embedder.get_inputs()[0].name
        chunks = []
        for i in range(0, len(images), BATCH):
            x = np.concatenate([preprocess(im) for im in images[i : i + BATCH]])
            chunks.append(self.embedder.run(None, {name: x})[0])
        return np.concatenate(chunks).astype(np.float32)

    def _answer(
        self, sess: ort.InferenceSession, classes: list[str], emb: np.ndarray
    ) -> np.ndarray:
        proba = sess.run(["probabilities"], {"x": emb})[0]
        return np.asarray(classes)[proba.argmax(axis=1)]

    def fold_answer(self, emb: np.ndarray) -> np.ndarray:
        return self._answer(self.fold, self.fold_classes, emb)

    def verdict_answer(self, check: str, emb: np.ndarray) -> np.ndarray:
        return self._answer(
            self.verdict[check], self.verdict_classes[check], emb
        )


def holdout_frames(data_dir: Path) -> pd.DataFrame:
    """Holdout frames the heads are scored on, with their targets."""
    f = pd.read_parquet(data_dir / FRAMES)
    f = f[(f["split"] == TEST) & f["frame_path"].notna()].copy()
    fold_ok = f["check"].isin(FOLD_CHECKS) & f["ok"]
    f["fold_target"] = np.where(fold_ok, f["check"].map(FOLD_CHECKS), None)
    f["verdict_target"] = np.where(
        f["check"].isin(VERDICT_CHECKS), np.where(f["ok"], PASS, FAIL), None
    )
    keep = f["fold_target"].notna() | f["verdict_target"].notna()
    return f[keep].reset_index(drop=True)


def score_all(data_dir: Path, models: Path) -> pd.DataFrame:
    """Long table: one row per frame and perturbation with the answers."""
    frames = holdout_frames(data_dir)
    images = [load_frame(data_dir / p) for p in frames["frame_path"]]
    heads = Heads(models)
    calibration = load_calibration()
    parts = []
    for p in perturbations():
        out = [
            apply(p, im, robot, calibration)
            for im, robot in zip(images, frames["robot"], strict=True)
        ]
        ok = np.array([o is not None for o in out])
        emb = heads.embed([o for o in out if o is not None])
        part = frames.loc[ok, ["run_id", "check", "attempt", "robot"]].copy()
        part["fold_target"] = frames.loc[ok, "fold_target"].to_numpy()
        part["verdict_target"] = frames.loc[ok, "verdict_target"].to_numpy()
        part["perturbation"], part["kind"] = p.name, p.kind
        part["strength"] = p.strength
        part["fold_answer"] = heads.fold_answer(emb)
        verdict = np.full(len(part), None, dtype=object)
        for check in VERDICT_CHECKS:
            m = (part["check"] == check).to_numpy()
            if m.any():
                verdict[m] = heads.verdict_answer(check, emb[m])
        part["verdict_answer"] = verdict
        parts.append(part)
    table = pd.concat(parts, ignore_index=True)
    base = table[table["perturbation"] == IDENTITY].set_index(
        ["run_id", "check", "attempt"]
    )[["fold_answer", "verdict_answer"]]
    table = table.join(
        base.add_suffix("_base"), on=["run_id", "check", "attempt"]
    )
    return table


def _balanced(y: pd.Series, pred: pd.Series) -> float | None:
    recalls = [float((pred[y == c] == c).mean()) for c in sorted(y.unique())]
    return float(np.mean(recalls)) if recalls else None


def _metrics(rows: pd.DataFrame, model: str) -> dict[str, Any]:
    target, answer = f"{model}_target", f"{model}_answer"
    rows = rows[rows[target].notna()]
    if rows.empty:
        return {"n": 0}
    y, pred = rows[target], rows[answer]
    out = {
        "n": len(rows),
        "balanced_accuracy": _balanced(y, pred),
        "accuracy": float((y == pred).mean()),
        "flip_rate": float((pred != rows[f"{answer}_base"]).mean()),
    }
    if model == "verdict":
        fails = y == FAIL
        out["n_fail"] = int(fails.sum())
        out["fail_recall"] = (
            float((pred[fails] == FAIL).mean()) if fails.any() else None
        )
    return out


def summarize(table: pd.DataFrame) -> dict[str, Any]:
    """Aggregates per model, perturbation, check and robot. No run ids."""
    out: dict[str, Any] = {
        "split": "robot-chrono-v2 holdout",
        "chance_balanced_accuracy": {
            "fold": majority_baseline(len(FOLD_CHECKS)),
            "verdict": majority_baseline(2),
        },
        "robots": sorted(table["robot"].unique()),
        "perturbations": {},
    }
    for name, rows in table.groupby("perturbation", sort=False):
        entry: dict[str, Any] = {}
        for model in ("fold", "verdict"):
            entry[model] = {
                "all": _metrics(rows, model),
                "per_check": {
                    c: _metrics(g, model)
                    for c, g in rows.groupby("check")
                    if g[f"{model}_target"].notna().any()
                },
                "per_robot": {
                    r: _metrics(g, model) for r, g in rows.groupby("robot")
                },
            }
        out["perturbations"][name] = entry
    return out


def build_robustness(data_dir: Path, models: Path) -> dict[str, Any]:
    table = score_all(data_dir, models)
    report = summarize(table)
    out = data_dir / OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    table.to_parquet(out.with_suffix(".parquet"))
    return report


EXAMPLE_ASSET = Path("assets") / "robustness-example.jpg"
EXAMPLE_CHECK = "fold1"
EXAMPLE_SHOWN = ("none", "rotation=90", "haze=0.6")
TILE_WIDTH = 450
LABEL_SCALE = 0.7
LABEL_THICKNESS = 2
LABEL_ORIGIN = (8, 26)
LABEL_LINE = 28
OK_COLOUR = (60, 200, 60)  # BGR
BAD_COLOUR = (40, 40, 230)


def _label(tile: np.ndarray, lines: list[tuple[str, tuple]]) -> None:
    x, y = LABEL_ORIGIN
    for i, (text, colour) in enumerate(lines):
        cv2.putText(
            tile,
            text,
            (x, y + i * LABEL_LINE),
            cv2.FONT_HERSHEY_SIMPLEX,
            LABEL_SCALE,
            colour,
            LABEL_THICKNESS,
        )


def example_figure(data_dir: Path, out: Path) -> Path:
    """One passed fold1 holdout frame that the fold head gets right
    unperturbed and wrong both turned and in steam: the failure case."""
    table = pd.read_parquet((data_dir / OUT).with_suffix(".parquet"))
    rows = table[
        (table["check"] == EXAMPLE_CHECK) & table["fold_target"].notna()
    ]
    key = ["run_id", "check", "attempt"]
    wide = rows.pivot_table(
        index=key, columns="perturbation", values="fold_answer", aggfunc="first"
    )
    target = rows.groupby(key)["fold_target"].first()
    right = wide[IDENTITY] == target
    wrong = [(wide[p] != target) for p in EXAMPLE_SHOWN[1:]]
    picks = wide[right & np.logical_and.reduce(wrong)]
    if picks.empty:
        raise ValueError("no frame fails every shown perturbation")
    pick = picks.index[0]
    frames = holdout_frames(data_dir).set_index(key)
    robot = frames.loc[pick, "robot"]
    img = load_frame(data_dir / frames.loc[pick, "frame_path"])
    calibration = load_calibration()
    by_name = {p.name: p for p in perturbations()}
    tiles = []
    for name in EXAMPLE_SHOWN:
        tile = apply(by_name[name], img, robot, calibration)
        h, w = tile.shape[:2]
        tile = cv2.resize(tile, (TILE_WIDTH, round(h * TILE_WIDTH / w)))
        answer = wide.loc[pick, name]
        colour = OK_COLOUR if answer == target[pick] else BAD_COLOUR
        _label(
            tile,
            [
                (name, OK_COLOUR),
                (f"head: {answer} (true {target[pick]})", colour),
            ],
        )
        tiles.append(tile)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.hstack(tiles))
    return out
