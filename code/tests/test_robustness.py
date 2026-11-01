"""Robustness eval on synthetic frames and fake heads (no models, no data)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("onnxruntime")

from maus_classify import robustness  # noqa: E402
from maus_classify.check_verdict import FAIL, PASS  # noqa: E402
from maus_classify.robustness import (  # noqa: E402
    IDENTITY,
    Perturbation,
    apply,
    brightness,
    gamma,
    haze,
    holdout_frames,
    perturbations,
    rotate_plate,
    score_all,
    summarize,
)
from maus_classify.split import TEST, TRAIN  # noqa: E402
from maus_export.fold_frames import TARGET_SOURCE_SIZE  # noqa: E402

WIDTH, HEIGHT = TARGET_SOURCE_SIZE
CENTRE = (WIDTH // 2, HEIGHT // 2)
RADIUS = 150
DOT_OFFSET = 100
ROBOT = "robot-a"
GREY = 128
BRIGHT = 200  # synthetic "crepe" brightness, above the fake head threshold
DARK = 40
THRESHOLD = 100.0


def calibration() -> dict:
    return {
        "image_size_px": [WIDTH, HEIGHT],
        "robot_corrections": {
            ROBOT: {
                "matrix_mean": np.eye(3).tolist(),
                "target_size_px": [WIDTH, HEIGHT],
            }
        },
        "hotplate_roi": {
            ROBOT: {"center_px": list(CENTRE), "radius_px": RADIUS}
        },
    }


def grey_image(value: int = GREY) -> np.ndarray:
    return np.full((HEIGHT, WIDTH, 3), value, np.uint8)


def test_perturbation_list_starts_with_identity() -> None:
    ps = perturbations()
    assert ps[0].name == IDENTITY
    names = [p.name for p in ps]
    assert len(names) == len(set(names))
    assert "rotation=90" in names
    assert {p.kind for p in ps} == {
        IDENTITY,
        "gamma",
        "brightness",
        "haze",
        "rotation",
    }


def test_light_perturbations_move_pixels_the_right_way() -> None:
    img = grey_image()
    assert gamma(img, 0.5).mean() > GREY > gamma(img, 2.0).mean()
    assert brightness(img, 1.5).mean() > GREY > brightness(img, 0.6).mean()
    assert gamma(img, 1.0).mean() == pytest.approx(GREY, abs=1)


def test_haze_lowers_contrast() -> None:
    img = grey_image(DARK)
    img[:, : WIDTH // 2] = BRIGHT
    hazy = haze(img, 0.6)
    assert hazy.std() < img.std()
    assert hazy.mean() > img.mean()  # veil is light grey


def test_rotate_plate_turns_only_the_disc() -> None:
    img = grey_image(DARK)
    cx, cy = CENTRE
    cv2.circle(img, (cx + DOT_OFFSET, cy), 10, (BRIGHT,) * 3, -1)
    corner = (5, 5)
    img[corner] = BRIGHT
    out = rotate_plate(img, 180, ROBOT, calibration())
    assert out is not None
    assert out[cy, cx - DOT_OFFSET].mean() > THRESHOLD
    assert out[cy, cx + DOT_OFFSET].mean() < THRESHOLD
    assert out[corner].mean() == BRIGHT  # outside the disc: untouched


def test_rotate_plate_without_calibration_is_none() -> None:
    assert rotate_plate(grey_image(), 90, "unknown", calibration()) is None
    assert rotate_plate(grey_image(), 90, ROBOT, {}) is None


def test_apply_identity_returns_frame() -> None:
    img = grey_image()
    out = apply(Perturbation(IDENTITY, 0.0), img, ROBOT, {})
    assert out is img


def frames_table(tmp_path: Path) -> pd.DataFrame:
    """Holdout frames: bright = flat/pass, dark = empty/fail."""
    rows = []
    specs = [
        ("spread", True, BRIGHT, TEST),
        ("spread", False, DARK, TEST),
        ("plate_clean", True, DARK, TEST),
        ("push_off", True, BRIGHT, TEST),
        ("push_off", False, DARK, TEST),
        ("fold1", True, BRIGHT, TRAIN),
    ]
    (tmp_path / "frames").mkdir()
    for i, (check, ok, value, split) in enumerate(specs):
        path = f"frames/{i}.jpg"
        cv2.imwrite(str(tmp_path / path), grey_image(value))
        rows.append(
            {
                "run_id": i,
                "check": check,
                "attempt": 1,
                "robot": ROBOT,
                "ok": ok,
                "split": split,
                "frame_path": path,
            }
        )
    rows.append({**rows[0], "run_id": 99, "frame_path": None})
    df = pd.DataFrame(rows)
    (tmp_path / "training").mkdir()
    df.to_parquet(tmp_path / robustness.FRAMES)
    return df


def test_holdout_frames_targets(tmp_path: Path) -> None:
    frames_table(tmp_path)
    f = holdout_frames(tmp_path)
    assert set(f["split"]) == {TEST}
    assert f["frame_path"].notna().all()
    spread = f[f["check"] == "spread"].set_index("ok")
    assert spread.loc[True, "fold_target"] == "flat"
    assert pd.isna(spread.loc[False, "fold_target"])  # failed: shape unknown
    assert spread.loc[False, "verdict_target"] == FAIL
    clean = f[f["check"] == "plate_clean"].iloc[0]
    assert clean["fold_target"] == "empty"
    assert pd.isna(clean["verdict_target"])
    push = f[f["check"] == "push_off"].set_index("ok")
    assert push.loc[True, "verdict_target"] == PASS


class FakeHeads:
    """Bright frame -> flat and pass; dark -> empty and fail."""

    def __init__(self, models: Path) -> None:
        pass

    def embed(self, images: list[np.ndarray]) -> np.ndarray:
        return np.array([[float(im.mean())] for im in images], np.float32)

    def fold_answer(self, emb: np.ndarray) -> np.ndarray:
        return np.where(emb[:, 0] > THRESHOLD, "flat", "empty")

    def verdict_answer(self, check: str, emb: np.ndarray) -> np.ndarray:
        return np.where(emb[:, 0] > THRESHOLD, PASS, FAIL)


@pytest.fixture
def scored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> pd.DataFrame:
    frames_table(tmp_path)
    monkeypatch.setattr(robustness, "Heads", FakeHeads)
    monkeypatch.setattr(robustness, "load_calibration", calibration)
    return score_all(tmp_path, tmp_path)


def test_score_all_identity_is_perfect(scored: pd.DataFrame) -> None:
    n_holdout = 5
    assert len(scored) == n_holdout * len(perturbations())
    base = scored[scored["perturbation"] == IDENTITY]
    fold = base[base["fold_target"].notna()]
    assert (fold["fold_answer"] == fold["fold_target"]).all()
    verdict = base[base["verdict_target"].notna()]
    assert (verdict["verdict_answer"] == verdict["verdict_target"]).all()
    assert base["verdict_answer"][base["check"] == "plate_clean"].isna().all()


def test_summarize_reports_flips_and_no_run_ids(scored: pd.DataFrame) -> None:
    report = summarize(scored)
    none = report["perturbations"][IDENTITY]
    assert none["fold"]["all"]["balanced_accuracy"] == 1.0
    assert none["fold"]["all"]["flip_rate"] == 0.0
    assert none["verdict"]["all"]["fail_recall"] == 1.0
    assert none["verdict"]["all"]["n_fail"] == 2
    assert report["robots"] == [ROBOT]
    # Heavy haze pulls dark frames toward the light veil: answers flip
    hazy = report["perturbations"]["haze=0.6"]["verdict"]["all"]
    assert hazy["flip_rate"] > 0
    text = json.dumps(report)
    assert "run_id" not in text
    assert "frames/" not in text


def test_metrics_empty_rows() -> None:
    empty = pd.DataFrame(
        columns=["fold_target", "fold_answer", "fold_answer_base"]
    )
    assert robustness._metrics(empty, "fold") == {"n": 0}
