from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("onnxruntime")

from maus_classify import check_verdict, fold_class  # noqa: E402
from maus_classify.fold_class import (  # noqa: E402
    CLASSES,
    EMBEDDINGS,
    FAIL,
    FRAMES,
    GEOMETRY_CASES,
    NO_DATA,
    PASS,
    VISION,
    FoldClassError,
    choose_c,
    confidence_summary,
    load_frames,
    no_change_frames,
    train_fold_class,
    verdict,
)

DIM = 6
N_RUNS = 30
TEST_RUNS = 6
NOISE = 0.3
ROBOT = "robot-a"
LABEL_OF = {"spread": "flat", "fold1": "half", "fold2": "quarter"}
# Mean direction of each class in the synthetic embedding
CENTRE = {"flat": 0, "half": 1, "quarter": 2, "empty": 3}
# check -> what the pan shows when the check passes
SHOWS = {
    "spread": "flat",
    "fold1": "half",
    "fold2": "quarter",
    "plate_clean": "empty",
    "plate_present": "empty",
}
AREA = {"spread": 0.6, "fold1": 0.3, "fold2": 0.15}
FOLD2_FAILS = 5  # every 5th run: fold2 failed and changed nothing
FOLD1_FAILS = 7  # runs 3, 10, 17, 24: fold1 failed and changed nothing
FOLD1_OFFSET = 3


def run_state(run: int) -> dict[str, tuple[bool, str]]:
    """check -> (ok, what the frame really shows) for one run."""
    fold1_fails = run % FOLD1_FAILS == FOLD1_OFFSET
    fold2_fails = run % FOLD2_FAILS == FOLD2_FAILS - 1
    state = {c: (True, s) for c, s in SHOWS.items()}
    if fold1_fails:
        state["fold1"] = (False, "flat")
        state["fold2"] = (True, "half")  # passed but unlabeled below
    elif fold2_fails:
        state["fold2"] = (False, "half")
    return state


def write_data(data_dir: Path) -> None:
    rng = np.random.default_rng(0)
    frames, emb, vision = [], [], []
    start = pd.Timestamp("2026-10-01", tz="UTC")
    for run in range(N_RUNS):
        split = "test" if run >= N_RUNS - TEST_RUNS else "train"
        state = run_state(run)
        for idx, (check, (ok, shows)) in enumerate(state.items()):
            label = LABEL_OF.get(check) if ok else None
            if check == "fold2" and state["fold1"][0] is False:
                label = None  # keep the synthetic labels clean
            frames.append(
                {
                    "run_id": run,
                    "check": check,
                    "check_index": idx,
                    "attempt": 1,
                    "ts_utc": start + pd.Timedelta(hours=run),
                    "robot": ROBOT,
                    "ok": ok,
                    "is_last_attempt": True,
                    "fold_class": label,
                    "split": split,
                }
            )
            vec = rng.normal(0, NOISE, DIM)
            vec[CENTRE[shows]] += 1
            emb.append(
                {"run_id": run, "check": check, "attempt": 1}
                | {f"e{j}": np.float32(v) for j, v in enumerate(vec)}
            )
        area = dict(AREA)
        if not state["fold1"][0]:
            area["fold1"] = area["spread"]
        if not state["fold2"][0]:
            area["fold2"] = area["fold1"]
        vision.append(
            {"id": run} | {f"vision.{c}.area_rel": a for c, a in area.items()}
        )
    (data_dir / "training").mkdir(parents=True)
    pd.DataFrame(frames).to_parquet(data_dir / FRAMES)
    pd.DataFrame(emb).to_parquet(data_dir / EMBEDDINGS)
    pd.DataFrame(vision).to_parquet(data_dir / VISION)


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    write_data(tmp_path)
    return tmp_path


def test_reuses_check_verdict_helper() -> None:
    assert fold_class.embedding_columns is check_verdict.embedding_columns


def test_load_frames_labels(data_dir: Path) -> None:
    df = load_frames(data_dir)
    assert len(df) == N_RUNS * len(SHOWS)
    labels = df.groupby("check")["label"].first().to_dict()
    assert labels["plate_clean"] == "empty"
    assert pd.isna(labels["plate_present"])  # held out as a test check
    assert set(df["label"].dropna()) == set(CLASSES)


def test_missing_inputs_raise(tmp_path: Path) -> None:
    with pytest.raises(FoldClassError):
        load_frames(tmp_path)


def test_no_change_frames_needs_passed_previous_step() -> None:
    df = pd.DataFrame(
        {
            "run_id": [1, 1, 2, 2, 3, 3],
            "check": ["fold1", "fold2"] * 3,
            "ok": [True, False, False, False, True, False],
            "is_last_attempt": [True, True, True, True, True, False],
        }
    )
    vision = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "vision.fold1.area_rel": [0.3, 0.3, 0.3],
            "vision.fold2.area_rel": [0.31, 0.3, 0.3],
        }
    )
    rows = no_change_frames(df, vision, GEOMETRY_CASES["fold2_no_change"])
    # run 2: fold1 failed; run 3: not the last attempt
    assert rows["run_id"].tolist() == [1]


def test_verdict() -> None:
    win = {"n": 4, "predicted_share": {"half": 0.75, "quarter": 0.25}}
    lose = {"n": 4, "predicted_share": {"half": 0.25, "quarter": 0.75}}
    assert verdict(win, "half", "quarter") == PASS
    assert verdict(lose, "half", "quarter") == FAIL
    assert verdict({"n": 0}, "half", "quarter") == NO_DATA


def test_choose_c_prefers_regularization_on_ties() -> None:
    rng = np.random.default_rng(1)
    y = np.repeat(np.array(CLASSES), 20)
    X = rng.normal(0, 1, (len(y), DIM))
    X[np.arange(len(y)), np.repeat(np.arange(len(CLASSES)), 20)] += 3
    groups = np.arange(len(y)) // 2
    c, scores = choose_c(X, y, groups)
    best = min(scores.values())
    assert scores[str(c)] <= best * 1.05 + 1e-12
    assert all(float(k) >= c or v > best * 1.05 for k, v in scores.items())


def test_confidence_summary() -> None:
    y = np.array(["flat", "half"])
    proba = np.array([[0.0, 0.9, 0.1, 0.0], [0.0, 0.6, 0.4, 0.0]])
    out = confidence_summary(y, proba, CLASSES)
    assert out["accuracy"] == 0.5
    assert out["mean_confidence"] == pytest.approx(0.75)
    assert out["per_true_class_accuracy"] == {"flat": 1.0, "half": 0.0}


def test_train_fold_class_card(data_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "model"
    card = train_fold_class(data_dir, out)
    assert (out / "model.onnx").is_file()
    assert json.loads((out / "model.json").read_text()) == card

    assert card["outputs"][1]["classes"] == CLASSES
    test = card["metrics"]["test"]
    assert set(test["per_class"]) == set(CLASSES)
    assert set(test["calibration"]["per_class"]) == set(CLASSES)
    assert test["confidence"]["accuracy"] > 0.9
    assert card["data"]["test"]["robots"] == [ROBOT]
    by_robot = card["selection"]["cv_by_robot"]
    assert set(by_robot) == {ROBOT}
    assert by_robot[ROBOT]["accuracy"] > 0.9

    sanity = card["sanity_checks"]
    geo = sanity["geometry"]
    assert geo["fold2_no_change"]["verdict"] == PASS
    assert geo["fold2_no_change"]["verdict_basis"] == "holdout"
    assert geo["fold1_no_change"]["verdict"] == PASS
    assert sanity["plate_present_passed"]["verdict"] == PASS
    assert "fold2" in sanity["failed_fold_checks"]

    caveats = card["caveats"]
    assert "#87" in caveats["triangle_fold"]
    assert "#103" in caveats["label_leakage"]
    assert "#91" in caveats["scope"]
    assert ROBOT in caveats["holdout"]


def test_geometry_skipped_without_vision(data_dir: Path, tmp_path) -> None:
    (data_dir / VISION).unlink()
    card = train_fold_class(data_dir, tmp_path / "model")
    assert "skipped" in card["sanity_checks"]["geometry"]
