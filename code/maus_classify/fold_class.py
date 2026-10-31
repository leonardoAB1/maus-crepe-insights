"""Fold state from a single frame: empty, flat, half or quarter (Task 3).

A logistic-regression head on the frozen frame embeddings (#90), trained
and exported with the shared harness. Labels are weak: a passed spread
check is flat, a passed fold1 is half, a passed fold2 is quarter (the
`fold_class` column of the frame table), and a passed plate_clean check is
an empty pan. Failed checks have no label.

The head names the fold state of a step that succeeded. It does not decide
whether a fold succeeded; that is the check verdict head (#91).

Needs the ml environment: `pixi run -e ml fold-class`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from .check_verdict import embedding_columns
from .harness import ONNX_INPUT, ONNX_PROBA, train_and_evaluate, write_card
from .split import TEST, TRAIN

NAME = "fold-class"
EMPTY = "empty"
FOLD_CLASSES = ["flat", "half", "quarter"]
CLASSES = [EMPTY, *FOLD_CLASSES]
LABEL = "fold_class"
EMPTY_CHECK = "plate_clean"  # passed: the pan is empty
KEY = ["run_id", "check", "attempt"]
FRAMES = Path("training") / "frames.parquet"
EMBEDDINGS = Path("training") / "frame_embeddings.parquet"
VISION = Path("vision.parquet")
FOLD_CHECKS = ["spread", "fold1", "fold2"]
V = "vision."
# A failed fold whose crepe area barely changed did nothing: the crepe keeps
# its previous shape. A shape-reading head names that shape; a head that
# learned which check took the frame names the check's class instead.
NO_CHANGE_BAND = 0.15  # area ratio within 1 +- this
GEOMETRY_CASES = {
    "fold2_no_change": {
        "check": "fold2",
        "previous": "fold1",
        "ratio": (f"{V}fold2.area_rel", f"{V}fold1.area_rel"),
        "expected": "half",
        "check_class": "quarter",
    },
    "fold1_no_change": {
        "check": "fold1",
        "previous": "spread",
        "ratio": (f"{V}fold1.area_rel", f"{V}spread.area_rel"),
        "expected": "flat",
        "check_class": "half",
    },
}
PASS_SHARE = 0.5  # expected class must win at least this share
# passed plate_present frames show an empty pan from a check the head never
# trains on
EMPTY_TEST_CHECK = "plate_present"
PASS, FAIL, NO_DATA = "pass", "fail", "no frames"
C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
# Pick the strongest regularization whose CV loss is within this share of
# the best: the loss curve is flat at the top, and a smaller C is less
# overconfident
LOSS_TOL = 0.05
CV_FOLDS = 5
MAX_ITER = 5000
SEED = 0


class FoldClassError(ValueError):
    """Missing or unusable input."""


def load_frames(data_dir: Path) -> pd.DataFrame:
    """Frame table joined with embeddings on (run_id, check, attempt),
    plus a `label` column. Frames without an embedding are dropped."""
    for rel in (FRAMES, EMBEDDINGS):
        if not (data_dir / rel).is_file():
            raise FoldClassError(f"missing {rel}; build it first")
    frames = pd.read_parquet(data_dir / FRAMES)
    emb = pd.read_parquet(data_dir / EMBEDDINGS)
    df = frames.merge(emb[KEY + embedding_columns(emb)], on=KEY, how="inner")
    empty = (df["check"] == EMPTY_CHECK) & df["ok"]
    df["label"] = df[LABEL].where(df[LABEL].isin(FOLD_CLASSES))
    df.loc[empty, "label"] = EMPTY
    return df


def build_estimator(c: float) -> Pipeline:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, max_iter=MAX_ITER, random_state=SEED),
    )


def choose_c(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray
) -> tuple[float, dict[str, float]]:
    """Regularization by run-grouped CV log loss, on train rows only.
    Log loss rewards calibrated probabilities, not just right labels."""
    folds = GroupKFold(n_splits=min(CV_FOLDS, len(set(groups))))
    scores: dict[str, float] = {}
    for c in C_GRID:
        losses = []
        for tr, va in folds.split(X, y, groups):
            model = build_estimator(c).fit(X[tr], y[tr])
            proba = model.predict_proba(X[va])
            losses.append(log_loss(y[va], proba, labels=model.classes_))
        scores[str(c)] = float(np.mean(losses))
    floor = min(scores.values()) * (1 + LOSS_TOL)
    chosen = min(c for c in C_GRID if scores[str(c)] <= floor)
    return chosen, scores


def predict(sess: ort.InferenceSession, X: np.ndarray) -> np.ndarray:
    return sess.run([ONNX_PROBA], {ONNX_INPUT: X.astype(np.float32)})[0]


def cv_by_robot(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    robots: np.ndarray,
    c: float,
) -> dict[str, Any]:
    """Out-of-fold accuracy and log loss per robot, from run-grouped CV on
    train rows. Every robot is in train, so this shows whether one robot
    is much harder; it is not a leave-one-robot-out test."""
    folds = GroupKFold(n_splits=min(CV_FOLDS, len(set(groups))))
    labels = np.unique(y)
    proba = cross_val_predict(
        build_estimator(c),
        X,
        y,
        groups=groups,
        cv=folds,
        method="predict_proba",
    )
    pred = labels[proba.argmax(axis=1)]
    out: dict[str, Any] = {}
    for robot in sorted(set(robots)):
        m = robots == robot
        out[str(robot)] = {
            "n": int(m.sum()),
            "accuracy": float((pred[m] == y[m]).mean()),
            "log_loss": float(log_loss(y[m], proba[m], labels=labels)),
            "label_counts": pd.Series(y[m]).value_counts().to_dict(),
        }
    return out


def confidence_summary(
    y: np.ndarray, proba: np.ndarray, classes: list[str]
) -> dict[str, Any]:
    """Mean top-class confidence against accuracy: the gap shows over- or
    under-confidence at a glance, overall and per predicted class."""
    pred = np.asarray(classes)[proba.argmax(axis=1)]
    conf = proba.max(axis=1)
    hit = pred == y
    per_class = {}
    for cls in classes:
        mask = pred == cls
        if mask.any():
            per_class[cls] = {
                "n_predicted": int(mask.sum()),
                "mean_confidence": float(conf[mask].mean()),
                "accuracy": float(hit[mask].mean()),
            }
    return {
        "accuracy": float(hit.mean()),
        "mean_confidence": float(conf.mean()),
        "per_true_class_accuracy": {
            cls: float(hit[y == cls].mean())
            for cls in classes
            if (y == cls).any()
        },
        "per_predicted_class": per_class,
    }


def shares(
    sess: ort.InferenceSession,
    rows: pd.DataFrame,
    cols: list[str],
    classes: list[str],
) -> dict[str, Any]:
    """n, predicted class shares and mean confidence on `rows`."""
    if rows.empty:
        return {"n": 0}
    proba = predict(sess, rows[cols].to_numpy())
    pred = np.asarray(classes)[proba.argmax(axis=1)]
    return {
        "n": len(pred),
        "predicted_share": {c: float((pred == c).mean()) for c in classes},
        "mean_confidence": float(proba.max(axis=1).mean()),
    }


def verdict(res: dict[str, Any], expected: str, rival: str | None) -> str:
    """pass when the expected class wins at least PASS_SHARE and beats the
    rival class; fail otherwise."""
    if not res["n"]:
        return NO_DATA
    share = res["predicted_share"]
    ok = share[expected] >= PASS_SHARE
    if rival is not None:
        ok = ok and share[expected] > share[rival]
    return PASS if ok else FAIL


def no_change_frames(
    df: pd.DataFrame, vision: pd.DataFrame, case: dict[str, Any]
) -> pd.DataFrame:
    """Failed last-attempt frames of the case's check whose crepe area
    ratio is within NO_CHANGE_BAND of 1, in runs where the previous step
    passed (so the crepe had the expected shape going in). vision.parquet
    scores the last attempt of each check, so only last attempts match."""
    num, den = case["ratio"]
    v = vision[["id", num, den]].rename(columns={"id": "run_id"})
    last = df[df["is_last_attempt"]]
    passed = last.loc[
        (last["check"] == case["previous"]) & last["ok"], "run_id"
    ]
    rows = last[
        (last["check"] == case["check"])
        & ~last["ok"]
        & last["run_id"].isin(passed)
    ].merge(v, on="run_id", how="inner")
    ratio = rows[num] / rows[den]
    return rows[(ratio - 1).abs() <= NO_CHANGE_BAND]


def geometry_checks(
    sess: ort.InferenceSession,
    df: pd.DataFrame,
    vision: pd.DataFrame | None,
    cols: list[str],
    classes: list[str],
) -> dict[str, Any]:
    """Leakage test that can fail: failed folds that changed nothing.

    The verdict uses holdout runs, or all runs when the holdout has none.
    Train runs hold a near-duplicate of the same crepe labeled with the
    expected class, which flatters a pass but makes a fail stronger."""
    if vision is None:
        return {"skipped": f"no {VISION}; run `pixi run vision`"}
    out: dict[str, Any] = {
        "rule": (
            f"failed last attempt after a passed previous step, crepe "
            f"area ratio within 1 +- {NO_CHANGE_BAND:g} of that step; "
            "pass if the "
            f"expected class wins >= {PASS_SHARE:.0%} and beats the "
            "check's own class"
        )
    }
    for name, case in GEOMETRY_CASES.items():
        rows = no_change_frames(df, vision, case)
        holdout = shares(sess, rows[rows["split"] == TEST], cols, classes)
        everything = shares(sess, rows, cols, classes)
        basis = "holdout" if holdout["n"] else "all_runs"
        out[name] = {
            "expected": case["expected"],
            "check_class": case["check_class"],
            "holdout": holdout,
            "all_runs": everything,
            "verdict": verdict(
                holdout if holdout["n"] else everything,
                case["expected"],
                case["check_class"],
            ),
            "verdict_basis": basis,
        }
    return out


def sanity_checks(
    sess: ort.InferenceSession,
    df: pd.DataFrame,
    vision: pd.DataFrame | None,
    cols: list[str],
    classes: list[str],
) -> dict[str, Any]:
    """Predictions on holdout frames the head never saw a label for."""
    test = df[df["split"] == TEST]
    empty = shares(
        sess,
        test[(test["check"] == EMPTY_TEST_CHECK) & test["ok"]],
        cols,
        classes,
    )
    failed = test[test["check"].isin(FOLD_CHECKS) & ~test["ok"]]
    return {
        "geometry": geometry_checks(sess, df, vision, cols, classes),
        f"{EMPTY_TEST_CHECK}_passed": {
            **empty,
            "expected": EMPTY,
            "verdict": verdict(empty, EMPTY, None),
        },
        "failed_fold_checks": {
            name: shares(sess, g, cols, classes)
            for name, g in failed.groupby("check")
        },
    }


CAVEATS = {
    "scope": (
        "Names the fold state of a step that succeeded. It does not detect "
        "an incomplete fold: on failed fold frames it spreads its answers "
        "over the classes at high confidence (see "
        "sanity_checks.failed_fold_checks). Use the check verdict head "
        "(#91) to decide whether a fold succeeded."
    ),
    "triangle_fold": (
        "Unsupported. No frame in the data shows a triangle fold, so the "
        "head can only answer empty, flat, half or quarter. Human fold "
        "labels from the labelling workstream (#87) are the path to add "
        "it, but its visual taxonomy (crepe-visual-v2) has no triangle "
        "label yet; the follow-up (#106) depends on adding one."
    ),
    "label_leakage": (
        "Weak labels: the class equals the check that took the frame. The "
        "head may learn check-specific camera timing or plate state "
        "instead of the fold shape. sanity_checks.geometry tests this "
        "with failed folds that changed nothing (small n, stated there). "
        "Still-flat crepes after a failed fold1 are named flat (pass). "
        "Still-half crepes after a failed fold2 are never named half "
        "(fail): by eye they are half folds turned sideways, which no "
        "training frame shows, and the head reads the wedge as quarter "
        "or the pan as empty. The likely cause is rotation: the head "
        "reads the wedge's orientation, not the fold, so half vs quarter "
        "is only shown to work in the robot's normal fold pose. Follow-up: "
        "rotation augmentation (#103) and rotation robustness plus the "
        "check against human labels on video frames (#104)."
    ),
    "empty": (
        "empty is learned from passed plate_clean frames only and tested "
        "on passed plate_present frames, a check it never trained on."
    ),
    "failed_checks": (
        "Failed checks are excluded from training because their true "
        "state is unknown."
    ),
}


def train_fold_class(data_dir: Path, out_dir: Path) -> dict[str, Any]:
    """Fit, score and export the fold head; returns the model card."""
    df = load_frames(data_dir)
    cols = embedding_columns(df)
    labeled = df[df["label"].notna()].reset_index(drop=True)
    if labeled.empty:
        raise FoldClassError("no labeled frames")
    X = labeled[cols].to_numpy(np.float64)
    y = labeled["label"].astype(str).to_numpy()
    split = labeled["split"].to_numpy()
    groups = labeled["run_id"].to_numpy()

    train = split == TRAIN
    c, cv_scores = choose_c(X[train], y[train], groups[train])
    card = train_and_evaluate(
        build_estimator(c),
        X,
        y,
        split,
        groups,
        name=NAME,
        feature_names=cols,
        out_dir=out_dir,
        timestamps=labeled["ts_utc"].to_numpy(),
        robots=labeled["robot"].fillna("unknown").to_numpy(),
        description=(
            "Fold state of the crepe from one raw check frame of a "
            "successful step: empty pan, flat, half or quarter, with "
            "probabilities. Logistic regression on the frozen "
            "frame-embedder output (code/models/frame-embedder)."
        ),
    )

    classes = card["outputs"][1]["classes"]
    sess = ort.InferenceSession(
        str(out_dir / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    vision_path = data_dir / VISION
    vision = pd.read_parquet(vision_path) if vision_path.is_file() else None
    test = split == TEST
    card["labels"] = {
        "rule": (
            "spread ok -> flat, fold1 ok -> half, fold2 ok -> quarter "
            f"(frames.{LABEL}); {EMPTY_CHECK} ok -> {EMPTY}"
        ),
        "counts": {
            side: labeled.loc[mask, "label"].value_counts().to_dict()
            for side, mask in ((TRAIN, train), (TEST, test))
        },
    }
    card["features"] = {
        "source": "data/training/frame_embeddings.parquet",
        "embedder": "code/models/frame-embedder (preprocess first)",
        "join": KEY,
    }
    card["selection"] = {
        "param": "LogisticRegression C",
        "chosen": c,
        "cv": f"{CV_FOLDS}-fold run-grouped, train rows only",
        "rule": f"smallest C within {LOSS_TOL:.0%} of the best CV log loss",
        "cv_log_loss": cv_scores,
        "cv_by_robot": cv_by_robot(
            X[train],
            y[train],
            groups[train],
            labeled.loc[train, "robot"].fillna("unknown").to_numpy(),
            c,
        ),
    }
    card["metrics"][TEST]["confidence"] = confidence_summary(
        y[test], predict(sess, X[test]), classes
    )
    card["sanity_checks"] = sanity_checks(sess, df, vision, cols, classes)
    card["caveats"] = CAVEATS | {"holdout": card["data"]["holdout_note"]}
    card["out_of_scope"] = "edge angle for plate alignment"
    write_card(card, out_dir / "model.json")
    return card


EXAMPLE_ASSET = Path("assets") / "fold-class-example.jpg"


def _typical(rows: pd.DataFrame) -> pd.Series:
    """The row with the median confidence: typical, not cherry-picked."""
    order = rows.sort_values("confidence", kind="stable")
    return order.iloc[len(order) // 2]


def example_figure(data_dir: Path, model_dir: Path, out: Path) -> Path:
    """Example grid for the PR and README: one typical correct holdout
    frame per class, the documented geometry FAIL and a holdout error."""
    from maus_export.frame_preprocess import crop_plate, load_frame

    from .figures import Example, example_grid

    df = load_frames(data_dir)
    cols = embedding_columns(df)
    sess = ort.InferenceSession(
        str(model_dir / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    card = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
    classes = card["outputs"][1]["classes"]

    def scored(rows: pd.DataFrame) -> pd.DataFrame:
        proba = predict(sess, rows[cols].to_numpy())
        return rows.assign(
            predicted=np.asarray(classes)[proba.argmax(axis=1)],
            confidence=proba.max(axis=1),
        )

    def tile(
        row: pd.Series, label: str, note: str = "", truth: str = ""
    ) -> Example:
        return Example(
            image=load_frame(data_dir / row["frame_path"]),
            truth=truth or label,
            predicted=row["predicted"],
            confidence=float(row["confidence"]),
            robot=str(row["robot"]),
            note=note,
            agree=row["predicted"] == label,
        )

    test = scored(df[(df["split"] == TEST) & df["label"].notna()])
    tiles = [
        tile(_typical(g), cls, "holdout")
        for cls in CLASSES
        if len(g := test[(test["label"] == cls) & (test["predicted"] == cls)])
    ]
    vision_path = data_dir / VISION
    if vision_path.is_file():
        case = GEOMETRY_CASES["fold2_no_change"]
        fails = scored(no_change_frames(df, pd.read_parquet(vision_path), case))
        fails = fails[fails["predicted"] != case["expected"]]
        if len(fails):
            row = fails.sort_values("confidence").iloc[-1]
            tiles.append(
                tile(
                    row,
                    case["expected"],
                    "known FAIL: sideways half",
                    f"{case['expected']} (fold2 did nothing)",
                )
            )
    errors = test[test["predicted"] != test["label"]]
    if len(errors):
        row = errors.sort_values("confidence").iloc[-1]
        tiles.append(tile(row, row["label"], "holdout error"))
    return example_grid(
        tiles,
        out,
        title="fold-class: crepe fold state from one frame",
        subtitle=(
            "Robot label vs model answer (confidence), robot. Green = "
            "agree, red = disagree. Frames cropped to the plate."
        ),
        crop=crop_plate,
    )
