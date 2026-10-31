"""Train the fold-risk model and export ONNX.

pixi run -e ml python models/fold-risk/train.py [--data-dir ../data]

Reads data/training/frames.parquet, data/vision.parquet and, when present,
data/training/frame_embeddings.parquet. Writes model.onnx and model.json next
to this file. The shipped model uses the spread overlay scores and the
robot's spread verdict. Two comparisons are scored on the same holdout and
recorded in the card: the verdict alone, and scores plus the spread frame
embedding.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from maus_classify.fold_risk import (
    FAIL,
    SCORE_FEATURES,
    VERDICT_FEATURE,
    embedding_columns,
    fold_risk_table,
    lift_report,
    robot_breakdown,
    signal_verdict,
    verdict_baseline,
)
from maus_classify.harness import (
    ONNX_INPUT,
    ONNX_PROBA,
    train_and_evaluate,
)
from maus_classify.split import TEST, TRAIN
from maus_export.analysis import find_data_dir
from maus_export.doe_analyze import dump_json

HERE = Path(__file__).resolve().parent
NAME = "fold-risk"
FEATURES = SCORE_FEATURES + [VERDICT_FEATURE]
C = 1.0  # L2 strength; default, not tuned on 94 train runs
MAX_ITER = 1000
EMBED_COMPONENTS = 8
PERM_REPEATS = 200
CV_SPLITS = 5
CV_REPEATS = 20
SEED = 0


def scores_model() -> Pipeline:
    return make_pipeline(
        StandardScaler(), LogisticRegression(C=C, max_iter=MAX_ITER)
    )


def embedding_model(n_scores: int, n_embed: int) -> Pipeline:
    scores = list(range(n_scores))
    embed = list(range(n_scores, n_scores + n_embed))
    prep = ColumnTransformer(
        [
            ("scores", StandardScaler(), scores),
            (
                "embed",
                make_pipeline(
                    StandardScaler(),
                    PCA(n_components=EMBED_COMPONENTS, random_state=SEED),
                ),
                embed,
            ),
        ]
    )
    return Pipeline(
        [("prep", prep), ("lr", LogisticRegression(C=C, max_iter=MAX_ITER))]
    )


def holdout_proba(path: Path, X: np.ndarray, classes: list[str]) -> np.ndarray:
    """Failure probability from the exported graph, as it will be served."""
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    proba = sess.run([ONNX_PROBA], {ONNX_INPUT: X.astype(np.float32)})[0]
    return proba[:, classes.index(FAIL)]


def fit(
    table: pd.DataFrame,
    features: list[str],
    estimator: Pipeline,
    out_dir: Path,
    description: str,
) -> tuple[dict[str, Any], np.ndarray]:
    card = train_and_evaluate(
        estimator,
        table[features].to_numpy(float),
        table["label"].to_numpy(),
        table["split"].to_numpy(),
        table.index.to_numpy(),
        name=NAME,
        feature_names=features,
        out_dir=out_dir,
        timestamps=table["ts_utc"].to_numpy(),
        description=description,
    )
    test = table[table["split"] == TEST]
    classes = card["outputs"][1]["classes"]
    p = holdout_proba(
        out_dir / "model.onnx", test[features].to_numpy(float), classes
    )
    return card, p


def importance(table: pd.DataFrame) -> dict[str, Any]:
    """Standardized coefficients (train fit) and holdout permutation drop in
    ROC AUC. Both are noisy at this sample size."""
    train = table[table["split"] == TRAIN]
    test = table[table["split"] == TEST]
    model = scores_model().fit(
        train[FEATURES].to_numpy(float), train["fold_fail"].to_numpy()
    )
    coef = model[-1].coef_[0]
    perm = permutation_importance(
        model,
        test[FEATURES].to_numpy(float),
        test["fold_fail"].to_numpy(),
        scoring="roc_auc",
        n_repeats=PERM_REPEATS,
        random_state=SEED,
    )
    rows = [
        {
            "feature": f,
            "std_coef": float(coef[i]),
            "perm_auc_drop_mean": float(perm.importances_mean[i]),
            "perm_auc_drop_std": float(perm.importances_std[i]),
        }
        for i, f in enumerate(FEATURES)
    ]
    rows.sort(key=lambda r: -abs(r["std_coef"]))
    return {
        "note": "std_coef: log-odds of fold failure per train SD; "
        "positive raises risk",
        "features": rows,
    }


def _cv_auc(rows: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    cv = RepeatedStratifiedKFold(
        n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=SEED
    )
    auc = cross_val_score(
        scores_model(),
        rows[features].to_numpy(float),
        rows["fold_fail"].to_numpy(),
        scoring="roc_auc",
        cv=cv,
    )
    return {
        "n_runs": len(rows),
        "n_fail": int(rows["fold_fail"].sum()),
        "roc_auc_mean": float(auc.mean()),
        "roc_auc_std": float(auc.std()),
    }


def train_cv(table: pd.DataFrame) -> dict[str, Any]:
    """Repeated stratified CV ROC AUC on train runs only (one row per run).
    Across robots it can learn the robot instead of the crepe, so it is
    repeated within the largest train robot."""
    train = table[table["split"] == TRAIN]
    largest = train["robot"].value_counts().idxmax()
    one = train[train["robot"] == largest]
    lightless = [f for f in FEATURES if not f.startswith("l_")]
    return {
        "scheme": f"{CV_REPEATS}x{CV_SPLITS}-fold stratified, train runs",
        "all_robots": _cv_auc(train, FEATURES),
        "largest_robot": {
            "all_features": _cv_auc(one, FEATURES),
            "verdict_only": _cv_auc(one, [VERDICT_FEATURE]),
            "without_lightness": _cv_auc(one, lightless),
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=None)
    args = p.parse_args()
    data_dir = args.data_dir or find_data_dir()

    frames = pd.read_parquet(data_dir / "training" / "frames.parquet")
    vision = pd.read_parquet(data_dir / "vision.parquet")
    table = fold_risk_table(frames, vision)
    test = table[table["split"] == TEST]
    y_test = test["fold_fail"].to_numpy(float)

    card, p_model = fit(
        table,
        FEATURES,
        scores_model(),
        HERE,
        "Risk that fold1 or fold2 fails, from the last spread check: "
        "overlay scores and the robot's spread verdict. Weak labels "
        "(robot verdicts). Small sample, see fold_risk.sample.",
    )
    report = lift_report(y_test, p_model)
    baseline = lift_report(y_test, verdict_baseline(table))
    comparisons: dict[str, Any] = {
        "spread_verdict_only": {
            "what": "train fold-fail rate by robot spread verdict",
            "holdout": baseline,
        }
    }

    emb_path = data_dir / "training" / "frame_embeddings.parquet"
    if emb_path.is_file():
        emb_table = fold_risk_table(frames, vision, pd.read_parquet(emb_path))
        emb_feats = FEATURES + embedding_columns(emb_table)
        emb_test = emb_table[emb_table["split"] == TEST]
        with tempfile.TemporaryDirectory() as tmp:
            _, p_emb = fit(
                emb_table,
                emb_feats,
                embedding_model(len(FEATURES), len(emb_feats) - len(FEATURES)),
                Path(tmp),
                "comparison only",
            )
        comparisons["scores_plus_embedding"] = {
            "what": f"scores + PCA({EMBED_COMPONENTS}) of the spread frame "
            "embedding (models/frame-embedder)",
            "n_runs": len(emb_table),
            "holdout": lift_report(
                emb_test["fold_fail"].to_numpy(float), p_emb
            ),
        }

    card["fold_risk"] = {
        "target": "any fold1 or fold2 attempt in the run not ok "
        "(robot verdict, weak label)",
        "positive_class": FAIL,
        "sample": {
            "n_runs": len(table),
            "n_fail": int(table["fold_fail"].sum()),
            TRAIN: int((table["split"] == TRAIN).sum()),
            TEST: len(test),
            "test_fail": int(y_test.sum()),
            "note": "small sample: holdout intervals are wide",
        },
        "holdout": report,
        "verdict": signal_verdict(report),
        "comparisons": comparisons,
        "robots": robot_breakdown(table),
        "train_cv": train_cv(table),
        "importance": importance(table),
    }
    dump_json(card, HERE / "model.json")
    auc = report["roc_auc"]
    print(
        f"{len(table)} runs ({len(test)} test, {int(y_test.sum())} fail). "
        f"Holdout ROC AUC {auc['value']:.2f} "
        f"[{auc['ci'][0]:.2f}, {auc['ci'][1]:.2f}]: "
        f"{card['fold_risk']['verdict']}"
    )


if __name__ == "__main__":
    main()
