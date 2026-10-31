"""Fold risk: robot-adjusted insight, power, model checks and ONNX export.

pixi run -e ml python models/fold-risk/train.py [--data-dir ../data]

Reads data/training/frames.parquet and data/vision.parquet. Writes
model.onnx and model.json next to this file and the effect plot to
assets/fold-risk-effects.png (aggregates only). See
documentation/analysis/fold-risk.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from maus_classify import fold_cv, fold_effects
from maus_classify.fold_effects import PRIMARY, ROBOT
from maus_classify.fold_risk import (
    FAIL,
    MECHANISMS,
    fold_risk_table,
    lift_report,
)
from maus_classify.harness import ONNX_INPUT, ONNX_PROBA, train_and_evaluate
from maus_classify.split import TEST
from maus_export.analysis import find_data_dir
from maus_export.doe_analyze import dump_json

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FIGURE = REPO / "assets" / "fold-risk-effects.png"
NAME = "fold-risk"
# Robots in the fit. maus-otto has 7 runs (6 fold failures) and is reported
# only, as the issue asks.
FIT_ROBOTS = ["maus-quin", "maus-pixi"]
REPORT_ROBOTS = ["maus-otto"]
C = 1.0
MAX_ITER = 1000
N_PERM = 100
PERM_REPEATS = 5
DEPLOYABLE = "deployable"
RESEARCH = "research only"


def model() -> Any:
    return make_pipeline(
        StandardScaler(), LogisticRegression(C=C, max_iter=MAX_ITER)
    )


def robot_counts(table: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for robot, g in table.groupby(ROBOT):
        rows.append(
            {
                "robot": str(robot),
                "in_fit": robot in FIT_ROBOTS,
                "n_runs": len(g),
                "first_utc": g["ts_utc"].min().isoformat(),
                "last_utc": g["ts_utc"].max().isoformat(),
                "spread_fail": int(g[fold_effects.SPREAD_FAIL].sum()),
                **{f"{m}_fail": int(g[t].sum()) for m, t in MECHANISMS.items()},
                "excluded_no_crepe": int(g[MECHANISMS["shape"]].isna().sum()),
            }
        )
    return rows


def permutation_null(table: pd.DataFrame, target: str) -> dict[str, Any]:
    """Within-robot delta AUC when labels are shuffled inside each robot."""
    rows = table[table[ROBOT].isin(FIT_ROBOTS) & table[target].notna()]
    rows = rows.copy()
    rng = np.random.default_rng(0)
    vals = []
    for i in range(N_PERM):
        rows["_perm"] = rows.groupby(ROBOT)[target].transform(
            lambda s: rng.permutation(s.to_numpy())
        )
        pm, _ = fold_cv.oof_predictions(
            rows, "_perm", PRIMARY, n_repeats=PERM_REPEATS, seed=i
        )
        auc = fold_cv.within_auc(
            rows["_perm"].to_numpy(float), pm, rows[ROBOT].to_numpy()
        )
        vals.append(auc - 0.5)
    v = np.asarray(vals)
    return {
        "n_perm": N_PERM,
        "mean": float(v.mean()),
        "q05_q95": [float(np.quantile(v, 0.05)), float(np.quantile(v, 0.95))],
    }


def shared_holdout(table: pd.DataFrame, card: dict[str, Any]) -> dict:
    test = table[table["split"] == TEST]
    sess = ort.InferenceSession(
        str(HERE / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    proba = sess.run(
        [ONNX_PROBA],
        {ONNX_INPUT: test[PRIMARY].to_numpy(np.float32)},
    )[0]
    classes = card["outputs"][1]["classes"]
    p = proba[:, classes.index(FAIL)]
    robots = sorted(test[ROBOT].astype(str).unique())
    return {
        "label": f"one robot ({', '.join(robots)})",
        "report": lift_report(test["fold_fail"].to_numpy(float), p),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=None)
    args = p.parse_args()
    data_dir = args.data_dir or find_data_dir()

    frames = pd.read_parquet(data_dir / "training" / "frames.parquet")
    vision = pd.read_parquet(data_dir / "vision.parquet")
    table = fold_effects.add_spread_fail(fold_risk_table(frames, vision))
    all_robots = FIT_ROBOTS + REPORT_ROBOTS

    eff = fold_effects.effects(table, FIT_ROBOTS)
    fold_effects.plot_effects(eff, FIGURE)
    checks = {
        mech: {
            "pooled_cv": fold_cv.pooled_cv(table, target, PRIMARY, FIT_ROBOTS),
            "within_robot_cv": fold_cv.within_robot_cv(
                table, target, PRIMARY, FIT_ROBOTS
            ),
            "leave_one_robot_out": fold_cv.leave_one_robot_out(
                table, target, PRIMARY, all_robots
            ),
        }
        for mech, target in MECHANISMS.items()
    }
    checks["any"]["permutation_null"] = permutation_null(
        table, MECHANISMS["any"]
    )
    delta = checks["any"]["pooled_cv"]["delta_auc_within_robot"]
    verdict = DEPLOYABLE if delta["ci95"][0] > 0 else RESEARCH

    card = train_and_evaluate(
        model(),
        table[PRIMARY].to_numpy(float),
        table["label"].to_numpy(),
        table["split"].to_numpy(),
        table.index.to_numpy(),
        name=NAME,
        feature_names=PRIMARY,
        out_dir=HERE,
        timestamps=table["ts_utc"].to_numpy(),
        description=f"{verdict.upper()}. Risk that fold1 or fold2 fails, "
        "from the last spread check (robot verdict and overlay scores). "
        "Weak labels (robot verdicts). See fold_risk.verdict.",
    )
    card["fold_risk"] = {
        "verdict": verdict,
        "verdict_rule": "deployable only if the 95% CI of within-robot "
        "delta AUC over robot-only (any fold failure) is above 0",
        "target": {
            "any": "any fold1 or fold2 attempt not ok",
            "residue": "a fold error 'batter left on the plate'",
            "shape": "a fold error 'not a half fold', 'not a quarter fold' "
            "or 'wrong fold shape'",
            "excluded": "runs with 'no crepe detected' are left out of "
            "residue and shape",
        },
        "features": PRIMARY,
        "fit_robots": FIT_ROBOTS,
        "report_only_robots": REPORT_ROBOTS,
        "robots": robot_counts(table),
        "pooled_spread_verdict": fold_effects.pooled_verdict(
            table, MECHANISMS["any"]
        ),
        "effects": eff,
        "confound_check": fold_effects.confound_check(table, FIT_ROBOTS),
        "power": fold_effects.power_analysis(table, FIT_ROBOTS),
        "model_checks": checks,
        "shared_holdout": shared_holdout(table, card),
    }
    dump_json(card, HERE / "model.json")
    print(
        f"{len(table)} runs. Within-robot delta AUC "
        f"{delta['value']:+.2f} [{delta['ci95'][0]:+.2f}, "
        f"{delta['ci95'][1]:+.2f}] -> {verdict}"
    )


if __name__ == "__main__":
    main()
