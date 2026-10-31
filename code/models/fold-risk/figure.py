"""Example result grid for the fold-risk model (PR #111, issue #117).

pixi run -e ml python models/fold-risk/figure.py [--data-dir ../data]

One tile per holdout run: the last spread-check frame, the robot's fold
outcome (truth) against the model's answer, and its fold failure risk.
Tiles are picked by rule, not by eye: the most confident catch, the
failure the model was surest would pass (a miss), the most confident false
alarm, and a typical pass and failure. Writes assets/fold-risk-example.jpg.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

from maus_classify.figures import Example, example_grid
from maus_classify.fold_effects import PRIMARY, ROBOT, add_spread_fail
from maus_classify.fold_risk import FAIL, PASS, fold_risk_table, last_spread
from maus_classify.harness import ONNX_INPUT, ONNX_PROBA
from maus_classify.split import TEST
from maus_export.analysis import find_data_dir
from maus_export.frame_preprocess import crop_plate, load_frame

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
EXAMPLE_ASSET = REPO / "assets" / "fold-risk-example.jpg"
FRAMES = Path("training") / "frames.parquet"
VISION = Path("vision.parquet")
COLS = 3
DECISION = 0.5  # the exported label's own cut-off


def scored_holdout(data_dir: Path, model_dir: Path) -> pd.DataFrame:
    """Holdout runs with `p_fail`, `predicted` and the spread frame path."""
    frames = pd.read_parquet(data_dir / FRAMES)
    vision = pd.read_parquet(data_dir / VISION)
    table = add_spread_fail(fold_risk_table(frames, vision))
    test = table[table["split"] == TEST]
    sess = ort.InferenceSession(
        str(model_dir / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    proba = sess.run(
        [ONNX_PROBA], {ONNX_INPUT: test[PRIMARY].to_numpy(np.float32)}
    )[0]
    classes = _classes(model_dir)
    p_fail = proba[:, classes.index(FAIL)]
    paths = last_spread(frames)["frame_path"]
    return test.assign(
        p_fail=p_fail,
        predicted=np.where(p_fail >= DECISION, FAIL, PASS),
        frame_path=paths.reindex(test.index),
    ).dropna(subset=["frame_path"])


def _classes(model_dir: Path) -> list[str]:
    card = json.loads((model_dir / "model.json").read_text())
    return card["outputs"][1]["classes"]


def pick(test: pd.DataFrame) -> list[tuple[pd.Series, str]]:
    """Rule-based tiles, each run used once, in reading order."""
    fails = test[test["fold_fail"]].sort_values("p_fail", kind="stable")
    passes = test[~test["fold_fail"]].sort_values("p_fail", kind="stable")
    wanted = [
        (fails.iloc[::-1], "caught: highest risk, failed"),
        (fails, "missed: lowest risk, failed"),
        (passes.iloc[::-1], "false alarm: highest risk, passed"),
        (passes.iloc[len(passes) // 2 :], "typical pass (median risk)"),
        (fails.iloc[len(fails) // 2 :], "typical failure (median risk)"),
    ]
    used: set = set()
    tiles = []
    for rows, note in wanted:
        rows = rows[~rows.index.isin(used)]
        if len(rows):
            row = rows.iloc[0]
            used.add(row.name)
            tiles.append((row, note))
    return tiles


def example_figure(data_dir: Path, model_dir: Path, out: Path) -> Path:
    """Write the fold-risk example grid to `out`."""
    test = scored_holdout(data_dir, model_dir)
    if test.empty:
        raise ValueError("no scored holdout runs with a spread frame")
    tiles = []
    for row, note in pick(test):
        spread = "spread fail" if row["spread_fail"] else "spread ok"
        tiles.append(
            Example(
                image=load_frame(data_dir / row["frame_path"]),
                truth=f"{FAIL if row['fold_fail'] else PASS} (folds)",
                predicted=row["predicted"],
                confidence=float(
                    row["p_fail"]
                    if row["predicted"] == FAIL
                    else 1 - row["p_fail"]
                ),
                robot=str(row[ROBOT]),
                note=f"{note}\np(fail) {row['p_fail']:.2f}, {spread}",
            )
        )
    robots = ", ".join(sorted(test[ROBOT].astype(str).unique()))
    n_fail = int(test["fold_fail"].sum())
    return example_grid(
        tiles,
        out,
        title="fold-risk: will a fold fail, judged at the spread check",
        subtitle=(
            f"Research only. Holdout: {len(test)} runs on one robot "
            f"({robots}), {n_fail} with a fold failure.\n"
            "Robot fold outcome vs model answer (confidence). "
            "Green = agree, red = disagree. Last spread frame."
        ),
        crop=crop_plate,
        cols=COLS,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--out", type=Path, default=EXAMPLE_ASSET)
    args = p.parse_args()
    out = example_figure(args.data_dir or find_data_dir(), HERE, args.out)
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
