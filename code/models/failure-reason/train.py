"""Train, score and export the failure-reason heads, one folder per check.

pixi run -e ml python models/failure-reason/train.py [--data-dir ../data]

Needs data/training/frames.parquet and frame_embeddings.parquet. Writes
<check>/model.json (check card) and <check>/<class>/model.onnx + model.json
(one binary head per modelled reason class) next to this file, plus
summary.json and assets/failure-reason-example.jpg.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import pandas as pd

from maus_classify import failure_reason as fr
from maus_classify.check_verdict import embedding_columns
from maus_classify.harness import write_card
from maus_classify.split import TEST
from maus_export.analysis import find_data_dir

HERE = Path(__file__).resolve().parent
EXAMPLE = HERE.parents[2] / "assets" / "failure-reason-example.jpg"
EXAMPLE_CHECK = "spread"
EXAMPLE_TILES = 6
EXAMPLE_WRONG = 2


def summary(cards: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for check, card in cards.items():
        out[check] = {
            "failed_frames": {
                s: sum(card["robots"][s].values()) for s in card["robots"]
            },
            "modelled": card["modelled"],
            "not_modelled": card["not_modelled"],
            "abstain_threshold": card.get("abstain", {}).get("threshold"),
            "verdict": card["verdict"],
            "heads": {c: h["verdict"] for c, h in card["heads"].items()},
        }
    return out


def example_figure(
    data_dir: Path, frames: pd.DataFrame, embeddings: pd.DataFrame
) -> Path:
    """Holdout spread frames: the robot's reason against the heads' answer.
    At least one disagreement is always shown."""
    from maus_classify.figures import Example, example_grid
    from maus_export.frame_preprocess import crop_plate, load_frame

    card_dir = HERE / EXAMPLE_CHECK
    card = json.loads((card_dir / "model.json").read_text(encoding="utf-8"))
    classes = card["modelled"]
    threshold = card["abstain"]["threshold"]
    rows = fr.failed_rows(frames, embeddings, EXAMPLE_CHECK)
    rows = rows[(rows["split"] == TEST) & rows["frame_path"].notna()]
    features = embedding_columns(embeddings)
    probas = {}
    for c in classes:
        sess = ort.InferenceSession(
            str(card_dir / c / "model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        _, proba = sess.run(None, {"x": rows[features].to_numpy(np.float32)})
        head = json.loads(
            (card_dir / c / "model.json").read_text(encoding="utf-8")
        )
        col = head["outputs"][1]["classes"].index(c)
        probas[c] = proba[:, col]
    preds = fr.predict_sets(probas, threshold)
    truths = fr.truth_sets(rows["codes"], classes)
    tiles = []
    for i, (_, row) in enumerate(rows.iterrows()):
        top = preds[i][0]
        conf = (
            float(probas[top][i])
            if top in probas
            else float(1 - max(p[i] for p in probas.values()))
        )
        truth = truths[i][0]
        if truth == fr.OTHER:
            truth = f"{fr.OTHER} ({'+'.join(row['codes'])})"
        tiles.append(
            (
                truth.split()[0] == top,
                Example(
                    image=load_frame(data_dir / row["frame_path"]),
                    truth=truth,
                    predicted=top,
                    confidence=conf,
                    robot=str(row["robot"]),
                ),
            )
        )
    wrong = [t for ok, t in tiles if not ok]
    # Correct reason calls first, then correct abstentions
    right = sorted(
        (t for ok, t in tiles if ok), key=lambda t: t.predicted == fr.OTHER
    )
    n_wrong = max(1, min(len(wrong), EXAMPLE_WRONG))
    chosen = wrong[:n_wrong] + right[: EXAMPLE_TILES - n_wrong]
    # show a second robot when the picks all come from one
    seen = {t.robot for t in chosen}
    other = [t for _, t in tiles if t.robot not in seen]
    if len(seen) < 2 and other:
        chosen = chosen[: EXAMPLE_TILES - 1] + other[:1]
    return example_grid(
        chosen,
        EXAMPLE,
        title="Why did the spread check fail? Holdout frames",
        subtitle=(
            f"Head for 'too_small' at threshold {threshold:.2f}; "
            "'other' means abstain (shown with 1 - p). Green: agrees with "
            "the robot's reason. Red: disagrees."
        ),
        crop=crop_plate,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=None)
    args = p.parse_args()
    data_dir = args.data_dir or find_data_dir()
    frames = pd.read_parquet(data_dir / "training" / "frames.parquet")
    embeddings = pd.read_parquet(
        data_dir / "training" / "frame_embeddings.parquet"
    )
    cards = {}
    for check in fr.CHECKS:
        cards[check] = fr.train_check(frames, embeddings, check, HERE / check)
        print(f"{check}: {cards[check]['verdict']}")
    write_card(summary(cards), HERE / "summary.json")
    print(example_figure(data_dir, frames, embeddings))


if __name__ == "__main__":
    main()
