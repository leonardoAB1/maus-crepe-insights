"""Train the per-check pass/fail heads on frame embeddings and export ONNX.

pixi run -e ml python models/check-verdict/train.py [--data-dir ../data]

Reads training/frames.parquet, training/frame_embeddings.parquet and
vision.parquet. Writes <check>/model.onnx and <check>/model.json next to this
file for spread, fold1, fold2 and push_off, and prints a holdout summary.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from maus_classify.check_verdict import CHECKS, summary_table, train_check
from maus_export.analysis import find_data_dir

HERE = Path(__file__).resolve().parent
FRAMES = Path("training") / "frames.parquet"
EMBEDDINGS = Path("training") / "frame_embeddings.parquet"
VISION = Path("vision.parquet")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=None)
    args = p.parse_args()
    data_dir = args.data_dir or find_data_dir()

    frames = pd.read_parquet(data_dir / FRAMES)
    embeddings = pd.read_parquet(data_dir / EMBEDDINGS)
    vision_path = data_dir / VISION
    vision = (
        pd.read_parquet(vision_path)
        if vision_path.is_file()
        else pd.DataFrame(columns=["id"])
    )
    cards = {
        check: train_check(frames, embeddings, vision, check, HERE / check)
        for check in CHECKS
    }
    print(summary_table(cards))
    for check, card in cards.items():
        print(f"{check}: {card['verdict']}")


if __name__ == "__main__":
    main()
