"""Train, score and export the fold state head (flat / half / quarter).

pixi run -e ml python models/fold-class/train.py [--data-dir ../data]
(same as `pixi run -e ml fold-class`)

Needs data/training/frames.parquet (`pixi run frames`) and
data/training/frame_embeddings.parquet (`pixi run -e ml embed`). Writes
model.onnx and model.json next to this file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from maus_classify.fold_class import train_fold_class
from maus_export.analysis import find_data_dir

HERE = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=None)
    args = p.parse_args()
    card = train_fold_class(args.data_dir or find_data_dir(), HERE)
    conf = card["metrics"]["test"]["confidence"]
    print(
        f"holdout accuracy {conf['accuracy']:.3f}, "
        f"mean confidence {conf['mean_confidence']:.3f}"
    )


if __name__ == "__main__":
    main()
