"""Example grid for the crêpe student: correct states and its failures.

Needs ``pixi run -e ml fold-student-eval`` first. Draws held-out check
frames with the student mask, cropped to the plate, and writes
``assets/crepe-student-example.jpg``:

    pixi run -e ml python models/crepe-student/figure.py \\
        --data-dir /abs/path/to/data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from maus_classify.figures import Example, example_grid
from maus_export.crepe_student import (
    CrepeStudent,
    area_share,
    is_test_run,
    overlay,
    read_frame,
)
from maus_export.fold_frames import video_path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "assets" / "crepe-student-example.jpg"
STATES = ("empty", "flat", "half", "quarter")
# Plate box in the raw frame, as shares of width and height. Keeps the
# background (and anyone standing there) out of the picture.
CROP_X = (0.28, 0.74)
CROP_Y = (0.16, 0.96)
WRONG_TILES = 2


def crop(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    x0, x1 = (round(width * share) for share in CROP_X)
    y0, y1 = (round(height * share) for share in CROP_Y)
    return image[y0:y1, x0:x1]


def pick(features: pd.DataFrame) -> pd.DataFrame:
    """One right frame per state from test runs, then the wrong ones."""
    passed = features[features["check_ok"].astype(bool)]
    test = passed[passed["run_id"].map(lambda run: is_test_run(int(run)))]
    right = test[test["predicted_state"] == test["expected_state"]]
    chosen = [right[right["expected_state"] == s].head(1) for s in STATES]
    wrong = passed[passed["predicted_state"] != passed["expected_state"]]
    chosen.append(wrong.head(WRONG_TILES))
    return pd.concat(chosen)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    features = pd.read_parquet(
        args.data_dir / "fold_segmentation" / "student" / "features.parquet"
    )
    student = CrepeStudent.load()
    examples = []
    for row in pick(features).itertuples(index=False):
        video = video_path(args.data_dir, int(row.run_id))
        frame = read_frame(video, int(row.decoded_frame_index))
        mask = student.mask(frame)
        split = "test run" if is_test_run(int(row.run_id)) else "train run"
        examples.append(
            Example(
                image=overlay(frame, mask, ""),
                truth=row.expected_state,
                predicted=row.predicted_state,
                confidence=area_share(mask),
                robot=row.machine_name,
                note=f"{row.check_name} check, {split}",
                score_name="area",
            )
        )
    path = example_grid(
        examples,
        args.out,
        "Crêpe student: mask and fold state at the robot checks",
        "green mask = student; state from the area relative to the run's "
        "flat crêpe; red captions are disagreements",
        crop=crop,
    )
    print(path)


if __name__ == "__main__":
    main()
