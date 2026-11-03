"""Create a private contact sheet for segmentation review."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

SHEET_COLUMNS = 4
FRAMES_PER_STATE = 8
THUMB_WIDTH = 240
THUMB_HEIGHT = 240
LABEL_HEIGHT = 36
STATES = ("empty", "flat", "half", "quarter")


def contact_sheet(data_dir: Path) -> int:
    """Write a deterministic, state-balanced sheet of local check frames."""
    root = data_dir / "fold_segmentation"
    features = pd.read_parquet(root / "features.parquet")
    output = root / "review" / "contact-sheet.jpg"
    if output.exists():
        raise FileExistsError("Contact sheet exists; move it aside first")
    selected = []
    for state in STATES:
        group = features[features["expected_state"] == state].copy()
        group["mismatch"] = group["predicted_state"] != group["expected_state"]
        group = group.sort_values(
            ["mismatch", "frame_id"], ascending=[False, True]
        )
        selected.extend(group.head(FRAMES_PER_STATE).to_dict("records"))
    if not selected:
        raise ValueError("No frames to review")
    rows = (len(selected) + SHEET_COLUMNS - 1) // SHEET_COLUMNS
    sheet = np.zeros(
        (rows * (THUMB_HEIGHT + LABEL_HEIGHT), SHEET_COLUMNS * THUMB_WIDTH, 3),
        dtype=np.uint8,
    )
    for index, item in enumerate(selected):
        frame_path = (root / item["path"]).resolve()
        if not frame_path.is_relative_to((root / "frames").resolve()):
            raise ValueError("Frame path escapes frames directory")
        image = cv2.imread(str(frame_path))
        if image is None:
            raise ValueError("Cannot read frame for review")
        x = (index % SHEET_COLUMNS) * THUMB_WIDTH
        y = (index // SHEET_COLUMNS) * (THUMB_HEIGHT + LABEL_HEIGHT)
        sheet[y : y + THUMB_HEIGHT, x : x + THUMB_WIDTH] = cv2.resize(
            image, (THUMB_WIDTH, THUMB_HEIGHT), interpolation=cv2.INTER_AREA
        )
        label = (
            f"{item['expected_state']} / {item.get('predicted_state') or '?'}"
        )
        cv2.putText(
            sheet,
            label,
            (x + 5, y + THUMB_HEIGHT + 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet):
        raise OSError("Cannot write contact sheet")
    return len(selected)
