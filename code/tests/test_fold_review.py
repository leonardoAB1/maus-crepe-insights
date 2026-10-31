"""Local-only contact sheet output from synthetic frames."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from maus_export.fold_review import contact_sheet


def test_private_contact_sheet_and_refuse_overwrite(tmp_path: Path) -> None:
    root = tmp_path / "fold_segmentation"
    (root / "frames").mkdir(parents=True)
    cv2.imwrite(
        str(root / "frames" / "image.jpg"), np.zeros((30, 30, 3), np.uint8)
    )
    pd.DataFrame(
        [
            {
                "frame_id": "synthetic",
                "path": "frames/image.jpg",
                "expected_state": "half",
                "predicted_state": "quarter",
            }
        ]
    ).to_parquet(root / "features.parquet")
    assert contact_sheet(tmp_path) == 1
    assert (root / "review" / "contact-sheet.jpg").is_file()
    with pytest.raises(FileExistsError):
        contact_sheet(tmp_path)
