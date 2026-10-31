"""Offline test: the example figure renders from synthetic overlays."""

import json
from pathlib import Path

import cv2
import pytest
from test_vision_metrics import disc_overlay, fold_overlay, spread_overlay

from maus_export.figures import vision_example

SPREAD_RUN = 1
FOLD_RUN = 2


def write_run(data_dir: Path, run_id: int, checks: dict) -> None:
    media = data_dir / "raw" / "media" / str(run_id)
    media.mkdir(parents=True, exist_ok=True)
    vcs = []
    for check, img in checks.items():
        name = f"{check}_overlay.jpg"
        cv2.imwrite(str(media / name), img)
        vcs.append(
            {"phase": check, "ok": False, "errors": ["x"], "overlay": name}
        )
    runs = data_dir / "raw" / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    run = {"id": run_id, "visionChecks": vcs}
    (runs / f"{run_id}.json").write_text(json.dumps(run), encoding="utf-8")


def test_vision_example_writes_image(tmp_path: Path) -> None:
    write_run(tmp_path, SPREAD_RUN, {"spread": spread_overlay(250, hole_r=60)})
    write_run(
        tmp_path,
        FOLD_RUN,
        {"fold1": disc_overlay(180), "fold2": fold_overlay(0)},
    )
    out = vision_example(
        tmp_path, tmp_path / "out" / "example.png", SPREAD_RUN, FOLD_RUN
    )
    img = cv2.imread(str(out))
    assert img is not None and img.shape[0] > 0


def test_missing_overlay_is_reported(tmp_path: Path) -> None:
    write_run(tmp_path, SPREAD_RUN, {"fold1": disc_overlay(180)})
    with pytest.raises(ValueError, match="no spread overlay"):
        vision_example(tmp_path, tmp_path / "x.png", SPREAD_RUN, SPREAD_RUN)
