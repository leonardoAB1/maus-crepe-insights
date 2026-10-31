"""Fit the DSD response model from a filled run sheet and export ONNX.

pixi run -e ml python models/doe-response/train.py <sheet.parquet>
    [--data-dir ../data]

Writes model.onnx and model.json next to this file, and checks onnxruntime
against the numpy prediction on the design points.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

from maus_export.analysis import find_data_dir
from maus_export.doe import CODED_SUFFIX
from maus_export.doe_analyze import (
    ONNX_INPUT,
    analyze_sheet,
    dump_json,
    model_card,
    predict,
    to_onnx,
)

HERE = Path(__file__).resolve().parent
ATOL = 1e-4  # float32 graph vs float64 numpy


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sheet", type=Path, help="filled run sheet (.parquet)")
    p.add_argument("--data-dir", type=Path, default=None)
    args = p.parse_args()
    data_dir = args.data_dir or find_data_dir()

    sheet = pd.read_parquet(args.sheet)
    runs = pd.read_parquet(data_dir / "runs.parquet")
    vision_path = data_dir / "vision.parquet"
    vision = (
        pd.read_parquet(vision_path)
        if vision_path.is_file()
        else pd.DataFrame()
    )
    result = analyze_sheet(sheet, runs, vision)
    factors = result["factors"]
    outcomes = list(result["effects"]["outcome"].unique())

    model = to_onnx(result["effects"], factors, outcomes)
    (HERE / "model.onnx").write_bytes(model.SerializeToString())

    coded = sheet[[f.name + CODED_SUFFIX for f in factors]].to_numpy(float)
    want = predict(result["effects"], factors, outcomes, coded)
    sess = ort.InferenceSession(str(HERE / "model.onnx"))
    got = sess.run(None, {ONNX_INPUT: coded.astype(np.float32)})[0]
    np.testing.assert_allclose(got, want, atol=ATOL)

    card = model_card(result, outcomes)
    card["commit"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    dump_json(card, HERE / "model.json")
    print(f"model.onnx for {len(outcomes)} outcomes, onnxruntime check ok")


if __name__ == "__main__":
    main()
