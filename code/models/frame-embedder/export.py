"""Export the frozen frame embedder to ONNX, check parity, time it.

pixi run -e ml python models/frame-embedder/export.py [--data-dir ../data]

Writes model.onnx and model.json next to this file. Nothing is trained:
the ImageNet MobileNetV3-Small backbone is frozen. Parity and latency use
a real raw frame when local media exists, else a seeded random image.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np

from maus_export.analysis import find_data_dir
from maus_export.doe_analyze import dump_json
from maus_export.frame_embedder import (
    LATENCY_TARGET_MS,
    build_embedder,
    export_onnx,
    latency_report,
    model_card,
    parity,
)
from maus_export.frame_embeddings import MEDIA_DIR
from maus_export.frame_preprocess import load_frame, preprocess

HERE = Path(__file__).resolve().parent
RAW_SHAPE = (509, 900, 3)  # raw camera frame, H x W x BGR
SEED = 0
PIXEL_LEVELS = 256


def sample_frame(data_dir: Path | None) -> tuple[np.ndarray, str]:
    if data_dir is not None:
        frames = sorted((data_dir / MEDIA_DIR).glob("*/*_frame.jpg"))
        if frames:
            return load_frame(frames[0]), "first local raw frame"
    rng = np.random.default_rng(SEED)
    img = rng.integers(0, PIXEL_LEVELS, RAW_SHAPE, dtype=np.uint8)
    return img, f"random image, seed {SEED}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument(
        "--no-weights",
        action="store_true",
        help="random init instead of ImageNet weights (offline smoke test)",
    )
    p.add_argument("--out", type=Path, default=HERE)
    args = p.parse_args()
    try:
        data_dir = args.data_dir or find_data_dir()
    except FileNotFoundError:
        data_dir = None

    pretrained = not args.no_weights
    model = build_embedder(pretrained)
    onnx_path = args.out / "model.onnx"
    export_onnx(model, onnx_path)

    bgr, sample = sample_frame(data_dir)
    diff = parity(model, onnx_path, preprocess(bgr))
    latency = latency_report(onnx_path, bgr)

    card = model_card(pretrained, diff, sample, latency)
    card["commit"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    dump_json(card, args.out / "model.json")

    single = latency["single_thread"]["preprocess_and_model"]["median_ms"]
    print(f"parity ok (max abs diff {diff:.2e}, {sample})")
    print(
        f"latency 1 thread {single} ms; Pi 5 estimate "
        f"{latency['pi5_estimate_ms']} ms (target < {LATENCY_TARGET_MS:g})"
    )


if __name__ == "__main__":
    main()
