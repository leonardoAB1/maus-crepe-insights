"""Merge the frame embedder and every embedding head into one ONNX graph.

Run with `pixi run -e ml vision-pipeline`. Writes model.onnx and model.json
next to this file and checks the merged outputs against the separate models.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

from maus_export.vision_pipeline import (
    EMBEDDER_DIR,
    EMBEDDING,
    HEADS,
    MODELS_DIR,
    PROBABILITIES,
    head_classes,
    merge,
)

HERE = Path(__file__).resolve().parent
OPSET = 17
PARITY_ATOL = 1e-4
PARITY_SAMPLES = 4
SEED = 0
INPUT_SHAPE = (3, 224, 224)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def main() -> None:
    embedder_path = EMBEDDER_DIR / "model.onnx"
    embedder = onnx.load(embedder_path)
    heads = {
        name: onnx.load(MODELS_DIR / folder / "model.onnx")
        for name, folder in HEADS.items()
    }
    model = merge(embedder, heads)
    onnx.checker.check_model(model)
    out = HERE / "model.onnx"
    onnx.save(model, out)

    rng = np.random.default_rng(SEED)
    x = rng.standard_normal((PARITY_SAMPLES, *INPUT_SHAPE)).astype(np.float32)
    merged = ort.InferenceSession(str(out))
    got = dict(
        zip(
            [o.name for o in merged.get_outputs()],
            merged.run(None, {merged.get_inputs()[0].name: x}),
            strict=True,
        )
    )
    emb = ort.InferenceSession(str(embedder_path)).run(None, {"x": x})[0]
    worst = float(np.abs(got[EMBEDDING] - emb).max())
    for name, folder in HEADS.items():
        sess = ort.InferenceSession(str(MODELS_DIR / folder / "model.onnx"))
        (want,) = sess.run([PROBABILITIES], {sess.get_inputs()[0].name: emb})
        worst = max(worst, float(np.abs(got[name] - want).max()))
    if worst > PARITY_ATOL:
        raise SystemExit(f"parity failed: max abs diff {worst}")

    classes = head_classes()
    sources = {"frame-embedder": "frame-embedder"} | {
        name: folder for name, folder in HEADS.items()
    }
    card = {
        "name": "vision-pipeline",
        "description": (
            "Frame embedder and every embedding head merged into one graph: "
            "one raw check frame in (frame_preprocess.preprocess), check "
            "verdicts, fold class and failure reasons out. No retraining; "
            "metrics and caveats are those of each source model card."
        ),
        "inputs": [
            {"name": "x", "dtype": "float32", "shape": [None, *INPUT_SHAPE]}
        ],
        "outputs": [
            {"name": EMBEDDING, "dtype": "float32", "shape": [None, 576]}
        ]
        + [
            {
                "name": name,
                "dtype": "float32",
                "shape": [None, len(classes[name])],
                "classes": classes[name],
                "source": f"code/models/{HEADS[name]}",
            }
            for name in HEADS
        ],
        "preprocess": "maus_export.frame_preprocess.preprocess",
        "sources": {
            name: {
                "path": f"code/models/{folder}/model.onnx",
                "sha256": _sha(MODELS_DIR / folder / "model.onnx"),
            }
            for name, folder in sources.items()
        },
        "companion": {
            "crepe-student": "runs beside it in maus_export.vision_pipeline "
            "(different input size), gives the crepe mask and area share"
        },
        "onnx": {
            "opset": OPSET,
            "parity_atol": PARITY_ATOL,
            "parity_max_abs_diff": worst,
            "parity": "ok",
        },
        "caveats": [
            "check-verdict and failure-reason heads learned the robot's own "
            "verdicts (weak labels) and do not all generalize across robots; "
            "read each source card",
            "trained on still check frames; video frames are not validated",
        ],
        "commit": _commit(),
    }
    (HERE / "model.json").write_text(json.dumps(card, indent=2) + "\n")
    print(f"wrote {out} ({out.stat().st_size} bytes), parity {worst:.2e}")


if __name__ == "__main__":
    main()
