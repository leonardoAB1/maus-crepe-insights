"""Export a frozen MobileNetV3-Small frame embedder to ONNX (ml env).

Torch is only needed here, at export time. Inference and caching use
onnxruntime (see frame_embeddings.py) with the same `preprocess`.
"""

from __future__ import annotations

import platform
import statistics
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from .doe_analyze import ONNX_INPUT, ONNX_OPSET, ONNX_OUTPUT
from .frame_preprocess import INPUT_SIZE, preprocess, preprocess_config

EMBED_DIM = 576  # MobileNetV3-Small channels after global average pooling
BACKBONE = "torchvision.models.mobilenet_v3_small"
WEIGHTS = "MobileNet_V3_Small_Weights.IMAGENET1K_V1"
PARITY_ATOL = 1e-4
WARMUP_RUNS = 5
TIMED_RUNS = 50
P95 = 0.95
MS_PER_S = 1000.0
# Rough Pi 5 (Cortex-A76 @ 2.4 GHz) vs one modern x86 laptop core, single
# thread. An estimate for planning only, not a measurement.
PI5_SLOWDOWN = 4.0
LATENCY_TARGET_MS = 300.0


def build_embedder(pretrained: bool = True) -> Any:
    """features -> global avg pool -> flatten, frozen, in eval mode."""
    import torch
    from torchvision.models import (
        MobileNet_V3_Small_Weights,
        mobilenet_v3_small,
    )

    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    net = mobilenet_v3_small(weights=weights)
    model = torch.nn.Sequential(
        net.features, net.avgpool, torch.nn.Flatten(1)
    ).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def export_onnx(model: Any, path: Path) -> None:
    """Write `model` as ONNX with a dynamic batch axis."""
    import torch

    x = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        torch.onnx.export(
            model,
            (x,),
            str(path),
            input_names=[ONNX_INPUT],
            output_names=[ONNX_OUTPUT],
            dynamic_axes={ONNX_INPUT: {0: "batch"}, ONNX_OUTPUT: {0: "batch"}},
            opset_version=ONNX_OPSET,
            dynamo=False,
        )


def session(path: Path, threads: int | None = None) -> Any:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    if threads:
        opts.intra_op_num_threads = threads
    return ort.InferenceSession(
        str(path), opts, providers=["CPUExecutionProvider"]
    )


def parity(model: Any, path: Path, x: np.ndarray) -> float:
    """Max abs difference between torch and onnxruntime on `x`; raises
    if above PARITY_ATOL."""
    import torch

    with torch.no_grad():
        want = model(torch.from_numpy(x)).numpy()
    got = session(path).run(None, {ONNX_INPUT: x})[0]
    np.testing.assert_allclose(got, want, atol=PARITY_ATOL)
    return float(np.abs(got - want).max())


def _stats(times_s: list[float]) -> dict[str, float]:
    ms = sorted(t * MS_PER_S for t in times_s)
    return {
        "median_ms": round(statistics.median(ms), 2),
        "p95_ms": round(ms[int(P95 * (len(ms) - 1))], 2),
    }


def benchmark(
    path: Path,
    bgr: np.ndarray,
    threads: int | None = None,
    runs: int = TIMED_RUNS,
) -> dict[str, Any]:
    """CPU latency at batch 1: model alone, and preprocess + model."""
    sess = session(path, threads)
    x = preprocess(bgr)
    for _ in range(WARMUP_RUNS):
        sess.run(None, {ONNX_INPUT: x})
    model_t, full_t = [], []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {ONNX_INPUT: x})
        model_t.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        sess.run(None, {ONNX_INPUT: preprocess(bgr)})
        full_t.append(time.perf_counter() - t0)
    return {
        "threads": threads or "default",
        "runs": runs,
        "model": _stats(model_t),
        "preprocess_and_model": _stats(full_t),
    }


def latency_report(
    path: Path, bgr: np.ndarray, runs: int = TIMED_RUNS
) -> dict[str, Any]:
    single = benchmark(path, bgr, threads=1, runs=runs)
    pi5 = round(single["preprocess_and_model"]["median_ms"] * PI5_SLOWDOWN)
    return {
        "host": {
            "cpu": platform.processor() or platform.machine(),
            "system": platform.platform(),
        },
        "batch": 1,
        "single_thread": single,
        "default_threads": benchmark(path, bgr, runs=runs),
        "pi5_estimate_ms": pi5,
        "pi5_estimate_method": (
            f"single-thread preprocess+model median x {PI5_SLOWDOWN:g} "
            "(rough Cortex-A76 vs x86 factor, not measured)"
        ),
        "target_ms": LATENCY_TARGET_MS,
    }


def model_card(
    pretrained: bool,
    parity_diff: float,
    parity_sample: str,
    latency: dict[str, Any],
) -> dict[str, Any]:
    """model.json content for code/models/frame-embedder/."""
    return {
        "name": "frame-embedder",
        "description": (
            "Frozen ImageNet MobileNetV3-Small; one 576-d embedding per "
            "raw check frame for small classifier heads"
        ),
        "backbone": BACKBONE,
        "weights": WEIGHTS if pretrained else None,
        "trainable": False,
        "inputs": [
            {
                "name": ONNX_INPUT,
                "dtype": "float32",
                "shape": [None, 3, INPUT_SIZE, INPUT_SIZE],
            }
        ],
        "outputs": [
            {
                "name": ONNX_OUTPUT,
                "dtype": "float32",
                "shape": [None, EMBED_DIM],
            }
        ],
        "embedding_dim": EMBED_DIM,
        "opset": ONNX_OPSET,
        "preprocess": preprocess_config(),
        "training": None,
        "parity": {
            "sample": parity_sample,
            "max_abs_diff": parity_diff,
            "atol": PARITY_ATOL,
        },
        "latency": latency,
    }
