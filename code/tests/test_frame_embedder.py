from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("onnxruntime")

from maus_export.frame_embedder import (  # noqa: E402
    EMBED_DIM,
    build_embedder,
    export_onnx,
    latency_report,
    model_card,
    parity,
    session,
)
from maus_export.frame_preprocess import preprocess  # noqa: E402

RAW_SHAPE = (509, 900, 3)
BATCH = 2
FEW_RUNS = 3


@pytest.fixture(scope="module")
def exported(tmp_path_factory) -> tuple[object, Path]:
    torch.manual_seed(0)
    model = build_embedder(pretrained=False)  # no weight download offline
    path = tmp_path_factory.mktemp("embedder") / "model.onnx"
    export_onnx(model, path)
    return model, path


def frame(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, RAW_SHAPE, dtype=np.uint8)


def test_embedder_is_frozen(exported) -> None:
    model, _ = exported
    assert not model.training
    assert not any(p.requires_grad for p in model.parameters())


def test_onnx_matches_torch(exported) -> None:
    model, path = exported
    assert parity(model, path, preprocess(frame(0))) < 1e-4


def test_dynamic_batch(exported) -> None:
    _, path = exported
    x = np.concatenate([preprocess(frame(i)) for i in range(BATCH)])
    y = session(path).run(None, {"x": x})[0]
    assert y.shape == (BATCH, EMBED_DIM)
    assert y.dtype == np.float32


def test_model_card_has_latency(exported) -> None:
    _, path = exported
    latency = latency_report(path, frame(1), runs=FEW_RUNS)
    card = model_card(False, 0.0, "random", latency)
    assert card["outputs"][0]["shape"] == [None, EMBED_DIM]
    assert card["latency"]["pi5_estimate_ms"] > 0
    assert card["preprocess"]["crop_xywh"]
