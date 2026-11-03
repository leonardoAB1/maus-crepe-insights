"""Vision pipeline: merged graph, per-frame results and the run CSV."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from maus_export import vision_pipeline as vp
from maus_export.experimental_fold.fold_dashboard import attach_models
from maus_export.experimental_fold.fold_runtime import main

FRAME_SHAPE = (509, 900, 3)
SHIPPED = vp.MODEL_DIR / "model.onnx"
SYNTHETIC_IR = 8  # readable by every onnxruntime we pin


class FakeInput:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeMerged:
    """Every check fails at 0.8, fold class is flat."""

    def get_inputs(self) -> list[FakeInput]:
        return [FakeInput("x")]

    def run(self, names: list[str], feeds: dict) -> list[np.ndarray]:
        assert feeds["x"].shape == (1, 3, 224, 224)
        out = []
        for name in names:
            if name == "fold_class":
                out.append(np.array([[0.1, 0.7, 0.1, 0.1]], np.float32))
            elif name.startswith(vp.VERDICT_PREFIX):
                out.append(np.array([[0.8, 0.2]], np.float32))
            else:
                out.append(np.array([[0.4, 0.6]], np.float32))
        return out


class FakeStudent:
    def get_inputs(self) -> list[FakeInput]:
        return [FakeInput("image")]

    def run(self, names: None, feeds: dict) -> list[np.ndarray]:
        assert feeds["image"].shape == (1, 3, 216, 384)
        mask = np.zeros((1, 1, 216, 384), np.uint8)
        mask[..., :108, :] = 1
        return [mask]


def _pipeline(tmp_path: Path, student: bool = True) -> vp.VisionPipeline:
    classes = vp.head_classes()
    card = {
        "outputs": [{"name": "embedding"}]
        + [{"name": n, "classes": c} for n, c in classes.items()]
    }
    (tmp_path / "model.json").write_text(json.dumps(card))
    return vp.VisionPipeline(
        tmp_path, sessions=(FakeMerged(), FakeStudent() if student else None)
    )


def _run(data_dir: Path, run_id: int = 7) -> None:
    run = {
        "visionChecks": [
            {"phase": "spread", "frame": "01_spread_frame.jpg", "ok": True},
            {"phase": "fold2", "frame": "02_fold2_frame.jpg", "ok": False},
            {"phase": "fold2", "frame": "03_fold2_frame.jpg", "ok": None},
            {"phase": "fold1", "frame": "04_fold1_frame.jpg", "ok": True},
        ]
    }
    runs = data_dir / "raw" / "runs"
    runs.mkdir(parents=True)
    (runs / f"{run_id}.json").write_text(json.dumps(run))
    media = data_dir / "raw" / "media" / str(run_id)
    media.mkdir(parents=True)
    for name in ("01_spread_frame.jpg", "02_fold2_frame.jpg"):
        cv2.imwrite(str(media / name), np.zeros(FRAME_SHAPE, np.uint8))
    cv2.imwrite(
        str(media / "03_fold2_frame.jpg"), np.zeros(FRAME_SHAPE, np.uint8)
    )


def test_heads_cover_every_shipped_card() -> None:
    classes = vp.head_classes()
    assert set(classes) == set(vp.HEADS)
    assert classes["fold_class"] == ["empty", "flat", "half", "quarter"]
    for name in classes:
        if name.startswith(vp.VERDICT_PREFIX):
            assert vp.FAIL_CLASS in classes[name]
        if name.startswith(vp.REASON_PREFIX):
            assert name.rsplit(".", 1)[1] in classes[name]


def test_frame_result_reads_every_model(tmp_path: Path) -> None:
    result = _pipeline(tmp_path).run(np.zeros(FRAME_SHAPE, np.uint8))
    assert result.p_fail("spread") == pytest.approx(0.8)
    assert result.p_fail("plate_present") is None
    assert result.fold_state() == ("flat", pytest.approx(0.7))
    assert result.reasons("fold2") == {
        "shape": pytest.approx(0.6),
        "residue": pytest.approx(0.6),
    }
    assert result.crepe_share == pytest.approx(0.5)


def test_without_student_there_is_no_mask(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path, student=False)
    result = pipeline.run(np.zeros(FRAME_SHAPE, np.uint8))
    assert result.crepe_share is None and result.mask is None


def test_classify_run_writes_csv_and_dashboard_reads_it(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    _run(data)
    rows = vp.classify_run(data, 7, _pipeline(tmp_path))
    assert [(r["check"], r["attempt"]) for r in rows] == [
        ("spread", 1),
        ("fold2", 1),
        ("fold2", 2),
    ]
    assert rows[0]["model_verdict"] == "fail" and rows[0]["agrees"] is False
    assert rows[1]["agrees"] is True
    assert rows[2]["agrees"] == ""
    root = data / "fold_segmentation" / "live" / "7"
    events = [{"name": "spread"}, {"name": "fold2"}, {"name": "fold2"}]
    events.append({"name": "fold1"})
    attach_models(events, root)
    assert events[0]["model"]["agrees"] is False
    assert events[1]["model"]["reasons"]["shape"] == pytest.approx(0.6)
    assert events[2]["model"]["agrees"] is None
    assert events[3]["model"] is None


def test_dashboard_without_results_sets_no_model(tmp_path: Path) -> None:
    events = [{"name": "spread"}]
    attach_models(events, tmp_path)
    assert events == [{"name": "spread", "model": None}]


def test_student_input_matches_card() -> None:
    card = json.loads(
        (vp.MODELS_DIR / "crepe-student" / "model.json").read_text()
    )
    pre = card["preprocess"]
    assert list(vp.STUDENT_SIZE) == pre["resize"]
    assert list(vp.IMAGENET_MEAN) == pre["mean"]
    assert list(vp.IMAGENET_STD) == pre["std"]
    frame = np.full(FRAME_SHAPE, 255, np.uint8)
    out = vp.student_input(frame)
    assert out.shape == (1, 3, 216, 384)
    want = (1 - np.asarray(pre["mean"])) / np.asarray(pre["std"])
    assert np.allclose(out[0, :, 0, 0], want, atol=1e-5)


def test_runtime_classifies_when_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    monkeypatch.setattr(
        "maus_export.experimental_fold.fold_video.analyse_video",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "maus_export.experimental_fold.fold_runtime.classify",
        lambda *args: calls.append(args),
    )
    main(["3", "--data-dir", str(tmp_path), "--analyze-only"])
    assert calls == []
    main(["3", "--data-dir", str(tmp_path), "--analyze-only", "--classify"])
    assert calls == [(tmp_path, 3, 1)]


def test_merge_feeds_every_head_from_the_embedder() -> None:
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper

    def linear(name: str, n_in: int, n_out: int, out: str) -> object:
        weight = helper.make_tensor(
            "w", TensorProto.FLOAT, [n_in, n_out], [0.5] * (n_in * n_out)
        )
        graph = helper.make_graph(
            [helper.make_node("MatMul", ["x", "w"], [out])],
            name,
            [
                helper.make_tensor_value_info(
                    "x", TensorProto.FLOAT, [None, n_in]
                )
            ],
            [
                helper.make_tensor_value_info(
                    out, TensorProto.FLOAT, [None, n_out]
                )
            ],
            [weight],
        )
        return helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 17)],
            ir_version=SYNTHETIC_IR,
        )

    embedder = linear("emb", 3, 4, "y")
    heads = {"a": linear("a", 4, 2, "probabilities")}
    merged = vp.merge(embedder, heads)
    onnx.checker.check_model(merged)
    sess = ort.InferenceSession(merged.SerializeToString())
    emb, a = sess.run(["embedding", "a"], {"x": np.ones((1, 3), np.float32)})
    assert np.allclose(emb, 1.5) and np.allclose(a, 3.0)


@pytest.mark.skipif(not SHIPPED.is_file(), reason="model not built")
def test_shipped_pipeline_runs() -> None:
    pytest.importorskip("onnxruntime")
    pipeline = vp.VisionPipeline()
    result = pipeline.run(np.zeros(FRAME_SHAPE, np.uint8))
    assert set(result.probabilities) == set(vp.HEADS)
    for probs in result.probabilities.values():
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-4)
