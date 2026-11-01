"""One vision pipeline for a raw check frame: every image model in one pass.

`code/models/vision-pipeline/model.onnx` merges the frame embedder with all
the small embedding heads (check verdict, fold class, failure reason), so a
single onnxruntime session turns one preprocessed frame into every
probability. The crepe-student mask runs next to it when its model is present.

Runtime needs numpy, cv2 and onnxruntime only (no pandas), so it also runs in
the `pi` environment. Building the merged graph needs `onnx` (ml env).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .frame_preprocess import load_frame, preprocess

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_DIR = MODELS_DIR / "vision-pipeline"
STUDENT_PATH = MODELS_DIR / "crepe-student" / "model.onnx"
EMBEDDER_DIR = MODELS_DIR / "frame-embedder"
EMBEDDING = "embedding"
EMBEDDER_OUTPUT = "y"
PROBABILITIES = "probabilities"
CPU_THREADS = 1

# merged output name -> head folder under code/models
HEADS = {
    "check_verdict.spread": "check-verdict/spread",
    "check_verdict.fold1": "check-verdict/fold1",
    "check_verdict.fold2": "check-verdict/fold2",
    "check_verdict.push_off": "check-verdict/push_off",
    "fold_class": "fold-class",
    "failure_reason.fold2.shape": "failure-reason/fold2/shape",
    "failure_reason.fold2.residue": "failure-reason/fold2/residue",
    "failure_reason.spread.too_small": "failure-reason/spread/too_small",
}
VERDICT_PREFIX = "check_verdict."
REASON_PREFIX = "failure_reason."
FAIL_CLASS = "fail"
FAIL_THRESHOLD = 0.5

# crepe-student input contract (models/crepe-student/model.json preprocess);
# kept here because maus_export.crepe_student imports pandas
STUDENT_SIZE = (384, 216)  # width, height
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PIXEL_MAX = 255.0

CSV_NAME = "pipeline.csv"
CSV_FIELDS = (
    "check",
    "attempt",
    "frame",
    "robot_ok",
    "p_fail",
    "model_verdict",
    "agrees",
    "fold_state",
    "fold_p",
    "reasons",
    "crepe_share",
)


def head_classes(models_dir: Path = MODELS_DIR) -> dict[str, list[str]]:
    """Class order of every head's probabilities, from its model card."""
    classes = {}
    for name, folder in HEADS.items():
        card = json.loads((models_dir / folder / "model.json").read_text())
        out = next(o for o in card["outputs"] if o["name"] == PROBABILITIES)
        classes[name] = list(out["classes"])
    return classes


def merge(embedder: Any, heads: dict[str, Any]) -> Any:
    """One ONNX graph: embedder output feeds every head.

    Outputs are the 576-d `embedding` plus one float probability tensor per
    head, named as in `HEADS`. The heads' string labels are dropped.
    """
    from onnx import TensorProto, compose, helper

    graph = embedder.graph
    nodes = list(graph.node)
    initializers = list(graph.initializer)
    opsets = {o.domain: o.version for o in embedder.opset_import}
    width = graph.output[0].type.tensor_type.shape.dim[1].dim_value
    nodes.append(helper.make_node("Identity", [EMBEDDER_OUTPUT], [EMBEDDING]))
    outputs = [
        helper.make_tensor_value_info(
            EMBEDDING, TensorProto.FLOAT, ["N", width]
        )
    ]
    ir_version = embedder.ir_version
    for name, head in heads.items():
        head = compose.add_prefix(head, f"{name}/")
        head_input = head.graph.input[0].name
        prob = next(
            o for o in head.graph.output if o.name.endswith(PROBABILITIES)
        )
        n_classes = prob.type.tensor_type.shape.dim[1].dim_value
        nodes.append(
            helper.make_node("Identity", [EMBEDDER_OUTPUT], [head_input])
        )
        nodes.extend(head.graph.node)
        initializers.extend(head.graph.initializer)
        nodes.append(helper.make_node("Identity", [prob.name], [name]))
        outputs.append(
            helper.make_tensor_value_info(
                name, TensorProto.FLOAT, ["N", n_classes]
            )
        )
        for opset in head.opset_import:
            opsets[opset.domain] = max(
                opsets.get(opset.domain, 0), opset.version
            )
        ir_version = max(ir_version, head.ir_version)
    merged = helper.make_graph(
        nodes, "vision_pipeline", list(graph.input), outputs, initializers
    )
    model = helper.make_model(
        merged,
        opset_imports=[
            helper.make_opsetid(domain, version)
            for domain, version in opsets.items()
        ],
        producer_name="maus_export.vision_pipeline",
    )
    model.ir_version = ir_version
    return model


def student_input(bgr: np.ndarray) -> np.ndarray:
    """float32 NCHW tensor for the crepe-student model."""
    small = cv2.resize(bgr, STUDENT_SIZE, interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb /= PIXEL_MAX
    rgb -= np.asarray(IMAGENET_MEAN, dtype=np.float32)
    rgb /= np.asarray(IMAGENET_STD, dtype=np.float32)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])


@dataclass
class FrameResult:
    """Every model's answer for one raw check frame."""

    probabilities: dict[str, dict[str, float]]
    crepe_share: float | None = None
    mask: np.ndarray | None = field(default=None, repr=False)

    def p_fail(self, check: str) -> float | None:
        """Model fail probability for a check, None without a head."""
        probs = self.probabilities.get(VERDICT_PREFIX + check)
        return None if probs is None else probs[FAIL_CLASS]

    def fold_state(self) -> tuple[str, float]:
        """Most likely fold state and its probability."""
        probs = self.probabilities["fold_class"]
        state = max(probs, key=probs.__getitem__)
        return state, probs[state]

    def reasons(self, check: str) -> dict[str, float]:
        """Failure-reason probabilities that apply to a check."""
        prefix = f"{REASON_PREFIX}{check}."
        out = {}
        for name, probs in self.probabilities.items():
            if name.startswith(prefix):
                reason = name.removeprefix(prefix)
                out[reason] = probs[reason]
        return out


class VisionPipeline:
    """Loads the merged graph (and the student) once, runs them per frame."""

    def __init__(
        self,
        model_dir: Path = MODEL_DIR,
        student_path: Path | None = STUDENT_PATH,
        threads: int = CPU_THREADS,
        sessions: tuple[Any, Any] | None = None,
    ) -> None:
        card = json.loads((model_dir / "model.json").read_text())
        self.classes = {
            o["name"]: o["classes"] for o in card["outputs"] if "classes" in o
        }
        if sessions is None:
            sessions = (
                _session(model_dir / "model.onnx", threads),
                _session(student_path, threads)
                if student_path is not None and student_path.is_file()
                else None,
            )
        self.session, self.student = sessions
        self.input_name = self.session.get_inputs()[0].name

    def run(self, bgr: np.ndarray) -> FrameResult:
        names = list(self.classes)
        values = self.session.run(names, {self.input_name: preprocess(bgr)})
        probabilities = {
            name: {
                cls: float(p)
                for cls, p in zip(self.classes[name], value[0], strict=True)
            }
            for name, value in zip(names, values, strict=True)
        }
        if self.student is None:
            return FrameResult(probabilities)
        student_name = self.student.get_inputs()[0].name
        (mask,) = self.student.run(None, {student_name: student_input(bgr)})
        crepe = mask[0, 0].astype(bool)
        return FrameResult(probabilities, float(crepe.mean()), crepe)


def _session(path: Path, threads: int) -> Any:
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise ImportError(
            "onnxruntime is missing: use `pixi run -e ml` or `-e pi`"
        ) from error
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    return ort.InferenceSession(
        str(path), options, providers=["CPUExecutionProvider"]
    )


def check_frames(data_dir: Path, run_id: int) -> list[dict[str, Any]]:
    """Robot vision checks of a run with their raw still, in run order."""
    run = json.loads((data_dir / "raw" / "runs" / f"{run_id}.json").read_text())
    media = data_dir / "raw" / "media" / str(run_id)
    attempts: dict[str, int] = {}
    rows = []
    for check in run.get("visionChecks") or []:
        name = check.get("phase")
        frame = check.get("frame")
        if not name or not frame:
            continue
        attempts[name] = attempts.get(name, 0) + 1
        rows.append(
            {
                "check": name,
                "attempt": attempts[name],
                "frame": media / frame,
                "robot_ok": check.get("ok"),
            }
        )
    return rows


def classify_run(
    data_dir: Path, run_id: int, pipeline: VisionPipeline
) -> list[dict[str, Any]]:
    """Run the pipeline on every check still of a run.

    Writes `fold_segmentation/live/<id>/pipeline.csv` and returns its rows.
    Checks without a cached still are skipped.
    """
    rows = []
    for check in check_frames(data_dir, run_id):
        if not check["frame"].is_file():
            continue
        result = pipeline.run(load_frame(check["frame"]))
        p_fail = result.p_fail(check["check"])
        verdict = (
            None
            if p_fail is None
            else "fail"
            if p_fail >= FAIL_THRESHOLD
            else "pass"
        )
        robot_ok = check["robot_ok"]
        state, state_p = result.fold_state()
        rows.append(
            {
                "check": check["check"],
                "attempt": check["attempt"],
                "frame": check["frame"].name,
                "robot_ok": "" if robot_ok is None else robot_ok,
                "p_fail": "" if p_fail is None else round(p_fail, 4),
                "model_verdict": verdict or "",
                "agrees": ""
                if verdict is None or robot_ok is None
                else (verdict == "pass") == bool(robot_ok),
                "fold_state": state,
                "fold_p": round(state_p, 4),
                "reasons": json.dumps(
                    {
                        k: round(v, 4)
                        for k, v in result.reasons(check["check"]).items()
                    }
                ),
                "crepe_share": ""
                if result.crepe_share is None
                else round(result.crepe_share, 4),
            }
        )
    out = data_dir / "fold_segmentation" / "live" / str(run_id)
    out.mkdir(parents=True, exist_ok=True)
    with (out / CSV_NAME).open("w", newline="") as file:
        writer = csv.DictWriter(file, CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def read_results(root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """`pipeline.csv` rows keyed by (check, attempt), empty when absent."""
    path = root / CSV_NAME
    if not path.is_file():
        return {}
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    out = {}
    for row in rows:
        out[(row["check"], int(row["attempt"]))] = {
            "p_fail": float(row["p_fail"]) if row["p_fail"] else None,
            "verdict": row["model_verdict"] or None,
            "agrees": None if not row["agrees"] else row["agrees"] == "True",
            "fold_state": row["fold_state"],
            "fold_p": float(row["fold_p"]),
            "reasons": json.loads(row["reasons"] or "{}"),
            "crepe_share": float(row["crepe_share"])
            if row["crepe_share"]
            else None,
        }
    return out
