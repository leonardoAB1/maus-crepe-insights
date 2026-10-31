"""Layer producers: what the viewer can ask for, per frame.

A producer takes one frame and its context and returns the fields and masks
that belong to it. The viewer never knows which producer it is talking to, so a
new model or a new visual recognition pass is added here and nowhere else.

Producers that need an extra environment or a model that does not exist yet must
fail with a readable message rather than silently returning nothing.
"""

from __future__ import annotations

import base64
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from .crepe_states import UnknownState, normalise
from .pixel_masks import DEFAULT_CONFIG, PixelMaskConfig, suggest_masks

MASK_ENCODING = ".png"
MATERIAL_LAYER = "material"
HOLE_LAYER = "holes"


class ProducerError(RuntimeError):
    """A producer cannot answer for this frame; the reason is shown."""


@dataclass(frozen=True)
class FrameContext:
    """What is known about the frame before any producer runs."""

    frame_index: int
    run_id: int | None = None
    record: Mapping[str, Any] | None = None


@dataclass
class LayerSet:
    """Everything the viewer draws for one frame."""

    state: str | None = None
    state_source: str | None = None
    prediction_state: str | None = None
    deviation: float | None = None
    check: str | None = None
    verdict: str | None = None
    phase_context: str | None = None
    notes: list[str] = field(default_factory=list)
    material: np.ndarray | None = None
    holes: np.ndarray | None = None
    producer: str = ""

    def _normalise_state(self, name: str) -> None:
        value = getattr(self, name)
        if value is None:
            return
        try:
            setattr(self, name, normalise(str(value)))
        except UnknownState as error:
            raise ProducerError(str(error)) from error

    def to_json(self) -> dict[str, Any]:
        """The layer set as the browser sees it, masks as data URLs."""
        self._normalise_state("state")
        self._normalise_state("prediction_state")
        payload: dict[str, Any] = {
            name: getattr(self, name)
            for name in (
                "state",
                "state_source",
                "prediction_state",
                "deviation",
                "check",
                "verdict",
                "phase_context",
                "notes",
                "producer",
            )
        }
        payload["masks"] = {}
        areas: dict[str, int] = {}
        shape = None
        for name, mask in (
            (MATERIAL_LAYER, self.material),
            (HOLE_LAYER, self.holes),
        ):
            if mask is not None:
                payload["masks"][name] = encode_mask(mask)
                areas[f"{name}_pixels"] = int(mask.sum())
                shape = mask.shape
        if shape is not None:
            areas["frame_pixels"] = int(shape[0] * shape[1])
        payload["areas"] = areas
        return payload


def encode_mask(mask: np.ndarray) -> str:
    """A boolean mask as a base64 PNG data URL."""
    ok, buffer = cv2.imencode(MASK_ENCODING, mask.astype(np.uint8) * 255)
    if not ok:
        raise ProducerError("cannot encode a mask")
    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class Producer(Protocol):
    """One source of layers for a frame."""

    name: str
    describes: str

    def layers(self, frame: np.ndarray, context: FrameContext) -> LayerSet:
        """Return the layers for this frame, or raise ProducerError."""


@dataclass
class RecordsProducer:
    """Layers already known from a records sidecar, plus its mask files."""

    records: Mapping[int, Mapping[str, Any]]
    masks_dir: Path | None = None
    name: str = "records"
    describes: str = "fields and masks from a records sidecar"

    def layers(self, frame: np.ndarray, context: FrameContext) -> LayerSet:
        record = self.records.get(context.frame_index)
        if record is None:
            raise ProducerError("no record for this frame")
        layers = LayerSet(
            state=record.get("state"),
            state_source=record.get("state_source"),
            prediction_state=record.get("prediction_state"),
            deviation=record.get("deviation"),
            check=record.get("check"),
            verdict=record.get("verdict"),
            phase_context=record.get("phase_context"),
            notes=list(record.get("notes") or []),
            producer=self.name,
        )
        layers.material = self._load(record.get("material_mask"), frame.shape)
        layers.holes = self._load(record.get("hole_mask"), frame.shape)
        return layers

    def _load(
        self, name: str | None, shape: tuple[int, ...]
    ) -> np.ndarray | None:
        if not name:
            return None
        if self.masks_dir is None:
            raise ProducerError("a mask path needs the mask directory")
        candidate = (self.masks_dir / name).resolve()
        if not candidate.is_relative_to(self.masks_dir.resolve()):
            raise ProducerError("mask path escapes the mask directory")
        mask = cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ProducerError(f"cannot read mask {name}")
        if mask.shape != shape[:2]:
            raise ProducerError("mask size differs from the frame")
        return mask > 0


@dataclass
class PixelMaskProducer:
    """Material and hole masks from the merged colour suggestion pass.

    It names no fold state on purpose: colour and connectivity cannot tell a
    flat crepe from a cooking one, so the panel says where a state would come
    from instead of inventing one.
    """

    config: PixelMaskConfig = DEFAULT_CONFIG
    roi: tuple[int, int, int, int] | None = None
    records: Mapping[int, Mapping[str, Any]] = field(default_factory=dict)
    name: str = "pixel_masks"
    describes: str = "colour and connectivity suggestions, no fold state"

    def layers(self, frame: np.ndarray, context: FrameContext) -> LayerSet:
        material, holes = suggest_masks(frame, roi=self.roi, config=self.config)
        record = self.records.get(context.frame_index, {})
        return LayerSet(
            state=record.get("state"),
            state_source=record.get("state_source"),
            prediction_state=record.get("prediction_state"),
            deviation=record.get("deviation"),
            check=record.get("check"),
            verdict=record.get("verdict"),
            phase_context=record.get("phase_context"),
            notes=[
                "material and holes are colour suggestions, review them",
                "no fold state yet: needs the segmentation or a fold model",
            ],
            material=material,
            holes=holes,
            producer=self.name,
        )


@dataclass
class CheckVerdictProducer:
    """Pass or fail from the ONNX check heads, when that environment is here.

    The heads need onnxruntime, so this producer stays unavailable in the
    default environment and says so instead of pretending to be missing data.
    """

    models_dir: Path | None = None
    frame_preprocess: Callable[[np.ndarray], np.ndarray] | None = None
    check: str = "spread"
    name: str = "check_verdict"
    describes: str = "ONNX check verdict, needs the ml environment"

    def layers(self, frame: np.ndarray, context: FrameContext) -> LayerSet:
        try:
            import onnxruntime  # noqa: F401
        except ImportError as error:
            raise ProducerError(
                "check verdict needs onnxruntime: run in the ml environment"
            ) from error
        raise ProducerError(
            "the check verdict producer is a placeholder: wire the model path "
            "and the preprocessing contract before using it"
        )


def default_producers(
    *,
    records: Mapping[int, Mapping[str, Any]] | None = None,
    masks_dir: Path | None = None,
) -> dict[str, Any]:
    """The producers the viewer offers, in display order."""
    registry: dict[str, Any] = {}
    if records:
        registry["records"] = RecordsProducer(
            records=records, masks_dir=masks_dir
        )
    registry["pixel_masks"] = PixelMaskProducer(records=records or {})
    registry["check_verdict"] = CheckVerdictProducer()
    return registry


def describe(registry: Mapping[str, Any]) -> list[dict[str, str]]:
    """Name and description of every producer, for the interface."""
    described = []
    for key, producer in registry.items():
        described.append(
            {
                "name": key,
                "describes": getattr(producer, "describes", ""),
            }
        )
    return described


def layer_fields() -> list[str]:
    """The scalar fields of a layer set, for documentation and tests."""
    return [item.name for item in fields(LayerSet)]


def as_dict(layers: LayerSet) -> dict[str, Any]:
    """The scalar part of a layer set, without the masks."""
    payload = asdict(layers)
    payload.pop("material", None)
    payload.pop("holes", None)
    return payload
