"""Draw every layer the pipeline knows onto a video, or onto a live stream.

One frame in, one frame out: the same image with the crepe state, the human and
model masks, the robot's check context and the deviation score drawn on top.
The renderer knows nothing about where a layer came from, so a new source only
has to emit records; nothing here is rewritten when a model, a label export or
the segmentation work lands.

Records are one JSON object per frame, next to the output video:

    {"version": 1, "run_id": 3350, "decoded_frame_index": 145,
     "requested_seconds": 25.0, "state": "flat", "state_source": "human",
     "material_mask": "run-000/000005-material.png", "hole_mask": null,
     "prediction_state": "flat", "prediction_mask": null, "deviation": 0.14,
     "check": "spread", "verdict": "pass", "phase_context": "init -> oil"}

Frames, masks and the rendered video stay in the private data directory.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .crepe_states import UnknownState, normalise

RECORD_VERSION = 1
FRAME_KEY = ("run_id", "decoded_frame_index")
UNALIGNED_MARKER = "UNALIGNED"

# Human layers keep the labeling colours so a reviewer reads them the same way.
MATERIAL_COLOUR = (126, 179, 54)  # BGR green, as in the mask project
HOLE_COLOUR = (48, 86, 255)  # BGR red
PREDICTION_COLOUR = (255, 176, 0)  # BGR blue
TINT_ALPHA = 0.45
OUTLINE_THICKNESS = 2
TEXT_COLOUR = (255, 255, 255)
TEXT_SHADOW = (0, 0, 0)
BANNER_HEIGHT = 26
BANNER_PADDING = 8
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 1
UNKNOWN_TEXT = "no annotation for this frame"
DEFAULT_FOURCC = "mp4v"
DEVICE_INDEX_PREFIX = "device:"
DEFAULT_FPS = 15.0


class OverlayError(ValueError):
    """A record or a frame cannot be drawn without guessing."""


@dataclass(frozen=True)
class FrameRecord:
    """What is known about one frame, from any source."""

    run_id: int
    decoded_frame_index: int
    requested_seconds: float | None = None
    state: str | None = None
    state_source: str | None = None
    material_mask: str | None = None
    hole_mask: str | None = None
    prediction_state: str | None = None
    prediction_mask: str | None = None
    deviation: float | None = None
    check: str | None = None
    verdict: str | None = None
    phase_context: str | None = None
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Hold the invariants whoever builds the record."""
        for name in ("state", "prediction_state"):
            value = getattr(self, name)
            if value is None:
                continue
            try:
                object.__setattr__(self, name, normalise(str(value)))
            except UnknownState as error:
                raise OverlayError(str(error)) from error
        if self.deviation is not None and not 0.0 <= self.deviation <= 1.0:
            raise OverlayError("deviation must be between 0 and 1")

    @property
    def key(self) -> tuple[int, int]:
        return (self.run_id, self.decoded_frame_index)

    @classmethod
    def from_mapping(cls, record: Mapping[str, Any]) -> FrameRecord:
        """Validate one record, rejecting an unknown state or a bad score."""
        if int(record.get("version", RECORD_VERSION)) != RECORD_VERSION:
            raise OverlayError("unsupported record version")
        missing = [name for name in FRAME_KEY if record.get(name) is None]
        if missing:
            raise OverlayError(f"record misses {missing}")
        deviation = record.get("deviation")
        return cls(
            run_id=int(record["run_id"]),
            decoded_frame_index=int(record["decoded_frame_index"]),
            requested_seconds=(
                None
                if record.get("requested_seconds") is None
                else float(record["requested_seconds"])
            ),
            state=record.get("state"),
            state_source=record.get("state_source"),
            material_mask=record.get("material_mask"),
            hole_mask=record.get("hole_mask"),
            prediction_state=record.get("prediction_state"),
            prediction_mask=record.get("prediction_mask"),
            deviation=None if deviation is None else float(deviation),
            check=record.get("check"),
            verdict=record.get("verdict"),
            phase_context=record.get("phase_context"),
            notes=list(record.get("notes") or []),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {"version": RECORD_VERSION, **asdict(self)}


def load_records(path: Path) -> dict[tuple[int, int], FrameRecord]:
    """Read one JSON object per line, keyed by frame identity."""
    records: dict[tuple[int, int], FrameRecord] = {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = FrameRecord.from_mapping(json.loads(line))
        except (json.JSONDecodeError, OverlayError) as error:
            raise OverlayError(f"{path}:{number}: {error}") from error
        if record.key in records:
            raise OverlayError(f"{path}:{number}: duplicate frame {record.key}")
        records[record.key] = record
    return records


def write_records(path: Path, records: Iterable[FrameRecord]) -> int:
    """Write the JSONL sidecar next to a rendered video."""
    if path.exists():
        raise OverlayError(f"refusing to overwrite {path}")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    written = 0
    with path.open("w") as stream:
        for record in records:
            stream.write(json.dumps(record.to_mapping()) + "\n")
            written += 1
    return written


def _resolve_mask(mask: str | np.ndarray, masks_dir: Path | None) -> np.ndarray:
    """Load a mask from an array or from a file inside the masks directory."""
    if isinstance(mask, np.ndarray):
        return mask > 0
    if masks_dir is None:
        raise OverlayError("a mask path needs --masks-dir")
    candidate = (masks_dir / mask).resolve()
    if not candidate.is_relative_to(masks_dir.resolve()):
        raise OverlayError("mask path escapes the masks directory")
    image = cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise OverlayError(f"cannot read mask {mask}")
    return image > 0


def _tint(
    frame: np.ndarray, mask: np.ndarray, colour: Sequence[int], alpha: float
) -> None:
    if mask.shape != frame.shape[:2]:
        raise OverlayError("mask size differs from the frame")
    if not mask.any():
        return
    layer = np.zeros_like(frame)
    layer[mask] = colour
    blended = cv2.addWeighted(frame, 1.0 - alpha, layer, alpha, 0.0)
    frame[mask] = blended[mask]


def _outline(
    frame: np.ndarray, mask: np.ndarray, colour: Sequence[int]
) -> None:
    if not mask.any():
        return
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(frame, contours, -1, tuple(colour), OUTLINE_THICKNESS)


def _text(frame: np.ndarray, lines: Sequence[str], row: int) -> None:
    origin = (BANNER_PADDING, BANNER_PADDING + BANNER_HEIGHT * (row + 1))
    cv2.putText(
        frame,
        lines[row],
        origin,
        FONT,
        FONT_SCALE,
        TEXT_SHADOW,
        FONT_THICKNESS + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        lines[row],
        origin,
        FONT,
        FONT_SCALE,
        TEXT_COLOUR,
        FONT_THICKNESS,
        cv2.LINE_AA,
    )


def record_lines(record: FrameRecord | None) -> list[str]:
    """The text layer, in the order it is drawn."""
    if record is None:
        return [UNKNOWN_TEXT]
    frame = f"frame {record.decoded_frame_index}"
    if record.requested_seconds is not None:
        frame += f" at {record.requested_seconds:g}s"
    state = (
        f"state {record.state} ({record.state_source})"
        if record.state and record.state_source
        else f"state {record.state}"
        if record.state
        else "state unknown"
    )
    prediction = (
        f"model {record.prediction_state}"
        if record.prediction_state
        else "model: none"
    )
    if record.deviation is not None:
        prediction += f", deviation {record.deviation:.2f}"
    context = "check: none"
    if record.check:
        context = f"check {record.check}"
        if record.verdict:
            context += f" recorded {record.verdict}"
    if record.phase_context:
        context += f" | phases {UNALIGNED_MARKER}: {record.phase_context}"
    lines = [frame, state, prediction, context]
    lines.extend(record.notes)
    return lines


def render_frame(
    frame: np.ndarray,
    record: FrameRecord | None,
    *,
    masks_dir: Path | None = None,
) -> np.ndarray:
    """Draw the layers for one frame. The input frame is not modified."""
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise OverlayError("expected a BGR frame")
    canvas = frame.copy()
    if record is not None:
        if record.material_mask is not None:
            material = _resolve_mask(record.material_mask, masks_dir)
            _tint(canvas, material, MATERIAL_COLOUR, TINT_ALPHA)
            _outline(canvas, material, MATERIAL_COLOUR)
        if record.hole_mask is not None:
            holes = _resolve_mask(record.hole_mask, masks_dir)
            _tint(canvas, holes, HOLE_COLOUR, TINT_ALPHA)
            _outline(canvas, holes, HOLE_COLOUR)
        if record.prediction_mask is not None:
            predicted = _resolve_mask(record.prediction_mask, masks_dir)
            _outline(canvas, predicted, PREDICTION_COLOUR)
    lines = record_lines(record)
    for row in range(len(lines)):
        _text(canvas, lines, row)
    return canvas


def _open_source(source: str) -> cv2.VideoCapture:
    """A video file, or `device:N` for a camera or stream index."""
    if source.startswith(DEVICE_INDEX_PREFIX):
        index = int(source.split(":", 1)[1])
        capture = cv2.VideoCapture(index)
    else:
        path = Path(source)
        if not path.is_file():
            raise OverlayError(f"cannot read video {path}")
        capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OverlayError(f"cannot open source {source}")
    return capture


def _iterate_frames(
    source: str, limit: int | None, start: int = 0
) -> Iterator[tuple[int, np.ndarray]]:
    capture = _open_source(source)
    try:
        if start and not capture.set(cv2.CAP_PROP_POS_FRAMES, start):
            raise OverlayError(f"cannot seek to frame {start} in {source}")
        index = start
        while limit is None or index < limit:
            ok, frame = capture.read()
            if not ok:
                break
            yield index, frame
            index += 1
    finally:
        capture.release()


def render_video(
    *,
    source: str,
    output: Path,
    records: Mapping[tuple[int, int], FrameRecord] | None = None,
    masks_dir: Path | None = None,
    run_id: int | None = None,
    fps: float = DEFAULT_FPS,
    start: int = 0,
    limit: int | None = None,
    progress_every: int = 200,
) -> dict[str, Any]:
    """Render a video with its overlays and write the JSONL sidecar.

    A frame without a record is still rendered, with a banner saying so, because
    a viewer must see the gaps rather than a silently shorter video.

    `start` and `limit` are absolute frame indices in the source, so a window of
    a long clip can be rendered and records keep their source numbering.
    """
    if output.exists():
        raise OverlayError(f"refusing to overwrite {output}")
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    written = 0
    annotated = 0
    used: list[FrameRecord] = []
    writer: cv2.VideoWriter | None = None
    try:
        for index, frame in _iterate_frames(source, limit, start):
            record = None
            if records is not None and run_id is not None:
                record = records.get((run_id, index))
            canvas = render_frame(frame, record, masks_dir=masks_dir)
            if writer is None:
                height, width = canvas.shape[:2]
                writer = cv2.VideoWriter(
                    str(output),
                    cv2.VideoWriter_fourcc(*DEFAULT_FOURCC),
                    fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise OverlayError(
                        "no video encoder available for the output file"
                    )
            writer.write(canvas)
            written += 1
            if record is not None:
                annotated += 1
                used.append(record)
            if progress_every and written % progress_every == 0:
                print(f"rendered {written} frames", flush=True)
    finally:
        if writer is not None:
            writer.release()
        if written == 0:
            output.unlink(missing_ok=True)
    sidecar = output.with_suffix(".records.jsonl")
    write_records(sidecar, used)
    return {
        "frames": written,
        "frames_with_records": annotated,
        "video": str(output),
        "records": str(sidecar),
    }


def main() -> int:
    """Command line entry point for the overlay renderer."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="maus-overlay",
        description="Draw the pipeline layers onto a video or a stream",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="video file, or device:N for a camera or stream index",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help=".mp4 output"
    )
    parser.add_argument(
        "--records", type=Path, default=None, help="JSONL records per frame"
    )
    parser.add_argument(
        "--masks-dir", type=Path, default=None, help="private mask directory"
    )
    parser.add_argument("--run-id", type=int, default=None)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument(
        "--start", type=int, default=0, help="first source frame to render"
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    records = None
    if args.records is not None:
        records = load_records(args.records)
    result = render_video(
        source=args.input,
        output=args.output,
        records=records,
        masks_dir=args.masks_dir,
        run_id=args.run_id,
        fps=args.fps,
        start=args.start,
        limit=args.limit,
    )
    print(json.dumps(result, indent=2))
    return 0
