"""Offline checks of the overlay renderer, on synthetic frames and records."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from maus_export.overlay import (
    UNALIGNED_MARKER,
    UNKNOWN_TEXT,
    FrameRecord,
    OverlayError,
    load_records,
    record_lines,
    render_frame,
    render_video,
    write_records,
)

WIDTH = 160
HEIGHT = 120
FRAMES = 6


def frame(colour: int = 40) -> np.ndarray:
    return np.full((HEIGHT, WIDTH, 3), colour, dtype=np.uint8)


def record(**overrides) -> FrameRecord:
    values = {
        "run_id": 3350,
        "decoded_frame_index": 3,
        "requested_seconds": 15.0,
        "state": "flat",
        "state_source": "human",
    }
    values.update(overrides)
    return FrameRecord(**values)


def write_mask(directory: Path, name: str, box: tuple[int, int, int, int]):
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    x, y, w, h = box
    mask[y : y + h, x : x + w] = 255
    directory.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(directory / name), mask)
    return name


def write_video(path: Path) -> Path:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (WIDTH, HEIGHT),
    )
    if not writer.isOpened():
        pytest.skip("no video encoder in this environment")
    for index in range(FRAMES):
        writer.write(frame(40 + index))
    writer.release()
    return path


def test_record_round_trip_normalises_the_state() -> None:
    parsed = FrameRecord.from_mapping(
        {"version": 1, "run_id": 7, "decoded_frame_index": 2, "state": "circle"}
    )
    assert parsed.state == "cooking"
    assert parsed.key == (7, 2)
    assert FrameRecord.from_mapping(parsed.to_mapping()) == parsed


def test_record_rejects_an_unknown_state() -> None:
    with pytest.raises(OverlayError):
        FrameRecord.from_mapping(
            {"run_id": 7, "decoded_frame_index": 2, "state": "banana"}
        )


def test_record_rejects_a_missing_frame_identity() -> None:
    with pytest.raises(OverlayError):
        FrameRecord.from_mapping({"run_id": 7})


def test_record_rejects_an_impossible_deviation() -> None:
    with pytest.raises(OverlayError):
        FrameRecord.from_mapping(
            {"run_id": 7, "decoded_frame_index": 2, "deviation": 1.4}
        )


def test_records_are_read_from_json_lines(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(
        json.dumps({"run_id": 1, "decoded_frame_index": 0, "state": "flat"})
        + "\n\n"
        + json.dumps({"run_id": 1, "decoded_frame_index": 1})
        + "\n"
    )
    records = load_records(path)
    assert set(records) == {(1, 0), (1, 1)}
    assert records[(1, 0)].state == "flat"


def test_duplicate_frames_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    line = json.dumps({"run_id": 1, "decoded_frame_index": 0})
    path.write_text(f"{line}\n{line}\n")
    with pytest.raises(OverlayError):
        load_records(path)


def test_bad_json_line_names_the_file_and_line(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(
        json.dumps({"run_id": 1, "decoded_frame_index": 0}) + "\nnot json\n"
    )
    with pytest.raises(OverlayError) as error:
        load_records(path)
    assert "records.jsonl:2" in str(error.value)


def test_sidecar_is_written_once(tmp_path: Path) -> None:
    path = tmp_path / "out.records.jsonl"
    assert write_records(path, [record()]) == 1
    assert json.loads(path.read_text().splitlines()[0])["run_id"] == 3350
    with pytest.raises(OverlayError):
        write_records(path, [record()])


def test_frame_without_a_record_says_so() -> None:
    assert record_lines(None) == [UNKNOWN_TEXT]
    canvas = render_frame(frame(), None)
    assert canvas.shape == (HEIGHT, WIDTH, 3)
    assert not np.array_equal(canvas, frame())


def test_banner_carries_the_layers_that_matter() -> None:
    lines = record_lines(
        record(
            prediction_state="half",
            deviation=0.42,
            check="fold1",
            verdict="pass",
            phase_context="init -> oil",
        )
    )
    joined = " | ".join(lines)
    assert "state flat (human)" in joined
    assert "model halffold" in joined
    assert "deviation 0.42" in joined
    assert "check fold1 recorded pass" in joined
    assert UNALIGNED_MARKER in joined


def test_material_mask_is_tinted_inside_and_not_outside(tmp_path: Path) -> None:
    name = write_mask(tmp_path, "material.png", (20, 20, 40, 30))
    canvas = render_frame(
        frame(), record(material_mask=name), masks_dir=tmp_path
    )
    inside = canvas[30, 40]
    outside = canvas[HEIGHT - 5, WIDTH - 5]
    assert not np.array_equal(inside, outside)
    assert tuple(int(value) for value in inside) != (40, 40, 40)


def test_mask_path_outside_the_masks_directory_is_refused(
    tmp_path: Path,
) -> None:
    with pytest.raises(OverlayError):
        render_frame(
            frame(), record(material_mask="../escape.png"), masks_dir=tmp_path
        )


def test_missing_mask_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(OverlayError):
        render_frame(
            frame(), record(material_mask="absent.png"), masks_dir=tmp_path
        )


def test_mask_of_the_wrong_size_is_reported(tmp_path: Path) -> None:
    small = np.zeros((10, 10), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "small.png"), small)
    with pytest.raises(OverlayError):
        render_frame(
            frame(), record(material_mask="small.png"), masks_dir=tmp_path
        )


def test_prediction_outline_is_drawn(tmp_path: Path) -> None:
    name = write_mask(tmp_path, "prediction.png", (60, 40, 40, 30))
    canvas = render_frame(
        frame(), record(prediction_mask=name), masks_dir=tmp_path
    )
    assert not np.array_equal(canvas, frame())


def test_render_video_writes_frames_and_a_sidecar(tmp_path: Path) -> None:
    source = write_video(tmp_path / "source.mp4")
    output = tmp_path / "rendered.mp4"
    records = {(3350, 1): record(decoded_frame_index=1)}
    result = render_video(
        source=str(source),
        output=output,
        records=records,
        run_id=3350,
        fps=10.0,
    )
    assert result["frames"] == FRAMES
    assert result["frames_with_records"] == 1
    rendered = cv2.VideoCapture(str(output))
    count = 0
    while rendered.read()[0]:
        count += 1
    rendered.release()
    assert count == FRAMES
    first = _read(output, 0)
    second = _read(output, 1)
    assert not np.array_equal(first, second)
    sidecar = Path(result["records"])
    assert json.loads(sidecar.read_text().splitlines()[0])["run_id"] == 3350


def _read(path: Path, index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, image = capture.read()
        if not ok:
            break
        frames.append(image)
    capture.release()
    return frames[index]


def test_render_video_refuses_to_overwrite(tmp_path: Path) -> None:
    source = write_video(tmp_path / "source.mp4")
    output = tmp_path / "rendered.mp4"
    output.write_bytes(b"keep me")
    with pytest.raises(OverlayError):
        render_video(source=str(source), output=output)


def test_render_video_reports_a_missing_source(tmp_path: Path) -> None:
    with pytest.raises(OverlayError):
        render_video(
            source=str(tmp_path / "absent.mp4"), output=tmp_path / "out.mp4"
        )


def test_render_video_can_start_inside_a_clip(tmp_path: Path) -> None:
    source = write_video(tmp_path / "source.mp4")
    output = tmp_path / "window.mp4"
    records = {(3350, 3): record(decoded_frame_index=3)}
    result = render_video(
        source=str(source),
        output=output,
        records=records,
        run_id=3350,
        fps=10.0,
        start=3,
        limit=5,
    )
    assert result["frames"] == 2
    assert result["frames_with_records"] == 1
    sidecar = json.loads(Path(result["records"]).read_text().splitlines()[0])
    assert sidecar["decoded_frame_index"] == 3
