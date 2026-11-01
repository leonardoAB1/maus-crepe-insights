"""Offline checks of the viewer app, without a browser."""

import http.client
import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import pytest

from maus_export.viewer import (
    ViewerApp,
    ViewerError,
    ViewerHandler,
    load_record_file,
    probe_video,
)

SIZE = 96
FRAMES = 8


def write_video(path: Path) -> Path:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (SIZE, SIZE)
    )
    if not writer.isOpened():
        pytest.skip("no video encoder in this environment")
    for index in range(FRAMES):
        image = np.full((SIZE, SIZE, 3), 30 + index, dtype=np.uint8)
        cv2.circle(image, (48, 48), 20, (170, 220, 245), -1)
        writer.write(image)
    writer.release()
    return path


def records_file(path: Path, frames: int = 3) -> Path:
    lines = []
    for index in range(frames):
        lines.append(
            json.dumps(
                {
                    "version": 1,
                    "run_id": 3350,
                    "decoded_frame_index": index,
                    "requested_seconds": float(index * 5),
                    "state": "flat" if index % 2 else "emptyplate",
                    "state_source": "human",
                    "check": "spread",
                    "verdict": "pass",
                    "phase_context": "init -> oil",
                }
            )
        )
    path.write_text("\n".join(lines) + "\n")
    return path


def make_app(tmp_path: Path, **overrides) -> ViewerApp:
    video = write_video(tmp_path / "run.mp4")
    records = load_record_file(records_file(tmp_path / "run.records.jsonl"))
    values = {
        "video": video,
        "records": records,
        "masks_dir": None,
        "producer": "pixel_masks",
        "run_id": 3350,
    }
    values.update(overrides)
    return ViewerApp(**values)


def test_probe_reports_the_clip(tmp_path: Path) -> None:
    info = probe_video(write_video(tmp_path / "run.mp4"))
    assert info["frames"] == FRAMES
    assert info["fps"] == pytest.approx(10.0, abs=0.5)
    assert (info["width"], info["height"]) == (SIZE, SIZE)


def test_records_are_keyed_by_frame_index(tmp_path: Path) -> None:
    records = load_record_file(records_file(tmp_path / "r.jsonl"))
    assert sorted(records) == [0, 1, 2]


def test_a_record_without_a_frame_index_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"run_id": 1}\n')
    with pytest.raises(ViewerError):
        load_record_file(path)


def test_status_lists_the_producers(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    status = app.status()
    assert status["frames"] == FRAMES
    assert status["records"] == 3
    names = [entry["name"] for entry in status["producers"]]
    assert "pixel_masks" in names and "records" in names
    assert status["video"] == "run.mp4"


def test_unknown_producer_is_refused_at_startup(tmp_path: Path) -> None:
    with pytest.raises(ViewerError):
        make_app(tmp_path, producer="not_a_producer")


def test_layers_come_from_the_pixel_producer(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    payload = app.frame_layers(1)
    assert payload["frame"] == 1
    assert payload["state"] == "flat"
    assert payload["state_source"] == "human"
    assert "material" in payload["masks"]
    assert payload["producer"] == "pixel_masks"


def test_layers_can_come_from_the_records_producer(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    payload = app.frame_layers(0, "records")
    assert payload["state"] == "emptyplate"
    assert payload["check"] == "spread"
    assert payload["masks"] == {}


def test_a_frame_without_a_record_reports_the_reason(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    payload = app.frame_layers(7, "records")
    assert "error" in payload
    assert "no record" in payload["error"]


def test_frame_outside_the_video_is_refused(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    with pytest.raises(ViewerError):
        app.frame_layers(999)


def test_records_list_drives_the_timeline(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    entries = app.records_list()
    assert [entry["frame"] for entry in entries] == [0, 1, 2]
    assert entries[0]["state"] == "emptyplate"


def test_video_serves_a_full_body(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    response = app.video_range(None)
    assert response.status == 200
    assert response.headers["Accept-Ranges"] == "bytes"
    assert len(response.body) == app.video.stat().st_size


def test_video_serves_a_byte_range(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    response = app.video_range("bytes=0-99")
    assert response.status == 206
    assert response.headers["Content-Range"].startswith("bytes 0-99/")
    assert len(response.body) == 100
    assert response.body == app.video.read_bytes()[:100]


def test_video_reports_an_impossible_range(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    response = app.video_range(f"bytes={app.video.stat().st_size + 10}-")
    assert response.status == 416
    assert response.headers["Content-Range"].startswith("bytes */")


def test_malformed_range_is_refused(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    with pytest.raises(ViewerError):
        app.video_range("bytes=abc-def")


def test_routes_serve_the_page_and_the_apis(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    page = app.route("GET", "/")
    assert page.status == 200
    assert b"crepe run viewer" in page.body.lower()
    status = app.route("GET", "/api/status")
    assert json.loads(status.body)["frames"] == FRAMES
    frame = app.route("GET", "/api/frame/2?producer=records")
    assert json.loads(frame.body)["state"] == "emptyplate"
    records = app.route("GET", "/api/records")
    assert len(json.loads(records.body)["records"]) == 3
    assert app.route("GET", "/nope").status == 404
    assert app.route("POST", "/").status == 405


def test_bad_frame_index_is_a_clear_error(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    with pytest.raises(ViewerError):
        app.route("GET", "/api/frame/abc")


def test_the_server_answers_over_a_socket(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    handler = type("Bound", (ViewerHandler,), {"app": app})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection(
            "127.0.0.1", server.server_port, timeout=10
        )
        connection.request("GET", "/api/status")
        response = connection.getresponse()
        assert response.status == 200
        payload = json.loads(response.read())
        assert payload["frames"] == FRAMES
        connection.request("GET", "/api/video", headers={"Range": "bytes=0-9"})
        partial = connection.getresponse()
        assert partial.status == 206
        assert len(partial.read()) == 10
        connection.request("HEAD", "/")
        head = connection.getresponse()
        assert head.status == 200
        assert head.read() == b""
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
