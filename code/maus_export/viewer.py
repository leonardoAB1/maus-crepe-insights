"""A local web app: one run video, with the pipeline layers beside it.

Start it, open the printed address, and the page plays the video while the side
panel and the overlay show what the pipeline knows about the frame under the
playhead. Layers come from a producer, so a model or a recognition pass is
plugged in by adding one there.

Everything stays local: the server binds to loopback, reads the video and the
masks from the private data directory, and writes nothing back.
"""

from __future__ import annotations

import json
import mimetypes
from collections.abc import Mapping
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

import cv2

from .producers import (
    FrameContext,
    ProducerError,
    default_producers,
    describe,
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8087
DEFAULT_PRODUCER = "pixel_masks"
HTML_NAME = "viewer.html"
VIDEO_ROUTE = "/api/video"
FRAME_ROUTE = "/api/frame/"
STATUS_ROUTE = "/api/status"
RECORDS_ROUTE = "/api/records"
RANGE_PREFIX = "bytes="
VIDEO_CHUNK_BYTES = 4 * 1024 * 1024
FALLBACK_FPS = 25.0
JSON_CONTENT = "application/json"
HTML_CONTENT = "text/html; charset=utf-8"
RANGE_NOT_SATISFIABLE = 416
PARTIAL_CONTENT = 206


@dataclass(frozen=True)
class Response:
    """What the handler sends back."""

    status: int
    content_type: str
    body: bytes
    headers: Mapping[str, str] | None = None


class ViewerError(ValueError):
    """The app cannot serve what was asked for."""


def load_record_file(path: Path) -> dict[int, dict[str, Any]]:
    """Read a records sidecar, keyed by the frame index in the video."""
    records: dict[int, dict[str, Any]] = {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ViewerError(f"{path}:{number}: {error}") from error
        index = record.get("decoded_frame_index")
        if index is None:
            raise ViewerError(f"{path}:{number}: record has no frame index")
        records[int(index)] = record
    return records


def probe_video(video: Path) -> dict[str, Any]:
    """Frame count, rate and size, so the timeline and seeking work."""
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ViewerError(f"cannot open video {video}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or FALLBACK_FPS
        return {
            "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": fps,
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        capture.release()


class ViewerApp:
    """The routes, without the socket, so tests can call them directly."""

    def __init__(
        self,
        *,
        video: Path,
        records: Mapping[int, Mapping[str, Any]] | None = None,
        masks_dir: Path | None = None,
        producer: str = DEFAULT_PRODUCER,
        run_id: int | None = None,
    ) -> None:
        self.video = video.resolve()
        if not self.video.is_file():
            raise ViewerError(f"cannot read video {video}")
        self.records = dict(records or {})
        self.masks_dir = None if masks_dir is None else masks_dir.resolve()
        self.run_id = run_id
        self.info = probe_video(self.video)
        self.producers = default_producers(
            records=self.records, masks_dir=self.masks_dir
        )
        if producer not in self.producers:
            raise ViewerError(f"unknown producer {producer}")
        self.producer = producer
        self.html = (Path(__file__).with_name(HTML_NAME)).read_text()

    @property
    def frames(self) -> int:
        return int(self.info["frames"])

    def status(self) -> dict[str, Any]:
        """What the page needs before it plays anything."""
        return {
            "video": self.video.name,
            "frames": self.frames,
            "fps": self.info["fps"],
            "width": self.info["width"],
            "height": self.info["height"],
            "records": len(self.records),
            "producer": self.producer,
            "producers": describe(self.producers),
            "run_id": self.run_id,
        }

    def records_list(self) -> list[dict[str, Any]]:
        """The scalar fields of every record, for the timeline strip."""
        return [
            {
                "frame": index,
                "state": record.get("state"),
                "prediction_state": record.get("prediction_state"),
                "deviation": record.get("deviation"),
            }
            for index, record in sorted(self.records.items())
        ]

    def frame_layers(
        self, index: int, producer: str | None = None
    ) -> dict[str, Any]:
        """The layers for one frame, from the chosen producer."""
        name = producer or self.producer
        if name not in self.producers:
            raise ViewerError(f"unknown producer {name}")
        if index < 0 or (self.frames and index >= self.frames):
            raise ViewerError(f"frame {index} is outside the video")
        frame = self._read_frame(index)
        context = FrameContext(
            frame_index=index,
            run_id=self.run_id,
            record=self.records.get(index),
        )
        try:
            layers = self.producers[name].layers(frame, context)
        except ProducerError as error:
            return {"frame": index, "producer": name, "error": str(error)}
        payload = layers.to_json()
        payload["frame"] = index
        return payload

    def _read_frame(self, index: int):
        capture = cv2.VideoCapture(str(self.video))
        if not capture.isOpened():
            raise ViewerError(f"cannot open video {self.video}")
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise ViewerError(f"cannot read frame {index}")
            return frame
        finally:
            capture.release()

    def video_range(self, range_header: str | None) -> Response:
        """The video, honouring a byte range so the player can seek."""
        size = self.video.stat().st_size
        common = {"Accept-Ranges": "bytes"}
        if not range_header or not range_header.startswith(RANGE_PREFIX):
            body = self.video.read_bytes()
            return Response(
                200,
                "video/mp4",
                body,
                {**common, "Content-Length": str(size)},
            )
        try:
            start_text, end_text = range_header[len(RANGE_PREFIX) :].split(
                "-", 1
            )
            start = int(start_text) if start_text else 0
            end = int(end_text) if end_text else size - 1
        except ValueError as error:
            raise ViewerError("malformed byte range") from error
        end = min(end, size - 1, start + VIDEO_CHUNK_BYTES - 1)
        if start < 0 or start >= size or end < start:
            return Response(
                RANGE_NOT_SATISFIABLE,
                "text/plain",
                b"range not satisfiable\n",
                {**common, "Content-Range": f"bytes */{size}"},
            )
        with self.video.open("rb") as stream:
            stream.seek(start)
            body = stream.read(end - start + 1)
        return Response(
            PARTIAL_CONTENT,
            "video/mp4",
            body,
            {
                **common,
                "Content-Length": str(len(body)),
                "Content-Range": f"bytes {start}-{end}/{size}",
            },
        )

    def route(
        self,
        method: str,
        target: str,
        headers: Mapping[str, str] | None = None,
    ) -> Response:
        """Resolve one request. `target` is the path with its query string."""
        headers = headers or {}
        parts = urlsplit(target)
        path = unquote(parts.path)
        query = parse_qs(parts.query)
        if method not in {"GET", "HEAD"}:
            return Response(405, "text/plain", b"method not allowed\n")
        if path in {"/", "/index.html"}:
            return Response(200, HTML_CONTENT, self.html.encode())
        if path == STATUS_ROUTE:
            return _json(self.status())
        if path == RECORDS_ROUTE:
            return _json({"records": self.records_list()})
        if path.startswith(FRAME_ROUTE):
            index_text = path[len(FRAME_ROUTE) :]
            try:
                index = int(index_text)
            except ValueError as error:
                raise ViewerError("frame index must be a number") from error
            producer = query.get("producer", [None])[0]
            return _json(self.frame_layers(index, producer))
        if path == VIDEO_ROUTE:
            return self.video_range(headers.get("Range"))
        return Response(404, "text/plain", b"not found\n")


def _json(payload: Any) -> Response:
    body = json.dumps(payload, allow_nan=False).encode()
    return Response(200, JSON_CONTENT, body)


class ViewerHandler(BaseHTTPRequestHandler):
    """The socket side of the app; the routes live in ViewerApp."""

    app: ViewerApp
    server_version = "maus-viewer"
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802
        self._respond("GET")

    def do_HEAD(self) -> None:  # noqa: N802
        self._respond("HEAD")

    def _respond(self, method: str) -> None:
        try:
            response = self.app.route(method, self.path, self.headers)
        except ViewerError as error:
            response = Response(400, "text/plain", f"{error}\n".encode())
        self.send_response(response.status)
        self.send_header("Content-Type", response.content_type)
        for name, value in (response.headers or {}).items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(response.body)))
        self.end_headers()
        if method == "GET":
            self.wfile.write(response.body)

    def log_message(self, format: str, *args: Any) -> None:
        """Keep the console for the printed address, not every request."""
        return


def serve(
    app: ViewerApp, *, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT
) -> None:
    """Serve until interrupted."""
    handler = type("BoundViewerHandler", (ViewerHandler,), {"app": app})
    with ThreadingHTTPServer((host, port), handler) as server:
        print(f"viewer on http://{host}:{server.server_port}", flush=True)
        print(
            "Ctrl-C stops it; nothing is uploaded or written back.", flush=True
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("stopped", flush=True)


def main() -> int:
    """Command line entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="maus-view",
        description="Watch a run with the pipeline layers beside it",
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--records", type=Path, default=None)
    parser.add_argument("--masks-dir", type=Path, default=None)
    parser.add_argument("--producer", default=DEFAULT_PRODUCER)
    parser.add_argument("--run-id", type=int, default=None)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    records = None if args.records is None else load_record_file(args.records)
    app = ViewerApp(
        video=args.video,
        records=records,
        masks_dir=args.masks_dir,
        producer=args.producer,
        run_id=args.run_id,
    )
    serve(app, host=args.host, port=args.port)
    return 0


def content_type_for(path: Path) -> str:
    """Best-effort type for a static file, used by tests."""
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"
