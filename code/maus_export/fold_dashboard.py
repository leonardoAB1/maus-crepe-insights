"""Local-only dashboard for private fold video analysis outputs."""

from __future__ import annotations

import csv
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .fold_frames import check_events, phase_clock_origin, video_time

CHUNK_SIZE = 1024 * 1024
PAGE = Path(__file__).with_name("fold_dashboard.html")


def run_details(data_dir: Path, run_id: int) -> dict:
    """Read one local timeline and robot check events for dashboard playback."""
    root = data_dir / "fold_segmentation" / "live" / str(run_id)
    with (root / "timeline.csv").open(newline="") as file:
        timeline = list(csv.DictReader(file))
    run = json.loads((data_dir / "raw" / "runs" / f"{run_id}.json").read_text())
    origin = phase_clock_origin(run["phases"])
    events = [
        {
            "time": video_time(event, origin),
            "name": event.name,
            "ok": event.check_ok,
            "expected": event.state,
        }
        for event in check_events(run["logs"])
    ]
    return {"id": run_id, "events": events, "timeline": timeline}


def make_handler(data_dir: Path) -> type[BaseHTTPRequestHandler]:
    """Make a read-only HTTP handler limited to known private artifacts."""
    root = data_dir.resolve()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def _send(self, body: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            if path == "/":
                self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
                return
            if path == "/api/runs":
                directory = root / "fold_segmentation" / "live"
                runs = []
                for item in sorted(directory.glob("*")):
                    if (
                        not item.name.isdecimal()
                        or not (item / "annotated.mp4").is_file()
                    ):
                        continue
                    try:
                        info = run_details(root, int(item.name))
                        runs.append(
                            {"id": info["id"], "events": info["events"]}
                        )
                    except (OSError, ValueError, KeyError):
                        continue
                self._send(json.dumps(runs).encode(), "application/json")
                return
            match = re.fullmatch(r"/(api/run|video)/(\d+)", path)
            if not match:
                self.send_error(404)
                return
            run_id = int(match.group(2))
            if match.group(1) == "api/run":
                try:
                    result = run_details(root, run_id)
                except (OSError, ValueError, KeyError):
                    self.send_error(404)
                    return
                self._send(json.dumps(result).encode(), "application/json")
                return
            video = (
                root
                / "fold_segmentation"
                / "live"
                / str(run_id)
                / "annotated.mp4"
            )
            if not video.is_file():
                self.send_error(404)
                return
            size = video.stat().st_size
            start, end = 0, size - 1
            header = self.headers.get("Range")
            if header:
                match_range = re.fullmatch(r"bytes=(\d+)-(\d*)", header)
                if not match_range:
                    self.send_error(416)
                    return
                start = int(match_range.group(1))
                end = min(int(match_range.group(2) or end), end)
                if start > end:
                    self.send_error(416)
                    return
            self.send_response(206 if header else 200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Cache-Control", "no-store")
            if header:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with video.open("rb") as stream:
                stream.seek(start)
                remaining = end - start + 1
                while remaining:
                    chunk = stream.read(min(CHUNK_SIZE, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

    return Handler


def serve_dashboard(data_dir: Path, port: int = 8765) -> None:
    """Serve private data only on localhost until interrupted."""
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(data_dir))
    print(f"Private dashboard: http://127.0.0.1:{server.server_port}/")
    try:
        server.serve_forever()
    finally:
        server.server_close()
