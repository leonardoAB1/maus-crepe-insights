"""Local-only dashboard for private fold video analysis outputs."""

from __future__ import annotations

import csv
import json
import re
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .fold_backend import analysis_root
from .fold_frames import check_starts, phase_clock_origin, video_time
from .fold_stages import INITIAL_CLEANING_LABEL, initial_cleaning_end

CHUNK_SIZE = 1024 * 1024
PAGE = Path(__file__).with_name("fold_dashboard.html")


def run_details(data_dir: Path, run_id: int, model: str = "classical") -> dict:
    """Read one local timeline and robot check events for dashboard playback."""
    root = analysis_root(data_dir, model) / str(run_id)
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
            "observation": None,
            "observed_at": None,
            "observation_status": "not_evaluated",
            "verdict": "unknown",
            "confidence": 0.0,
        }
        for event in check_starts(run["logs"])
    ]
    observations_path = root / "observations.csv"
    if observations_path.is_file():
        with observations_path.open(newline="") as file:
            observations = list(csv.DictReader(file))
        for event, observation in zip(events, observations, strict=True):
            if (
                observation["check"] != event["name"]
                or abs(float(observation["start"]) - event["time"]) > 1e-6
            ):
                raise ValueError("Observation does not match the cached log")
            event["observation"] = observation["observation"] or None
            event["observed_at"] = (
                float(observation["observed_at"])
                if observation["observed_at"]
                else None
            )
            event["observation_status"] = observation["status"]
            event["verdict"] = observation.get("verdict", "unknown")
            event["confidence"] = float(observation.get("confidence") or 0)
    phases = [
        {"time": float(phase["elapsed_sec"]), "name": phase["phase"]}
        for phase in run["phases"]
        if phase.get("phase") and phase.get("elapsed_sec") is not None
    ]
    cleaning_end = initial_cleaning_end(
        [(phase["time"], phase["name"]) for phase in phases]
    )
    highlights = (
        [
            {
                "start": 0.0,
                "end": cleaning_end,
                "name": INITIAL_CLEANING_LABEL,
                "source": "inferred_preparation",
            }
        ]
        if cleaning_end > 0
        else []
    )
    dispense = None
    summary_path = root / "summary.json"
    if summary_path.is_file():
        value = json.loads(summary_path.read_text()).get("dispense")
        if isinstance(value, dict):
            dispense = value
    return {
        "id": run_id,
        "model": model,
        "events": events,
        "phases": phases,
        "highlights": highlights,
        "dispense": dispense,
        "timeline": timeline,
    }


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
            request = urlsplit(self.path)
            path = request.path
            values = parse_qs(request.query, keep_blank_values=True).get(
                "model", ["classical"]
            )
            try:
                if len(values) != 1:
                    raise ValueError("Ambiguous model")
                model = values[0]
                directory = analysis_root(root, model)
            except ValueError:
                self.send_error(400, "Unknown segmentation model")
                return
            if path == "/":
                self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
                return
            if path == "/api/runs":
                runs = []
                for item in sorted(directory.glob("*")):
                    if (
                        not item.name.isdecimal()
                        or not (item / "annotated.mp4").is_file()
                    ):
                        continue
                    try:
                        info = run_details(root, int(item.name), model)
                        runs.append(
                            {
                                "id": info["id"],
                                "events": info["events"],
                                "visual_failures": sum(
                                    event["verdict"] == "failure"
                                    for event in info["events"]
                                ),
                                "visual_unknown": sum(
                                    event["verdict"] == "unknown"
                                    for event in info["events"]
                                ),
                                "dispense_ok": (info["dispense"] or {}).get(
                                    "ok"
                                ),
                            }
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
                    result = run_details(root, run_id, model)
                except (OSError, ValueError, KeyError):
                    self.send_error(404)
                    return
                self._send(json.dumps(result).encode(), "application/json")
                return
            video = directory / str(run_id) / "annotated.mp4"
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


def serve_dashboard(
    data_dir: Path, port: int = 8765, *, open_browser: bool = False
) -> None:
    """Serve private data only on localhost until interrupted."""
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(data_dir))
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"Private dashboard: {url}", flush=True)
    try:
        if open_browser:
            webbrowser.open(url)
        server.serve_forever()
    finally:
        server.server_close()
