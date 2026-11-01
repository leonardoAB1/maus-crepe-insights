"""Local dashboard routes and video range requests."""

import gzip
import json
import re
import shutil
import subprocess
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

import maus_export
from maus_export.fold_dashboard import make_handler, run_details
from maus_export.fold_playback import source_signature


def test_dashboard_routes_and_private_video_ranges(tmp_path: Path) -> None:
    live = tmp_path / "fold_segmentation" / "live" / "1"
    runs = tmp_path / "raw" / "runs"
    live.mkdir(parents=True)
    runs.mkdir(parents=True)
    (live / "annotated.mp4").write_bytes(b"0123456789")
    (live / "timeline.csv").write_text("video_t,state\n0,\n")
    (live / "observations.csv").write_text(
        "check,start,observation,observed_at,status\nfold1,1.6,flat,2.6,observed\n"
    )
    (live / "summary.json").write_text(
        json.dumps(
            {
                "frames": 10,
                "dispense": {
                    "frames": 20,
                    "first_fraction": 0.1,
                    "peak_fraction": 0.7,
                    "final_fraction": 0.68,
                    "ok": True,
                },
            }
        )
    )
    (runs / "1.json").write_text(
        json.dumps(
            {
                "phases": [
                    {
                        "timestamp": "2026-01-01T10:00:00",
                        "elapsed_sec": 0,
                        "phase": "fold_carefully",
                    },
                    {
                        "timestamp": "2026-01-01T10:00:05",
                        "elapsed_sec": 5,
                        "phase": "cook",
                    },
                    {
                        "timestamp": "2026-01-01T10:00:09",
                        "elapsed_sec": 9,
                        "phase": "oil",
                    },
                ],
                "logs": [
                    {
                        "timestamp": "2026-01-01T10:00:02",
                        "msg": "check_fold1 started",
                    },
                    {
                        "timestamp": "2026-01-01T10:00:03",
                        "msg": "check_fold1: first fold completed",
                    },
                ],
            }
        )
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/api/runs") as response:
            first_run = json.load(response)[0]
            assert first_run["id"] == 1
            assert first_run["dispense_ok"] is True
        with urlopen(base + "/api/run/1") as response:
            detail = json.load(response)
            assert detail["timeline"][0]["state"] == ""
            assert detail["events"][0]["expected"] == "half"
            assert detail["events"][0]["ok"] is True
            assert detail["events"][0]["observation"] == "flat"
            assert detail["events"][0]["observed_at"] == 2.6
            assert [phase["name"] for phase in detail["phases"]] == [
                "fold_carefully",
                "cook",
                "oil",
            ]
            assert [phase["time"] for phase in detail["phases"]] == [0, 5, 9]
            assert detail["dispense"]["final_fraction"] == 0.68
            assert detail["dispense"]["ok"] is True
        compressed = Request(base + "/", headers={"Accept-Encoding": "gzip"})
        with urlopen(compressed) as response:
            assert response.headers["Content-Encoding"] == "gzip"
            assert b"Fold insights" in gzip.decompress(response.read())
        disabled = Request(base + "/", headers={"Accept-Encoding": "gzip;q=0"})
        with urlopen(disabled) as response:
            assert response.headers.get("Content-Encoding") is None
        request = Request(base + "/video/1", headers={"Range": "bytes=2-4"})
        with urlopen(request) as response:
            assert response.status == 206
            assert response.read() == b"234"
        with pytest.raises(HTTPError) as error:
            urlopen(base + "/video/../raw/runs/1.json")
        assert error.value.code == 404
        (live / "playback.mp4").write_bytes(b"abcdefghij")
        (live / "playback.json").write_text(
            json.dumps(
                {
                    "source": source_signature(live / "annotated.mp4"),
                }
            )
        )
        suffix = Request(base + "/video/1", headers={"Range": "bytes=-3"})
        with urlopen(suffix) as response:
            assert response.status == 206
            assert response.read() == b"hij"
        with urlopen(base + "/original-video/1") as response:
            assert response.read() == b"0123456789"
        invalid = Request(base + "/video/1", headers={"Range": "bytes=-0"})
        with pytest.raises(HTTPError) as error:
            urlopen(invalid)
        assert error.value.code == 416
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_initial_cleaning_highlight_preserves_logged_phases(
    tmp_path: Path,
) -> None:
    live = tmp_path / "fold_segmentation/live/1"
    runs = tmp_path / "raw/runs"
    live.mkdir(parents=True)
    runs.mkdir(parents=True)
    (live / "timeline.csv").write_text("video_t,state\n0,\n")
    phases = [
        {
            "phase": name,
            "elapsed_sec": time,
            "timestamp": f"2026-01-01T10:00:{time:02d}",
        }
        for time, name in ((0, "init"), (3, "prepare"), (8, "oil"))
    ]
    (runs / "1.json").write_text(json.dumps({"phases": phases, "logs": []}))
    detail = run_details(tmp_path, 1)
    assert detail["highlights"] == [
        {
            "start": 0,
            "end": 8,
            "name": "initial cleaning/prep (inferred)",
            "source": "inferred_preparation",
        }
    ]
    assert [p["name"] for p in detail["phases"]] == [
        "init",
        "prepare",
        "oil",
    ]
    assert detail["events"] == []


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_dashboard_page_script_parses(tmp_path: Path) -> None:
    """The embedded dashboard script must stay valid JavaScript."""
    html = (
        Path(maus_export.__file__).with_name("fold_dashboard.html").read_text()
    )
    match = re.search(r"<script>([\s\S]*?)</script>", html)
    assert match is not None, "no <script> block found"
    script = tmp_path / "dashboard.js"
    script.write_text(match.group(1))
    result = subprocess.run(
        [shutil.which("node"), "--check", str(script)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
