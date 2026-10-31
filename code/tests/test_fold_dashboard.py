"""Local dashboard routes and video range requests."""

import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from maus_export.fold_dashboard import make_handler


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
    (runs / "1.json").write_text(
        json.dumps(
            {
                "phases": [
                    {
                        "timestamp": "2026-01-01T10:00:00",
                        "elapsed_sec": 0,
                        "phase": "fold_carefully",
                    }
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
            assert json.load(response)[0]["id"] == 1
        with urlopen(base + "/api/run/1") as response:
            detail = json.load(response)
            assert detail["timeline"][0]["state"] == ""
            assert detail["events"][0]["expected"] == "half"
            assert detail["events"][0]["ok"] is True
            assert detail["events"][0]["observation"] == "flat"
            assert detail["events"][0]["observed_at"] == 2.6
            assert detail["phases"][0]["name"] == "fold_carefully"
        request = Request(base + "/video/1", headers={"Range": "bytes=2-4"})
        with urlopen(request) as response:
            assert response.status == 206
            assert response.read() == b"234"
        with pytest.raises(HTTPError) as error:
            urlopen(base + "/video/../raw/runs/1.json")
        assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
