"""Validated background launches without running real video inference."""

import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from maus_export.experimental_fold.fold_dashboard import make_handler
from maus_export.experimental_fold.fold_web_runner import RunnerBusy, WebRunner


class Process:
    def __init__(self) -> None:
        self.code = None

    def poll(self) -> int | None:
        return self.code


@pytest.fixture
def runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> WebRunner:
    runs = tmp_path / "raw/runs"
    runs.mkdir(parents=True)
    (runs / "1.json").write_text("{}", newline="\n")
    media = tmp_path / "raw/media/1"
    media.mkdir(parents=True)
    (media / "video.mp4").write_bytes(b"synthetic")
    instance = WebRunner(tmp_path)
    calls = []

    def launch(command: list[str], **kwargs: object) -> Process:
        calls.append((command, kwargs))
        return Process()

    monkeypatch.setattr(
        "maus_export.experimental_fold.fold_web_runner.subprocess.Popen", launch
    )
    monkeypatch.setattr(
        "maus_export.experimental_fold.fold_web_runner.importlib.util.find_spec",
        lambda name: object(),
    )
    instance.calls = calls
    return instance


@pytest.mark.parametrize("threads", [1, 4, 16])
def test_launch_selected_video_and_prevent_duplicates(
    runner: WebRunner, threads: int
) -> None:
    result = runner.start({"nthreads": threads, "model": "onnx", "run_id": 1})
    assert result["running"] is True
    command, options = runner.calls[0]
    assert command[command.index("--threads") + 1] == str(threads)
    assert command[command.index("--model") + 1] == "onnx"
    assert command[-3:] == ["--ids", "1", "--replace"]
    assert "shell" not in options
    with pytest.raises(RunnerBusy):
        runner.start({})
    runner.process.code = 0
    assert runner.snapshot()["running"] is False
    runner.start({})
    command = runner.calls[-1][0]
    assert command[command.index("--threads") + 1] == "1"
    assert "--replace" not in command


@pytest.mark.parametrize(
    "options",
    [
        {"nthreads": 0},
        {"nthreads": 2},
        {"nthreads": True},
        {"nthreads": "16"},
        {"model": "../raw"},
        {"run_id": True},
        {"run_id": -1},
        {"run_id": 2},
        {"command": "anything"},
        [],
    ],
)
def test_invalid_launch_never_spawns(
    runner: WebRunner, options: object
) -> None:
    with pytest.raises(ValueError):
        runner.start(options)
    assert runner.calls == []


def test_missing_onnx_environment_does_not_spawn(
    runner: WebRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "maus_export.experimental_fold.fold_web_runner.importlib.util.find_spec",
        lambda name: None,
    )
    with pytest.raises(ValueError, match="ml environment"):
        runner.start({"model": "onnx"})
    assert runner.calls == []


def test_runner_http_requires_json_custom_header_and_same_origin(
    tmp_path: Path, runner: WebRunner
) -> None:
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), make_handler(tmp_path, runner=runner)
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/api/runner") as response:
            assert json.load(response)["runs"] == [1]
        for headers, code in (
            ({"Content-Type": "application/json"}, 403),
            ({"X-Maus-Runner": "1", "Origin": "https://elsewhere.test"}, 403),
            ({"X-Maus-Runner": "1"}, 415),
        ):
            with pytest.raises(HTTPError) as error:
                urlopen(
                    Request(base + "/api/runner", data=b"{}", headers=headers)
                )
            assert error.value.code == code
        headers = {"Content-Type": "application/json", "X-Maus-Runner": "1"}
        with urlopen(
            Request(base + "/api/runner", data=b"{}", headers=headers)
        ) as response:
            assert response.status == 202
            assert json.load(response)["nthreads"] == 1
        with pytest.raises(HTTPError) as error:
            urlopen(Request(base + "/api/runner", data=b"{}", headers=headers))
        assert error.value.code == 409
        assert len(runner.calls) == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
