"""Launch one bounded offline analysis runner from a review website."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from threading import Lock
from typing import Any

from .fold_backend import MODELS
from .fold_batch import discover_runs

THREAD_COUNTS = (1, 4, 16)


class RunnerBusy(ValueError):
    """This website already has a running analysis process."""


class WebRunner:
    """Serialize browser launches without blocking video or progress requests.

    The subprocess uses this server's Python environment. It only reads cached
    inputs and writes private analysis artifacts; it never fetches or controls
    a robot. This guards this server, not separately launched CLI processes.
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir.resolve()
        self.lock = Lock()
        self.process: subprocess.Popen | None = None
        self.options: dict[str, Any] = {}

    def _status(self) -> dict[str, Any]:
        code = self.process.poll() if self.process is not None else None
        return {
            "running": self.process is not None and code is None,
            "returncode": code,
            **self.options,
        }

    def snapshot(self) -> dict[str, Any]:
        """Expose available cached videos and the server-owned runner status."""
        with self.lock:
            return {**self._status(), "runs": discover_runs(self.data_dir)}

    def start(self, request: object) -> dict[str, Any]:
        """Validate browser choices and spawn a shell-free background runner."""
        if not isinstance(request, dict) or set(request) - {
            "nthreads",
            "model",
            "run_id",
        }:
            raise ValueError("Invalid runner options")
        threads = request.get("nthreads", 1)
        model = request.get("model", "classical")
        run_id = request.get("run_id")
        if type(threads) is not int or threads not in THREAD_COUNTS:
            raise ValueError("nthreads must be 1, 4 or 16")
        if model not in (*MODELS, "both"):
            raise ValueError("Unknown segmentation model")
        if run_id is not None and (type(run_id) is not int or run_id < 0):
            raise ValueError("Invalid run identifier")
        with self.lock:
            if self._status()["running"]:
                raise RunnerBusy("An analysis runner is already active")
            runs = discover_runs(self.data_dir)
            if not runs or (run_id is not None and run_id not in runs):
                raise ValueError("Requested video is not cached")
            if (
                model in {"onnx", "both"}
                and importlib.util.find_spec("onnxruntime") is None
            ):
                raise ValueError(
                    "ONNX requires starting the server in the ml environment"
                )
            command = [
                sys.executable,
                "-m",
                "maus_export.cli",
                "fold-batch",
                "--data-dir",
                str(self.data_dir),
                "--threads",
                str(threads),
                "--model",
                model,
            ]
            if run_id is not None:
                command.extend(["--ids", str(run_id), "--replace"])
            log = self.data_dir / "fold_segmentation" / "web-runner.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("ab") as output:
                self.process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            self.options = {
                "nthreads": threads,
                "workers": max(threads, len(MODELS) if model == "both" else 1),
                "model": model,
                "run_id": run_id,
            }
            return self._status()
