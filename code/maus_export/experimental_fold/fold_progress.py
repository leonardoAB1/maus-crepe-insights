"""Atomic, private progress snapshots shared by runners and the dashboard."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path


class VideoProgress:
    """Report end-to-end throughput (including decoding and output writes)."""

    def __init__(self, data_dir: Path, run_id: int) -> None:
        self.path = (
            data_dir / "fold_segmentation" / "progress" / f"{run_id}.json"
        )
        self.started = time.perf_counter()
        self.last_write = float("-inf")
        self.value = {
            "id": run_id,
            "state": "starting",
            "frames_done": 0,
            "frames_total": None,
            "fraction": 0.0,
        }
        self.update(force=True)

    def update(self, *, force: bool = False, **values: object) -> None:
        self.value.update(values)
        now = time.perf_counter()
        if not force and now - self.last_write < 0.5:
            return
        elapsed = now - self.started
        frames = self.value["frames_done"]
        fraction = self.value["fraction"]
        self.value.update(
            elapsed_seconds=elapsed,
            fps=frames / elapsed if frames else None,
            seconds_per_frame=elapsed / frames if frames else None,
            eta_seconds=(elapsed * (1 - fraction) / fraction)
            if fraction and self.value["state"] == "running"
            else None,
            updated_at=time.time(),
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            dir=self.path.parent, prefix=".progress-"
        )
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(self.value, stream, allow_nan=False)
            os.replace(temporary, self.path)
        finally:
            Path(temporary).unlink(missing_ok=True)
        self.last_write = now


def read_progress(data_dir: Path) -> list[dict]:
    """Skip corrupt snapshots without breaking dashboard polling."""
    result = []
    directory = data_dir / "fold_segmentation" / "progress"
    for path in sorted(directory.glob("*.json")):
        if not path.stem.isdecimal():
            continue
        try:
            value = json.loads(path.read_text())
            if isinstance(value, dict) and value.get("id") == int(path.stem):
                value["stale"] = (
                    value.get("state") in {"starting", "running", "finalizing"}
                    and time.time() - value.get("updated_at", 0) > 30
                )
                result.append(value)
        except (OSError, ValueError, TypeError):
            continue
    return result
