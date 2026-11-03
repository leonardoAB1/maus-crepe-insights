"""Pi-specific offline detection and automatic low-bandwidth playback."""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path

from .fold_playback import playback_video, prepare_playback

SCAN_INTERVAL_SECONDS = 15.0
WORKER_JOIN_SECONDS = 2.0


def ensure_pi_playback(
    data_dir: Path, run_id: int, *, blocking: bool = True
) -> bool:
    """Prepare a missing proxy, serializing Pi encoders across processes.

    The POSIX lock releases even after a crash. It is global to this data
    directory so a website worker cannot compete with another Pi encoder.
    """
    import fcntl

    root = data_dir / "fold_segmentation"
    directory = root / "live" / str(run_id)
    original = directory / "annotated.mp4"
    if not original.is_file():
        return False
    with (root / ".playback.lock").open("a") as lock:
        flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(lock, flags)
        except BlockingIOError:
            return False
        try:
            if playback_video(directory) == original:
                print(
                    f"Pi: preparing low-bandwidth playback for {run_id}",
                    flush=True,
                )
                prepare_playback(data_dir, run_id)
            return True
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


class PiPlaybackWorker:
    """Prepare completed analyses in the background without delaying HTTP."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def scan_once(self) -> None:
        """Ignore staging directories and let an active encoder finish first."""
        root = self.data_dir / "fold_segmentation/live"
        for directory in sorted(root.glob("*")):
            if self.stop_event.is_set():
                break
            if not directory.name.isdecimal():
                continue
            if not all(
                (directory / name).is_file()
                for name in ("summary.json", "annotated.mp4")
            ):
                continue
            try:
                ensure_pi_playback(
                    self.data_dir, int(directory.name), blocking=False
                )
            except (
                OSError,
                ValueError,
                RuntimeError,
                subprocess.SubprocessError,
            ) as error:
                print(
                    f"Pi playback unavailable: {type(error).__name__}",
                    file=sys.stderr,
                    flush=True,
                )

    def _run(self) -> None:
        while not self.stop_event.is_set():
            self.scan_once()
            self.stop_event.wait(SCAN_INTERVAL_SECONDS)

    def start(self) -> None:
        """Start one background scanner; the HTTP server need not wait."""
        print("Pi: automatic low-bandwidth playback enabled", flush=True)
        self.thread.start()

    def stop(self) -> None:
        """Stop scheduling new jobs; systemd also stops in-flight encoders."""
        self.stop_event.set()
        self.thread.join(timeout=WORKER_JOIN_SECONDS)


def main(argv: list[str] | None = None) -> None:
    """Use the Pi profile rather than changing generic desktop defaults."""
    from .fold_runtime import main as runtime_main

    runtime_main(argv, pi_profile=True)


if __name__ == "__main__":
    main()
