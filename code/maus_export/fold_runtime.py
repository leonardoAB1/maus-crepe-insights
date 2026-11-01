"""Minimal offline fold detection and website, without export/auth clients."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_PORT = 8766
DEFAULT_THREADS = 1
DEFAULT_FPS = 10.0


def main(argv: list[str] | None = None, *, pi_profile: bool = False) -> None:
    """Analyze cached videos and/or serve their local review website."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_ids", type=int, nargs="*")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="bind address; non-loopback exposes unauthenticated private data",
    )
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.set_defaults(compress_playback=False)
    if pi_profile:
        parser.add_argument(
            "--compress-playback",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="automatically prepare low-bandwidth Pi video",
        )
    args = parser.parse_args(argv)
    if args.threads < DEFAULT_THREADS:
        parser.error("--threads must be positive")
    if not args.run_ids and args.analyze_only:
        parser.error("--analyze-only requires at least one run id")
    if args.run_ids:
        import cv2

        from .fold_video import analyse_video

        cv2.setNumThreads(args.threads)
        for run_id in dict.fromkeys(args.run_ids):
            result = analyse_video(
                args.data_dir,
                run_id,
                replace=args.replace,
                analysis_fps=args.fps,
            )
            print(f"Run {run_id}: {result}", flush=True)
            if args.compress_playback:
                from .fold_pi import ensure_pi_playback

                try:
                    ensure_pi_playback(args.data_dir, run_id)
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
    if not args.analyze_only:
        from .fold_dashboard import serve_dashboard

        worker = None
        if args.compress_playback:
            from .fold_pi import PiPlaybackWorker

            worker = PiPlaybackWorker(args.data_dir)
            worker.start()
        try:
            serve_dashboard(args.data_dir, args.port, host=args.host)
        finally:
            if worker is not None:
                worker.stop()


if __name__ == "__main__":
    main()
