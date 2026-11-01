"""Prepare a small browser video without altering analysis or its timings."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

DEFAULT_WIDTH = 480
DEFAULT_BITRATE_KBPS = 120
ENCODER_THREADS = 1
KEYFRAME_INTERVAL = 20


def source_signature(path: Path) -> dict[str, int]:
    """Identify the original so stale proxies are not served after rebuilds."""
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def playback_video(directory: Path) -> Path:
    """Select a complete matching playback proxy, or the original video."""
    original = directory / "annotated.mp4"
    proxy = directory / "playback.mp4"
    try:
        metadata = json.loads((directory / "playback.json").read_text())
        if (
            metadata["source"] == source_signature(original)
            and proxy.is_file()
            and proxy.stat().st_size > 0
        ):
            return proxy
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return original


def prepare_playback(
    data_dir: Path,
    run_id: int,
    *,
    replace: bool = False,
    width: int = DEFAULT_WIDTH,
    bitrate_kbps: int = DEFAULT_BITRATE_KBPS,
) -> dict:
    """Encode H.264/yuv420p fast-start playback as a private artifact."""
    if run_id < 0 or width < 2 or width % 2 or bitrate_kbps <= 0:
        raise ValueError("Require nonnegative id, positive bitrate, even width")
    encoder = shutil.which("ffmpeg")
    if encoder is None:
        raise FileNotFoundError("ffmpeg is required to prepare playback")
    directory = data_dir / "fold_segmentation/live" / str(run_id)
    original = directory / "annotated.mp4"
    signature = source_signature(original)
    if playback_video(directory) != original and not replace:
        raise FileExistsError("Playback exists; use --replace to rebuild")
    with tempfile.TemporaryDirectory(
        prefix=".playback-", dir=directory
    ) as temp:
        proxy = Path(temp) / "playback.mp4"
        subprocess.run(
            [
                encoder,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-threads",
                str(ENCODER_THREADS),
                "-i",
                str(original),
                "-map",
                "0:v:0",
                "-an",
                "-vf",
                f"scale={width}:-2",
                "-c:v",
                "libx264",
                "-threads",
                str(ENCODER_THREADS),
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
                "-b:v",
                f"{bitrate_kbps}k",
                "-maxrate",
                f"{bitrate_kbps}k",
                "-bufsize",
                f"{2 * bitrate_kbps}k",
                "-g",
                str(KEYFRAME_INTERVAL),
                "-movflags",
                "+faststart",
                str(proxy),
            ],
            check=True,
        )
        if not proxy.is_file() or not proxy.stat().st_size:
            raise OSError("Encoder produced no playback video")
        if source_signature(original) != signature:
            raise RuntimeError("Analysis changed during playback preparation")
        metadata = {
            "source": signature,
            "codec": "h264",
            "width": width,
            "bitrate_kbps": bitrate_kbps,
            "bytes": proxy.stat().st_size,
        }
        manifest = Path(temp) / "playback.json"
        manifest.write_text(json.dumps(metadata, indent=2))
        # Invalidate the old signature before replacing a different proxy.
        (directory / "playback.json").unlink(missing_ok=True)
        os.replace(proxy, directory / "playback.mp4")
        os.replace(manifest, directory / "playback.json")
    return metadata


def main() -> None:
    """Prepare playback for already analyzed runs, without rerunning vision."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_ids", nargs="+", type=int)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument(
        "--bitrate-kbps", type=int, default=DEFAULT_BITRATE_KBPS
    )
    args = parser.parse_args()
    for run_id in dict.fromkeys(args.run_ids):
        print(
            prepare_playback(
                args.data_dir,
                run_id,
                replace=args.replace,
                width=args.width,
                bitrate_kbps=args.bitrate_kbps,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
