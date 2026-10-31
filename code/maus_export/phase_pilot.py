"""Prepare a private phase-first frame pilot from a local raw snapshot."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from fractions import Fraction
from pathlib import Path

RUN_COUNT = 50
INTERVAL_SECONDS = 5
MIN_FREE_BYTES = 2 * 1024**3
MAX_FRAME_BYTES_PER_PIXEL = 8
FRAME_OVERHEAD_BYTES = 65536


class InvalidVideo(Exception):
    """Video cannot be decoded safely."""


class UnsafeTiming(Exception):
    """Decoded timestamps could not be independently verified."""


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_runs(raw: Path) -> list[tuple[datetime, dict, Path, str]]:
    selected = []
    ids = set()
    paths = list((raw / "runs").glob("*.json"))
    if not paths:
        raise ValueError("no run records")
    for path in paths:
        content = path.read_bytes()
        run = json.loads(content)
        run_id = run["id"]
        if type(run_id) is not int or run_id < 0 or path.stem != str(run_id):
            raise ValueError("invalid run identity")
        if run_id in ids:
            raise ValueError("duplicate run identity")
        ids.add(run_id)
        start = datetime.fromisoformat(run["tsStart"].replace("Z", "+00:00"))
        if start.utcoffset() is None:
            raise ValueError("run start lacks timezone")
        selected.append((start, run, path, hashlib.sha256(content).hexdigest()))
    if len(selected) < RUN_COUNT:
        raise ValueError("fewer than 50 locally recorded runs")
    selected.sort(key=lambda row: (row[0], row[1]["id"]), reverse=True)
    return selected[:RUN_COUNT]


def _probe(video: Path) -> tuple[dict, list[int]]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=time_base,start_pts,duration_ts,width,height:"
            "frame=pts,best_effort_timestamp",
            "-of",
            "json",
            str(video),
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode or result.stderr:
        raise InvalidVideo
    try:
        data = json.loads(result.stdout)
        stream = data["streams"][0]
        pts = [int(frame["best_effort_timestamp"]) for frame in data["frames"]]
        base = Fraction(stream["time_base"])
        start, duration = int(stream["start_pts"]), int(stream["duration_ts"])
        width, height = int(stream["width"]), int(stream["height"])
        if width <= 0 or height <= 0:
            raise ValueError
    except (ValueError, KeyError, IndexError, TypeError, ZeroDivisionError):
        raise InvalidVideo from None
    if any(
        "pts" not in frame or frame["pts"] != frame["best_effort_timestamp"]
        for frame in data["frames"]
    ):
        raise UnsafeTiming
    if (
        not pts
        or base <= 0
        or duration <= 0
        or pts[0] != start
        or any(a >= b for a, b in zip(pts, pts[1:], strict=False))
        or pts[-1] >= start + duration
    ):
        raise UnsafeTiming
    return stream, pts


def _extract(video: Path, directory: Path) -> dict:
    stream, pts = _probe(video)
    base = Fraction(stream["time_base"])
    start = int(stream["start_pts"])
    duration = int(stream["duration_ts"])
    requested = []
    second = 0
    while Fraction(second) < duration * base:
        index = bisect.bisect_right(pts, Fraction(start) + second / base) - 1
        requested.append((second, index))
        second += INTERVAL_SECONDS
    indices = sorted({index for _, index in requested})
    bytes_per_frame = (
        int(stream["width"]) * int(stream["height"]) * MAX_FRAME_BYTES_PER_PIXEL
        + FRAME_OVERHEAD_BYTES
    )
    if shutil.disk_usage(directory.parent).free < (
        MIN_FREE_BYTES + bytes_per_frame * len(indices)
    ):
        raise ValueError("insufficient free disk space")
    directory.mkdir(mode=0o700)
    select = "+".join(f"eq(n,{index})" for index in indices)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-xerror",
        "-copyts",
        "-i",
        str(video),
        "-map",
        "0:v:0",
        "-vf",
        f"select='{select}',showinfo",
        "-fps_mode",
        "passthrough",
        "-start_number",
        "0",
        str(directory / "%06d.png"),
    ]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode:
        raise InvalidVideo
    log = result.stderr.decode("utf-8", errors="replace")
    bases = re.findall(r"config in time_base: (\d+/\d+)", log)
    observed = [
        int(value) for value in re.findall(r"\bn:\s*\d+\s+pts:\s*(-?\d+)", log)
    ]
    if (
        len(bases) != 1
        or Fraction(bases[0]) != base
        or observed != [pts[index] for index in indices]
        or len(list(directory.glob("*.png"))) != len(indices)
    ):
        raise UnsafeTiming
    names = {index: f"{number:06d}.png" for number, index in enumerate(indices)}
    frames = []
    for second, index in requested:
        frames.append(
            {
                "requested_seconds": second,
                "decoded_frame_index": index,
                "decoded_pts": pts[index],
                "image": f"{directory.name}/{names[index]}",
                "image_sha256": _hash_file(directory / names[index]),
                "phase_suggestion": None,
                "suggestion_status": "unavailable_unaligned",
                "suggestion_shown": False,
                "phase_context_shown": False,
                "human_label": None,
                "review_status": "unreviewed",
            }
        )
    return {
        "time_base": str(base),
        "start_pts": start,
        "duration_ticks": duration,
        "frames": frames,
    }


def prepare(data_dir: Path, output: Path) -> dict:
    """Build a private pilot and return counts without logging run IDs."""
    data_dir = data_dir.resolve(strict=True)
    output = output.absolute()
    parent = data_dir / "labeling"
    if output.parent != parent or output.name in ("", ".", ".."):
        raise ValueError("output must be a direct child of data/labeling")
    os.umask(0o077)
    parent.mkdir(mode=0o700, exist_ok=True)
    if parent.is_symlink() or parent.stat().st_mode & 0o077:
        raise ValueError("labeling directory must be private and not a symlink")
    if shutil.disk_usage(parent).free < MIN_FREE_BYTES:
        raise ValueError("insufficient free disk space")
    chosen = _select_runs(data_dir / "raw")
    output.mkdir(mode=0o700)
    try:
        runs = []
        classes = {"unclear"}
        for ordinal, (_, run, path, raw_hash) in enumerate(chosen):
            if _hash_file(path) != raw_hash:
                raise ValueError("raw source changed during preparation")
            phases = run.get("phases") or []
            if not isinstance(phases, list):
                raise ValueError("invalid phase list")
            for phase in phases:
                if not isinstance(phase, dict):
                    raise ValueError("invalid phase entry")
                name = phase.get("phase")
                if isinstance(name, str) and name:
                    classes.add(name)
            entry = {
                "run_id": run["id"],
                "ts_start": run["tsStart"],
                "raw_path": str(path.relative_to(data_dir)),
                "raw_sha256": raw_hash,
                "phase_sequence": phases,
                "phase_alignment": "unverified",
            }
            video = data_dir / "raw" / "media" / str(run["id"]) / "video.mp4"
            if not video.is_file():
                entry["status"] = "missing_video"
            else:
                entry["video_path"] = str(video.relative_to(data_dir))
                entry["video_sha256"] = _hash_file(video)
                try:
                    entry.update(_extract(video, output / f"run-{ordinal:03d}"))
                    entry["status"] = "ok"
                except InvalidVideo:
                    shutil.rmtree(
                        output / f"run-{ordinal:03d}", ignore_errors=True
                    )
                    entry["status"] = "invalid_video"
                if _hash_file(video) != entry["video_sha256"]:
                    raise ValueError("video source changed during preparation")
            if _hash_file(path) != raw_hash:
                raise ValueError("raw source changed during preparation")
            runs.append(entry)
            if shutil.disk_usage(parent).free < MIN_FREE_BYTES:
                raise ValueError("insufficient free disk space")
        manifest = {
            "version": 1,
            "selection": (
                "50 newest locally cached runs by tsStart; "
                "ties by id descending"
            ),
            "sampling": (
                "frame displayed at 0,5,10,... seconds before stream end"
            ),
            "classes": sorted(classes),
            "runs": runs,
        }
        temporary = output / ".manifest.tmp"
        temporary.write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n"
        )
        os.replace(temporary, output / "manifest.json")
        return {
            "selected": len(runs),
            "ok": sum(run["status"] == "ok" for run in runs),
            "missing_video": sum(
                run["status"] == "missing_video" for run in runs
            ),
            "invalid_video": sum(
                run["status"] == "invalid_video" for run in runs
            ),
            "frames": sum(len(run.get("frames", [])) for run in runs),
        }
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise


def main() -> int:
    """Run the local frame preparation command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        summary = prepare(args.data_dir, args.output)
    except (OSError, ValueError, KeyError, TypeError, UnsafeTiming) as exc:
        print(
            f"pilot preparation failed ({type(exc).__name__})", file=sys.stderr
        )
        return 1
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
