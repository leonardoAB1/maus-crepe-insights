"""Convert raw run JSON (+ media) into one ROS 2 MCAP file per run."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from mcap_ros2.writer import Writer

from .msgdefs import SCHEMAS

ZURICH = ZoneInfo("Europe/Zurich")


def _utc(ts: str) -> datetime:
    """Parse an ISO UTC timestamp like 2026-10-30T16:45:37.000Z."""
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)


class Clock:
    """Turns robot-local naive timestamps into UTC.

    Logs/phases/checks are written by the robot in local time without an offset,
    while tsStart is UTC. We measure the offset from the first log line (rounded to
    15 min) and fall back to Europe/Zurich if there are no logs.
    """

    def __init__(self, run: dict[str, Any]):
        self.start = _utc(run["tsStart"])
        self.offset: timedelta | None = None
        logs = run.get("logs") or []
        if logs:
            first = datetime.fromisoformat(logs[0]["timestamp"])
            diff = (first - self.start.replace(tzinfo=None)).total_seconds()
            self.offset = timedelta(minutes=round(diff / 900) * 15)

    def local(self, ts: str) -> datetime:
        naive = datetime.fromisoformat(ts)
        if self.offset is not None:
            return (naive - self.offset).replace(tzinfo=timezone.utc)
        return naive.replace(tzinfo=ZURICH).astimezone(timezone.utc)


_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _ns(dt: datetime) -> int:
    """Exact nanoseconds since epoch (integer math, no float rounding)."""
    return (dt - _EPOCH) // timedelta(microseconds=1) * 1000


def _stamp(dt: datetime) -> dict[str, int]:
    ns = _ns(dt)
    return {"sec": ns // 1_000_000_000, "nanosec": ns % 1_000_000_000}


def _header(dt: datetime, frame_id: str) -> dict[str, Any]:
    return {"stamp": _stamp(dt), "frame_id": frame_id}


def _flag(v: Any) -> int:
    return -1 if v is None else int(bool(v))


def _f(v: Any) -> float:
    return float("nan") if v is None else float(v)


def run_to_mcap(run: dict[str, Any], out: Path, media_dir: Path | None, include_video: bool = False) -> dict[str, int]:
    clock = Clock(run)
    robot = run.get("machineName") or f"machine-{run.get('machineId')}"
    start = clock.start
    end = _utc(run["tsEnd"]) if run.get("tsEnd") else start
    counts: dict[str, int] = {}

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        w = Writer(f)
        schemas = {name: w.register_msgdef(name, text) for name, text in SCHEMAS.items()}

        def write(topic: str, datatype: str, msg: dict[str, Any], t: datetime) -> None:
            w.write_message(topic, schemas[datatype], msg, log_time=_ns(t), sequence=counts.get(topic, 0))
            counts[topic] = counts.get(topic, 0) + 1

        # Run summary at the start of the run
        write(
            "/maus/run_summary",
            "maus_msgs/msg/RunSummary",
            {
                "header": _header(start, robot),
                "run_id": run["id"],
                "machine_id": run.get("machineId") or 0,
                "robot": robot,
                "topping": run.get("topping") or "",
                "is_test": bool(run.get("isTest")),
                "order_id": run.get("orderId") or "",
                "terminal_status": run.get("terminal_status") or run.get("terminalStatus") or "",
                "success_spread": _flag(run.get("successSpread")),
                "success_topping": _flag(run.get("successTopping")),
                "success_fold": _flag(run.get("successFold")),
                "success_handover": _flag(run.get("successHandover")),
                "duration_sec": (end - start).total_seconds(),
                "hotplate_temp": _f(run.get("hotplateTemp")),
                "env_temp_start": _f(run.get("envTempStart")),
                "env_humidity_start": _f(run.get("envHumidityStart")),
                "comments": run.get("comments") or "",
                "video_url": run.get("videoUrl") or "",
            },
            start,
        )

        # Sensors recorded at the start
        if run.get("envTempStart") is not None:
            write("/maus/env/temperature", "sensor_msgs/msg/Temperature",
                  {"header": _header(start, robot), "temperature": float(run["envTempStart"]), "variance": 0.0}, start)
        if run.get("envHumidityStart") is not None:
            write("/maus/env/humidity", "sensor_msgs/msg/RelativeHumidity",
                  {"header": _header(start, robot), "relative_humidity": float(run["envHumidityStart"]) / 100.0,
                   "variance": 0.0}, start)
        if run.get("hotplateTemp") is not None:
            write("/maus/hotplate/setpoint", "sensor_msgs/msg/Temperature",
                  {"header": _header(start, robot), "temperature": float(run["hotplateTemp"]), "variance": 0.0}, start)

        # Machine logs -> /rosout
        for entry in run.get("logs") or []:
            t = clock.local(entry["timestamp"])
            write("/rosout", "rcl_interfaces/msg/Log",
                  {"stamp": _stamp(t), "level": int(entry.get("level") or 20), "name": entry.get("name") or "",
                   "msg": entry.get("msg") or "", "file": "", "function": "", "line": 0}, t)

        # Phases: placed at tsStart + elapsed_sec (sub-second precision)
        for ph in run.get("phases") or []:
            t = start + timedelta(seconds=float(ph.get("elapsed_sec") or 0.0))
            write("/maus/phase", "maus_msgs/msg/Phase",
                  {"stamp": _stamp(t), "robot": robot, "run_id": run["id"], "phase": ph.get("phase") or "",
                   "detail": ph.get("detail") or "", "progress": float(ph.get("progress") or 0.0),
                   "elapsed_sec": float(ph.get("elapsed_sec") or 0.0)}, t)

        # Vision checks (+ their images when downloaded)
        run_media = media_dir / str(run["id"]) if media_dir else None
        for i, vc in enumerate(run.get("visionChecks") or []):
            t = clock.local(vc["timestamp"]) if vc.get("timestamp") else start
            write("/maus/vision_check", "maus_msgs/msg/VisionCheck",
                  {"header": _header(t, robot), "run_id": run["id"], "index": i, "phase": vc.get("phase") or "",
                   "ok": bool(vc.get("ok")), "message": vc.get("message") or "", "errors": list(vc.get("errors") or []),
                   "frame_image": vc.get("frame") or "", "overlay_image": vc.get("overlay") or ""}, t)
            if run_media:
                for key, topic in (("frame", "/maus/camera/check/compressed"),
                                   ("overlay", "/maus/camera/check_overlay/compressed")):
                    img = run_media / (vc.get(key) or "")
                    if vc.get(key) and img.is_file():
                        write(topic, "sensor_msgs/msg/CompressedImage",
                              {"header": _header(t, vc.get("phase") or robot), "format": "jpeg",
                               "data": img.read_bytes()}, t)

        # Metadata records
        w.add_metadata("run", {
            "run_id": str(run["id"]),
            "robot": robot,
            "machine_id": str(run.get("machineId")),
            "topping": str(run.get("topping")),
            "is_test": str(run.get("isTest")),
            "order_id": run.get("orderId") or "",
            "terminal_status": str(run.get("terminal_status")),
            "ts_start": run["tsStart"],
            "ts_end": run.get("tsEnd") or "",
            "local_clock_offset_min": str(int(clock.offset.total_seconds() // 60)) if clock.offset else "zurich",
            "video_url": run.get("videoUrl") or "",
            "prev_id": str(run.get("prevId")),
            "next_id": str(run.get("nextId")),
            "batch": json.dumps(run.get("batch")),
        })
        params = run.get("parameters") or {}
        w.add_metadata("parameters", {k: v if isinstance(v, str) else json.dumps(v) for k, v in params.items()})

        if include_video and run_media and (run_media / "video.mp4").is_file():
            w.add_attachment(0, _ns(start), "video.mp4", "video/mp4", (run_media / "video.mp4").read_bytes())

        w.finish()
    return counts


def convert_all(data_dir: Path, include_video: bool = False, only: list[int] | None = None) -> None:
    runs_dir = data_dir / "raw" / "runs"
    media_dir = data_dir / "raw" / "media"
    out_dir = data_dir / "mcap"
    files = sorted(runs_dir.glob("*.json"), key=lambda p: int(p.stem))
    if only:
        files = [p for p in files if int(p.stem) in set(only)]
    for i, path in enumerate(files, 1):
        run = json.loads(path.read_text(encoding="utf-8"))
        run_to_mcap(run, out_dir / f"run_{run['id']}.mcap", media_dir, include_video)
        if i % 50 == 0 or i == len(files):
            print(f"  {i}/{len(files)} MCAP files")
