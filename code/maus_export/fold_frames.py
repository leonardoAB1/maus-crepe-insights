"""Clock alignment and check-event parsing for fold analysis.

The dashboard phase and log timestamps share a naive local clock. These
functions deliberately do not interpret that clock as UTC: only differences
between timestamps on the same clock are used for video alignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

PHASE_PRECISION = timedelta(seconds=1)
CHECK_DELAY = timedelta(seconds=0.1)
CHECKS = {
    "plate_clean": ("empty", "hotplate clean"),
    "spread": ("flat", "crepe spread successfully"),
    "fold1": ("half", "first fold completed"),
    "fold2": ("quarter", "second fold completed"),
}


@dataclass(frozen=True)
class CheckEvent:
    """A completed robot check and the time at which it started."""

    name: str
    state: str
    timestamp: datetime
    check_ok: bool


def _local_time(value: str) -> datetime:
    timestamp = datetime.fromisoformat(value)
    if timestamp.tzinfo is not None:
        raise ValueError("Expected a naive local timestamp")
    return timestamp


def phase_clock_origin(phases: list[dict[str, Any]]) -> datetime:
    """Intersect one-second phase clock bounds and return their midpoint.

    A missing or inconsistent phase sequence cannot safely be aligned and is
    rejected instead of falling back to an unrelated clock.
    """
    if not phases:
        raise ValueError("At least one phase is required")
    bounds = [
        _local_time(phase["timestamp"])
        - timedelta(seconds=float(phase["elapsed_sec"]))
        for phase in phases
    ]
    lower = max(bounds)
    upper = min(bound + PHASE_PRECISION for bound in bounds)
    if lower >= upper:
        raise ValueError("Phase clock bounds do not overlap")
    return lower + (upper - lower) / 2


def check_events(logs: list[dict[str, Any]]) -> list[CheckEvent]:
    """Match each started check to its next result; omit unfinished checks."""
    pending: dict[str, datetime] = {}
    events: list[CheckEvent] = []
    for log in logs:
        message = log["msg"]
        for name, (state, success) in CHECKS.items():
            prefix = f"check_{name}"
            if message.startswith(f"{prefix} started"):
                pending[name] = _local_time(log["timestamp"])
            elif message.startswith(f"{prefix}:") and name in pending:
                result = message[len(prefix) + 1 :].strip()
                events.append(
                    CheckEvent(
                        name,
                        state,
                        pending.pop(name),
                        result.startswith(success),
                    )
                )
    return events


def video_time(event: CheckEvent, origin: datetime) -> float:
    """Return the check start in video seconds with the specified delay."""
    return (event.timestamp - origin + CHECK_DELAY).total_seconds()
