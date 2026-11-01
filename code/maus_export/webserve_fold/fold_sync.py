"""Evaluate visual estimates only after robot check-start log events."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .fold_frames import CheckEvent
from .fold_state import FoldState

CHECK_WINDOW_SECONDS = 1.0
MIN_OBSERVATION_FRAMES = 3
TIME_EPSILON = 1e-9


@dataclass
class CheckObservation:
    """A robot-requested check and an independent, unverified visual vote."""

    event: CheckEvent
    start: float
    target: float
    observed_at: float | None = None
    observation: FoldState | None = None
    status: str = "pending"
    votes: list[FoldState] = field(default_factory=list)
    flat_areas: list[float] = field(default_factory=list)


class CheckWindowEvaluator:
    """Collect only post-check-start frames; never infer success from logs."""

    def __init__(self, events: list[tuple[float, CheckEvent]]) -> None:
        self.checks = [
            CheckObservation(event, start, start + CHECK_WINDOW_SECONDS)
            for start, event in events
        ]
        for check in self.checks:
            if check.start < 0:
                check.status = "outside_video"

    def update(
        self,
        timestamp: float,
        prediction: FoldState | None,
        area: float | None = None,
        *,
        flat_reference_available: bool = False,
    ) -> list[CheckObservation]:
        """Return evaluations completed at this frame; ties stay unknown."""
        completed = []
        for check in self.checks:
            if (
                check.status != "pending"
                or timestamp + TIME_EPSILON < check.start
            ):
                continue
            if prediction is not None:
                check.votes.append(prediction)
                if prediction == "flat" and area is not None and area > 0:
                    check.flat_areas.append(area)
            if timestamp + TIME_EPSILON < check.target:
                continue
            check.observed_at = timestamp
            counts = Counter(check.votes)
            if len(check.votes) < MIN_OBSERVATION_FRAMES:
                check.status = "insufficient_frames"
            elif (
                check.event.name in {"fold1", "fold2"}
                and not flat_reference_available
            ):
                check.status = "no_flat_reference"
            else:
                winner, count = counts.most_common(1)[0]
                if count * 2 <= len(check.votes):
                    check.status = "ambiguous"
                else:
                    check.observation = winner
                    check.status = "observed"
            completed.append(check)
        return completed

    def finish(self) -> None:
        """Mark checks that ended before their evaluation point as unknown."""
        for check in self.checks:
            if check.status == "pending":
                check.status = "outside_video"
