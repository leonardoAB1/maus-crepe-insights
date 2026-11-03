"""Supervise dough dispensing from per-frame video measurements.

The robot spreads batter while the plate rotates, so the dough blob
changes shape and position during the ``spread_batter`` phase. Only
coverage of the plate is rotation-invariant, so this module judges the
dispensing from the frame plate fractions: the dough must grow to a
substantial share of the plate and must not shrink again once spread.
The thresholds are provisional supervisor limits, not fitted ground
truth.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

MIN_FINAL_FRACTION = 0.4
STABLE_FINAL_RATIO = 0.7
SETTLE_SECONDS = 1.0


@dataclass(frozen=True)
class DispenseSummary:
    """Aggregated dough coverage over one dispensing phase."""

    frames: int
    first_fraction: float | None
    peak_fraction: float | None
    final_fraction: float | None
    ok: bool | None


def _validate(samples: list[tuple[float, float]]) -> None:
    for t, fraction in samples:
        if not math.isfinite(t) or t < 0:
            raise ValueError("Sample time must be finite and nonnegative")
        if not math.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("Plate fraction must be finite in [0, 1]")


def summarise_dispense(
    samples: Iterable[tuple[float, float]],
    *,
    min_final_fraction: float = MIN_FINAL_FRACTION,
    stable_final_ratio: float = STABLE_FINAL_RATIO,
    settle_seconds: float = SETTLE_SECONDS,
) -> DispenseSummary:
    """Summarize (time, plate fraction) samples of one dispensing phase.

    The final coverage is the median of the last ``settle_seconds`` so a
    single outlier frame does not decide the verdict. ``ok`` is None when
    no coverage was measured, otherwise it requires the final coverage to
    reach ``min_final_fraction`` of the plate and to keep at least
    ``stable_final_ratio`` of the measured peak (the dough must not
    collapse after spreading).
    """
    if (
        not math.isfinite(min_final_fraction)
        or not 0 < min_final_fraction <= 1
        or not math.isfinite(stable_final_ratio)
        or not 0 < stable_final_ratio <= 1
        or not math.isfinite(settle_seconds)
        or settle_seconds <= 0
    ):
        raise ValueError("Invalid dispensing limits")
    samples = list(samples)
    _validate(samples)
    if not samples:
        return DispenseSummary(0, None, None, None, None)
    ordered = sorted(samples)
    fractions = [fraction for _, fraction in ordered]
    peak = max(fractions)
    cutoff = ordered[-1][0] - settle_seconds
    tail = [fraction for t, fraction in ordered if t >= cutoff] or fractions[
        -1:
    ]
    final = float(sorted(tail)[len(tail) // 2])
    stable = final >= stable_final_ratio * peak
    return DispenseSummary(
        len(ordered),
        fractions[0],
        peak,
        final,
        final >= min_final_fraction and stable,
    )
