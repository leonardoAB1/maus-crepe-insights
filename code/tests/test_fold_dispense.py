"""Dough-dispensing coverage summaries from video samples."""

import math

import pytest

from maus_export.fold_dispense import (
    MIN_FINAL_FRACTION,
    summarise_dispense,
)


def test_empty_phase_has_no_verdict() -> None:
    summary = summarise_dispense([])
    assert summary.frames == 0
    assert summary.first_fraction is None
    assert summary.peak_fraction is None
    assert summary.final_fraction is None
    assert summary.ok is None


def test_stable_growth_is_ok() -> None:
    samples = [(t, min(0.7, 0.1 + 0.06 * i)) for i, t in enumerate(range(12))]
    summary = summarise_dispense(samples)
    assert summary.frames == 12
    assert summary.first_fraction == pytest.approx(0.1)
    assert summary.peak_fraction == pytest.approx(0.7)
    assert summary.final_fraction == pytest.approx(0.7)
    assert summary.ok is True


def test_small_final_coverage_fails() -> None:
    samples = [(float(i), 0.2) for i in range(10)]
    assert summarise_dispense(samples).ok is False
    assert summarise_dispense(samples, min_final_fraction=0.2).ok is True


def test_collapse_after_spreading_fails() -> None:
    samples = [
        (float(i), f) for i, f in enumerate([0.6, 0.7, 0.2, 0.19, 0.2, 0.2])
    ]
    summary = summarise_dispense(samples, settle_seconds=2.0)
    assert summary.peak_fraction == pytest.approx(0.7)
    assert summary.final_fraction == pytest.approx(0.2)
    assert summary.ok is False


def test_final_is_the_median_of_the_settle_window() -> None:
    samples = [(float(i), f) for i, f in enumerate([0.7, 0.7, 0.1, 0.7])]
    summary = summarise_dispense(samples, settle_seconds=2.0)
    # Window [2, 3] covers 0.1 and 0.7; the median is the upper middle.
    assert summary.final_fraction == pytest.approx(0.7)


def test_samples_are_sorted_before_summarising() -> None:
    samples = [(3.0, 0.7), (0.0, 0.1), (1.0, 0.2), (2.0, 0.3)]
    summary = summarise_dispense(samples)
    assert summary.first_fraction == pytest.approx(0.1)
    assert summary.final_fraction == pytest.approx(0.7)


@pytest.mark.parametrize(
    "samples",
    [
        [(-1.0, 0.5)],
        [(float("nan"), 0.5)],
        [(0.0, 1.5)],
        [(0.0, float("inf"))],
        [(0.0, -0.1)],
    ],
)
def test_rejects_invalid_samples(samples: list[tuple[float, float]]) -> None:
    with pytest.raises(ValueError):
        summarise_dispense(samples)


@pytest.mark.parametrize(
    ("min_final", "stable_ratio", "settle"),
    [
        (0.0, 1.0, 1.0),
        (1.5, 1.0, 1.0),
        (MIN_FINAL_FRACTION, 0.0, 1.0),
        (MIN_FINAL_FRACTION, 1.0, 0.0),
        (MIN_FINAL_FRACTION, math.nan, 1.0),
    ],
)
def test_rejects_invalid_limits(
    min_final: float, stable_ratio: float, settle: float
) -> None:
    with pytest.raises(ValueError):
        summarise_dispense(
            [(0.0, 0.5)],
            min_final_fraction=min_final,
            stable_final_ratio=stable_ratio,
            settle_seconds=settle,
        )
