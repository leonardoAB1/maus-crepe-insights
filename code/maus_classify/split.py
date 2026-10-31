"""Chronological, run-grouped holdout split, fixed before any modelling."""

from __future__ import annotations

import math

import pandas as pd

TRAIN = "train"
TEST = "test"
# Share of runs (latest by start time) reserved for the holdout
TEST_FRAC = 0.2


def holdout_split(
    run_starts: pd.Series, test_frac: float = TEST_FRAC
) -> pd.Series:
    """Map run id -> "train" / "test".

    `run_starts` is indexed by run id and holds each run's UTC start time.
    Pass every run that has vision checks, before filtering on verdicts,
    labels or features, so the split never depends on them. The latest
    `test_frac` of runs form the test set. Runs that start at the cutoff
    time all go to test, so tied timestamps never straddle the split.
    """
    if not 0 < test_frac < 1:
        raise ValueError("test_frac must be between 0 and 1")
    if run_starts.index.has_duplicates:
        raise ValueError("run ids must be unique")
    if run_starts.isna().any():
        raise ValueError("every run needs a start time")
    if run_starts.empty:
        return pd.Series([], dtype=object, name="split")
    ordered = run_starts.sort_values(kind="stable")
    n_test = max(1, math.ceil(len(ordered) * test_frac))
    cutoff = ordered.iloc[len(ordered) - n_test]
    split = (run_starts >= cutoff).map({True: TEST, False: TRAIN})
    return split.rename("split")


# Version of the split rule, recorded in frames.parquet and model cards.
# v1: latest TEST_FRAC of all runs. v2: latest TEST_FRAC within each robot.
SPLIT_VERSION = "robot-chrono-v2"
# A robot needs at least this many runs to give any to test
MIN_RUNS_FOR_TEST = 5
UNKNOWN_ROBOT = "unknown"


def robot_split(
    run_starts: pd.Series,
    robots: pd.Series,
    test_frac: float = TEST_FRAC,
    min_runs: int = MIN_RUNS_FOR_TEST,
) -> pd.Series:
    """Map run id -> "train" / "test", chronological within each robot.

    Each robot's latest `test_frac` of runs form its test part, with ties
    kept together as in `holdout_split`, so test mixes every robot and is
    still later data within each one. A robot with fewer than `min_runs`
    runs stays in train. `robots` is indexed like `run_starts`; a missing
    robot counts as UNKNOWN_ROBOT.
    """
    if not run_starts.index.equals(robots.index):
        raise ValueError("run_starts and robots need the same run ids")
    if run_starts.empty:
        return pd.Series([], dtype=object, name="split")
    robots = robots.fillna(UNKNOWN_ROBOT).astype(str)
    parts = []
    for robot in sorted(robots.unique()):
        starts = run_starts[robots == robot]
        if len(starts) < min_runs:
            parts.append(pd.Series(TRAIN, index=starts.index))
        else:
            parts.append(holdout_split(starts, test_frac))
    return pd.concat(parts).reindex(run_starts.index).rename("split")
