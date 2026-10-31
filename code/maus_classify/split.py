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
