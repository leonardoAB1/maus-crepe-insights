"""Tests for the chronological, run-grouped holdout split."""

import pandas as pd
import pytest

from maus_classify.split import TEST, TRAIN, holdout_split


def starts(*times: str, ids=None) -> pd.Series:
    index = ids or list(range(1, len(times) + 1))
    return pd.Series(pd.to_datetime(list(times), utc=True), index=index)


def test_latest_runs_are_test() -> None:
    s = starts(*[f"2026-10-{d:02d}T10:00:00Z" for d in range(1, 11)])
    split = holdout_split(s, test_frac=0.2)
    assert split.tolist() == [TRAIN] * 8 + [TEST] * 2


def test_order_follows_time_not_id() -> None:
    s = starts(
        "2026-10-05T00:00:00Z",
        "2026-10-01T00:00:00Z",
        "2026-10-03T00:00:00Z",
        ids=[1, 2, 3],
    )
    assert holdout_split(s, test_frac=0.3).to_dict() == {
        1: TEST,
        2: TRAIN,
        3: TRAIN,
    }


def test_tied_starts_stay_together() -> None:
    s = starts(
        "2026-10-01T00:00:00Z",
        "2026-10-02T00:00:00Z",
        "2026-10-03T00:00:00Z",
        "2026-10-03T00:00:00Z",
    )
    split = holdout_split(s, test_frac=0.25)
    assert split.tolist() == [TRAIN, TRAIN, TEST, TEST]


def test_empty_and_bad_input() -> None:
    assert holdout_split(starts()).empty
    with pytest.raises(ValueError):
        holdout_split(starts("2026-10-01T00:00:00Z"), test_frac=1.0)
    with pytest.raises(ValueError):
        holdout_split(
            starts("2026-10-01T00:00:00Z", "2026-10-02T00:00:00Z", ids=[1, 1])
        )


def robots(*names: str, ids=None) -> pd.Series:
    return pd.Series(list(names), index=ids or list(range(1, len(names) + 1)))


def test_robot_split_puts_every_robot_in_test() -> None:
    from maus_classify.split import robot_split

    # quin runs first, then pixi: a global cutoff would test pixi only
    s = starts(*[f"2026-10-{d:02d}T10:00:00Z" for d in range(1, 21)])
    r = robots(*(["quin"] * 10 + ["pixi"] * 10))
    split = robot_split(s, r, test_frac=0.2)
    for name in ("quin", "pixi"):
        part = split[r == name]
        assert part.tolist() == [TRAIN] * 8 + [TEST] * 2


def test_robot_split_keeps_small_robots_in_train() -> None:
    from maus_classify.split import robot_split

    s = starts(*[f"2026-10-{d:02d}T10:00:00Z" for d in range(1, 9)])
    r = robots(*(["quin"] * 5 + ["otto"] * 3))
    split = robot_split(s, r, test_frac=0.2, min_runs=5)
    assert (split[r == "otto"] == TRAIN).all()
    assert (split[r == "quin"] == TEST).sum() == 1


def test_robot_split_missing_robot_counts_as_unknown() -> None:
    from maus_classify.split import robot_split

    s = starts(*[f"2026-10-{d:02d}T10:00:00Z" for d in range(1, 6)])
    r = pd.Series([None] * 5, index=s.index, dtype=object)
    assert (robot_split(s, r, min_runs=5) == TEST).sum() == 1
    with pytest.raises(ValueError):
        robot_split(s, r.iloc[:4])
