"""Synthetic causal averaging and recovery from missing measurements."""

import numpy as np

from maus_export.fold_temporal import TemporalMask


def test_one_frame_reflection_is_suppressed() -> None:
    average = TemporalMask()
    empty = np.zeros((20, 20), np.uint8)
    food = empty.copy()
    food[4:16, 4:16] = 255
    assert average.update(food) is None
    assert average.update(food) is None
    transient = food.copy()
    transient[0:3, 0:3] = 255
    result = average.update(transient)
    assert result is not None
    assert np.array_equal(result, food)
    assert np.array_equal(average.update(food), food)


def test_majority_transition_is_causal_and_reset_drops_old_food() -> None:
    average = TemporalMask()
    food = np.full((20, 20), 255, np.uint8)
    empty = np.zeros_like(food)
    for _ in range(5):
        average.update(food)
    assert average.update(empty).all()
    assert average.update(empty).all()
    assert not average.update(empty).any()
    average.reset()
    assert average.update(empty) is None
    assert average.update(empty) is None
    assert not average.update(empty).any()


def test_shape_change_requires_new_warmup() -> None:
    average = TemporalMask()
    for _ in range(3):
        average.update(np.ones((20, 20), np.uint8))
    assert average.update(np.zeros((10, 10), np.uint8)) is None
