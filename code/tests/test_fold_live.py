"""Synthetic temporal filter and forward transition tests."""

import pytest

from maus_export.fold_live import (
    EMPTY_HOLD_FRAMES,
    VOTE_WINDOW,
    FoldStateMachine,
)


def test_transitions_skip_folds_but_require_flat_after_empty() -> None:
    machine = FoldStateMachine()
    for _ in range(VOTE_WINDOW):
        assert machine.update("quarter") == "empty"
    for _ in range(VOTE_WINDOW):
        machine.update("flat")
    assert machine.state == "flat"
    for _ in range(VOTE_WINDOW):
        machine.update("quarter")
    assert machine.state == "quarter"
    for _ in range(VOTE_WINDOW):
        machine.update("half")
    assert machine.state == "quarter"


def test_empty_requires_sustained_vote_and_ignores_single_noise() -> None:
    machine = FoldStateMachine()
    for _ in range(VOTE_WINDOW):
        machine.update("flat")
    assert machine.update("empty") == "flat"
    for _ in range(VOTE_WINDOW):
        machine.update("flat")
    for _ in range(EMPTY_HOLD_FRAMES - 1):
        assert machine.update("empty") == "flat"
    for _ in range(VOTE_WINDOW):
        machine.update("empty")
    assert machine.state == "empty"
    with pytest.raises(ValueError):
        machine.update("invalid")  # type: ignore[arg-type]
