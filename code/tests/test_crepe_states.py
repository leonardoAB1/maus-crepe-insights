"""Offline checks of the unified crepe-state vocabulary."""

import pytest

from maus_export.crepe_states import (
    COOKING,
    EMPTY_PLATE,
    FLAT,
    HALF_FOLD,
    LEGACY_ALIASES,
    MODEL_STATE_MAP,
    QUARTER_FOLD,
    STATE_MEANINGS,
    STATE_TAXONOMY_VERSION,
    UGLY,
    UNIFIED_STATES,
    UnknownState,
    comparable_states,
    is_known,
    normalise,
    normalise_all,
    state_table,
)


def test_unified_states_are_the_agreed_six() -> None:
    assert UNIFIED_STATES == (
        EMPTY_PLATE,
        FLAT,
        COOKING,
        HALF_FOLD,
        QUARTER_FOLD,
        UGLY,
    )


def test_legacy_pilot_names_map_onto_the_unified_ones() -> None:
    assert LEGACY_ALIASES == {"spreading": FLAT, "circle": COOKING}
    assert normalise("spreading") == "flat"
    assert normalise("circle") == "cooking"


def test_model_labels_map_onto_the_unified_ones() -> None:
    assert normalise("empty") == "emptyplate"
    assert normalise("half") == "halffold"
    assert normalise("quarter") == "quarter fold"
    assert comparable_states() == (
        "emptyplate",
        "flat",
        "halffold",
        "quarter fold",
    )


def test_unified_names_pass_through_unchanged() -> None:
    for state in UNIFIED_STATES:
        assert normalise(state) == state
        assert is_known(state)


def test_unknown_label_is_rejected() -> None:
    assert not is_known("banana")
    with pytest.raises(UnknownState):
        normalise("banana")
    with pytest.raises(UnknownState):
        normalise_all(["flat", "banana"])


def test_every_state_carries_a_meaning() -> None:
    assert set(STATE_MEANINGS) == set(UNIFIED_STATES)
    for meaning in STATE_MEANINGS.values():
        assert meaning.strip()


def test_state_table_is_self_describing() -> None:
    table = state_table()
    assert table["version"] == STATE_TAXONOMY_VERSION
    assert table["states"] == list(UNIFIED_STATES)
    assert table["legacy_aliases"] == LEGACY_ALIASES
    assert table["model_labels"] == {"fold-class": MODEL_STATE_MAP}
