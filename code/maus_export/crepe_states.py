"""One crepe-state vocabulary for labels, models and analysis.

The vocabulary names what is physically on the plate, not what the robot was
doing, so a human label and a model prediction can be compared directly.

Two names used by the phase labeling pilot predate this module: `spreading`
means the unified `flat`, and `circle` means the unified `cooking`. `normalise`
accepts the old names, so stored labels stay usable without being rewritten.
"""

from __future__ import annotations

from collections.abc import Iterable

STATE_TAXONOMY_VERSION = "crepe-state-v1"

EMPTY_PLATE = "emptyplate"
FLAT = "flat"
COOKING = "cooking"
HALF_FOLD = "halffold"
QUARTER_FOLD = "quarter fold"
UGLY = "ugly"

# Order of reporting and of the rows and columns of a confusion matrix.
UNIFIED_STATES: tuple[str, ...] = (
    EMPTY_PLATE,
    FLAT,
    COOKING,
    HALF_FOLD,
    QUARTER_FOLD,
    UGLY,
)

STATE_MEANINGS: dict[str, str] = {
    EMPTY_PLATE: "bare plate, no crepe material in view",
    FLAT: "unfolded batter or crepe spread flat on the plate",
    COOKING: "flat crepe past spreading, browning or bubbling, not folded",
    HALF_FOLD: "crepe folded once, into a half",
    QUARTER_FOLD: "crepe folded twice, into quarters",
    UGLY: "visible crepe whose shape or condition failed the run",
}

# Names stored by earlier labeling rounds; the meaning is unchanged.
LEGACY_ALIASES: dict[str, str] = {"spreading": FLAT, "circle": COOKING}

# Labels of the fold-class head, mapped onto this vocabulary.
MODEL_STATE_SOURCE = "fold-class"
MODEL_STATE_MAP: dict[str, str] = {
    "empty": EMPTY_PLATE,
    "flat": FLAT,
    "half": HALF_FOLD,
    "quarter": QUARTER_FOLD,
}


class UnknownState(ValueError):
    """A label that is not part of the unified vocabulary."""


def is_known(label: str) -> bool:
    """True for a unified name, a legacy alias or a model label."""
    return (
        label in UNIFIED_STATES
        or label in LEGACY_ALIASES
        or label in MODEL_STATE_MAP
    )


def normalise(label: str) -> str:
    """Map a unified, legacy or model label onto the unified vocabulary."""
    if label in UNIFIED_STATES:
        return label
    if label in LEGACY_ALIASES:
        return LEGACY_ALIASES[label]
    if label in MODEL_STATE_MAP:
        return MODEL_STATE_MAP[label]
    raise UnknownState(f"unknown crepe state: {label!r}")


def normalise_all(labels: Iterable[str]) -> list[str]:
    """Map every label, failing on the first unknown one."""
    return [normalise(label) for label in labels]


def comparable_states() -> tuple[str, ...]:
    """Unified states that a fold-class prediction can express."""
    return tuple(dict.fromkeys(MODEL_STATE_MAP.values()))


def state_table() -> dict[str, object]:
    """The vocabulary as data, for reports, manifests and model cards."""
    return {
        "version": STATE_TAXONOMY_VERSION,
        "states": list(UNIFIED_STATES),
        "meanings": dict(STATE_MEANINGS),
        "legacy_aliases": dict(LEGACY_ALIASES),
        "model_labels": {MODEL_STATE_SOURCE: dict(MODEL_STATE_MAP)},
    }
