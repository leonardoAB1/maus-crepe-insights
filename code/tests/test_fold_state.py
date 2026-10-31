"""Exercise ordering and boundary conditions in the fold-state rule."""

import pytest

from maus_export.fold_state import classify_state


@pytest.mark.parametrize(
    ("area", "edges", "reference", "expected"),
    [
        (0, 0, None, "empty"),
        (49, 0, None, "flat"),
        (48.9, 2, None, "half"),
        (15, 1, 20, "flat"),
        (35, 1, 50, "half"),
        (32.5, 1, 100, "half"),
        (27.9, 1, 100, "half"),
        (20, 1, 50, "half"),
        (15, 1, 100, "quarter"),
        (10, 1, 40, "quarter"),
        (21, 1, None, "half"),
        (20.9, 0, 100, "quarter"),
    ],
)
def test_state_rule(
    area: float, edges: int, reference: float | None, expected: str
) -> None:
    assert classify_state(area, 100, edges, reference) == expected


@pytest.mark.parametrize(
    ("area", "plate", "edges", "reference"),
    [
        (-1, 100, 0, None),
        (0, 0, 0, None),
        (1, 100, -1, None),
        (1, 100, 0, 0),
        (float("nan"), 100, 0, None),
    ],
)
def test_rejects_invalid_inputs(
    area: float, plate: float, edges: int, reference: float | None
) -> None:
    with pytest.raises(ValueError):
        classify_state(area, plate, edges, reference)
