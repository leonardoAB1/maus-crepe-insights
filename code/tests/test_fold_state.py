"""Exercise ordering and boundary conditions in the fold-state rule."""

import pytest

from maus_export.fold_state import classify_state


@pytest.mark.parametrize(
    ("area", "edges", "reference", "aspect", "expected"),
    [
        (0, 0, None, None, "empty"),
        (49, 0, None, None, "flat"),
        (48.9, 2, None, None, "half"),
        (15, 1, 20, None, "flat"),
        (35, 1, 50, None, "half"),
        (32.5, 1, 100, None, "half"),
        (27.9, 1, 100, None, "half"),
        (20, 1, 50, None, "half"),
        (15, 1, 100, None, "quarter"),
        (10, 1, 40, None, "quarter"),
        (21, 1, None, None, "half"),
        (20.9, 0, 100, None, "quarter"),
        (21, 1, None, 1.4, "half"),
        (21, 1, None, 1.39, "quarter"),
        (32.5, 1, 100, 1.2, "quarter"),
    ],
)
def test_state_rule(
    area: float,
    edges: int,
    reference: float | None,
    aspect: float | None,
    expected: str,
) -> None:
    assert classify_state(area, 100, edges, reference, aspect) == expected


@pytest.mark.parametrize(
    ("area", "plate", "edges", "reference", "aspect"),
    [
        (-1, 100, 0, None, None),
        (0, 0, 0, None, None),
        (1, 100, -1, None, None),
        (1, 100, 0, 0, None),
        (float("nan"), 100, 0, None, None),
        (1, 100, 0, None, 0),
        (1, 100, 0, None, float("nan")),
    ],
)
def test_rejects_invalid_inputs(
    area: float,
    plate: float,
    edges: int,
    reference: float | None,
    aspect: float | None,
) -> None:
    with pytest.raises(ValueError):
        classify_state(area, plate, edges, reference, aspect)
