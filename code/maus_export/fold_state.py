"""Area-based fold-state rule, independent of robot check verdicts."""

from __future__ import annotations

import math
from typing import Literal

FoldState = Literal["empty", "flat", "half", "quarter"]
FLAT_PLATE_FRACTION = 0.24
FLAT_REFERENCE_FRACTION = 0.75
FLAT_NO_EDGE_FRACTION = 0.20
HALF_REFERENCE_FRACTION = 0.28
HALF_PLATE_FRACTION = 0.11


def classify_state(
    area: float,
    plate_area: float,
    straight_edges: int,
    flat_area: float | None = None,
) -> FoldState:
    """Classify a measured mask; zero area denotes no detected crepe.

    The caller must not use the frame under evaluation as its flat reference.
    Areas must use the same coordinate system and plate mask.
    """
    if (
        not math.isfinite(area)
        or not math.isfinite(plate_area)
        or area < 0
        or plate_area <= 0
        or straight_edges < 0
        or (
            flat_area is not None
            and (not math.isfinite(flat_area) or flat_area <= 0)
        )
    ):
        raise ValueError("Invalid areas or edge count")
    if area == 0:
        return "empty"
    fraction = area / plate_area
    relative = area / flat_area if flat_area is not None else None
    if fraction >= FLAT_PLATE_FRACTION:
        return "flat"
    if relative is not None and relative >= FLAT_REFERENCE_FRACTION:
        return "flat"
    if straight_edges == 0 and fraction >= FLAT_NO_EDGE_FRACTION:
        return "flat"
    if relative is not None:
        return "half" if relative >= HALF_REFERENCE_FRACTION else "quarter"
    return "half" if fraction >= HALF_PLATE_FRACTION else "quarter"
