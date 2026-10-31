"""Area-based fold-state rule for rectified frames.

The thresholds describe the top-down rectified view, where a full crepe
covers roughly 0.7 of the plate, a half fold about 0.3 and a quarter fold
about 0.17. They were fitted on the local rectified export and need
validation on held-out data before deployment.
"""

from __future__ import annotations

import math
from typing import Literal

FoldState = Literal["empty", "flat", "half", "quarter"]
STATE_ORDER: tuple[FoldState, ...] = ("empty", "flat", "half", "quarter")
FLAT_PLATE_FRACTION = 0.49
FLAT_REFERENCE_FRACTION = 0.75
HALF_REFERENCE_FRACTION = 0.32
HALF_PLATE_FRACTION = 0.21


def classify_state(
    area: float,
    plate_area: float,
    straight_edges: int,
    flat_area: float | None = None,
) -> FoldState:
    """Classify a measured mask; zero area denotes no detected crepe.

    The caller must not use the frame under evaluation as its flat reference.
    Areas must use the same coordinate system and plate mask. Straight edges
    do not influence the decision on rectified frames, where flat crepes
    often expose a fold edge and half folds often expose none.
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
    if relative is not None and relative >= HALF_REFERENCE_FRACTION:
        return "half"
    if fraction >= HALF_PLATE_FRACTION:
        return "half"
    return "quarter"
