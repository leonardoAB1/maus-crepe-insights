"""Display-only inferred preparation highlights; never classifier inputs."""

INITIAL_PREPARATION_PHASES = {"init", "prepare", "check_serving_plate"}
INITIAL_CLEANING_LABEL = "initial cleaning/prep (inferred)"


def initial_cleaning_end(phases: list[tuple[float, str]]) -> float:
    """End of an initial preparation prefix, or zero if no such prefix exists.

    The logs do not promise a cleaning action. Highlight preparation with
    explicit inferred provenance instead of fabricating a logged event.
    """
    ordered = sorted(phases)
    if not ordered or ordered[0][1] not in INITIAL_PREPARATION_PHASES:
        return 0.0
    for timestamp, name in ordered:
        if name not in INITIAL_PREPARATION_PHASES:
            return max(timestamp, 0.0)
    return 0.0
