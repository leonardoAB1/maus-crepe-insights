"""Temporal voting and state transitions for sequential video inference."""

from __future__ import annotations

from collections import Counter, deque

from .fold_state import FoldState

VOTE_WINDOW = 9
EMPTY_HOLD_FRAMES = 30
STATE_ORDER: tuple[FoldState, ...] = ("empty", "flat", "half", "quarter")


class FoldStateMachine:
    """Filter transient predictions and enforce forward cooking transitions."""

    def __init__(self) -> None:
        self.state: FoldState = "empty"
        self._votes: deque[FoldState] = deque(maxlen=VOTE_WINDOW)
        self._empty_count = 0

    def update(self, prediction: FoldState) -> FoldState:
        """Update with one analysed frame and return the stable state."""
        if prediction not in STATE_ORDER:
            raise ValueError("Invalid fold state")
        self._votes.append(prediction)
        counts = Counter(self._votes)
        highest = max(counts.values())
        tied = {state for state, count in counts.items() if count == highest}
        voted = self.state if self.state in tied else prediction
        if voted not in tied:
            voted = next(
                state for state in reversed(self._votes) if state in tied
            )
        if voted == "empty":
            self._empty_count += 1
            if self._empty_count >= EMPTY_HOLD_FRAMES:
                self.state = "empty"
            return self.state
        self._empty_count = 0
        if self.state == "empty":
            if voted == "flat":
                self.state = "flat"
        elif STATE_ORDER.index(voted) > STATE_ORDER.index(self.state):
            self.state = voted
        return self.state
