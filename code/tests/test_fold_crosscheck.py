"""Fold-class vs segmentation cross-check summary on synthetic rows."""

from __future__ import annotations

import json

import pandas as pd
import pytest

pytest.importorskip("onnxruntime")

from maus_classify.fold_crosscheck import (  # noqa: E402
    OURS,
    STATES,
    THEIRS,
    _confusion,
    summarize,
)
from maus_classify.split import TEST, TRAIN  # noqa: E402

PASSED = [
    # check, robot, split, ours, theirs
    ("plate_clean", "robot-a", TEST, "empty", "empty"),
    ("spread", "robot-a", TEST, "flat", "flat"),
    ("fold1", "robot-a", TEST, "half", "half"),
    ("fold2", "robot-b", TEST, "quarter", "half"),
    ("fold2", "robot-b", TRAIN, "quarter", "quarter"),
]
# A failed fold2 that left the crepe half: a check-reading head says quarter
FAILED = [
    ("fold2", "robot-a", TEST, "quarter", "half"),
    ("fold2", "robot-b", TRAIN, "quarter", "half"),
    ("fold1", "robot-a", TRAIN, "flat", "flat"),
]


def rows() -> pd.DataFrame:
    out = []
    for ok, spec in ((True, PASSED), (False, FAILED)):
        for i, (check, robot, split, ours, theirs) in enumerate(spec):
            out.append(
                {
                    "run_id": i if ok else 100 + i,
                    "check": check,
                    "robot": robot,
                    "split": split,
                    "ok": ok,
                    OURS: ours,
                    THEIRS: theirs,
                }
            )
    return pd.DataFrame(out)


def test_confusion_covers_all_states() -> None:
    a = pd.Series(["half", "flat"], name=OURS)
    b = pd.Series(["half", "half"], name=THEIRS)
    c = _confusion(a, b)
    assert c["labels"] == list(STATES)
    assert len(c["matrix"]) == len(STATES)
    half = STATES.index("half")
    assert c["matrix"][half][half] == 1
    assert c["matrix"][STATES.index("flat")][half] == 1
    assert sum(map(sum, c["matrix"])) == 2


def test_summarize_agreement_and_leakage() -> None:
    report = summarize(rows(), vision=None)
    assert report["n_frames"] == len(PASSED) + len(FAILED)
    valid = report["their_validity_on_our_frames"]
    assert valid["n"] == len(PASSED)
    assert valid["agreement"] == pytest.approx(4 / 5)
    assert valid["per_state"]["quarter"] == pytest.approx(1 / 2)
    held = report["agreement"]["holdout"]["passed"]
    assert held == {"n": 4, "agreement": pytest.approx(3 / 4)}
    assert report["agreement"]["all"]["per_robot_passed"]["robot-a"][
        "agreement"
    ] == pytest.approx(1.0)
    leak = report["leakage"]
    assert leak["fold2"]["n"] == 2
    assert leak["fold2"]["agreement"] == 0.0
    assert leak["fold2_failed_theirs_half"]["ours"]["quarter"] == 1.0
    assert "fold2_no_change" not in leak  # no vision table given


def test_summarize_has_no_run_ids() -> None:
    text = json.dumps(summarize(rows(), vision=None))
    assert "run_id" not in text
    assert "100" not in text
