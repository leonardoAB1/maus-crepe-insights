"""Robot check timing gates independent visual observations."""

from datetime import datetime

from maus_export.experimental_fold.fold_frames import CheckEvent
from maus_export.experimental_fold.fold_sync import CheckWindowEvaluator


def event(name: str, state: str, verdict: bool | None) -> CheckEvent:
    return CheckEvent(name, state, datetime(2026, 1, 1), verdict)


def test_no_early_transition_and_no_trust_in_robot_verdict() -> None:
    evaluator = CheckWindowEvaluator([(10.0, event("fold1", "half", True))])
    for timestamp in (8.0, 9.0, 9.9):
        assert evaluator.update(timestamp, "half") == []
    for timestamp in (10.0, 10.2, 10.4, 10.6, 10.8):
        assert evaluator.update(timestamp, "flat") == []
    check = evaluator.update(11.0, "flat", flat_reference_available=True)[0]
    assert check.event.state == "half"
    assert check.event.check_ok is True
    assert check.observation == "flat"
    assert check.status == "observed"
    assert check.observed_at == 11.0
    assert check.verdict == "failure"
    assert check.confidence == 1.0


def test_failed_robot_verdict_does_not_override_visual_result() -> None:
    evaluator = CheckWindowEvaluator([(1.0, event("fold2", "quarter", False))])
    for timestamp in (1.0, 1.2, 1.4, 1.6, 1.8):
        evaluator.update(timestamp, "quarter")
    result = evaluator.update(2.0, "quarter", flat_reference_available=True)[0]
    assert result.event.check_ok is False
    assert result.observation == "quarter"
    assert result.verdict == "success"
    assert result.confidence == 1.0


def test_fold_without_flat_reference_is_unknown() -> None:
    evaluator = CheckWindowEvaluator([(1.0, event("fold1", "half", True))])
    for timestamp in (1.0, 1.2, 1.4, 1.6, 1.8):
        evaluator.update(timestamp, "half")
    result = evaluator.update(2.0, "half")[0]
    assert result.status == "no_flat_reference"
    assert result.observation is None
    assert result.verdict == "unknown"
    assert result.confidence == 0.0


def test_confidence_counts_invalid_samples_and_rejects_low_coverage() -> None:
    evaluator = CheckWindowEvaluator([(0.0, event("spread", "flat", False))])
    for time, prediction in (
        (0.0, "flat"),
        (0.2, "flat"),
        (0.4, None),
        (0.6, "quarter"),
        (0.8, "flat"),
        (1.0, "flat"),
    ):
        evaluator.update(time, prediction)
    result = evaluator.checks[0]
    assert result.verdict == "success"
    assert result.confidence == 4 / 6
    missing = CheckWindowEvaluator([(0.0, event("spread", "flat", True))])
    for index in range(11):
        missing.update(index / 10, "flat" if index < 3 else None)
    assert missing.checks[0].verdict == "unknown"
    assert missing.checks[0].confidence == 0.0


def test_ties_missing_frames_and_missing_check_are_unknown() -> None:
    evaluator = CheckWindowEvaluator([(0.0, event("fold1", "half", None))])
    evaluator.update(0.0, "flat")
    evaluator.update(0.2, "half")
    evaluator.update(0.4, "flat")
    evaluator.update(0.6, "half")
    result = evaluator.update(1.0, None, flat_reference_available=True)[0]
    assert result.status == "ambiguous"
    assert result.observation is None
    missing = CheckWindowEvaluator([(2.0, event("fold2", "quarter", True))])
    missing.update(2.0, None)
    assert missing.update(3.0, "quarter")[0].status == "insufficient_frames"
    no_check = CheckWindowEvaluator([])
    assert no_check.update(100.0, "quarter") == []
    incomplete = CheckWindowEvaluator([(4.0, event("fold1", "half", True))])
    incomplete.update(4.5, "half")
    incomplete.finish()
    assert incomplete.checks[0].status == "outside_video"
