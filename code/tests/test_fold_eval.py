"""Offline fold evaluation with synthetic checks."""

from pathlib import Path

import pandas as pd
import pytest

from maus_export.webserve_fold.fold_eval import evaluate, rank_auc


def test_auc_handles_ties_and_missing_classes() -> None:
    assert rank_auc(
        pd.Series([1, 2, 2, 3]), pd.Series([False, False, True, True])
    ) == pytest.approx(0.875)
    assert rank_auc(pd.Series([1, 2]), pd.Series([True, True])) is None


def test_report_lists_classifier_flips(tmp_path: Path) -> None:
    root = tmp_path / "fold_segmentation"
    root.mkdir()
    metrics = {
        "diameter_fraction": 1.0,
        "offset_fraction": 0.1,
        "hole_fraction": 0.0,
        "circularity": 0.8,
        "solidity": 0.9,
        "centre_x": 10.0,
        "edge_angle": None,
        "corner_x": None,
        "plate_fraction": 0.3,
    }
    rows = [
        # rule said half, model flipped to quarter (correct).
        {
            **metrics,
            "robot": "synthetic",
            "status": "ok",
            "check_ok": True,
            "expected_state": "quarter",
            "rule_state": "half",
            "predicted_state": "quarter",
        },
        # rule and model agree.
        {
            **metrics,
            "robot": "synthetic",
            "status": "ok",
            "check_ok": True,
            "expected_state": "half",
            "rule_state": "half",
            "predicted_state": "half",
        },
    ]
    pd.DataFrame(rows).to_parquet(root / "features.parquet", index=False)
    evaluate(tmp_path)
    report = (root / "report.md").read_text()
    assert "Boundary-zone classifier flips" in report
    assert "changed the rule's decision on 1 passed" in report
    assert "1 of those changes were correct" in report
    assert "rule agreement was 50.0%" in report
    assert "final agreement 100.0%" in report


def test_report_without_rule_state_skips_flip_section(tmp_path: Path) -> None:
    root = tmp_path / "fold_segmentation"
    root.mkdir()
    metrics = {
        "diameter_fraction": 1.0,
        "offset_fraction": 0.1,
        "hole_fraction": 0.0,
        "circularity": 0.8,
        "solidity": 0.9,
        "centre_x": 10.0,
        "edge_angle": None,
        "corner_x": None,
        "plate_fraction": 0.3,
    }
    pd.DataFrame(
        [
            {
                **metrics,
                "robot": "synthetic",
                "status": "ok",
                "check_ok": True,
                "expected_state": "half",
                "predicted_state": "half",
            }
        ]
    ).to_parquet(root / "features.parquet", index=False)
    evaluate(tmp_path)
    assert (
        "Boundary-zone classifier flips" not in (root / "report.md").read_text()
    )


def test_evaluation_excludes_failed_detection_and_retains_failed_checks(
    tmp_path: Path,
) -> None:
    root = tmp_path / "fold_segmentation"
    root.mkdir()
    metrics = {
        "diameter_fraction": 1.0,
        "offset_fraction": 0.1,
        "hole_fraction": 0.0,
        "circularity": 0.8,
        "solidity": 0.9,
        "centre_x": 10.0,
        "edge_angle": None,
        "corner_x": None,
    }
    rows = [
        {
            **metrics,
            "robot": "synthetic",
            "status": "ok",
            "check_ok": True,
            "expected_state": "flat",
            "predicted_state": "flat",
        },
        {
            **metrics,
            "robot": "synthetic",
            "status": "ok",
            "check_ok": False,
            "expected_state": "half",
            "predicted_state": "quarter",
        },
        {
            **metrics,
            "robot": "synthetic",
            "status": "plate_not_found",
            "check_ok": True,
            "expected_state": "empty",
            "predicted_state": None,
        },
    ]
    pd.DataFrame(rows).to_parquet(root / "features.parquet", index=False)
    assert evaluate(tmp_path) == {
        "frames": 3,
        "usable": 2,
        "passed": 1,
        "correct": 1,
    }
    fold = pd.read_parquet(root / "evaluation" / "fold_verdicts.parquet")
    assert fold.iloc[0]["check_ok"] == False  # noqa: E712
    assert fold.iloc[0]["our_match"] == False  # noqa: E712
    assert "100.0%" in (root / "report.md").read_text()
    with pytest.raises(FileExistsError):
        evaluate(tmp_path)
