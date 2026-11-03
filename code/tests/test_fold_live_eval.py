"""Synthetic stratified sampling and live check scoring."""

import json
from pathlib import Path

import pandas as pd
import pytest

from maus_export.webserve_fold.fold_live_eval import evaluate_live, sample_runs


def test_sample_is_balanced_reproducible_and_strict() -> None:
    candidates = [{"id": i, "robot": "a" if i % 2 else "b"} for i in range(8)]
    first = sample_runs(candidates, 4, 42)
    assert first == sample_runs(list(reversed(candidates)), 4, 42)
    assert sum(item["robot"] == "a" for item in first) == 2
    with pytest.raises(ValueError):
        sample_runs(candidates, 9, 42)


def test_live_scoring_uses_check_synchronized_observations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "fold_segmentation"
    runs = tmp_path / "raw" / "runs"
    root.mkdir()
    runs.mkdir(parents=True)
    pd.DataFrame({"run_id": [1, 2]}).to_parquet(root / "frames.parquet")
    for run_id in (1, 2):
        messages = []
        for name, result in (
            ("plate_clean", "hotplate clean"),
            ("spread", "crepe spread successfully"),
            ("fold1", "first fold completed"),
            ("fold2", "second fold completed"),
        ):
            messages.extend(
                [
                    {
                        "timestamp": "2026-01-01T10:00:02",
                        "msg": f"check_{name} started",
                    },
                    {
                        "timestamp": "2026-01-01T10:00:02",
                        "msg": f"check_{name}: "
                        + (
                            "failed"
                            if run_id == 2 and name == "spread"
                            else result
                        ),
                    },
                ]
            )
        run = {
            "machineName": "synthetic",
            "terminal_status": "succeeded",
            "logs": messages,
            "phases": [{"timestamp": "2026-01-01T10:00:00", "elapsed_sec": 0}],
        }
        (runs / f"{run_id}.json").write_text(json.dumps(run))
        live = root / "live" / str(run_id)
        live.mkdir(parents=True)
        pd.DataFrame({"video_t": [2.5, 2.6, 2.7]}).to_csv(
            live / "timeline.csv", index=False
        )
        pd.DataFrame(
            [
                {
                    "check": name,
                    "robot_verdict": "unknown",
                    "expected_state": state,
                    "observed_at": 2.6,
                    "observation": "half",
                    "status": "observed",
                }
                for name, state in (
                    ("plate_clean", "empty"),
                    ("spread", "flat"),
                    ("fold1", "half"),
                    ("fold2", "quarter"),
                )
            ]
        ).to_csv(live / "observations.csv", index=False)
        (live / "summary.json").write_text(
            json.dumps(
                {
                    "frames": 10,
                    "dispense": {
                        "frames": 20,
                        "first_fraction": 0.1,
                        "peak_fraction": 0.7,
                        "final_fraction": 0.68 if run_id == 1 else 0.3,
                        "ok": run_id == 1,
                    },
                }
            )
        )
    monkeypatch.setattr(
        "maus_export.webserve_fold.fold_live_eval.SAMPLE_PER_GROUP", 1
    )
    result = evaluate_live(tmp_path)
    assert result == {"runs": 2, "checks": 8, "available": 8}
    table = pd.read_parquet(root / "live_checks.parquet")
    assert set(table.predicted_state) == {"half"}
    report = (root / "live_report.md").read_text()
    assert "Dispensing: 1/2 sampled runs" in report
