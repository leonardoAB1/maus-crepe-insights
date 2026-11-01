"""The CLI surface of the state and deviation pipeline."""

import json
import sys
from pathlib import Path

from maus_export import cli
from maus_export.label_deviation import DEFAULT_SCORE_THRESHOLD, SPACES


def test_cli_space_names_match_the_module() -> None:
    assert tuple(cli.LABEL_SPACES) == tuple(SPACES)


def test_cli_score_threshold_matches_the_module() -> None:
    assert cli.DEFAULT_SCORE_THRESHOLD == DEFAULT_SCORE_THRESHOLD


def test_label_states_prints_and_writes_the_vocabulary(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    target = tmp_path / "states.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["maus-export", "label-states", "--out", str(target)],
    )
    cli.main()
    printed = json.loads(capsys.readouterr().out)
    assert printed["states"] == [
        "emptyplate",
        "flat",
        "cooking",
        "halffold",
        "quarter fold",
        "ugly",
    ]
    assert json.loads(target.read_text()) == printed
