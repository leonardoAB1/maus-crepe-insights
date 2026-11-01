"""Both fold pipelines retain distinct CLI and package interfaces."""

import subprocess
import sys
from pathlib import Path

import pytest

from maus_export.experimental_fold.fold_dataset import CALIBRATION_PATH

CLI_TIMEOUT_SECONDS = 30


@pytest.mark.parametrize(
    "command",
    [
        "fold-frames",
        "fold-segment",
        "fold-eval",
        "fold-live",
        "fold-dashboard",
        "fold-experimental-frames",
        "fold-experimental-segment",
        "fold-experimental-eval",
        "fold-experimental-live",
        "fold-experimental-dashboard",
        "fold-watch",
        "fold-live-eval",
        "fold-review",
        "view",
    ],
)
def test_fold_commands_coexist(command: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "maus_export.cli", command, "--help"],
        capture_output=True,
        text=True,
        timeout=CLI_TIMEOUT_SECONDS,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert f"maus-export {command}" in result.stdout


def test_experimental_calibration_path_survives_package_move() -> None:
    root = Path(__file__).resolve().parents[2]
    assert CALIBRATION_PATH == root / "robot-image-calibration.json"
    assert CALIBRATION_PATH.is_file()
