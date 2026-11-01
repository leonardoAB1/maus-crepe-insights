"""Thin offline entry point shared by desktop and minimal Pi launchers."""

from pathlib import Path

import pytest

from maus_export.experimental_fold import fold_batch


@pytest.mark.parametrize("failed", [0, 1])
def test_batch_entry_forwards_options_and_exit_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed: int
) -> None:
    calls = []

    def run(data_dir: Path, **options: object) -> dict[str, int]:
        calls.append((data_dir, options))
        return {"failed": failed}

    monkeypatch.setattr(fold_batch, "run_batch", run)
    args = [
        "--data-dir",
        str(tmp_path),
        "--threads",
        "4",
        "--fps",
        "2",
        "--model",
        "classical",
        "--ids",
        "1",
        "2",
        "--replace",
    ]
    if failed:
        with pytest.raises(SystemExit) as error:
            fold_batch.main(args)
        assert error.value.code == 1
    else:
        fold_batch.main(args)
    assert calls == [
        (
            tmp_path,
            {
                "threads": 4,
                "analysis_fps": 2.0,
                "model": "classical",
                "run_ids": [1, 2],
                "replace": True,
            },
        )
    ]


def test_batch_entry_reports_invalid_data_directory(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as error:
        fold_batch.main(["--data-dir", str(tmp_path)])
    assert error.value.code == 2
