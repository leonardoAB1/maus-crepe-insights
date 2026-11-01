"""Minimal detection entry point does not load export or ML dependencies."""

import subprocess
import sys
from pathlib import Path

import pytest

from maus_export.fold_runtime import main


def test_video_imports_without_export_dependencies() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
for name in ('pandas', 'pyarrow', 'mcp', 'httpx', 'httpx2', 'onnxruntime'):
    sys.modules[name] = None
import maus_export.fold_video
import maus_export.fold_dashboard
""",
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr


def test_analyze_only_does_not_start_website(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    monkeypatch.setattr(
        "maus_export.fold_video.analyse_video",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    main(["1", "1", "--data-dir", str(tmp_path), "--analyze-only"])
    assert len(calls) == 1
    assert calls[0][1]["analysis_fps"] == 10


def test_website_only_does_not_load_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    monkeypatch.setattr(
        "maus_export.fold_dashboard.serve_dashboard",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    main(["--data-dir", str(tmp_path), "--port", "8767"])
    assert calls == [((tmp_path, 8767), {"host": "127.0.0.1"})]
    main(["--data-dir", str(tmp_path), "--host", "0.0.0.0", "--port", "80"])
    assert calls[-1] == ((tmp_path, 80), {"host": "0.0.0.0"})


def test_invalid_runtime_arguments(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["--data-dir", str(tmp_path), "--threads", "0"])
    with pytest.raises(SystemExit):
        main(["--data-dir", str(tmp_path), "--analyze-only"])
