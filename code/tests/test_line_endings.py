"""LF line endings: writers emit LF, and git stores no CRLF text."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_dump_json_writes_lf(tmp_path: Path) -> None:
    pytest.importorskip("scipy")  # doe_analyze runs in the ml environment
    from maus_export.doe_analyze import dump_json

    out = tmp_path / "card.json"
    dump_json({"x": 1}, out)
    assert out.read_bytes() == b'{\n  "x": 1\n}\n'


def test_no_crlf_text_in_the_index() -> None:
    if shutil.which("git") is None:
        pytest.skip("git not available")
    res = subprocess.run(
        ["git", "ls-files", "--eol"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent,
    )
    if res.returncode != 0:
        pytest.skip("not a git checkout")
    crlf = [
        line.split("\t")[-1]
        for line in res.stdout.splitlines()
        if line.startswith("i/crlf")
    ]
    assert crlf == [], f"stored with CRLF: {crlf}"
