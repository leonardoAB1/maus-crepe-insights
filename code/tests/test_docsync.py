"""Tests for the shared-diagram sync (offline, synthetic files)."""

from pathlib import Path

import pytest

from docsync import REPO_ROOT, UnknownDiagramError, sync

DIAGRAM = "flowchart LR\n    a --> b\n"
EMPTY_BLOCK = "intro\n<!-- diagram: demo -->\n<!-- /diagram -->\nend\n"


def make_repo(tmp_path: Path, readme: str) -> Path:
    diagrams = tmp_path / "documentation" / "diagrams"
    diagrams.mkdir(parents=True)
    (diagrams / "demo.mmd").write_text(DIAGRAM, encoding="utf-8")
    (tmp_path / "readme.md").write_text(readme, encoding="utf-8")
    (tmp_path / "documentation" / "page.md").write_text(
        readme, encoding="utf-8"
    )
    return tmp_path


def test_fills_blocks_in_readme_and_docs(tmp_path: Path) -> None:
    root = make_repo(tmp_path, EMPTY_BLOCK)
    changed = sync(root)
    assert len(changed) == 2
    text = (root / "readme.md").read_text(encoding="utf-8")
    assert "```mermaid\nflowchart LR\n    a --> b\n```\n" in text
    assert text.startswith("intro\n") and text.endswith("end\n")


def test_second_run_changes_nothing(tmp_path: Path) -> None:
    root = make_repo(tmp_path, EMPTY_BLOCK)
    sync(root)
    assert sync(root) == []


def test_check_reports_stale_without_writing(tmp_path: Path) -> None:
    root = make_repo(tmp_path, EMPTY_BLOCK)
    sync(root)
    (root / "documentation" / "diagrams" / "demo.mmd").write_text(
        "flowchart LR\n    a --> c\n", encoding="utf-8"
    )
    stale = sync(root, check=True)
    assert len(stale) == 2
    assert "a --> b" in (root / "readme.md").read_text(encoding="utf-8")


def test_keeps_crlf_line_endings(tmp_path: Path) -> None:
    root = make_repo(tmp_path, "")
    readme = root / "readme.md"
    readme.write_bytes(EMPTY_BLOCK.replace("\n", "\r\n").encode("utf-8"))
    sync(root)
    raw = readme.read_bytes()
    assert raw.count(b"\r\n") == raw.count(b"\n")
    assert b"a --> b\r\n" in raw


def test_unknown_diagram_raises(tmp_path: Path) -> None:
    root = make_repo(tmp_path, "<!-- diagram: missing -->\n<!-- /diagram -->\n")
    with pytest.raises(UnknownDiagramError, match="missing"):
        sync(root)


def test_repo_diagrams_are_in_sync() -> None:
    assert sync(REPO_ROOT, check=True) == [], "run `pixi run docs`"
