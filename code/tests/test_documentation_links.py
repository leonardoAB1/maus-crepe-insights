"""Check local documentation links and the original section URLs."""

from __future__ import annotations

import re
import subprocess
import unicodedata
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "documentation"
PAGES = {
    "data-preparation-handoff.md": (
        "training",
        [
            "data-preparation-handoff",
            "what-was-done",
            "meaning-and-limits",
            "reproduce-privately",
        ],
    ),
    "doe.md": (
        "analysis",
        [
            "designed-experiment-doe-batch",
            "the-control-view",
            "what-varies-and-what-does-not",
            "controllability",
            "settings-varied",
            "why-a-definitive-screening-design",
            "run-sheet",
            "analysis",
            "model",
            "next",
        ],
    ),
    "data.md": (
        "data",
        [
            "maus-dashboard-data",
            "access-window",
            "what-is-there",
            "a-crêpe-run-crepe_get",
            "quirks-to-know",
            "more-diagrams",
            "handling",
        ],
    ),
    "mcap.md": (
        "data",
        [
            "mcap-format",
            "topics",
            "reading-in-python",
            "viewing",
        ],
    ),
    "pipeline.md": (
        "data",
        [
            "data-pipeline-and-data-model",
            "pipeline",
            "run-data-model",
            "run-table-columns",
            "parameter-identity-and-compatibility",
            "vision-score-columns",
            "example-figure",
        ],
    ),
    "training-dataset.md": (
        "training",
        [
            "candidate-run-start-training-dataset",
            "build",
            "schema-and-manifest",
            "evaluation-and-limits",
        ],
    ),
    "training-audit.md": (
        "training",
        [
            "training-data-audit",
            "reproduce",
            "report-schema",
            "interpretation-and-privacy",
            "failures-and-troubleshooting",
            "handoff-checklist",
        ],
    ),
    "vision-targets.md": (
        "training",
        [
            "vision-target-training-dataset",
            "reproduce-the-dataset-privately",
            "schema-and-provenance",
            "evaluation",
        ],
    ),
    "analysis.md": (
        "analysis",
        [
            "which-settings-matter-history-analysis",
            "data-used",
            "three-views",
            "shortlist-dataanalysisshortlistjson",
            "caveats",
        ],
    ),
    "system.md": (
        "system",
        [
            "robot-system",
            "hardware-and-control-architecture",
            "software-ros-2-nodes",
            "customer-flow",
        ],
    ),
    "crepe-stages.md": (
        "system",
        [
            "crepe-stages",
            "stage-machine",
            "stage-table",
            "timeline-of-a-full-cycle",
            "where-runs-end",
        ],
    ),
}
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
REFERENCE = re.compile(r"^\s*\[[^\]]+\]:\s*(\S+)")
HTML_LINK = re.compile(r"\b(?:href|src)=[\"']([^\"']+)[\"']")


def prose_lines(path: Path) -> list[str]:
    """Omit fenced code, which may contain non-document links and headings."""
    result = []
    fence_char = None
    fence_length = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        fence = FENCE.match(line)
        if fence:
            marker = fence.group(1)
            if fence_char is None:
                fence_char, fence_length = marker[0], len(marker)
            elif marker[0] == fence_char and len(marker) >= fence_length:
                fence_char = None
            continue
        if fence_char is None:
            result.append(line)
    return result


def headings(path: Path) -> list[tuple[str, str]]:
    """Return GitHub heading fragments and their immediately following text."""
    sections: list[tuple[str, str]] = []
    counts: dict[str, int] = {}
    for line in prose_lines(path):
        match = HEADING.match(line)
        if match:
            text = re.sub(r"<[^>]+>", "", match.group(2)).lower()
            slug = "".join(
                char
                for char in text
                if char in "-_"
                or not unicodedata.category(char).startswith(("P", "S"))
            ).replace(" ", "-")
            number = counts.get(slug, 0)
            counts[slug] = number + 1
            if number:
                slug += f"-{number}"
            sections.append((slug, ""))
        elif sections:
            slug, body = sections[-1]
            sections[-1] = (slug, body + line + "\n")
    return sections


@pytest.mark.parametrize("name", PAGES)
def test_original_sections_link_to_the_matching_canonical_section(
    name: str,
) -> None:
    folder, expected = PAGES[name]
    legacy = DOCS / name
    canonical = DOCS / folder / name
    assert canonical.is_file()
    old_sections = dict(headings(legacy))
    new_sections = dict(headings(canonical))
    assert list(old_sections) == expected
    assert list(new_sections) == expected
    for slug in expected:
        assert f"({folder}/{name}#{slug})" in old_sections[slug]


def test_latest_doe_diagrams_are_preserved() -> None:
    """The shared overview and the detailed view both survive the move."""
    source = DOCS / "diagrams" / "doe-loop.mmd"
    doe = (DOCS / "analysis" / "doe.md").read_text(encoding="utf-8")
    readme = (ROOT / "readme.md").read_text(encoding="utf-8")
    assert source.is_file()
    assert "<!-- diagram: doe-loop -->" in doe
    assert "<!-- diagram: doe-loop -->" in readme
    assert "flowchart TB\n    r([Quality target r])" in doe
    assert "```mermaid" in doe


def test_canonical_pages_are_not_gitignored() -> None:
    for name, (folder, _) in PAGES.items():
        page = f"documentation/{folder}/{name}"
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", "-q", "--", page],
            cwd=ROOT,
            check=False,
        )
        assert result.returncode == 1, f"canonical page is ignored: {page}"


@pytest.mark.parametrize(
    "path",
    ["data/private.json", "documentation/data/private.json", "other/data/x"],
)
def test_private_data_paths_remain_ignored(path: str) -> None:
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", "--", path],
        cwd=ROOT,
        check=False,
    )
    assert result.returncode == 0


def test_all_local_markdown_links_and_fragments_resolve() -> None:
    files = (
        subprocess.check_output(
            [
                "git",
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
                "--",
                "*.md",
            ],
            cwd=ROOT,
        )
        .decode("utf-8")
        .split("\0")
    )
    errors = []
    for filename in filter(None, files):
        page = ROOT / filename
        for number, line in enumerate(prose_lines(page), start=1):
            destinations = LINK.findall(line) + HTML_LINK.findall(line)
            reference = REFERENCE.match(line)
            if reference:
                destinations.append(reference.group(1))
            for destination in destinations:
                url = urlsplit(destination.strip("<>"))
                if url.scheme or url.netloc:
                    continue
                target = (page.parent / unquote(url.path)).resolve()
                if not target.exists():
                    errors.append(f"{filename}:{number}: {destination}")
                elif url.fragment and target.suffix.lower() == ".md":
                    fragments = {slug for slug, _ in headings(target)}
                    if unquote(url.fragment) not in fragments:
                        errors.append(f"{filename}:{number}: {destination}")
    assert not errors, "\n" + "\n".join(errors)
