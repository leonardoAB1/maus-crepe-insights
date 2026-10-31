"""Keep shared Mermaid diagrams in sync across markdown files.

Each diagram lives once in ``documentation/diagrams/<name>.mmd``. Any
scanned markdown file can show it with a marked block::

    <!-- diagram: system -->
    <!-- /diagram -->

``maus-docs`` fills every block with the current ``.mmd`` content, and
``maus-docs --check`` only reports blocks that are out of date.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DIAGRAM_DIR = Path("documentation") / "diagrams"
DIAGRAM_SUFFIX = ".mmd"
SCANNED_GLOBS = ("readme.md", "documentation/*.md")
FENCE = "```"
LF = "\n"
CRLF = "\r\n"

BLOCK_RE = re.compile(
    r"(?P<open><!-- diagram: (?P<name>[\w-]+) -->\n)"
    r"(?P<body>.*?)"
    r"(?P<close><!-- /diagram -->)",
    re.DOTALL,
)

EXIT_STALE = 1


class UnknownDiagramError(ValueError):
    """A marker names a diagram with no matching ``.mmd`` file."""


def render_block(source: str) -> str:
    """Wrap diagram source in a mermaid code fence."""
    return f"{FENCE}mermaid\n{source.rstrip()}\n{FENCE}\n"


def sync_text(text: str, diagrams: dict[str, str], where: str = "") -> str:
    """Return ``text`` with every marked block filled from ``diagrams``."""

    def fill(match: re.Match[str]) -> str:
        name = match["name"]
        if name not in diagrams:
            raise UnknownDiagramError(f"{where}: no diagram named {name!r}")
        body = render_block(diagrams[name])
        return f"{match['open']}{body}{match['close']}"

    return BLOCK_RE.sub(fill, text)


def load_diagrams(root: Path) -> dict[str, str]:
    """Read every ``.mmd`` file in the diagram folder, keyed by stem."""
    folder = root / DIAGRAM_DIR
    return {
        path.stem: path.read_bytes().decode("utf-8")
        for path in sorted(folder.glob(f"*{DIAGRAM_SUFFIX}"))
    }


def markdown_files(root: Path) -> list[Path]:
    """Markdown files that may hold diagram blocks."""
    files: list[Path] = []
    for pattern in SCANNED_GLOBS:
        files.extend(sorted(root.glob(pattern)))
    return files


def sync(root: Path = REPO_ROOT, check: bool = False) -> list[Path]:
    """Fill diagram blocks under ``root``.

    Returns the files that changed, or with ``check`` the files that
    would change (nothing is written then).
    """
    diagrams = {
        name: source.replace(CRLF, LF)
        for name, source in load_diagrams(root).items()
    }
    stale: list[Path] = []
    for path in markdown_files(root):
        raw = path.read_bytes().decode("utf-8")
        # Keep each file's own line endings so a sync never rewrites
        # every line of a CRLF file
        eol = CRLF if CRLF in raw else LF
        text = raw.replace(CRLF, LF)
        new = sync_text(text, diagrams, where=str(path))
        if new == text:
            continue
        stale.append(path)
        if not check:
            path.write_bytes(new.replace(LF, eol).encode("utf-8"))
    return stale


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="maus-docs",
        description="Copy shared Mermaid diagrams into markdown files",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="only report out-of-date files, exit 1 if any",
    )
    args = parser.parse_args()

    stale = sync(check=args.check)
    for path in stale:
        rel = path.relative_to(REPO_ROOT)
        print(f"{'out of date' if args.check else 'updated'}: {rel}")
    if args.check and stale:
        print("run `pixi run docs` to fix", file=sys.stderr)
        sys.exit(EXIT_STALE)
