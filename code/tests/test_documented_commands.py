"""Check that every documented `pixi run` command exists.

Markdown links are covered by `test_documentation_links.py` and shared diagrams
by `test_docsync.py`. This test covers commands, because the status page told
readers to run `pixi run training`, `pixi run vision-targets` and
`pixi run audit`, and no pixi task has those names: the real commands are the
`training`, `vision-targets` and `audit` subcommands of `pixi run export`.

Every `pixi run` command is read from a fenced code block or from an inline
code span, so prose that mentions a command in passing is not parsed as one.
For each command:

- the task name must be a task of the manifest the command names (the shared
  `code/pyproject.toml`, any of its environment features, or a manifest given
  with `--manifest-path`), or one of the entry points pixi can run directly;
- when the command runs a `maus-` entry point, a following bare word must be a
  real subcommand of that entry point, so `pixi run export training` is checked
  all the way down to the CLI.

Examples that use shell variables or placeholders are skipped.
"""

from __future__ import annotations

import difflib
import re
import shlex
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "readme.md"
DOCS = ROOT / "documentation"
MAIN_MANIFEST = ROOT / "code" / "pyproject.toml"
LABELING_MANIFEST = ROOT / "code" / "labeling" / "pixi.toml"

INLINE_RE = re.compile(r"`pixi run(?P<rest>[^`]*)`")
FENCE_RE = re.compile(r"^\s*(```+|~~~+)")
FENCED_RE = re.compile(r"^\s*(?:\$ )?pixi run(?P<rest>.*)$")
COMMENT_RE = re.compile(r"\s#")
SUBCOMMAND_RE = re.compile(r'add_parser\(\s*"([a-z][a-z0-9-]*)"')
BARE_WORD_RE = re.compile(r"^[a-z][a-z0-9-]*$")

# Flags that take a separate value, so the next word is not the task name
VALUE_FLAGS = {"-e", "--environment", "--manifest-path"}
# Entry points pixi runs directly, without a task of the same name
ENTRY_POINTS = {
    "maus-export": ROOT / "code" / "maus_export" / "cli.py",
    "maus-classify": ROOT / "code" / "maus_classify" / "cli.py",
    "maus-docs": ROOT / "code" / "docsync" / "__init__.py",
}
# Commands the environment provides, which need no task
ENVIRONMENT_COMMANDS = {"python", "pytest", "jupyter", "ruff", "pip"}
MIN_SUBCOMMANDS = 5  # guards the pattern above against a silent change


@dataclass(frozen=True)
class Invocation:
    """One documented `pixi run` command."""

    path: Path
    line: int
    tokens: tuple[str, ...]
    manifest: Path


def _tracked_markdown() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "*.md"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    return [ROOT / name for name in out]


def _logical_lines(text: str) -> list[tuple[int, str, bool]]:
    """Split into lines, joining continuations, and mark code fences."""
    joined: list[tuple[int, str, bool]] = []
    pending = ""
    start = 0
    fenced = False
    for number, raw in enumerate(text.splitlines(), start=1):
        if FENCE_RE.match(raw):
            fenced = not fenced
            continue
        if not pending:
            start = number
        stripped = raw.rstrip()
        if stripped.endswith("\\"):
            pending += stripped[:-1] + " "
            continue
        joined.append((start, pending + stripped, fenced))
        pending = ""
    if pending:
        joined.append((start, pending, False))
    return joined


def _manifest_for(tokens: list[str], path: Path) -> Path:
    """The manifest a documented command names, defaulting to the main one."""
    for index, token in enumerate(tokens):
        if not token.startswith("--manifest-path"):
            continue
        if "=" in token:
            value = token.split("=", 1)[1]
        else:
            value = tokens[index + 1] if index + 1 < len(tokens) else ""
        if "labeling/pixi.toml" in value:
            return LABELING_MANIFEST
        if value.endswith("pyproject.toml"):
            return MAIN_MANIFEST
        if path == LABELING_MANIFEST:
            return LABELING_MANIFEST
    return MAIN_MANIFEST if path != LABELING_MANIFEST else LABELING_MANIFEST


def _split(rest: str) -> list[str]:
    """Split a command into words, dropping a trailing shell comment."""
    body = COMMENT_RE.split(rest, maxsplit=1)[0]
    try:
        return shlex.split(body)
    except ValueError:  # an unbalanced quote in prose
        return body.split()


def _invocations() -> list[Invocation]:
    found: list[Invocation] = []
    for path in _tracked_markdown():
        text = path.read_text(encoding="utf-8")
        for number, line, fenced in _logical_lines(text):
            if fenced:
                match = FENCED_RE.match(line)
                if match is not None:
                    tokens = _split(match.group("rest"))
                    if tokens:
                        manifest = _manifest_for(tokens, path)
                        found.append(
                            Invocation(path, number, tuple(tokens), manifest)
                        )
        for match in INLINE_RE.finditer(text):
            tokens = _split(match.group("rest"))
            number = text[: match.start()].count("\n") + 1
            if tokens:
                manifest = _manifest_for(tokens, path)
                found.append(Invocation(path, number, tuple(tokens), manifest))
    return found


def _task_name(tokens: list[str]) -> str | None:
    """The task name in a command, skipping pixi's own flags."""
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("-"):
            return token
        index += 2 if token in VALUE_FLAGS else 1
    return None


def _declared_tasks(manifest: Path) -> set[str]:
    document = tomllib.loads(manifest.read_text(encoding="utf-8"))
    if "tasks" in document:  # the labeling manifest has a flat table
        return set(document["tasks"])
    table = document.get("tool", {}).get("pixi", {})
    tasks = set(table.get("tasks", {}))
    for feature in table.get("feature", {}).values():
        tasks |= set(feature.get("tasks", {}))
    return tasks


def _task_command(manifest: Path, task: str) -> str | None:
    """The command a task runs, or none when the manifest is not a task one."""
    document = tomllib.loads(manifest.read_text(encoding="utf-8"))
    if "tasks" in document:
        entry = document["tasks"].get(task)
    else:
        table = document.get("tool", {}).get("pixi", {})
        groups = [table.get("tasks", {})]
        groups += [
            feature.get("tasks", {})
            for feature in table.get("feature", {}).values()
        ]
        entry = next((g[task] for g in groups if task in g), None)
    if entry is None:
        return None
    if isinstance(entry, dict):
        return str(entry.get("cmd", ""))
    return str(entry)


def _entry_point_of(manifest: Path, task: str) -> str | None:
    """The entry point a task or a direct command runs, if it is one."""
    if task in ENTRY_POINTS:
        return task
    command = _task_command(manifest, task)
    if not command:
        return None
    name = command.split()[0]
    return name if name in ENTRY_POINTS else None


def _subcommands(cli: Path) -> set[str]:
    names = set(SUBCOMMAND_RE.findall(cli.read_text(encoding="utf-8")))
    assert len(names) >= MIN_SUBCOMMANDS, (
        f"{cli} declares only {len(names)} subcommands, so this test can no "
        "longer read them; update the pattern"
    )
    return names


def documented_commands() -> list[Invocation]:
    invocations = _invocations()
    assert len(invocations) >= 40, (
        f"only {len(invocations)} documented pixi commands found, which means "
        "the scan or the documentation layout changed"
    )
    return invocations


COMMANDS = documented_commands()
IDS = [f"{i.path.name}:{i.line}" for i in COMMANDS]


@pytest.mark.parametrize("invocation", COMMANDS, ids=IDS)
def test_documented_task_exists(invocation: Invocation) -> None:
    task = _task_name(list(invocation.tokens))
    assert task is not None, f"{invocation.path}:{invocation.line} has no task"
    if not BARE_WORD_RE.match(task):
        pytest.skip(f"placeholder task {task!r}")
    if task in ENVIRONMENT_COMMANDS or task in ENTRY_POINTS:
        return
    if task in _declared_tasks(invocation.manifest):
        return
    known = sorted(
        _declared_tasks(MAIN_MANIFEST) | _declared_tasks(LABELING_MANIFEST)
    )
    hint = difflib.get_close_matches(task, known, n=3)
    raise AssertionError(
        f"{invocation.path}:{invocation.line} runs 'pixi run {task}', which "
        f"{invocation.manifest.name} does not declare. Nearby commands in the "
        f"repository: {hint or 'none'}"
    )


@pytest.mark.parametrize("invocation", COMMANDS, ids=IDS)
def test_documented_subcommand_exists(invocation: Invocation) -> None:
    tokens = list(invocation.tokens)
    task = _task_name(tokens)
    if task is None or not BARE_WORD_RE.match(task):
        pytest.skip("no task name")
    assert task is not None
    entry_point = _entry_point_of(invocation.manifest, task)
    if entry_point is None:
        pytest.skip(f"{task} has no subcommands to check")
    if entry_point == "maus-docs":
        # maus-docs takes flags, not subcommands
        return
    cli = ENTRY_POINTS[entry_point]
    # The first bare word after the task is its subcommand; flags, paths and
    # flag values such as `--data-dir data/` are skipped.
    words = [
        word
        for word in tokens[tokens.index(task) + 1 :]
        if BARE_WORD_RE.match(word)
    ]
    if not words:
        pytest.skip("no subcommand in this example")
    assert entry_point in ENTRY_POINTS
    subcommand = words[0]
    known = _subcommands(cli)
    if subcommand in known:
        return
    hint = difflib.get_close_matches(subcommand, sorted(known), n=3)
    raise AssertionError(
        f"{invocation.path}:{invocation.line} runs 'pixi run {task} "
        f"{subcommand}', but {entry_point} has no '{subcommand}' subcommand. "
        f"Nearby: {hint or sorted(known)}"
    )
