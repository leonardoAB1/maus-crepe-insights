"""Preview-first pixel-mask assistance for either private labeling set."""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import cv2

from labeling.workflow import prefill_context

from .pixel_mask_tasks import (
    PREFILL_CONFIG,
    prepare_task_suggestions,
    print_progress,
    untouched_tasks,
)
from .pixel_masks import ALGORITHM_VERSION, PixelMaskConfig

DEFAULT_SAMPLE = 12
LABELING_MANIFEST = Path(__file__).resolve().parents[1] / "labeling/pixi.toml"


def workspace_status(workspace: Path, set_number: int) -> dict:
    """Report local protection counts without changing the app or images."""
    state, plan = prefill_context(workspace, set_number)
    project = state["projects"]["masks"]["project_id"]
    database = workspace / "app/label_studio.sqlite3"
    eligible = len(untouched_tasks(database, project))
    with sqlite3.connect("file:" + str(database) + "?mode=ro", uri=True) as db:
        counts = {}
        for name, table in (
            ("annotations", "task_completion"),
            ("drafts", "tasks_annotationdraft"),
            ("locks", "tasks_tasklock"),
            ("predictions", "prediction"),
        ):
            counts[name] = db.execute(
                "select count(distinct t.id) from task t join "
                + table
                + " x on x.task_id=t.id where t.project_id=?",
                (project,),
            ).fetchone()[0]
    return {
        "set_number": set_number,
        "tasks": len(plan["tasks"]),
        "eligible_untouched": eligible,
        "protected_or_prefilled": len(plan["tasks"]) - eligible,
        "overlapping_protection_counts": counts,
        "image_sizes": sorted(
            {tuple(t["data"]["image_size"]) for t in plan["tasks"]}
        ),
        "mask_project_url": (
            f"http://localhost:{state.get('port', 8081 + set_number)}"
            f"/projects/{project}"
        ),
    }


def prepare_reviewed_job(
    workspace: Path,
    set_number: int,
    preview: Path,
    output: Path,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Copy reviewed preview parameters into a fresh full, live-filtered job."""
    _, plan = prefill_context(workspace, set_number)
    metadata = json.loads((preview / "manifest.json").read_text())
    if (
        metadata.get("scope") != "sample"
        or metadata["algorithm"] != ALGORITHM_VERSION
        or metadata["assignment"] != plan["assignment"]
        or metadata["pilot_sha256"] != plan["digest"]
        or metadata.get("set_number") != set_number
    ):
        raise ValueError("Preview belongs to another set, pilot or algorithm")
    return prepare_task_suggestions(
        workspace,
        output,
        set_number=set_number,
        roi=tuple(metadata["roi"]) if metadata["roi"] is not None else None,
        config=PixelMaskConfig(**metadata["parameters"]),
        progress=progress,
    )


def _labeling_task(task: str, arguments: list[str]) -> None:
    subprocess.run(
        [
            "pixi",
            "run",
            "--locked",
            "--manifest-path",
            str(LABELING_MANIFEST),
            task,
            *arguments,
        ],
        check=True,
    )


def main() -> int:
    """Inspect, preview, prepare or explicitly apply pixel suggestions."""
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    for name in ("status", "start", "preview", "prepare", "check", "apply"):
        child = actions.add_parser(name)
        child.add_argument(
            "--set", dest="set_number", type=int, choices=(1, 2), required=True
        )
        child.add_argument("--workspace", type=Path, required=True)
        if name == "start":
            child.add_argument("--port", type=int)
        if name in {"preview", "prepare"}:
            child.add_argument("--output", type=Path, required=True)
        if name == "prepare":
            child.add_argument("--from-preview", type=Path, required=True)
        if name in {"check", "apply"}:
            child.add_argument("--suggestions", type=Path, required=True)
        if name == "apply":
            child.add_argument(
                "--yes",
                action="store_true",
                required=True,
                help="confirm adding predictions, not human labels",
            )
        if name == "preview":
            child.add_argument("--limit", type=int, default=DEFAULT_SAMPLE)
            area = child.add_mutually_exclusive_group()
            area.add_argument(
                "--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H")
            )
            area.add_argument("--full-image", action="store_true")
            for field, value in asdict(PREFILL_CONFIG).items():
                if field != "keep_largest":
                    child.add_argument(
                        "--" + field.replace("_", "-"), type=int, default=value
                    )
            child.add_argument("--keep-all", action="store_true")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    common = [
        "--set",
        str(args.set_number),
        "--workspace",
        str(args.workspace.resolve()),
    ]
    try:
        if args.workspace.absolute() != args.workspace.resolve():
            raise ValueError("Use a canonical absolute workspace path")
        if args.action == "status":
            result = workspace_status(args.workspace, args.set_number)
        elif args.action == "start":
            if args.port is not None:
                common += ["--port", str(args.port)]
            _labeling_task("label-set", common)
            return 0
        elif args.action == "preview":
            config = PixelMaskConfig(
                **{
                    field: getattr(args, field)
                    for field in asdict(PREFILL_CONFIG)
                    if field != "keep_largest"
                },
                keep_largest=not args.keep_all,
            )
            result = prepare_task_suggestions(
                args.workspace,
                args.output,
                set_number=args.set_number,
                roi=tuple(args.roi) if args.roi else None,
                config=config,
                limit=args.limit,
                progress=print_progress,
            )
        elif args.action == "prepare":
            result = prepare_reviewed_job(
                args.workspace,
                args.set_number,
                args.from_preview,
                args.output,
                progress=print_progress,
            )
        else:
            common += ["--suggestions", str(args.suggestions.resolve())]
            if args.action == "apply":
                common.append("--apply")
            _labeling_task("prefill-pixels", common)
            return 0
    except KeyboardInterrupt:
        return 0
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        cv2.error,
        subprocess.CalledProcessError,
    ) as error:
        parser.exit(1, f"Mask assistance stopped: {error}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
