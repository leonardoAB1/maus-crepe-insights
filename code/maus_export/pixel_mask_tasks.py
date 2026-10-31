"""Prepare pixel suggestions for untouched tasks, without app writes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from labeling.workflow import build_plan

from .pixel_masks import ALGORITHM_VERSION, PixelMaskConfig, suggest_masks

PREFILL_CONFIG = PixelMaskConfig(min_area=10000)
PREFILL_ROI = (360, 100, 640, 620)
BYTE_MAX = 255


def task_digest(data: dict) -> str:
    """Pin unchanged task provenance without serializing account metadata."""
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def untouched_tasks(database: Path, project: int) -> list[tuple[int, dict]]:
    """Conservatively protect annotations, drafts, locks and any predictions."""
    with sqlite3.connect("file:" + str(database) + "?mode=ro", uri=True) as db:
        rows = db.execute(
            "select t.id,t.data from task t where t.project_id=? "
            "and not exists(select 1 from task_completion a "
            "where a.task_id=t.id) "
            "and not exists(select 1 from tasks_annotationdraft d "
            "where d.task_id=t.id) "
            "and not exists(select 1 from tasks_tasklock l "
            "where l.task_id=t.id) "
            "and not exists(select 1 from prediction p where p.task_id=t.id) "
            "order by t.id",
            (project,),
        ).fetchall()
    return [(pk, json.loads(data)) for pk, data in rows]


def prepare_task_suggestions(
    workspace: Path,
    output: Path,
    *,
    roi: tuple[int, int, int, int] | None = PREFILL_ROI,
    config: PixelMaskConfig = PREFILL_CONFIG,
) -> dict:
    """Produce private native-size mask PNGs; do not create annotations."""
    workspace = workspace.resolve(strict=True)
    pilot = workspace / "pilot"
    state = json.loads((workspace / "workspace.json").read_text())
    plan = build_plan(pilot, kind="masks")
    if plan.get("assignment", {}).get("set_number") != 1:
        raise ValueError("This operation is restricted to set 1")
    if plan["digest"] != state["pilot_sha256"]:
        raise ValueError("Workspace pilot identity changed")
    output = output.absolute()
    source_root = Path(__file__).resolve().parents[2]
    if (
        output != output.resolve()
        or output.is_relative_to(workspace)
        or (
            output.is_relative_to(source_root)
            and not output.is_relative_to(source_root / "data")
        )
    ):
        raise ValueError("Use a new private data/ folder outside the workspace")
    expected = {
        (
            t["data"]["run_id"],
            t["data"]["decoded_frame_index"],
            t["data"]["requested_seconds"],
        ): t["data"]
        for t in plan["tasks"]
    }
    images = {
        (
            run["run_id"],
            frame["decoded_frame_index"],
            frame["requested_seconds"],
        ): frame["image"]
        for run in json.loads((pilot / "manifest.json").read_text())["runs"]
        for frame in run.get("frames", [])
    }
    project = state["projects"]["masks"]["project_id"]
    candidates = untouched_tasks(
        workspace / "app/label_studio.sqlite3", project
    )
    os.umask(0o077)
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    output.mkdir(mode=0o700)
    records = []
    empty = 0
    try:
        for pk, data in candidates:
            key = (
                data["run_id"],
                data["decoded_frame_index"],
                data["requested_seconds"],
            )
            if (
                data != expected.get(key)
                or data["coordinate_space"] != "original_video"
            ):
                raise ValueError("Task does not match the original set-1 pilot")
            content = (pilot / images[key]).read_bytes()
            if hashlib.sha256(content).hexdigest() != data["image_sha256"]:
                raise ValueError("Source image changed")
            image = cv2.imdecode(
                np.frombuffer(content, np.uint8), cv2.IMREAD_UNCHANGED
            )
            material, holes = suggest_masks(image, roi=roi, config=config)
            if not material.any():
                empty += 1
                continue
            folder = output / f"task-{pk}"
            folder.mkdir(mode=0o700)
            masks = {}
            for name, pixels in (("material", material), ("holes", holes)):
                path = folder / f"{name}.png"
                if not cv2.imwrite(
                    str(path), pixels.astype(np.uint8) * BYTE_MAX
                ):
                    raise OSError("Cannot write suggestion mask")
                masks[name] = {
                    "path": str(path.relative_to(output)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            records.append(
                {
                    "task_id": pk,
                    "task_data_sha256": task_digest(data),
                    "image_sha256": data["image_sha256"],
                    "image_size": data["image_size"],
                    "masks": masks,
                }
            )
        manifest = {
            "algorithm": ALGORITHM_VERSION,
            "human_annotation": False,
            "pilot_sha256": plan["digest"],
            "assignment": plan["assignment"],
            "project_id": project,
            "roi": roi,
            "parameters": asdict(config),
            "candidates": len(candidates),
            "empty_not_imported": empty,
            "records": records,
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    except BaseException:
        shutil.rmtree(output)
        raise
    return {
        "candidates": len(candidates),
        "nonempty_suggestions": len(records),
        "empty_not_imported": empty,
    }


def main() -> int:
    """Generate masks; the separate importer rechecks live task protection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    result = prepare_task_suggestions(args.workspace, args.output)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
