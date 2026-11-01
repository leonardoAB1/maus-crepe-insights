"""Prepare pixel suggestions for untouched tasks, without app writes."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import shutil
import sqlite3
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from labeling.workflow import prefill_context

from .pixel_masks import (
    ALGORITHM_VERSION,
    PixelMaskConfig,
    preview_masks,
    suggest_masks,
)

PREFILL_CONFIG = PixelMaskConfig(min_area=10000)
PREFILL_ROI = (360, 100, 640, 620)
BYTE_MAX = 255
PREVIEW_COUNT = 12
TILE_WIDTH = 640
TILE_HEIGHT = 220
PANEL_WIDTH = 310
PANEL_HEIGHT = 180
SHEET_COLUMNS = 2
PROGRESS_INTERVAL = 25


def print_progress(done: int, total: int) -> None:
    """Show aggregate progress without exposing frame identities."""
    if done % PROGRESS_INTERVAL == 0 or done == total:
        print(
            f"Prepared {done}/{total} frames; nothing imported yet.",
            file=sys.stderr,
            flush=True,
        )


def _sample_indices(count: int, limit: int) -> list[int]:
    return (
        sorted(set(np.linspace(0, count - 1, min(count, limit)).astype(int)))
        if count
        else []
    )


def _review_tile(
    image: np.ndarray, material: np.ndarray, holes: np.ndarray, pk: int
) -> np.ndarray:
    tile = np.full((TILE_HEIGHT, TILE_WIDTH, 3), 35, np.uint8)
    for offset, pixels in (
        (0, image),
        (TILE_WIDTH // 2, preview_masks(image, material, holes)),
    ):
        h, w = pixels.shape[:2]
        scale = min(PANEL_WIDTH / w, PANEL_HEIGHT / h)
        thumb = cv2.resize(
            pixels, (max(1, round(w * scale)), max(1, round(h * scale)))
        )
        tile[: thumb.shape[0], offset : offset + thumb.shape[1]] = thumb
    text = f"Task {pk}: original | suggestion, material={int(material.sum())}"
    cv2.putText(
        tile,
        text,
        (5, TILE_HEIGHT - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (230, 230, 230),
        1,
    )
    return tile


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
    set_number: int,
    roi: tuple[int, int, int, int] | None = None,
    config: PixelMaskConfig = PREFILL_CONFIG,
    limit: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Produce private native-size mask PNGs; do not create annotations."""
    state, plan = prefill_context(workspace, set_number)
    workspace = workspace.resolve(strict=True)
    pilot = workspace / "pilot"
    if limit is not None and limit < 1:
        raise ValueError("Sample limit must be positive")
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
    total_candidates = len(candidates)
    if limit is not None:
        candidates = [
            candidates[i] for i in _sample_indices(total_candidates, limit)
        ]
    preview_indices = set(_sample_indices(len(candidates), PREVIEW_COUNT))
    os.umask(0o077)
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    output.mkdir(mode=0o700)
    records = []
    empty = 0
    tiles = []
    review = []
    try:
        for position, (pk, data) in enumerate(candidates):
            key = (
                data["run_id"],
                data["decoded_frame_index"],
                data["requested_seconds"],
            )
            if (
                data != expected.get(key)
                or data["coordinate_space"] != "original_video"
            ):
                raise ValueError(
                    "Task does not match the selected original pilot"
                )
            content = (pilot / images[key]).read_bytes()
            if hashlib.sha256(content).hexdigest() != data["image_sha256"]:
                raise ValueError("Source image changed")
            image = cv2.imdecode(
                np.frombuffer(content, np.uint8), cv2.IMREAD_UNCHANGED
            )
            material, holes = suggest_masks(image, roi=roi, config=config)
            if progress is not None:
                progress(position + 1, len(candidates))
            if position in preview_indices:
                tile = _review_tile(image, material, holes, pk)
                tiles.append(tile)
                review.append(
                    {
                        "task_id": pk,
                        "image_size": data["image_size"],
                        "material_pixels": int(material.sum()),
                        "hole_pixels": int(holes.sum()),
                    }
                )
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
            "opencv_version": cv2.__version__,
            "human_annotation": False,
            "pilot_sha256": plan["digest"],
            "assignment": plan["assignment"],
            "project_id": project,
            "set_number": set_number,
            "scope": "sample" if limit is not None else "full",
            "candidates_total": total_candidates,
            "roi": roi,
            "parameters": asdict(config),
            "candidates": len(candidates),
            "empty_not_imported": empty,
            "records": records,
            "preview_records": review,
        }
        if tiles:
            rows = (len(tiles) + SHEET_COLUMNS - 1) // SHEET_COLUMNS
            sheet = np.full(
                (rows * TILE_HEIGHT, SHEET_COLUMNS * TILE_WIDTH, 3),
                35,
                np.uint8,
            )
            for index, tile in enumerate(tiles):
                y, x = (
                    (index // SHEET_COLUMNS) * TILE_HEIGHT,
                    (index % SHEET_COLUMNS) * TILE_WIDTH,
                )
                sheet[y : y + TILE_HEIGHT, x : x + TILE_WIDTH] = tile
            if not cv2.imwrite(str(output / "contact-sheet.jpg"), sheet):
                raise OSError("Cannot write private review sheet")
        (output / "preview.html").write_text(
            "<!doctype html><meta charset='utf-8'>"
            "<title>Pixel-mask review</title>"
            f"<h1>Set {set_number}: suggestions only</h1>"
            + "<pre>"
            + html.escape(
                json.dumps(
                    {
                        "scope": "sample" if limit is not None else "full",
                        "roi": roi,
                        "parameters": asdict(config),
                    },
                    indent=2,
                )
            )
            + "</pre>"
            + "<p>Original left, suggestion right. Green: material; red: hole. "
            "Dark stains, glare and tools can be wrong. "
            "Nothing here is a human label.</p>"
            + (
                "<img style='max-width:100%' src='contact-sheet.jpg' "
                "alt='Review sheet'>"
                if tiles
                else "<p>No untouched tasks. Existing work was not changed.</p>"
            )
        )
        temporary = output / ".manifest.tmp"
        temporary.write_text(json.dumps(manifest, indent=2))
        temporary.replace(output / "manifest.json")
    except BaseException:
        shutil.rmtree(output)
        raise
    return {
        "candidates": len(candidates),
        "nonempty_suggestions": len(records),
        "empty_not_imported": empty,
        "set_number": set_number,
        "scope": manifest["scope"],
        "candidates_total": total_candidates,
        "preview": str(output / "preview.html"),
    }


def main() -> int:
    """Generate masks; the separate importer rechecks live task protection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument(
        "--set", dest="set_number", type=int, choices=(1, 2), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H")
    )
    args = parser.parse_args()
    cv2.setNumThreads(1)
    result = prepare_task_suggestions(
        args.workspace,
        args.output,
        set_number=args.set_number,
        roi=tuple(args.roi) if args.roi else None,
        limit=args.limit,
        progress=print_progress,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
