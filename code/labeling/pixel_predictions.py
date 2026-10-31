"""Install editable pixel-mask predictions, never replacing manual work."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

if __package__:
    from .workflow import (
        PIXEL_SUGGESTION_MARKER,
        PIXEL_SUGGESTION_VERSION,
        prefill_context,
        runtime,
        validate_pixel_result,
    )
else:
    from workflow import (
        PIXEL_SUGGESTION_MARKER,
        PIXEL_SUGGESTION_VERSION,
        prefill_context,
        runtime,
        validate_pixel_result,
    )

BYTE_MAX = 255


def _result(folder: Path, record: dict) -> list:
    from label_studio_sdk.converter.brush import mask2rle

    width, height = record["image_size"]
    result, masks = [], []
    for name, label in (("material", "crepe_material"), ("holes", "hole")):
        artifact = record["masks"][name]
        expected = f"task-{record['task_id']}/{name}.png"
        path = folder / artifact["path"]
        if artifact["path"] != expected or path.absolute() != path.resolve(
            strict=True
        ):
            raise ValueError("Unsafe mask artifact path")
        if hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]:
            raise ValueError("Suggestion mask hash changed")
        with Image.open(path) as image:
            if image.mode != "L" or image.size != (width, height):
                raise ValueError(
                    "Suggestion dimensions differ from original pixels"
                )
            pixels = np.array(image)
        if not set(np.unique(pixels)).issubset({0, BYTE_MAX}):
            raise ValueError("Suggestion masks must be binary")
        mask = pixels > 0
        masks.append(mask)
        if not mask.any():
            continue
        result.append(
            {
                "id": "pixel-" + name,
                "from_name": "segmentation",
                "to_name": "image",
                "type": "brushlabels",
                "origin": "prediction",
                "original_width": width,
                "original_height": height,
                "image_rotation": 0,
                "value": {
                    "format": "rle",
                    "rle": mask2rle(pixels),
                    "brushlabels": [label],
                },
            }
        )
    if np.any(masks[0] & masks[1]):
        raise ValueError("Overlapping proposal masks")
    validate_pixel_result(result, record["image_size"])
    return result


def install_predictions(
    workspace: Path, suggestions: Path, *, set_number: int, apply: bool = False
) -> dict:
    """Recheck live annotations/drafts/locks before each prediction insert."""
    from django.conf import settings
    from django.db import OperationalError, transaction
    from projects.models import Project
    from tasks.models import Prediction, Task
    from tasks.serializers import PredictionSerializer

    state, plan = prefill_context(workspace, set_number)
    workspace = workspace.resolve(strict=True)
    pilot = workspace / "pilot"
    if Path(settings.LOCAL_FILES_DOCUMENT_ROOT) != pilot:
        raise ValueError("App root differs from the selected workspace")
    job = json.loads((suggestions / "manifest.json").read_text())
    if (
        job["algorithm"] != PIXEL_SUGGESTION_VERSION
        or job["human_annotation"] is not False
        or job["pilot_sha256"] != plan["digest"]
        or job["assignment"] != plan["assignment"]
        or job["project_id"] != state["projects"]["masks"]["project_id"]
        or job.get("set_number", job["assignment"]["set_number"]) != set_number
    ):
        raise ValueError("Wrong prediction source, project or assignment")
    if apply and job.get("scope", "full") != "full":
        raise ValueError(
            "Sample previews cannot be applied; prepare a full job first"
        )
    project = Project.objects.get(pk=job["project_id"])
    descriptions = {
        plan["description"],
        plan["description"] + PIXEL_SUGGESTION_MARKER,
    }
    if (
        project.title != plan["title"]
        or project.label_config != plan["config"]
        or project.description not in descriptions
    ):
        raise ValueError("Target is not the selected managed mask project")
    if project.predictions.exclude(
        model_version=PIXEL_SUGGESTION_VERSION
    ).exists():
        raise ValueError(
            "Unrelated predictions already exist; refusing to alter them"
        )
    expected = {
        (
            t["data"]["run_id"],
            t["data"]["decoded_frame_index"],
            t["data"]["requested_seconds"],
        ): t["data"]
        for t in plan["tasks"]
    }
    counts = {
        "created": 0,
        "eligible": 0,
        "protected": 0,
        "existing": 0,
        "busy": 0,
    }
    seen = set()
    for record in job["records"]:
        pk = record["task_id"]
        if type(pk) is not int or pk in seen:
            raise ValueError("Invalid or duplicate task ID")
        seen.add(pk)
        task = Task.objects.get(pk=pk, project=project)
        if (
            task.annotations.exists()
            or task.drafts.exists()
            or task.locks.exists()
        ):
            counts["protected"] += 1
            continue
        if task.predictions.exists():
            counts["existing"] += 1
            continue
        result = _result(suggestions, record)
        try:
            with transaction.atomic():
                task = Task.objects.select_for_update().get(
                    pk=pk, project=project
                )
                if (
                    task.annotations.exists()
                    or task.drafts.exists()
                    or task.locks.exists()
                ):
                    counts["protected"] += 1
                    continue
                if task.predictions.exists():
                    counts["existing"] += 1
                    continue
                data = task.data
                key = (
                    data["run_id"],
                    data["decoded_frame_index"],
                    data["requested_seconds"],
                )
                digest = hashlib.sha256(
                    json.dumps(
                        data, sort_keys=True, separators=(",", ":")
                    ).encode()
                ).hexdigest()
                if (
                    data != expected.get(key)
                    or digest != record["task_data_sha256"]
                    or data["image_sha256"] != record["image_sha256"]
                    or data["image_size"] != record["image_size"]
                ):
                    raise ValueError("Task source data changed")
                counts["eligible"] += 1
                if apply:
                    serializer = PredictionSerializer(
                        data={
                            "task": pk,
                            "project": project.id,
                            "result": result,
                            "model_version": PIXEL_SUGGESTION_VERSION,
                        }
                    )
                    serializer.is_valid(raise_exception=True)
                    project.show_collab_predictions = True
                    project.model_version = PIXEL_SUGGESTION_VERSION
                    project.description = (
                        plan["description"] + PIXEL_SUGGESTION_MARKER
                    )
                    project.save(
                        update_fields=[
                            "show_collab_predictions",
                            "model_version",
                            "description",
                        ]
                    )
                    serializer.save()
                    counts["created"] += 1
        except OperationalError:
            counts["busy"] += 1
    if (
        apply
        and project.predictions.filter(
            model_version=PIXEL_SUGGESTION_VERSION
        ).exists()
    ):
        project.show_collab_predictions = True
        project.model_version = PIXEL_SUGGESTION_VERSION
        project.description = plan["description"] + PIXEL_SUGGESTION_MARKER
        project.save(
            update_fields=[
                "show_collab_predictions",
                "model_version",
                "description",
            ]
        )
        project.update_tasks_counters_and_task_states(
            tasks_queryset=project.tasks.all(),
            maximum_annotations_changed=False,
            overlap_cohort_percentage_changed=False,
            tasks_number_changed=False,
            recalculate_stats_counts={
                "prediction_count": Prediction.objects.filter(
                    project=project
                ).count()
            },
        )
    return {
        **counts,
        "applied": apply,
        "empty_not_imported": job["empty_not_imported"],
    }


def main() -> int:
    """Dry-run first; explicit --apply adds proposals to untouched tasks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument(
        "--set", dest="set_number", type=int, choices=(1, 2), required=True
    )
    parser.add_argument("--suggestions", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    os.umask(0o077)
    workspace = args.workspace.resolve(strict=True)
    state, _ = prefill_context(workspace, args.set_number)
    if (
        args.apply
        and json.loads((args.suggestions / "manifest.json").read_text()).get(
            "scope", "full"
        )
        != "full"
    ):
        parser.error(
            "Sample previews cannot be applied; prepare a full job first"
        )
    app = workspace / "app"
    command, env = runtime(
        workspace / "pilot", app, state.get("port", 8081 + args.set_number)
    )
    if args.apply:
        backups = app / "backups"
        backups.mkdir(mode=0o700, exist_ok=True)
        name = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S%fZ")
        with sqlite3.connect(
            "file:" + str(app / "label_studio.sqlite3") + "?mode=ro", uri=True
        ) as source:
            with sqlite3.connect(
                backups / f"before-pixel-{name}.sqlite3"
            ) as target:
                source.backup(target)
    source_root = Path(__file__).resolve().parents[1]
    code = (
        f"import sys\nsys.path.insert(0,{str(source_root)!r})\n"
        "import json\nfrom pathlib import Path\n"
        "from labeling.pixel_predictions import install_predictions\n"
        f"r=install_predictions(Path({str(workspace)!r}),"
        f"Path({str(args.suggestions.resolve())!r}),set_number={args.set_number!r},apply={args.apply!r})\n"
        "print('PIXEL_IMPORT='+json.dumps(r))\n"
    )
    result = subprocess.run(
        command[:3] + ["shell"],
        input=f"exec({code!r})\n",
        text=True,
        capture_output=True,
        env=env,
        cwd=app,
    )
    (app / "pixel-import.log").write_text(result.stdout + result.stderr)
    summaries = [
        s.split("PIXEL_IMPORT=", 1)[1]
        for s in result.stdout.splitlines()
        if "PIXEL_IMPORT=" in s
    ]
    if result.returncode or not summaries:
        print(
            "Import failed; inspect private app/pixel-import.log",
            file=sys.stderr,
        )
        return 1
    print(summaries[-1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
