"""Private visual-phase Label Studio workflow; no automatic labels."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Literal
from urllib.parse import quote
from xml.etree import ElementTree as ET

HUMAN_TAXONOMY_VERSION = "crepe-visual-v2"
MASK_TAXONOMY_VERSION = "crepe-material-holes-v1"
PIXEL_SUGGESTION_VERSION = "pixel-colour-components-v1"
PIXEL_SUGGESTION_MARKER = "; suggestions=" + PIXEL_SUGGESTION_VERSION
ORIGINAL_FRAME_VERSION = "original-video-frames-v1"
ProjectKind = Literal["phase", "masks"]
RECTIFICATION_VERSION = "upstream-rectified-v1"
EXPERIMENTAL_RECTIFICATION_VERSION = "upstream-rectified-experimental-v1"
RECTIFIED_COORDINATE_SPACE = "rectified_upstream"
SHA256_HEX_LENGTH = 64
MIN_TEMPORAL_SAMPLES = 2
VALID_MASK_VALUES = {0, 255}
ASSIGNMENT_VERSION = "labeling-two-sets-v1"
HUMAN_LABELS = (
    "emptyplate",
    "spreading",
    "circle",
    "halffold",
    "quarter fold",
    "ugly",
)


def runtime(pilot: Path, app: Path, port: int) -> tuple[list[str], dict]:
    """Return a private loopback command and environment, no credentials."""
    pilot = pilot.absolute()
    app = app.absolute()
    source = Path(__file__).resolve().parents[2]
    for path in (pilot, app):
        if path != path.resolve():
            raise ValueError("symlinked runtime paths are not supported")
        if path.is_relative_to(source) and not path.is_relative_to(
            source / "data"
        ):
            raise ValueError("use private data, not source directories")
    if app.is_relative_to(pilot) or pilot.is_relative_to(app):
        raise ValueError("app data and pilot must be separate directories")
    build_plan(pilot)
    if pilot.stat().st_mode & 0o077:
        raise ValueError("pilot must be private (0700)")
    app.mkdir(mode=0o700, parents=True, exist_ok=True)
    if app.stat().st_mode & 0o077:
        raise ValueError("app data must be private (0700)")
    env = {
        key: os.environ[key]
        for key in (
            "PATH",
            "HOME",
            "LANG",
            "TMPDIR",
            "SSL_CERT_FILE",
            "SSL_CERT_DIR",
            "LD_LIBRARY_PATH",
        )
        if key in os.environ
    }
    env.update(
        {
            "LABEL_STUDIO_BASE_DATA_DIR": str(app),
            "LABEL_STUDIO_DATABASE_NAME": str(app / "label_studio.sqlite3"),
            "LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT": str(pilot),
            "LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED": "true",
            "DJANGO_DB": "sqlite",
            "COLLECT_ANALYTICS": "False",
            "LATEST_VERSION_CHECK": "False",
            "SENTRY_DSN": "",
            "FRONTEND_SENTRY_DSN": "",
            "LABEL_STUDIO_DEBUG": "False",
            "LABEL_STUDIO_ALLOWED_HOSTS": "127.0.0.1,localhost",
            "LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK": "true",
        }
    )
    return [
        sys.executable,
        "-m",
        "label_studio.server",
        "start",
        "--internal-host",
        "127.0.0.1",
        "--port",
        str(port),
        "--no-browser",
    ], env


def setup_project(
    pilot: Path, owner_email: str, *, kind: ProjectKind = "phase"
) -> dict:
    """Import through installed serializers, inside Label Studio's shell."""
    from data_import.serializers import ImportApiSerializer
    from django.conf import settings
    from django.db import transaction
    from io_storages.localfiles.models import LocalFilesImportStorage
    from io_storages.localfiles.serializers import (
        LocalFilesImportStorageSerializer,
    )
    from projects.models import Project
    from projects.serializers import ProjectSerializer
    from users.models import User

    plan = build_plan(pilot, kind=kind)
    if Path(settings.LOCAL_FILES_DOCUMENT_ROOT) != pilot.resolve():
        raise ValueError("server local-file root does not match pilot")
    owner = User.objects.filter(email=owner_email, is_active=True).first()
    if owner is None or owner.active_organization is None:
        raise ValueError("sign up in this app first, then supply your email")
    organization = owner.active_organization
    if not organization.has_permission(owner):
        raise ValueError("owner lacks active organization membership")
    title = plan["title"]
    description = plan["description"]
    with transaction.atomic():
        existing = Project.objects.filter(
            title=title,
            organization=organization,
        )
        if existing.count() > 1:
            raise ValueError("ambiguous pilot projects")
        project = existing.first()
        created = project is None
        if created:
            serializer = ProjectSerializer(
                data={
                    "title": title,
                    "label_config": plan["config"],
                    "description": description,
                    "show_collab_predictions": False,
                    "show_skip_button": True,
                    "sampling": Project.SEQUENCE,
                    "maximum_annotations": 1,
                    "start_training_on_annotation_update": False,
                },
                context={"created_by": owner},
            )
            serializer.is_valid(raise_exception=True)
            project = serializer.save(organization=organization)
            for root in plan["storage_roots"]:
                storage = LocalFilesImportStorageSerializer(
                    data={
                        "project": project.id,
                        "path": root,
                        "title": "Pilot frames (do not sync)",
                        "use_blob_urls": True,
                        "recursive_scan": False,
                        "regex_filter": r".*\.png$",
                    }
                )
                storage.is_valid(raise_exception=True)
                storage.save()
            imported = ImportApiSerializer(
                data=plan["tasks"],
                many=True,
                context={"project": project, "user": owner},
            )
            imported.is_valid(raise_exception=True)
            tasks = imported.save(project_id=project.id)
            project.update_tasks_counters_and_task_states(
                tasks_queryset=tasks,
                maximum_annotations_changed=False,
                overlap_cohort_percentage_changed=False,
                tasks_number_changed=True,
                recalculate_stats_counts={
                    "task_count": len(tasks),
                    "annotation_count": 0,
                    "prediction_count": 0,
                },
            )
            project.summary.update_data_columns(plan["tasks"])
        project.refresh_from_db()
        pixel_suggestions = kind == "masks" and (
            project.description == description + PIXEL_SUGGESTION_MARKER
        )
        if (
            project.label_config != plan["config"]
            or project.description
            != description
            + (PIXEL_SUGGESTION_MARKER if pixel_suggestions else "")
            or project.created_by_id != owner.id
            or project.show_collab_predictions != pixel_suggestions
            or (
                pixel_suggestions
                and project.model_version != PIXEL_SUGGESTION_VERSION
            )
            or project.sampling != Project.SEQUENCE
            or project.maximum_annotations != 1
            or project.min_annotations_to_start_training
        ):
            raise ValueError("existing project configuration differs")
        roots = sorted(
            LocalFilesImportStorage.objects.filter(
                project=project,
            ).values_list("path", flat=True)
        )
        if roots != plan["storage_roots"]:
            raise ValueError("project local storage roots differ")
        actual = list(
            project.tasks.order_by("id").values_list("data", flat=True)
        )
        if actual != [task["data"] for task in plan["tasks"]]:
            raise ValueError(
                "existing project tasks differ; refusing to overwrite"
            )
        if project.tasks.filter(predictions__isnull=False).exists():
            if not pixel_suggestions:
                raise ValueError(
                    "predictions require explicit pixel-suggestion opt-in"
                )
            for prediction in project.predictions.select_related("task"):
                if prediction.model_version != PIXEL_SUGGESTION_VERSION:
                    raise ValueError("unexpected prediction source")
                validate_pixel_result(
                    prediction.result, prediction.task.data["image_size"]
                )
        if created and (
            project.tasks.filter(annotations__isnull=False).exists()
            or project.tasks.filter(drafts__isnull=False).exists()
        ):
            raise ValueError("new project unexpectedly contains labels")
        return {
            "project_id": project.id,
            "tasks": len(actual),
            "storage_roots": len(roots),
            "created": created,
        }


def validate_pixel_result(result: list, size: list[int]) -> None:
    """Permit brush proposals only, never preselected human review choices."""
    if not result:
        raise ValueError("empty output must not become a reviewed negative")
    width, height = size
    for region in result:
        value = region.get("value", {})
        rle = value.get("rle", [])
        if (
            region.get("from_name") != "segmentation"
            or region.get("to_name") != "image"
            or region.get("type") != "brushlabels"
            or region.get("original_width") != width
            or region.get("original_height") != height
            or region.get("image_rotation", 0) != 0
            or value.get("brushlabels") not in [["crepe_material"], ["hole"]]
            or value.get("format") != "rle"
            or len(rle) < 4
            or any(type(v) is not int or not 0 <= v <= 255 for v in rle)
            or int.from_bytes(bytes(rle[:4]), "big") != width * height * 4
        ):
            raise ValueError("invalid pixel brush prediction")


def _validate_manifest(pilot: Path, manifest: dict) -> None:
    from PIL import Image

    if manifest["version"] != 1:
        raise ValueError("unsupported pilot version")
    classes = {"unclear"}
    images = set()
    for run in manifest["runs"]:
        classes.update(p["phase"] for p in run["phase_sequence"])
        if run["phase_alignment"] != "unverified":
            raise ValueError("only unaligned pilots are supported")
        if run["status"] not in {
            "ok",
            "missing_video",
            "invalid_video",
            "alignment_rejected",
        }:
            raise ValueError("unknown video status")
        if run["status"] == "alignment_rejected" and (
            not manifest.get("preprocessing") or run.get("frames")
        ):
            raise ValueError("rejected alignment must not contain frames")
        for frame in run.get("frames", []):
            if (
                not manifest.get("preprocessing")
                and frame.get("coordinate_space", "raw") != "raw"
            ):
                raise ValueError(
                    "rectified frames require preprocessing metadata"
                )
            relative = Path(frame["image"])
            image = pilot / relative
            if (
                len(relative.parts) != 2
                or relative.is_absolute()
                or ".." in relative.parts
                or image.is_symlink()
                or image.parent.is_symlink()
                or not image.is_file()
                or image.suffix != ".png"
            ):
                raise ValueError("unsafe or missing frame")
            if (
                hashlib.sha256(image.read_bytes()).hexdigest()
                != (frame["image_sha256"])
            ):
                raise ValueError("frame hash mismatch")
            with Image.open(image) as decoded:
                decoded.verify()
            if (
                frame["human_label"] is not None
                or frame["phase_suggestion"] is not None
                or frame["suggestion_shown"] is not False
                or frame["phase_context_shown"] is not False
                or frame["review_status"] != "unreviewed"
                or frame["suggestion_status"] != "unavailable_unaligned"
            ):
                raise ValueError("pilot must be unannotated and unaligned")
            images.add(image)
    if sorted(classes) != manifest["classes"]:
        raise ValueError("taxonomy differs from recorded phase sequence")
    if not images:
        raise ValueError("pilot has no frames")
    for directory in {image.parent for image in images}:
        if set(directory.iterdir()) != {
            p for p in images if p.parent == directory
        }:
            raise ValueError("frame storage contains unexpected files")
    if manifest.get("preprocessing"):
        _validate_rectified(pilot, manifest)
    if manifest.get("assignment"):
        assignment = manifest["assignment"]
        run_ids = [run["run_id"] for run in manifest["runs"]]
        if (
            assignment["version"] != ASSIGNMENT_VERSION
            or assignment["set_number"] not in {1, 2}
            or len(assignment["assignment_id"]) != SHA256_HEX_LENGTH
            or len(assignment["source_manifest_sha256"]) != SHA256_HEX_LENGTH
            or len(set(run_ids)) != len(run_ids)
            or sorted(assignment["run_ids"]) != sorted(run_ids)
            or manifest.get("preprocessing")
        ):
            raise ValueError("invalid original-frame assignment")


def _validate_rectified(pilot: Path, manifest: dict) -> None:
    from PIL import Image

    preprocessing = manifest["preprocessing"]
    version = preprocessing["version"]
    space = RECTIFIED_COORDINATE_SPACE
    if (
        version
        not in {RECTIFICATION_VERSION, EXPERIMENTAL_RECTIFICATION_VERSION}
        or preprocessing["coordinate_space"] != space
        or len(preprocessing["source_manifest_sha256"]) != SHA256_HEX_LENGTH
    ):
        raise ValueError("unsupported rectified frame contract")
    if version == EXPERIMENTAL_RECTIFICATION_VERSION and (
        preprocessing.get("experimental") is not True
    ):
        raise ValueError("experimental rectification must be explicit")
    for run in manifest["runs"]:
        for frame in run.get("frames", []):
            if (
                frame["coordinate_space"] != space
                or frame["preprocessing_version"] != version
                or len(frame["source_image_sha256"]) != SHA256_HEX_LENGTH
            ):
                raise ValueError("mixed raw and rectified frames")
            directory = Path(frame["image"]).parent.name
            artifacts = (
                ("transform", f"transforms/{directory}.json"),
                ("valid_mask", f"valid/{frame['image']}"),
            )
            for key, expected in artifacts:
                relative = frame[key]
                path = pilot / relative
                if (
                    relative != expected
                    or path.absolute() != path.resolve(strict=True)
                    or hashlib.sha256(path.read_bytes()).hexdigest()
                    != frame[f"{key}_sha256"]
                ):
                    raise ValueError("unsafe or changed rectification artifact")
            transform = json.loads((pilot / frame["transform"]).read_text())
            if (
                transform["target_size"] != frame["image_size"]
                or transform["video_sha256"] != run["video_sha256"]
                or len(set(transform["video_validation_frames"]))
                < MIN_TEMPORAL_SAMPLES
            ):
                raise ValueError("rectification provenance mismatch")
            with Image.open(pilot / frame["image"]) as image:
                if list(image.size) != frame["image_size"]:
                    raise ValueError("rectified image size mismatch")
            with Image.open(pilot / frame["valid_mask"]) as valid:
                if list(valid.size) != frame["image_size"] or valid.mode != "L":
                    raise ValueError("validity mask dimensions or mode differ")
                if not set(valid.tobytes()).issubset(VALID_MASK_VALUES):
                    raise ValueError("validity mask must be binary")


def build_plan(pilot: Path, *, kind: ProjectKind = "phase") -> dict:
    """Validate a prepared pilot and build unannotated tasks in memory."""
    if kind not in {"phase", "masks"}:
        raise ValueError("unknown project kind")
    pilot = pilot.resolve(strict=True)
    content = (pilot / "manifest.json").read_bytes()
    manifest = json.loads(content)
    _validate_manifest(pilot, manifest)
    view = ET.Element("View")
    ET.SubElement(view, "Image", name="image", value="$image")
    if manifest.get("preprocessing", {}).get("experimental"):
        ET.SubElement(
            view,
            "Header",
            value="Experimental rectification: review geometry before labeling",
        )
    ET.SubElement(view, "Header", value="Recorded context: UNALIGNED")
    ET.SubElement(view, "Text", name="context", value="$phase_sequence")
    choices = ET.SubElement(
        view,
        "Choices",
        name="visual_label",
        toName="image",
        choice="single",
        required="true",
    )
    for name in HUMAN_LABELS:
        ET.SubElement(choices, "Choice", value=name)
    tasks = []
    roots = set()
    for run in manifest["runs"]:
        if run["status"] != "ok":
            continue
        for frame in run["frames"]:
            roots.add(str((pilot / frame["image"]).parent))
            tasks.append(
                {
                    "data": {
                        **frame,
                        "image": "/data/local-files/?d="
                        + quote(frame["image"]),
                        "run_id": run["run_id"],
                        "time_base": run["time_base"],
                        "start_pts": run["start_pts"],
                        "phase_sequence": "UNALIGNED recorded run context: "
                        + " -> ".join(
                            p["phase"] for p in run["phase_sequence"]
                        ),
                        "phase_context_shown": True,
                        "human_taxonomy_version": HUMAN_TAXONOMY_VERSION,
                    },
                    "annotations": [],
                    "predictions": [],
                }
            )
    digest = hashlib.sha256(content).hexdigest()
    plan = {
        "digest": digest,
        "config": ET.tostring(view, encoding="unicode"),
        "tasks": tasks,
        "storage_roots": sorted(roots),
        "title": "Visual pilot v1 " + digest[:32],
        "description": HUMAN_TAXONOMY_VERSION + "; SHA256=" + digest,
    }
    if manifest.get("assignment"):
        assignment = manifest["assignment"]
        number = assignment["set_number"]
        plan["assignment"] = assignment
        plan["title"] = f"Set {number} phase " + digest[:32]
        plan["description"] += (
            f"; set={number}; assignment=" + assignment["assignment_id"]
        )
        for task in tasks:
            task["data"].update(
                {
                    "assignment_id": assignment["assignment_id"],
                    "assignment_set": number,
                    "parent_pilot_sha256": assignment["source_manifest_sha256"],
                }
            )
    return _mask_plan(plan, manifest, pilot) if kind == "masks" else plan


def _mask_plan(plan: dict, manifest: dict, pilot: Path) -> dict:
    from PIL import Image

    view = ET.Element("View")
    ET.SubElement(view, "Image", name="image", value="$image")
    ET.SubElement(
        view,
        "Header",
        value="Paint visible crepe material and enclosed bare-plate holes. "
        "Exclude utensils, shadows and glare. Erase overlaps; skip if unsure.",
    )
    if manifest.get("preprocessing", {}).get("experimental"):
        ET.SubElement(
            view,
            "Header",
            value="Experimental rectification: review geometry before labeling",
        )
    elif not manifest.get("preprocessing"):
        ET.SubElement(
            view, "Header", value="Original camera frames: no rectification"
        )
    labels = ET.SubElement(
        view, "BrushLabels", name="segmentation", toName="image"
    )
    for name, colour in (("crepe_material", "#36b37e"), ("hole", "#ff5630")):
        ET.SubElement(labels, "Label", value=name, background=colour)
    state = ET.SubElement(
        view,
        "Choices",
        name="mask_state",
        toName="image",
        choice="single",
        required="true",
    )
    for name in ("crepe_present", "no_crepe"):
        ET.SubElement(state, "Choice", value=name)
    original_images = {
        "/data/local-files/?d=" + quote(frame["image"]): frame["image"]
        for run in manifest["runs"]
        for frame in run.get("frames", [])
    }
    for task in plan["tasks"]:
        if not manifest.get("preprocessing"):
            with Image.open(
                pilot / original_images[task["data"]["image"]]
            ) as image:
                task["data"].update(
                    {
                        "image_size": list(image.size),
                        "coordinate_space": "original_video",
                        "preprocessing_version": ORIGINAL_FRAME_VERSION,
                    }
                )
        task["data"].pop("phase_sequence", None)
        task["data"].pop("human_taxonomy_version", None)
        task["data"].update(
            {
                "phase_context_shown": False,
                "mask_taxonomy_version": MASK_TAXONOMY_VERSION,
                "source_pilot_sha256": plan["digest"],
            }
        )
    result = {
        **plan,
        "config": ET.tostring(view, encoding="unicode"),
        "title": (
            "Crepe masks v1 "
            if manifest.get("preprocessing")
            else "Original masks v1 "
        )
        + plan["digest"][:32],
        "description": MASK_TAXONOMY_VERSION + "; SHA256=" + plan["digest"],
    }
    if plan.get("assignment"):
        assignment = plan["assignment"]
        number = assignment["set_number"]
        result["title"] = f"Set {number} masks " + plan["digest"][:32]
        result["description"] += (
            f"; set={number}; assignment=" + assignment["assignment_id"]
        )
    return result


def main() -> int:
    """Start the local app or configure a signed-up owner's pilot project."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["start", "setup"])
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--app-data", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--owner-email")
    parser.add_argument("--allow-signup", action="store_true")
    parser.add_argument("--kind", choices=["phase", "masks"], default="phase")
    args = parser.parse_args()
    os.umask(0o077)
    try:
        pilot, app = args.pilot.absolute(), args.app_data.absolute()
        command, env = runtime(pilot, app, args.port)
        if args.action == "start":
            if args.allow_signup:
                env["LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK"] = "false"
            os.chdir(app)
            os.execve(command[0], command, env)
        if not args.owner_email:
            raise ValueError("setup requires --owner-email after signup")
        code = (
            "import json, runpy\nfrom pathlib import Path\n"
            f"workflow = runpy.run_path({str(Path(__file__).resolve())!r})\n"
            f"result = workflow['setup_project'](Path({str(pilot)!r}), "
            f"{args.owner_email!r}, kind={args.kind!r})\n"
            "print('PHASE_SETUP_RESULT=' + json.dumps(result))\n"
        )
        log_name = "phase" if args.kind == "phase" else "masks"
        with (app / f".{log_name}-setup.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            result = subprocess.run(
                command[:3] + ["shell"],
                env=env,
                cwd=app,
                input=f"exec({code!r})\n",
                capture_output=True,
                text=True,
                check=False,
            )
            (app / f"{log_name}-setup.log").write_text(
                result.stdout + result.stderr
            )
        marker = "PHASE_SETUP_RESULT="
        for line in result.stdout.splitlines():
            if marker in line and not result.returncode:
                summary = json.loads(line.split(marker, 1)[1])
                print(json.dumps(summary, sort_keys=True))
                return 0
        print(
            f"Setup failed; inspect private app-data/{log_name}-setup.log.",
            file=sys.stderr,
        )
        return 1
    except (OSError, ValueError, KeyError, TypeError):
        print(
            "Invalid or unsafe setup. Check paths, pilot and arguments.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
