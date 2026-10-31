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
from urllib.parse import quote
from xml.etree import ElementTree as ET

HUMAN_TAXONOMY_VERSION = "crepe-visual-v1"
HUMAN_LABELS = ("emptyplate", "circle", "halffold", "quarter fold", "ugly")


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


def setup_project(pilot: Path, owner_email: str) -> dict:
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

    plan = build_plan(pilot)
    if Path(settings.LOCAL_FILES_DOCUMENT_ROOT) != pilot.resolve():
        raise ValueError("server local-file root does not match pilot")
    owner = User.objects.filter(email=owner_email, is_active=True).first()
    if owner is None or owner.active_organization is None:
        raise ValueError("sign up in this app first, then supply your email")
    organization = owner.active_organization
    if not organization.has_permission(owner):
        raise ValueError("owner lacks active organization membership")
    title = "Visual pilot v1 " + plan["digest"][:32]
    description = HUMAN_TAXONOMY_VERSION + "; SHA256=" + plan["digest"]
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
        if (
            project.label_config != plan["config"]
            or project.description != description
            or project.created_by_id != owner.id
            or project.show_collab_predictions
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
            raise ValueError("predictions are not allowed in phase pilot")
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
        if run["status"] not in {"ok", "missing_video", "invalid_video"}:
            raise ValueError("unknown video status")
        for frame in run.get("frames", []):
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


def build_plan(pilot: Path) -> dict:
    """Validate a prepared pilot and build unannotated tasks in memory."""
    pilot = pilot.resolve(strict=True)
    content = (pilot / "manifest.json").read_bytes()
    manifest = json.loads(content)
    _validate_manifest(pilot, manifest)
    view = ET.Element("View")
    ET.SubElement(view, "Image", name="image", value="$image")
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
    return {
        "digest": hashlib.sha256(content).hexdigest(),
        "config": ET.tostring(view, encoding="unicode"),
        "tasks": tasks,
        "storage_roots": sorted(roots),
    }


def main() -> int:
    """Start the local app or configure a signed-up owner's pilot project."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["start", "setup"])
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--app-data", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--owner-email")
    parser.add_argument("--allow-signup", action="store_true")
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
            f"{args.owner_email!r})\n"
            "print('PHASE_SETUP_RESULT=' + json.dumps(result))\n"
        )
        with (app / ".phase-setup.lock").open("w") as lock:
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
            (app / "phase-setup.log").write_text(result.stdout + result.stderr)
        marker = "PHASE_SETUP_RESULT="
        for line in result.stdout.splitlines():
            if marker in line and not result.returncode:
                summary = json.loads(line.split(marker, 1)[1])
                print(json.dumps(summary, sort_keys=True))
                return 0
        print(
            "Setup failed; inspect private app-data/phase-setup.log.",
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
