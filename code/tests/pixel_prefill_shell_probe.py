"""Exercise protected tasks and prediction visibility in a synthetic app."""

import datetime as dt
import hashlib
import json
import runpy
import sys
from pathlib import Path

import numpy as np
from django.utils import timezone
from label_studio_sdk.converter.brush import decode_rle
from organizations.models import Organization
from PIL import Image
from projects.models import Project
from tasks.models import Annotation, AnnotationDraft, Prediction, TaskLock
from users.models import User

source = Path(globals()["SOURCE"])
pilot = Path(globals()["PILOT"])
sys.path.insert(0, str(source))
from labeling.pixel_predictions import install_predictions  # noqa: E402

workflow = runpy.run_path(str(source / "labeling/workflow.py"))
setup = workflow["setup_project"]
user = User.objects.create_user(
    email="pixel-test@example.invalid", password=None
)
user.active_organization = Organization.create_organization(
    created_by=user, title="Pixels"
)
user.save()
phase = Project.objects.get(pk=setup(pilot, user.email)["project_id"])
project = Project.objects.get(
    pk=setup(pilot, user.email, kind="masks")["project_id"]
)
tasks = list(project.tasks.order_by("id"))
original_data = [task.data for task in tasks]
annotation = Annotation.objects.create(
    task=tasks[0], project=project, completed_by=user, result=[]
)
draft = AnnotationDraft.objects.create(task=tasks[1], user=user, result=[])
TaskLock.objects.create(
    task=tasks[2], user=user, expire_at=timezone.now() + dt.timedelta(minutes=5)
)
Annotation.objects.create(
    task=tasks[3],
    project=project,
    completed_by=user,
    result=[],
    was_cancelled=True,
)
plan = workflow["build_plan"](pilot, kind="masks")
(pilot.parent / "workspace.json").write_text(
    json.dumps(
        {
            "projects": {"masks": {"project_id": project.id}},
        }
    )
)
folder = pilot.parent / "pixel-job"
folder.mkdir(mode=0o700)
records = []
for task in tasks:
    destination = folder / f"task-{task.id}"
    destination.mkdir()
    material = np.zeros((8, 8), np.uint8)
    material[1:6, 1:6] = 255
    material[3, 3] = 0
    holes = np.zeros_like(material)
    holes[3, 3] = 255
    artifacts = {}
    for name, pixels in (("material", material), ("holes", holes)):
        path = destination / f"{name}.png"
        Image.fromarray(pixels).save(path)
        artifacts[name] = {
            "path": str(path.relative_to(folder)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    records.append(
        {
            "task_id": task.id,
            "task_data_sha256": hashlib.sha256(
                json.dumps(
                    task.data, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest(),
            "image_sha256": task.data["image_sha256"],
            "image_size": [8, 8],
            "masks": artifacts,
        }
    )
(folder / "manifest.json").write_text(
    json.dumps(
        {
            "algorithm": "pixel-colour-components-v1",
            "human_annotation": False,
            "pilot_sha256": plan["digest"],
            "assignment": plan["assignment"],
            "project_id": project.id,
            "empty_not_imported": 0,
            "records": records,
        }
    )
)
dry = install_predictions(pilot.parent, folder)
assert dry["eligible"] == 3 and dry["protected"] == 4 and dry["created"] == 0
assert not Prediction.objects.exists()
# Work saved after generation/dry-run must win over any proposed pixels.
late = AnnotationDraft.objects.create(task=tasks[5], user=user, result=[])
applied = install_predictions(pilot.parent, folder, apply=True)
assert applied["created"] == 2 and applied["protected"] == 5
assert set(Prediction.objects.values_list("task_id", flat=True)) == {
    tasks[4].id,
    tasks[6].id,
}
assert Annotation.objects.get(pk=annotation.id).result == []
assert AnnotationDraft.objects.filter(pk__in=[draft.id, late.id]).count() == 2
assert [t.data for t in project.tasks.order_by("id")] == original_data
assert not phase.tasks.filter(predictions__isnull=False).exists()
for prediction in Prediction.objects.all():
    assert all(r["from_name"] == "segmentation" for r in prediction.result)
    assert all(
        r["original_width"] == r["original_height"] == 8
        for r in prediction.result
    )
    for region in prediction.result:
        decoded = decode_rle(region["value"]["rle"]).reshape(8, 8, 4)[..., -1]
        expected = (
            material
            if region["value"]["brushlabels"] == ["crepe_material"]
            else holes
        )
        assert np.array_equal(decoded, expected)
assert setup(pilot, user.email, kind="masks")["created"] is False
again = install_predictions(pilot.parent, folder, apply=True)
assert again["created"] == 0 and again["existing"] == 2
assert Prediction.objects.count() == 2
project.refresh_from_db()
assert project.show_collab_predictions is True
assert project.model_version == "pixel-colour-components-v1"
first = Prediction.objects.first()
first.model_version = "unrelated"
first.save()
try:
    setup(pilot, user.email, kind="masks")
except ValueError:
    pass
else:
    raise AssertionError("unrelated predictions must remain forbidden")
print("PIXEL_PREFILL_OK")
