"""Synthetic phase/mask coexistence inside an isolated management shell."""

import copy
import json
import runpy
import sys
import time
from pathlib import Path

import numpy as np
from django.conf import settings
from django.test import Client
from label_studio_sdk.converter.brush import encode_rle
from organizations.models import Organization
from projects.models import Project
from users.models import User

source = Path(globals()["SOURCE"])
pilot = Path(globals()["PILOT"])
sys.path.insert(0, str(source))
from labeling.mask_export import export_masks  # noqa: E402

workflow = runpy.run_path(str(source / "labeling" / "workflow.py"))
setup = workflow["setup_project"]
user = User.objects.create_user(
    email="mask-test@example.invalid", password=None
)
org = Organization.create_organization(created_by=user, title="Synthetic masks")
user.active_organization = org
user.save()
phase_result = setup(pilot, user.email)
phase = Project.objects.get(pk=phase_result["project_id"])
phase_config = phase.label_config
phase_data = list(phase.tasks.order_by("id").values_list("data", flat=True))
client = Client(HTTP_HOST="127.0.0.1")
client.force_login(user, backend="django.contrib.auth.backends.ModelBackend")
session = client.session
session["last_login"] = time.time()
session.save()
client.cookies[settings.SESSION_COOKIE_NAME] = session.session_key
response = client.post(
    f"/api/tasks/{phase.tasks.first().id}/annotations/",
    data=json.dumps(
        {
            "result": [
                {
                    "from_name": "visual_label",
                    "to_name": "image",
                    "type": "choices",
                    "value": {"choices": ["circle"]},
                }
            ],
            "was_cancelled": False,
        }
    ),
    content_type="application/json",
)
assert response.status_code == 201, response.content
saved_phase_annotation = copy.deepcopy(response.json())
created = setup(pilot, user.email, kind="masks")
mask = Project.objects.get(pk=created["project_id"])
assert mask.id != phase.id
assert mask.tasks.count() == phase.tasks.count() == 3
assert setup(pilot, user.email, kind="masks")["created"] is False
assert Project.objects.count() == 2
phase.refresh_from_db()
assert phase.label_config == phase_config
assert (
    list(phase.tasks.order_by("id").values_list("data", flat=True))
    == phase_data
)
phase_annotation = phase.tasks.first().annotations.get()
assert phase_annotation.result == saved_phase_annotation["result"]
tasks = list(mask.tasks.order_by("id"))
assert tasks[0].data["image"] == phase_data[0]["image"]
response = client.get(tasks[0].data["image"])
assert response.status_code == 200
pixels = (
    b"".join(response.streaming_content)
    if response.streaming
    else response.content
)
assert pixels == (pilot / "run-000" / "000000.png").read_bytes()
regions = []
for label, position in (("crepe_material", (1, 1)), ("hole", (4, 4))):
    rgba = np.zeros((8, 8, 4), np.uint8)
    rgba[position[0], position[1], -1] = 255
    regions.append(
        {
            "from_name": "segmentation",
            "to_name": "image",
            "type": "brushlabels",
            "original_width": 8,
            "original_height": 8,
            "image_rotation": 0,
            "value": {
                "format": "rle",
                "rle": encode_rle(rgba.ravel()),
                "brushlabels": [label],
            },
        }
    )
regions.append(
    {
        "from_name": "mask_state",
        "to_name": "image",
        "type": "choices",
        "value": {"choices": ["crepe_present"]},
    }
)
response = client.post(
    f"/api/tasks/{tasks[0].id}/annotations/",
    data=json.dumps({"result": regions, "was_cancelled": False}),
    content_type="application/json",
)
assert response.status_code == 201, response.content
assert setup(pilot, user.email, kind="masks")["created"] is False
response = client.get(
    f"/api/projects/{mask.id}/export",
    {
        "exportType": "JSON",
        "download_all_tasks": "true",
        "download_resources": "false",
    },
)
assert response.status_code == 200, response.content
exported = (
    b"".join(response.streaming_content)
    if response.streaming
    else response.content
)
annotation_file = pilot.parent / "human-masks.json"
annotation_file.write_bytes(exported)
summary = export_masks(
    pilot, annotation_file, pilot.parent / "exports" / "reviewed"
)
assert summary == {"exported": 1, "tasks_in_json": 3}
assert phase.tasks.first().annotations.count() == 1
mask.label_config = mask.label_config.replace('value="hole"', 'value="other"')
mask.save()
try:
    setup(pilot, user.email, kind="masks")
except ValueError:
    pass
else:
    raise AssertionError("changed mask configuration must fail closed")
assert tasks[0].annotations.count() == 1
assert (
    phase.tasks.first().annotations.get().result
    == saved_phase_annotation["result"]
)
print("MASK_INTEGRATION_OK")
