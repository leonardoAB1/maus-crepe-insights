"""Execute only inside an isolated synthetic Label Studio management shell."""

import json
import runpy
import time
from pathlib import Path

from django.conf import settings
from django.test import Client
from io_storages.localfiles.models import LocalFilesImportStorage
from organizations.models import Organization
from projects.models import Project
from tasks.models import Annotation, AnnotationDraft, Prediction, Task
from users.models import User

pilot = Path(globals()["PILOT"])
workflow = runpy.run_path(globals()["WORKFLOW"])
setup = workflow["setup_project"]
assert not User.objects.exists()
try:
    setup(pilot, "missing@example.invalid")
except ValueError:
    pass
else:
    raise AssertionError("setup must require an existing account")
assert not Project.objects.exists()
user = User.objects.create_user(
    email="synthetic@example.invalid", password=None
)
org = Organization.create_organization(created_by=user, title="Synthetic")
user.active_organization = org
user.save()
first = setup(pilot, user.email)
project = Project.objects.get(pk=first["project_id"])
assert project.created_by_id == user.id
assert project.organization_id == org.id
assert project.tasks.count() == 3
assert not Annotation.objects.exists()
assert not AnnotationDraft.objects.exists()
assert not Prediction.objects.exists()
assert LocalFilesImportStorage.objects.get(project=project).path == str(
    pilot / "run-000"
)
assert setup(pilot, user.email)["project_id"] == project.id
assert Project.objects.count() == 1
assert Task.objects.count() == 3
assert LocalFilesImportStorage.objects.count() == 1
assert Path(settings.LOCAL_FILES_DOCUMENT_ROOT) == pilot
assert settings.COLLECT_ANALYTICS is False
assert settings.LATEST_VERSION_CHECK is False
assert not settings.SENTRY_DSN
assert not settings.FRONTEND_SENTRY_DSN
assert (
    Path(settings.DATABASES["default"]["NAME"]).parent == pilot.parent / "app"
)
client = Client(HTTP_HOST="127.0.0.1")
task, skipped, unlabelled = list(project.tasks.order_by("id"))
url = task.data["image"]
assert client.get(url).status_code != 200
client.force_login(user, backend="django.contrib.auth.backends.ModelBackend")
session = client.session
session["last_login"] = time.time()
session.save()
client.cookies[settings.SESSION_COOKIE_NAME] = session.session_key
response = client.get(url)
assert response.status_code == 200, response.content
content = (
    b"".join(response.streaming_content)
    if response.streaming
    else response.content
)
assert content == (pilot / "run-000/000000.png").read_bytes()
for forbidden in (
    "manifest.json",
    "../app/label_studio.sqlite3",
    "../raw/secret",
):
    assert client.get("/data/local-files/", {"d": forbidden}).status_code != 200
# Community is single-organization: all app accounts must be trusted.
# Submit a synthetic human choice through the authenticated API, not import.
annotation = {
    "result": [
        {
            "from_name": "visual_label",
            "to_name": "image",
            "type": "choices",
            "value": {"choices": ["quarter fold"]},
        }
    ],
    "was_cancelled": False,
    "ground_truth": False,
}
response = client.post(
    f"/api/tasks/{task.id}/annotations/",
    data=json.dumps(annotation),
    content_type="application/json",
)
assert response.status_code == 201, response.content
assert setup(pilot, user.email)["project_id"] == project.id
assert Annotation.objects.count() == 1
response = client.post(
    f"/api/tasks/{skipped.id}/annotations/",
    data=json.dumps(
        {"result": [], "was_cancelled": True, "ground_truth": False}
    ),
    content_type="application/json",
)
assert response.status_code == 201, response.content
response = client.get(
    f"/api/projects/{project.id}/export",
    {
        "exportType": "JSON",
        "download_all_tasks": "true",
        "download_resources": "false",
    },
)
assert response.status_code == 200, response.content
exported = json.loads(
    b"".join(response.streaming_content)
    if response.streaming
    else response.content
)
exported.sort(key=lambda row: row["id"])
assert len(exported) == 3
assert exported[0]["annotations"][0]["result"][0]["value"] == {
    "choices": ["quarter fold"]
}
assert exported[0]["data"]["human_label"] is None
assert exported[0]["predictions"] == []
assert exported[1]["annotations"][0]["was_cancelled"] is True
assert exported[1]["annotations"][0]["result"] == []
assert exported[2]["annotations"] == []
# A changed project must fail closed without replacing human annotations.
project.label_config = project.label_config.replace(
    'value="circle"',
    'value="other"',
)
project.save()
try:
    setup(pilot, user.email)
except ValueError:
    pass
else:
    raise AssertionError("changed project config accepted")
assert Annotation.objects.count() == 2
print("SYNTHETIC_INTEGRATION_OK")
project.label_config = workflow["build_plan"](pilot)["config"]
project.save()
