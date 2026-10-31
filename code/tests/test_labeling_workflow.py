"""Synthetic-only contract and installed Label Studio integration tests."""

import hashlib
import importlib.util
import json
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType
from urllib.error import URLError
from urllib.request import urlopen
from xml.etree import ElementTree

import pytest

WORKFLOW = Path(__file__).parents[1] / "labeling" / "workflow.py"
pytestmark = pytest.mark.skipif(
    sys.platform not in {"linux", "darwin"},
    reason="labeling workflow requires POSIX file permissions and locking",
)


def load_workflow() -> ModuleType:
    assert WORKFLOW.is_file(), "portable workflow is not implemented"
    spec = importlib.util.spec_from_file_location("phase_workflow", WORKFLOW)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=["raw", "rectified"])
def pilot(tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    from PIL import Image

    root = tmp_path / "private" / "labeling" / "pilot"
    frames = root / "run-000"
    frames.mkdir(parents=True, mode=0o700)
    root.chmod(0o700)
    image = frames / "000000.png"
    Image.new("RGB", (8, 8), "red").save(image)
    manifest = {
        "version": 1,
        "classes": ["cook", "unclear"],
        "runs": [
            {
                "run_id": 1,
                "status": "ok",
                "phase_alignment": "unverified",
                "phase_sequence": [{"phase": "cook", "elapsed_sec": 3}],
                "time_base": "1/10",
                "start_pts": 0,
                "frames": [
                    {
                        "image": "run-000/000000.png",
                        "image_sha256": hashlib.sha256(
                            image.read_bytes()
                        ).hexdigest(),
                        "requested_seconds": 0,
                        "decoded_pts": 0,
                        "human_label": None,
                        "phase_suggestion": None,
                        "suggestion_status": "unavailable_unaligned",
                        "suggestion_shown": False,
                        "phase_context_shown": False,
                        "review_status": "unreviewed",
                    }
                ],
            }
        ],
    }
    frames_data = manifest["runs"][0]["frames"]
    frames_data.extend(
        [
            {**frames_data[0], "requested_seconds": 5},
            {**frames_data[0], "requested_seconds": 10},
        ]
    )
    if request.param == "rectified":
        manifest["preprocessing"] = {
            "version": "upstream-rectified-v1",
            "coordinate_space": "rectified_upstream",
            "source_manifest_sha256": "0" * 64,
        }
        transform = root / "transforms" / "run-000.json"
        transform.parent.mkdir()
        transform.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "source_size": [8, 8],
                    "target_size": [8, 8],
                    "video_sha256": "1" * 64,
                    "video_validation_frames": [0, 1],
                }
            )
        )
        valid = root / "valid" / "run-000" / "000000.png"
        valid.parent.mkdir(parents=True)
        Image.new("L", (8, 8), 255).save(valid)
        manifest["runs"][0]["video_sha256"] = "1" * 64
        for frame in frames_data:
            frame.update(
                {
                    "coordinate_space": "rectified_upstream",
                    "preprocessing_version": "upstream-rectified-v1",
                    "source_image_sha256": frame["image_sha256"],
                    "image_size": [8, 8],
                    "transform": "transforms/run-000.json",
                    "transform_sha256": hashlib.sha256(
                        transform.read_bytes()
                    ).hexdigest(),
                    "valid_mask": "valid/run-000/000000.png",
                    "valid_mask_sha256": hashlib.sha256(
                        valid.read_bytes()
                    ).hexdigest(),
                }
            )
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def test_plan_uses_fixed_visual_taxonomy_without_automatic_labels(
    pilot: Path,
) -> None:
    source = (pilot / "manifest.json").read_bytes()
    plan = load_workflow().build_plan(pilot)
    config = ElementTree.fromstring(plan["config"])
    choices = config.find("Choices")
    assert choices is not None
    assert choices.attrib["choice"] == "single"
    assert choices.attrib["required"] == "true"
    assert choices.attrib["name"] == "visual_label"
    assert [c.attrib["value"] for c in choices] == [
        "emptyplate",
        "spreading",
        "circle",
        "halffold",
        "quarter fold",
        "ugly",
    ]
    task = plan["tasks"][0]
    assert task["annotations"] == task["predictions"] == []
    assert task["data"]["image"] == ("/data/local-files/?d=run-000/000000.png")
    assert "UNALIGNED" in task["data"]["phase_sequence"]
    assert "cook" in task["data"]["phase_sequence"]
    assert task["data"]["phase_context_shown"] is True
    assert task["data"]["phase_suggestion"] is None
    assert task["data"]["human_label"] is None
    assert task["data"]["human_taxonomy_version"] == "crepe-visual-v2"
    assert (pilot / "manifest.json").read_bytes() == source
    assert plan["storage_roots"] == [str(pilot / "run-000")]


@pytest.mark.parametrize(
    "damage",
    [
        "hash",
        "traversal",
        "symlink",
        "taxonomy",
        "prediction",
        "label",
        "missing",
        "version",
        "empty",
        "directory_secret",
    ],
)
def test_plan_rejects_unsafe_or_changed_pilot(pilot: Path, damage: str) -> None:
    manifest_path = pilot / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    frame = manifest["runs"][0]["frames"][0]
    image = pilot / frame["image"]
    if damage == "hash":
        image.write_bytes(b"changed")
    elif damage == "traversal":
        frame["image"] = "../outside.png"
    elif damage == "symlink":
        outside = pilot.parent / "outside.png"
        image.rename(outside)
        image.symlink_to(outside)
    elif damage == "taxonomy":
        manifest["classes"].append("invented")
    elif damage == "prediction":
        frame["phase_suggestion"] = "cook"
    elif damage == "label":
        frame["human_label"] = "cook"
    elif damage == "missing":
        image.unlink()
    elif damage == "version":
        manifest["version"] = 2
    elif damage == "empty":
        manifest["runs"] = []
    elif damage == "directory_secret":
        (image.parent / "secret.txt").write_text("synthetic secret")
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_workflow().build_plan(pilot)


def test_runtime_is_private_loopback_and_ignores_ambient_settings(
    pilot: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_workflow()
    assert hasattr(module, "runtime"), "safe startup is not implemented"
    monkeypatch.setenv("LABEL_STUDIO_DATABASE_NAME", "/wrong.sqlite3")
    monkeypatch.setenv("LABEL_STUDIO_USERNAME", "must-not-create@example.org")
    monkeypatch.setenv("LABEL_STUDIO_PASSWORD", "must-not-use")
    app = pilot.parent / "app"
    command, env = module.runtime(pilot, app, 8080)
    assert command[command.index("--internal-host") + 1] == "127.0.0.1"
    assert "--no-browser" in command
    assert env["LABEL_STUDIO_BASE_DATA_DIR"] == str(app)
    assert env["LABEL_STUDIO_DATABASE_NAME"] == str(
        app / "label_studio.sqlite3"
    )
    assert env["LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"] == str(pilot)
    assert env["LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED"] == "true"
    assert env["COLLECT_ANALYTICS"] == "False"
    assert env["LATEST_VERSION_CHECK"] == "False"
    assert env.get("SENTRY_DSN") == ""
    assert env.get("FRONTEND_SENTRY_DSN") == ""
    assert env["LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK"] == "true"
    assert "LABEL_STUDIO_USERNAME" not in env
    assert "LABEL_STUDIO_PASSWORD" not in env
    assert app.stat().st_mode & 0o077 == 0
    with pytest.raises(ValueError):
        module.runtime(pilot, pilot / "db", 8080)
    with pytest.raises(ValueError):
        module.runtime(pilot, WORKFLOW.parent / "db", 8080)
    public = pilot.parent / "public"
    public.mkdir(mode=0o755)
    public.chmod(0o755)
    with pytest.raises(ValueError):
        module.runtime(pilot, public, 8080)


def test_installed_app_import_serves_images_and_preserves_human_export(
    pilot: Path,
) -> None:
    pytest.importorskip("label_studio")
    module = load_workflow()
    assert hasattr(module, "setup_project"), "project setup is not implemented"
    assert hasattr(module, "main"), "workflow CLI is not implemented"
    app = pilot.parent / "app"
    _, env = module.runtime(pilot, app, 8080)
    probe = Path(__file__).with_name("labeling_shell_probe.py")
    script = (
        "import runpy\n"
        f"runpy.run_path({str(probe)!r}, init_globals="
        f"{{'PILOT': {str(pilot)!r}, 'WORKFLOW': {str(WORKFLOW)!r}}})\n"
    )
    result = subprocess.run(
        [sys.executable, "-m", "label_studio.server", "shell"],
        input=f"exec({script!r})\n",
        text=True,
        capture_output=True,
        env=env,
        cwd=app,
        timeout=180,
    )
    assert "SYNTHETIC_INTEGRATION_OK" in result.stdout, (
        result.stdout + result.stderr
    )
    command = [
        sys.executable,
        str(WORKFLOW),
        "setup",
        "--pilot",
        str(pilot),
        "--app-data",
        str(app),
        "--owner-email",
        "synthetic@example.invalid",
    ]
    for _ in range(2):
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=90
        )
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["created"] is False
    result = subprocess.run(
        command[:-1] + ["missing@example.invalid"],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode != 0
    assert "missing@example.invalid" not in result.stdout + result.stderr


def test_start_cli_serves_login_on_loopback_and_stops(pilot: Path) -> None:
    pytest.importorskip("label_studio")
    assert hasattr(load_workflow(), "main"), "workflow CLI is not implemented"
    app = pilot.parent / "app"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = [
        sys.executable,
        str(WORKFLOW),
        "start",
        "--pilot",
        str(pilot),
        "--app-data",
        str(app),
        "--port",
        str(port),
        "--allow-signup",
    ]
    log = pilot.parent / "startup.log"
    with log.open("w") as output:
        process = subprocess.Popen(command, stdout=output, stderr=output)
        try:
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    pytest.fail(log.read_text())
                try:
                    with urlopen(
                        f"http://127.0.0.1:{port}/user/login/", timeout=1
                    ) as response:
                        assert response.status == 200
                    break
                except (URLError, TimeoutError):
                    time.sleep(0.2)
            else:
                pytest.fail(log.read_text())
            if sys.platform == "darwin":
                result = subprocess.run(
                    [
                        "/usr/sbin/lsof",
                        "-nP",
                        "-a",
                        "-p",
                        str(process.pid),
                        f"-iTCP:{port}",
                        "-sTCP:LISTEN",
                        "-Fn",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=10,
                )
                listeners = [
                    line[1:]
                    for line in result.stdout.splitlines()
                    if line.startswith("n")
                ]
                assert listeners == [f"127.0.0.1:{port}"]
            else:
                listeners = [
                    line.split()[1]
                    for line in Path("/proc/net/tcp")
                    .read_text()
                    .splitlines()[1:]
                    if line.split()[3] == "0A"
                    and line.split()[1].endswith(f":{port:04X}")
                ]
                assert listeners == [f"0100007F:{port:04X}"]
            assert (app / "label_studio.sqlite3").is_file()
            assert (app / "label_studio.sqlite3").stat().st_mode & 0o077 == 0
        finally:
            process.terminate()
            process.wait(timeout=15)
