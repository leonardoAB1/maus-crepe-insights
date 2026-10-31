"""Private handoffs and assignment startup use synthetic data only."""

import io
import json
import socket
import sqlite3
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pytest
from PIL import Image

if sys.platform not in {"linux", "darwin"}:
    pytest.skip("POSIX-only labeling environment", allow_module_level=True)

from labeling.assignments import (  # noqa: E402
    bundle_workspace,
    file_hash,
    receive_bundle,
    split_pilot,
    validate_workspace,
)
from labeling.collaboration import setup_assignment  # noqa: E402
from labeling.workflow import build_plan, runtime  # noqa: E402

SOURCE = Path(__file__).parents[1]


@pytest.fixture
def source_pilot(tmp_path: Path) -> Path:
    pilot = tmp_path / "data" / "labeling" / "original"
    pilot.mkdir(parents=True, mode=0o700)
    pilot.parent.chmod(0o700)
    runs = []
    for index, count in enumerate((8, 5, 4, 3, 2, 1)):
        folder = pilot / f"run-{index:03d}"
        folder.mkdir(mode=0o700)
        frames = []
        for number in range(count):
            path = folder / f"{number:06d}.png"
            Image.new("RGB", (8, 8), (index * 20, number * 20, 30)).save(path)
            frames.append(
                {
                    "image": str(path.relative_to(pilot)),
                    "image_sha256": file_hash(path),
                    "requested_seconds": number * 5,
                    "decoded_frame_index": number,
                    "decoded_pts": number * 50,
                    "human_label": None,
                    "phase_suggestion": None,
                    "suggestion_status": "unavailable_unaligned",
                    "suggestion_shown": False,
                    "phase_context_shown": False,
                    "review_status": "unreviewed",
                    "private_note": "do-not-share",
                }
            )
        runs.append(
            {
                "run_id": index,
                "status": "ok",
                "phase_alignment": "unverified",
                "phase_sequence": [
                    {"phase": "cook", "private_note": "do-not-share"}
                ],
                "time_base": "1/10",
                "start_pts": 0,
                "frames": frames,
                "raw_path": "do-not-share",
                "video_path": "do-not-share",
            }
        )
    (pilot / "manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "classes": ["cook", "unclear"],
                "runs": runs,
                "private_note": "do-not-share",
            }
        )
    )
    return pilot


def test_split_is_disjoint_deterministic_and_original_pixels(
    source_pilot: Path,
) -> None:
    first = source_pilot.parent / "assignment-a"
    second = source_pilot.parent / "assignment-b"
    before = (source_pilot / "manifest.json").read_bytes()
    summary = split_pilot(source_pilot, first)
    assert summary == split_pilot(source_pilot, second)
    source = json.loads(before)
    observed = []
    counts = []
    for number in (1, 2):
        pilot = first / f"set-{number}" / "pilot"
        manifest = json.loads((pilot / "manifest.json").read_text())
        assert "do-not-share" not in json.dumps(manifest)
        members = {r["run_id"] for r in manifest["runs"]}
        observed.append(members)
        counts.append(sum(len(r["frames"]) for r in manifest["runs"]))
        phase, masks = build_plan(pilot), build_plan(pilot, kind="masks")
        assert phase["title"].startswith(f"Set {number} phase")
        assert masks["title"].startswith(f"Set {number} masks")
        assert len(phase["title"]) <= 50 and len(masks["title"]) <= 50
        for task in masks["tasks"]:
            assert task["data"]["assignment_set"] == number
        for run in manifest["runs"]:
            for frame in run["frames"]:
                original, copied = (
                    source_pilot / frame["image"],
                    pilot / frame["image"],
                )
                assert original.read_bytes() == copied.read_bytes()
                assert not original.samefile(copied)
        assert (pilot / "manifest.json").read_bytes() == (
            second / f"set-{number}" / "pilot/manifest.json"
        ).read_bytes()
    assert not observed[0] & observed[1]
    assert observed[0] | observed[1] == {r["run_id"] for r in source["runs"]}
    assert abs(counts[0] - counts[1]) <= 1
    assert sum(counts) == 23
    assert (source_pilot / "manifest.json").read_bytes() == before
    with pytest.raises(FileExistsError):
        split_pilot(source_pilot, first)


def test_bundle_is_minimal_and_portable(source_pilot: Path) -> None:
    assignment = source_pilot.parent / "sets"
    split_pilot(source_pilot, assignment)
    workspace = assignment / "set-2"
    (workspace / "app").mkdir(mode=0o700)
    (workspace / "app/label_studio.sqlite3").write_text("synthetic secret")
    (workspace / "pilot/.env").write_text("synthetic secret")
    archive = source_pilot.parent / "handoff/set-2.tar.gz"
    result = bundle_workspace(workspace, 2, archive)
    assert (
        archive.with_name(archive.name + ".sha256")
        .read_text()
        .startswith(result["sha256"])
    )
    with tarfile.open(archive) as packed:
        for member in packed.getmembers():
            assert member.isfile()
            assert member.name.startswith("pilot/")
            assert member.name.endswith(("manifest.json", ".png"))
            assert member.uname == member.gname == ""
    received = source_pilot.parent / "received-set-2"
    summary = receive_bundle(archive, result["sha256"], 2, received)
    assert summary["tasks"] == len(validate_workspace(workspace, 2)["tasks"])
    assert (
        build_plan(received / "pilot")["tasks"]
        == build_plan(workspace / "pilot")["tasks"]
    )
    assert not (received / "app").exists()
    with pytest.raises(FileExistsError):
        receive_bundle(archive, result["sha256"], 2, received)
    with pytest.raises(ValueError, match="checksum"):
        receive_bundle(archive, "0" * 64, 2, source_pilot.parent / "bad-hash")
    wrong = source_pilot.parent / "wrong-set"
    with pytest.raises(ValueError, match="Wrong"):
        receive_bundle(archive, result["sha256"], 1, wrong)
    assert not wrong.exists()


@pytest.mark.parametrize(
    "damage", ["traversal", "symlink", "database", "duplicate", "hash"]
)
def test_unsafe_bundles_fail_closed(source_pilot: Path, damage: str) -> None:
    assignment = source_pilot.parent / "sets"
    split_pilot(source_pilot, assignment)
    archive = source_pilot.parent / "set-2.tar.gz"
    bundle_workspace(assignment / "set-2", 2, archive)
    with tarfile.open(archive) as packed:
        files = [
            (m.name, packed.extractfile(m).read()) for m in packed.getmembers()
        ]
    if damage == "hash":
        files[-1] = (files[-1][0], b"changed PNG")
    if damage == "duplicate":
        files.append(files[0])
    bad = source_pilot.parent / "bad.tar.gz"
    with tarfile.open(bad, "w:gz") as packed:
        for name, data in files:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            packed.addfile(info, io.BytesIO(data))
        if damage in {"traversal", "database", "symlink"}:
            name = {
                "traversal": "../escape",
                "database": "app/label_studio.sqlite3",
                "symlink": "pilot/link",
            }[damage]
            info = tarfile.TarInfo(name)
            if damage == "symlink":
                info.type = tarfile.SYMTYPE
                info.linkname = "../../escape"
            packed.addfile(info)
    destination = source_pilot.parent / "refused"
    with pytest.raises(ValueError):
        receive_bundle(bad, file_hash(bad), 2, destination)
    assert not destination.exists()
    assert not (source_pilot.parent / "escape").exists()


def test_startup_installs_assigned_projects_and_preserves_reruns(
    source_pilot: Path,
) -> None:
    pytest.importorskip("label_studio")
    assignment = source_pilot.parent / "sets"
    split_pilot(source_pilot, assignment)
    workspace = assignment / "set-2"
    pilot, app = workspace / "pilot", workspace / "app"
    with socket.socket() as check:
        check.bind(("127.0.0.1", 0))
        port = check.getsockname()[1]
    command, env = runtime(pilot, app, port)
    signup = (
        "from users.models import User\n"
        "from organizations.models import Organization\n"
        "u=User.objects.create_user(email='set-test@example.invalid',password=None)\n"
        "u.active_organization=Organization.create_organization(created_by=u,title='Synthetic')\n"
        "u.save()\nprint('LOCAL_OWNER_CREATED')\n"
    )
    # Installed-app startup creates a fresh DB before simulating local signup.
    launcher = [
        sys.executable,
        str(SOURCE / "labeling/collaboration.py"),
        "start",
        "--set",
        "2",
        "--workspace",
        str(workspace),
        "--port",
        str(port),
    ]
    with (workspace / "test-launcher.log").open("w") as log:
        process = subprocess.Popen(launcher, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                assert process.poll() is None
                try:
                    with urlopen(
                        f"http://127.0.0.1:{port}/user/login/", timeout=1
                    ) as response:
                        if response.status == 200:
                            break
                except (URLError, TimeoutError):
                    time.sleep(0.2)
            else:
                pytest.fail("assignment login page did not start")
            result = subprocess.run(
                command[:3] + ["shell"],
                env=env,
                cwd=app,
                input=f"exec({signup!r})\n",
                capture_output=True,
                text=True,
                timeout=120,
            )
            assert "LOCAL_OWNER_CREATED" in result.stdout, (
                result.stdout + result.stderr
            )
            state_file = workspace / "workspace.json"
            while time.monotonic() < deadline and not state_file.exists():
                assert process.poll() is None
                time.sleep(0.2)
            assert state_file.is_file(), (
                workspace / "test-launcher.log"
            ).read_text()
            state = json.loads(state_file.read_text())
            expected = len(build_plan(pilot)["tasks"])
            assert state["projects"]["phase"]["tasks"] == expected
            assert state["projects"]["masks"]["tasks"] == expected
            repeated = setup_assignment(
                pilot, app, port, "set-test@example.invalid"
            )
            assert (
                not repeated["phase"]["created"]
                and not repeated["masks"]["created"]
            )
            with sqlite3.connect(app / "label_studio.sqlite3") as db:
                assert (
                    db.execute("select count(*) from project").fetchone()[0]
                    == 2
                )
                rows = db.execute("select data from task").fetchall()
                assert len(rows) == expected * 2
                assert all(
                    json.loads(row[0])["assignment_set"] == 2 for row in rows
                )
        finally:
            process.terminate()
            process.wait(timeout=30)
