"""Pixel prefill protects saved work and stays separate from human labels."""

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from test_labeling_workflow import load_workflow
from test_labeling_workflow import pilot as pilot


def assigned_pilot(pilot: Path, set_number: int = 1) -> dict:
    path = pilot / "manifest.json"
    manifest = json.loads(path.read_text())
    if manifest.get("preprocessing"):
        pytest.skip("prefill is for original frames only")
    manifest["assignment"] = {
        "version": "labeling-two-sets-v1",
        "set_number": set_number,
        "assignment_id": "a" * 64,
        "source_manifest_sha256": "b" * 64,
        "run_ids": [1],
    }
    frames = manifest["runs"][0]["frames"]
    frames.extend(
        {**frames[0], "requested_seconds": i * 5} for i in range(3, 7)
    )
    path.write_text(json.dumps(manifest))
    return load_workflow().build_plan(pilot, kind="masks")


@pytest.mark.parametrize("set_number", [1, 2])
def test_generation_excludes_annotations_drafts_locks_and_predictions(
    pilot: Path,
    set_number: int,
) -> None:
    pytest.importorskip("cv2")
    from maus_export.pixel_mask_tasks import (
        prepare_task_suggestions,
        untouched_tasks,
    )
    from maus_export.pixel_masks import PixelMaskConfig

    plan = assigned_pilot(pilot, set_number)
    workspace = pilot.parent
    app = workspace / "app"
    app.mkdir(mode=0o700)
    database = app / "label_studio.sqlite3"
    with sqlite3.connect(database) as db:
        db.execute("create table task(id integer,project_id integer,data text)")
        for table in (
            "task_completion",
            "tasks_annotationdraft",
            "tasks_tasklock",
            "prediction",
        ):
            db.execute("create table " + table + "(task_id integer)")
        for index, task in enumerate(plan["tasks"], 1):
            db.execute(
                "insert into task values(?,?,?)",
                (index, 2, json.dumps(task["data"])),
            )
        for index, table in enumerate(
            (
                "task_completion",
                "tasks_annotationdraft",
                "tasks_tasklock",
                "prediction",
            ),
            1,
        ):
            db.execute("insert into " + table + " values(?)", (index,))
    (workspace / "workspace.json").write_text(
        json.dumps(
            {
                "pilot_sha256": plan["digest"],
                "assignment": plan["assignment"],
                "projects": {"masks": {"project_id": 2}},
            }
        )
    )
    assert [pk for pk, _ in untouched_tasks(database, 2)] == [5, 6, 7]
    output = workspace.parent / "data" / "suggestions"
    counts = prepare_task_suggestions(
        workspace,
        output,
        set_number=set_number,
        roi=None,
        config=PixelMaskConfig(
            min_area=1, opening=1, closing=1, max_saturation=255
        ),
    )
    assert {
        name: counts[name]
        for name in ("candidates", "nonempty_suggestions", "empty_not_imported")
    } == {
        "candidates": 3,
        "nonempty_suggestions": 3,
        "empty_not_imported": 0,
    }
    job = json.loads((output / "manifest.json").read_text())
    assert job["human_annotation"] is False
    assert job["set_number"] == set_number
    assert (output / "contact-sheet.jpg").is_file()
    assert [r["task_id"] for r in job["records"]] == [5, 6, 7]
    with sqlite3.connect(database) as db:
        assert db.execute("select count(*) from prediction").fetchone()[0] == 1
    from labeling.workflow import prefill_context
    from maus_export.mask_assist import prepare_reviewed_job, workspace_status

    with pytest.raises(ValueError):
        prefill_context(workspace, 3 - set_number)
    invalid = workspace.parent / "data" / "invalid-limit"
    with pytest.raises(ValueError):
        prepare_task_suggestions(
            workspace, invalid, set_number=set_number, limit=0
        )
    assert not invalid.exists()
    status = workspace_status(workspace, set_number)
    assert status["eligible_untouched"] == 3
    preview = workspace.parent / "data" / "sample"
    preview_counts = prepare_task_suggestions(
        workspace,
        preview,
        set_number=set_number,
        roi=None,
        limit=2,
        config=PixelMaskConfig(
            min_area=1, opening=1, closing=1, max_saturation=255
        ),
    )
    assert preview_counts["scope"] == "sample"
    assert preview_counts["candidates"] == 2
    reviewed = workspace.parent / "data" / "reviewed-full"
    assert (
        prepare_reviewed_job(workspace, set_number, preview, reviewed)["scope"]
        == "full"
    )
    full = json.loads((reviewed / "manifest.json").read_text())
    assert (
        full["parameters"]
        == json.loads((preview / "manifest.json").read_text())["parameters"]
    )
    assert len(full["records"]) == 3
    with pytest.raises(ValueError):
        prepare_reviewed_job(
            workspace,
            3 - set_number,
            preview,
            workspace.parent / "data/wrong-set",
        )


def test_human_review_choice_cannot_be_a_pixel_prediction() -> None:
    with pytest.raises(ValueError):
        load_workflow().validate_pixel_result(
            [
                {
                    "from_name": "mask_state",
                    "to_name": "image",
                    "type": "choices",
                    "value": {"choices": ["no_crepe"]},
                }
            ],
            [8, 8],
        )


def test_apply_command_requires_confirmation_and_forwards_the_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("cv2")
    from maus_export import mask_assist

    calls = []
    monkeypatch.setattr(
        mask_assist, "_labeling_task", lambda *args: calls.append(args)
    )
    command = [
        "mask-assist",
        "apply",
        "--set",
        "2",
        "--workspace",
        str(tmp_path / "set-2"),
        "--suggestions",
        str(tmp_path / "job"),
    ]
    monkeypatch.setattr(sys, "argv", command)
    with pytest.raises(SystemExit) as error:
        mask_assist.main()
    assert error.value.code == 2 and calls == []
    monkeypatch.setattr(sys, "argv", command + ["--yes"])
    assert mask_assist.main() == 0
    assert calls[0][0] == "prefill-pixels"
    arguments = calls[0][1]
    assert arguments[arguments.index("--set") + 1] == "2"
    assert "--apply" in arguments


@pytest.mark.parametrize("set_number", [1, 2])
def test_installed_import_rechecks_live_work_and_is_idempotent(
    pilot: Path,
    set_number: int,
) -> None:
    pytest.importorskip("label_studio")
    assigned_pilot(pilot, set_number)
    workflow = load_workflow()
    app = pilot.parent / "app"
    _, env = workflow.runtime(pilot, app, 8080)
    probe = Path(__file__).with_name("pixel_prefill_shell_probe.py")
    code = (
        "import runpy\n"
        f"runpy.run_path({str(probe)!r},init_globals="
        f"{{'PILOT':{str(pilot)!r},'SOURCE':{str(probe.parents[1])!r},'SET':{set_number!r}}})\n"
    )
    result = subprocess.run(
        [sys.executable, "-m", "label_studio.server", "shell"],
        input=f"exec({code!r})\n",
        text=True,
        capture_output=True,
        env=env,
        cwd=app,
        timeout=180,
    )
    assert "PIXEL_PREFILL_OK" in result.stdout, result.stdout + result.stderr
