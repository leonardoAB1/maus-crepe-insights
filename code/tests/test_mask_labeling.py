"""Mask projects and exports use synthetic pixels, never real annotations."""

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest
from PIL import Image
from test_labeling_workflow import load_workflow
from test_labeling_workflow import pilot as pilot

from labeling.mask_export import export_masks

pytestmark = pytest.mark.skipif(
    sys.platform not in {"linux", "darwin"}, reason="POSIX labeling workflow"
)


def mask_plan(pilot: Path) -> dict:
    manifest = json.loads((pilot / "manifest.json").read_text())
    if not manifest.get("preprocessing"):
        pytest.skip("mask project requires rectified frames")
    return load_workflow().build_plan(pilot, kind="masks")


def annotation(material: np.ndarray, holes: np.ndarray) -> dict:
    brush = pytest.importorskip("label_studio_sdk.converter.brush")
    height, width = material.shape
    regions = []
    for label, mask in (("crepe_material", material), ("hole", holes)):
        rgba = np.zeros((height, width, 4), np.uint8)
        rgba[..., -1] = mask.astype(np.uint8) * 255
        regions.append(
            {
                "from_name": "segmentation",
                "to_name": "image",
                "type": "brushlabels",
                "original_width": width,
                "original_height": height,
                "image_rotation": 0,
                "value": {
                    "format": "rle",
                    "rle": brush.encode_rle(rgba.ravel()),
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
    return {"id": 1, "was_cancelled": False, "result": regions}


def export_tasks(pilot: Path) -> list:
    plan = mask_plan(pilot)
    material = np.zeros((8, 8), bool)
    material[:2, :2] = True
    holes = np.zeros_like(material)
    holes[4, 4] = True
    tasks = copy.deepcopy(plan["tasks"])
    for index, task in enumerate(tasks, start=1):
        task["id"] = index
    tasks[0]["annotations"] = [annotation(material, holes)]
    tasks[1]["annotations"] = [
        {
            "id": 2,
            "was_cancelled": False,
            "result": [
                {
                    "from_name": "mask_state",
                    "to_name": "image",
                    "type": "choices",
                    "value": {"choices": ["no_crepe"]},
                }
            ],
        }
    ]
    tasks[2]["annotations"] = [{"id": 3, "was_cancelled": True, "result": []}]
    return tasks


def test_mask_plan_preserves_phase_plan_and_identical_image_identity(
    pilot: Path,
) -> None:
    workflow = load_workflow()
    phase = workflow.build_plan(pilot)
    before = copy.deepcopy(phase)
    manifest = json.loads((pilot / "manifest.json").read_text())
    if not manifest.get("preprocessing"):
        with pytest.raises(ValueError, match="shared rectified"):
            workflow.build_plan(pilot, kind="masks")
        return
    masks = workflow.build_plan(pilot, kind="masks")
    assert phase == before == workflow.build_plan(pilot)
    assert masks["title"] != phase["title"]
    assert masks["digest"] == phase["digest"]
    assert masks["storage_roots"] == phase["storage_roots"]
    config = ET.fromstring(masks["config"])
    assert [label.attrib["value"] for label in config.find("BrushLabels")] == [
        "crepe_material",
        "hole",
    ]
    assert config.find("Choices").attrib["required"] == "true"
    for mask_task, phase_task in zip(
        masks["tasks"], phase["tasks"], strict=True
    ):
        data = mask_task["data"]
        assert data["image"] == phase_task["data"]["image"]
        assert data["image_sha256"] == phase_task["data"]["image_sha256"]
        assert data["phase_context_shown"] is False
        assert "phase_sequence" not in data
        assert data["mask_taxonomy_version"] == "crepe-material-holes-v1"
        assert mask_task["annotations"] == mask_task["predictions"] == []


def test_reviewed_masks_export_in_original_coordinates(pilot: Path) -> None:
    tasks = export_tasks(pilot)
    source = pilot.parent / "human-masks.json"
    source.write_text(json.dumps(tasks))
    before = source.read_bytes()
    output = pilot.parent / "exports" / "masks-v1"
    assert export_masks(pilot, source, output) == {
        "exported": 2,
        "tasks_in_json": 3,
    }
    labels = np.asarray(Image.open(output / "task-1-labels.png"))
    assert labels.shape == (8, 8)
    assert labels[0, 0] == 1
    assert labels[4, 4] == 2
    assert labels[7, 7] == 0
    assert not np.asarray(Image.open(output / "task-2-labels.png")).any()
    assert source.read_bytes() == before
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(manifest["records"]) == 2
    assert (
        manifest["records"][0]["image_sha256"]
        == tasks[0]["data"]["image_sha256"]
    )
    with pytest.raises(FileExistsError):
        export_masks(pilot, source, output)


def test_unsupported_pixels_are_ignored_not_holes(pilot: Path) -> None:
    mask_plan(pilot)
    manifest_path = pilot / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    frame = manifest["runs"][0]["frames"][0]
    valid_path = pilot / frame["valid_mask"]
    with Image.open(valid_path) as image:
        valid = np.array(image)
    valid[0, 0] = 0
    Image.fromarray(valid).save(valid_path)
    digest = hashlib.sha256(valid_path.read_bytes()).hexdigest()
    for candidate in manifest["runs"][0]["frames"]:
        candidate["valid_mask_sha256"] = digest
    manifest_path.write_text(json.dumps(manifest))
    tasks = export_tasks(pilot)
    source = pilot.parent / "human-masks.json"
    source.write_text(json.dumps(tasks))
    output = pilot.parent / "exports" / "validity"
    export_masks(pilot, source, output)
    labels = np.asarray(Image.open(output / "task-1-labels.png"))
    holes = np.asarray(Image.open(output / "task-1-holes.png"))
    assert labels[0, 0] == 255
    assert holes[0, 0] == 0


@pytest.mark.parametrize(
    "damage",
    [
        "overlap",
        "rotation",
        "dimensions",
        "rle",
        "provenance",
        "multiple_annotations",
        "duplicate_id",
        "phase_export",
        "empty",
    ],
)
def test_bad_masks_fail_closed_without_partial_exports(
    pilot: Path,
    damage: str,
) -> None:
    tasks = export_tasks(pilot)
    regions = tasks[0]["annotations"][0]["result"]
    if damage == "overlap":
        regions[1]["value"]["rle"] = regions[0]["value"]["rle"]
    elif damage == "rotation":
        regions[0]["image_rotation"] = 90
    elif damage == "dimensions":
        regions[0]["original_width"] += 1
    elif damage == "rle":
        regions[0]["value"]["rle"] = [0, 0, 0, 0]
    elif damage == "provenance":
        tasks[0]["data"]["image_sha256"] = "0" * 64
    elif damage == "multiple_annotations":
        tasks[0]["annotations"] *= 2
    elif damage == "duplicate_id":
        tasks[1]["id"] = tasks[0]["id"]
    elif damage == "phase_export":
        del tasks[0]["data"]["mask_taxonomy_version"]
    elif damage == "empty":
        for task in tasks:
            task["annotations"] = []
    source = pilot.parent / "human-masks.json"
    source.write_text(json.dumps(tasks))
    output = pilot.parent / "exports" / "invalid"
    with pytest.raises((ValueError, KeyError)):
        export_masks(pilot, source, output)
    assert not output.exists()


def test_installed_app_keeps_phase_annotations_and_exports_brushes(
    pilot: Path,
) -> None:
    pytest.importorskip("label_studio")
    mask_plan(pilot)
    workflow = load_workflow()
    app = pilot.parent / "app"
    _, env = workflow.runtime(pilot, app, 8080)
    probe = Path(__file__).with_name("mask_labeling_shell_probe.py")
    code = (
        "import runpy\n"
        f"runpy.run_path({str(probe)!r}, init_globals="
        f"{{'PILOT': {str(pilot)!r}, 'SOURCE': {str(probe.parents[1])!r}}})\n"
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
    assert "MASK_INTEGRATION_OK" in result.stdout, result.stdout + result.stderr
