"""Shared rectified labeling frames use synthetic artifacts only."""

import hashlib
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from test_labeling_workflow import load_workflow

from maus_export.labeling_rectification import prepare_rectified

SIZE = 64
SHARED_UMASK = 0o022  # a typical default, distinct from the private 0o077
pytestmark = pytest.mark.skipif(
    sys.platform not in {"linux", "darwin"},
    reason="labeling workflow requires POSIX file permissions and locking",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def source_pilot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    data = tmp_path / "data"
    pilot = data / "labeling" / "raw-pilot"
    pilot.mkdir(mode=0o700, parents=True)
    pilot.parent.chmod(0o700)
    entries = []
    for index in range(2):
        raw = data / "raw" / "runs" / f"{index}.json"
        video = data / "raw" / "media" / str(index) / "video.mp4"
        raw.parent.mkdir(parents=True, exist_ok=True)
        video.parent.mkdir(parents=True)
        raw.write_text(json.dumps({"id": index, "machineId": 1}))
        video.write_bytes(b"synthetic video; decoding stubbed in this test")
        image = pilot / f"run-{index:03d}" / "000000.png"
        image.parent.mkdir()
        pixels = np.full((SIZE, SIZE, 3), (30, 60, 90), np.uint8)
        cv2.imwrite(str(image), pixels)
        if index == 0:
            overlay = video.with_name("01_spread_overlay.jpg")
            cv2.imwrite(str(overlay), np.zeros_like(pixels))
        entries.append(
            {
                "run_id": index,
                "status": "ok",
                "phase_alignment": "unverified",
                "phase_sequence": [{"phase": "cook"}],
                "raw_path": str(raw.relative_to(data)),
                "raw_sha256": digest(raw),
                "video_path": str(video.relative_to(data)),
                "video_sha256": digest(video),
                "time_base": "1/12000",
                "start_pts": 36000,
                "frames": [
                    {
                        "image": str(image.relative_to(pilot)),
                        "image_sha256": digest(image),
                        "requested_seconds": 5,
                        "decoded_frame_index": 10,
                        "decoded_pts": 96000,
                        "human_label": None,
                        "phase_suggestion": None,
                        "suggestion_status": "unavailable_unaligned",
                        "suggestion_shown": False,
                        "phase_context_shown": False,
                        "review_status": "unreviewed",
                    }
                ],
            }
        )
    (pilot / "manifest.json").write_text(
        json.dumps(
            {"version": 1, "classes": ["cook", "unclear"], "runs": entries}
        )
    )

    def alignment(video: Path, references: list[Path]) -> dict:
        return {
            "schema_version": 1,
            "source_size": [SIZE, SIZE],
            "target_size": [SIZE, SIZE],
            "matrix": np.eye(3).tolist(),
            "video_reference": references[0].name,
            "video_validation_frames": [0, 10],
            "expected_frames": 20,
        }

    monkeypatch.setattr(
        "maus_export.labeling_rectification.video_alignment", alignment
    )
    return data, pilot


def test_shared_frames_preserve_timing_and_export_only_raw_pixels(
    source_pilot: tuple,
) -> None:
    data, pilot = source_pilot
    before = (pilot / "manifest.json").read_bytes()
    output = pilot.parent / "rectified"
    assert prepare_rectified(data, pilot, output) == {
        "selected": 2,
        "ok": 1,
        "alignment_rejected": 1,
        "source_changed": 0,
        "unavailable": 0,
        "frames": 1,
    }
    result = json.loads((output / "manifest.json").read_text())
    accepted, rejected = result["runs"]
    frame = accepted["frames"][0]
    assert (pilot / "manifest.json").read_bytes() == before
    assert result["preprocessing"]["source_manifest_sha256"] == digest(
        pilot / "manifest.json"
    )
    assert frame["decoded_pts"] == 96000
    assert frame["requested_seconds"] == 5
    assert frame["source_image_sha256"] == digest(pilot / frame["image"])
    assert frame["image_sha256"] == digest(output / frame["image"])
    assert np.array_equal(
        cv2.imread(str(output / frame["image"])),
        cv2.imread(str(pilot / frame["image"])),
    )
    assert rejected["status"] == "alignment_rejected"
    assert rejected["frames"] == []
    assert not (output / "run-001").exists()
    plan = load_workflow().build_plan(output)
    assert len(plan["tasks"]) == 1
    assert plan["tasks"][0]["data"]["coordinate_space"] == "rectified_upstream"
    assert output.stat().st_mode & 0o077 == 0
    with pytest.raises(FileExistsError):
        prepare_rectified(data, pilot, output)


@pytest.mark.parametrize("artifact", ["transform", "valid_mask"])
def test_changed_rectification_artifacts_cannot_be_imported(
    source_pilot: tuple, artifact: str
) -> None:
    data, pilot = source_pilot
    output = pilot.parent / "rectified"
    prepare_rectified(data, pilot, output)
    result = json.loads((output / "manifest.json").read_text())
    frame = result["runs"][0]["frames"][0]
    (output / frame[artifact]).write_bytes(b"changed")
    with pytest.raises(ValueError):
        load_workflow().build_plan(output)


def test_rejected_warp_does_not_publish_partial_frames(
    source_pilot: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    data, pilot = source_pilot

    def reject(image: np.ndarray, transform: dict) -> tuple:
        raise ValueError("Synthetic invalid geometry")

    monkeypatch.setattr("maus_export.labeling_rectification.rectify", reject)
    output = pilot.parent / "rectified"
    assert prepare_rectified(data, pilot, output)["frames"] == 0
    assert not list(output.rglob("*.png"))
    assert not (output / ".pending").exists()
    with pytest.raises(ValueError, match="no frames"):
        load_workflow().build_plan(output)


def test_mixed_coordinate_spaces_cannot_be_imported(
    source_pilot: tuple,
) -> None:
    data, pilot = source_pilot
    output = pilot.parent / "rectified"
    prepare_rectified(data, pilot, output)
    manifest = output / "manifest.json"
    result = json.loads(manifest.read_text())
    result["runs"][0]["frames"][0]["coordinate_space"] = "raw"
    manifest.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="mixed raw"):
        load_workflow().build_plan(output)


def test_rectified_frames_require_preprocessing_metadata(
    source_pilot: tuple,
) -> None:
    data, pilot = source_pilot
    output = pilot.parent / "rectified"
    prepare_rectified(data, pilot, output)
    manifest = output / "manifest.json"
    result = json.loads(manifest.read_text())
    del result["preprocessing"]
    result["runs"] = [result["runs"][0]]
    manifest.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="require preprocessing"):
        load_workflow().build_plan(output)


def test_changed_source_image_is_rejected_without_raw_fallback(
    source_pilot: tuple,
) -> None:
    data, pilot = source_pilot
    image = pilot / "run-000" / "000000.png"
    image.write_bytes(b"changed source")
    output = pilot.parent / "rectified"
    summary = prepare_rectified(data, pilot, output)
    assert summary["frames"] == 0
    assert summary["source_changed"] == 1
    assert summary["alignment_rejected"] == 1  # run 1 has no reference
    assert not list(output.rglob("*.png"))
    runs = json.loads((output / "manifest.json").read_text())["runs"]
    assert runs[0]["status"] == "alignment_rejected"
    assert runs[0]["rectification_rejection_kind"] == "source_changed"


def test_unavailable_videos_are_counted_apart(source_pilot: tuple) -> None:
    data, pilot = source_pilot
    manifest = pilot / "manifest.json"
    content = json.loads(manifest.read_text())
    content["runs"][1]["status"] = "missing_video"
    content["runs"][1]["frames"] = []
    manifest.write_text(json.dumps(content))
    output = pilot.parent / "rectified"
    summary = prepare_rectified(data, pilot, output)
    assert (summary["ok"], summary["unavailable"]) == (1, 1)
    assert summary["alignment_rejected"] == 0


def test_annotated_pilot_is_refused_before_any_output(
    source_pilot: tuple,
) -> None:
    data, pilot = source_pilot
    manifest = pilot / "manifest.json"
    content = json.loads(manifest.read_text())
    content["runs"][0]["frames"][0]["human_label"] = "cook"
    manifest.write_text(json.dumps(content))
    output = pilot.parent / "rectified"
    with pytest.raises(ValueError, match="unannotated"):
        prepare_rectified(data, pilot, output)
    assert not output.exists()


def test_process_umask_is_restored(source_pilot: tuple) -> None:
    data, pilot = source_pilot
    before = os.umask(SHARED_UMASK)
    try:
        prepare_rectified(data, pilot, pilot.parent / "rectified")
        assert os.umask(SHARED_UMASK) == SHARED_UMASK
        image = pilot / "run-000" / "000000.png"
        image.write_bytes(b"changed source")
        prepare_rectified(data, pilot, pilot.parent / "after-change")
        assert os.umask(SHARED_UMASK) == SHARED_UMASK
    finally:
        os.umask(before)


def test_public_or_unrelated_output_is_refused(source_pilot: tuple) -> None:
    data, pilot = source_pilot
    output = data.parent / "public"
    with pytest.raises(ValueError, match="private direct children"):
        prepare_rectified(data, pilot, output)
    assert not output.exists()


def test_experimental_transfer_is_opt_in_and_visible_in_project(
    source_pilot: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    data, pilot = source_pilot

    def transfer(frames: list, references: list[Path]) -> dict:
        return {
            "schema_version": 1,
            "source_size": [SIZE, SIZE],
            "target_size": [SIZE, SIZE],
            "matrix": np.eye(3).tolist(),
            "reference_path": str(references[0]),
            "video_validation_frames": [0, 10],
            "kind": "experimental_photo_video_transfer",
        }

    monkeypatch.setattr(
        "maus_export.labeling_rectification.experimental_alignment", transfer
    )
    output = pilot.parent / "experimental"
    summary = prepare_rectified(data, pilot, output, experimental=True)
    assert summary["ok"] == 2
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["preprocessing"]["experimental"] is True
    assert manifest["preprocessing"]["version"] == (
        "upstream-rectified-experimental-v1"
    )
    plan = load_workflow().build_plan(output)
    assert "Experimental rectification" in plan["config"]
    assert len(plan["tasks"]) == 2
    assert all(
        task["data"]["preprocessing_version"]
        == "upstream-rectified-experimental-v1"
        for task in plan["tasks"]
    )
