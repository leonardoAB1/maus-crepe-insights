"""Private, original-pixel labeling assignments and bounded handoff bundles."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path
from typing import Any

if __package__:
    from .workflow import ASSIGNMENT_VERSION, build_plan
else:
    from workflow import ASSIGNMENT_VERSION, build_plan

SET_NUMBERS = (1, 2)
GROUP_COUNT = len(SET_NUMBERS)
PRIVATE_MODE = 0o700
HASH_CHUNK = 1024 * 1024
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_IMAGE_BYTES = 32 * 1024 * 1024
MAX_TOTAL_BYTES = 8 * 1024**3
MAX_MEMBERS = 20000
FRAME_KEYS = (
    "image",
    "image_sha256",
    "requested_seconds",
    "decoded_frame_index",
    "decoded_pts",
    "human_label",
    "phase_suggestion",
    "suggestion_status",
    "suggestion_shown",
    "phase_context_shown",
    "review_status",
)
RUN_KEYS = (
    "run_id",
    "ts_start",
    "status",
    "phase_alignment",
    "time_base",
    "start_pts",
    "duration_ticks",
    "video_sha256",
)


def file_hash(path: Path) -> str:
    """Hash an artifact without loading the complete bundle into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def private_path(path: Path) -> Path:
    """Reject symlinked destinations and writes into public source folders."""
    path = path.absolute()
    source = Path(__file__).resolve().parents[2]
    if path != path.resolve() or (
        path.is_relative_to(source) and not path.is_relative_to(source / "data")
    ):
        raise ValueError("Use a canonical private path under data/, not source")
    return path


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2))


def _tar_metadata(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mode = 0o600
    info.mtime = 0
    info.pax_headers = {}
    return info


def _image_paths(manifest: dict) -> dict[str, str]:
    paths = {}
    for run in manifest["runs"]:
        for frame in run["frames"]:
            relative = Path(frame["image"])
            if (
                len(relative.parts) != 2
                or relative.is_absolute()
                or ".." in relative.parts
                or relative.suffix != ".png"
            ):
                raise ValueError("Unsafe assignment image path")
            previous = paths.setdefault(
                relative.as_posix(), frame["image_sha256"]
            )
            if previous != frame["image_sha256"]:
                raise ValueError("Conflicting hashes for one image")
    return paths


def split_pilot(pilot: Path, output: Path) -> dict[str, Any]:
    """Balance complete runs into two disjoint original-image workspaces."""
    pilot = pilot.resolve(strict=True)
    output = private_path(output)
    if output.is_relative_to(pilot):
        raise ValueError("Do not write assignments inside the source pilot")
    content = (pilot / "manifest.json").read_bytes()
    manifest = json.loads(content)
    source_hash = hashlib.sha256(content).hexdigest()
    if manifest.get("preprocessing") or manifest.get("assignment"):
        raise ValueError(
            "Split the original full pilot, not a transformed/subset pilot"
        )
    if build_plan(pilot)["digest"] != source_hash:
        raise ValueError("Source manifest changed during validation")
    runs = [r for r in manifest["runs"] if r["status"] == "ok" and r["frames"]]
    ids = [r["run_id"] for r in runs]
    if len(runs) < GROUP_COUNT or len(set(ids)) != len(ids):
        raise ValueError("Need at least two independent usable runs")
    ordered = sorted(
        runs,
        key=lambda r: (
            -len(r["frames"]),
            hashlib.sha256(f"{source_hash}:{r['run_id']}".encode()).hexdigest(),
        ),
    )
    groups: list[list[dict]] = [[] for _ in SET_NUMBERS]
    loads = [0] * GROUP_COUNT
    capacity = (len(runs) + GROUP_COUNT - 1) // GROUP_COUNT
    for run in ordered:
        eligible = [
            i for i, group in enumerate(groups) if len(group) < capacity
        ]
        index = min(eligible, key=lambda i: (loads[i], len(groups[i]), i))
        groups[index].append(run)
        loads[index] += len(run["frames"])
    assignment = {
        "version": ASSIGNMENT_VERSION,
        "source_manifest_sha256": source_hash,
        "policy": "whole-run-largest-first-balanced-v1",
        "sets": {
            str(number): {
                "run_ids": sorted(r["run_id"] for r in group),
                "runs": len(group),
                "frames": load,
            }
            for number, group, load in zip(
                SET_NUMBERS, groups, loads, strict=True
            )
        },
        "excluded_runs": len(manifest["runs"]) - len(runs),
    }
    assignment_id = hashlib.sha256(
        json.dumps(assignment, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assignment["assignment_id"] = assignment_id
    os.umask(0o077)
    output.parent.mkdir(mode=PRIVATE_MODE, parents=True, exist_ok=True)
    output.mkdir(mode=PRIVATE_MODE)
    try:
        for number in SET_NUMBERS:
            membership = set(assignment["sets"][str(number)]["run_ids"])
            subset_runs = []
            for run in runs:
                if run["run_id"] not in membership:
                    continue
                subset = {key: run[key] for key in RUN_KEYS if key in run}
                subset["phase_sequence"] = [
                    {"phase": phase["phase"]} for phase in run["phase_sequence"]
                ]
                subset["frames"] = [
                    {key: frame[key] for key in FRAME_KEYS}
                    for frame in run["frames"]
                ]
                subset_runs.append(subset)
            subset_manifest = {
                "version": manifest["version"],
                "runs": subset_runs,
                "classes": sorted(
                    {"unclear"}
                    | {
                        p["phase"]
                        for run in subset_runs
                        for p in run["phase_sequence"]
                    }
                ),
                "assignment": {
                    "version": ASSIGNMENT_VERSION,
                    "assignment_id": assignment_id,
                    "source_manifest_sha256": source_hash,
                    "set_number": number,
                    "run_ids": sorted(membership),
                    "policy": assignment["policy"],
                },
            }
            destination = output / f"set-{number}" / "pilot"
            destination.mkdir(mode=PRIVATE_MODE, parents=True)
            for relative, digest in _image_paths(subset_manifest).items():
                target = destination / relative
                target.parent.mkdir(mode=PRIVATE_MODE, exist_ok=True)
                shutil.copyfile(pilot / relative, target)
                if file_hash(target) != digest:
                    raise ValueError("Copied image differs from original bytes")
            _write(destination / "manifest.json", subset_manifest)
            build_plan(destination)
        if (pilot / "manifest.json").read_bytes() != content:
            raise ValueError("Source manifest changed while splitting")
        _write(output / "assignment.json", assignment)
    except BaseException:
        shutil.rmtree(output)
        raise
    return {
        "assignment_id": assignment_id,
        "sets": {
            number: {
                key: value for key, value in record.items() if key != "run_ids"
            }
            for number, record in assignment["sets"].items()
        },
    }


def validate_workspace(workspace: Path, number: int) -> dict:
    """Check the assigned set and every source image before app/bundle use."""
    workspace = private_path(workspace)
    if workspace.stat().st_mode & 0o077:
        raise ValueError("Workspace must be private (0700)")
    plan = build_plan(workspace / "pilot")
    if plan.get("assignment", {}).get("set_number") != number:
        raise ValueError("Wrong or missing assignment set")
    return plan


def bundle_workspace(workspace: Path, number: int, output: Path) -> dict:
    """Package only the assigned pilot manifest and verified PNG files."""
    workspace = workspace.resolve(strict=True)
    validate_workspace(workspace, number)
    output = private_path(output)
    if output.is_relative_to(workspace / "pilot"):
        raise ValueError("Do not put bundles inside an immutable pilot")
    manifest = json.loads((workspace / "pilot/manifest.json").read_text())
    paths = ["pilot/manifest.json"] + [
        "pilot/" + name for name in sorted(_image_paths(manifest))
    ]
    os.umask(0o077)
    output.parent.mkdir(mode=PRIVATE_MODE, parents=True, exist_ok=True)
    checksum = private_path(output.with_name(output.name + ".sha256"))
    if checksum.exists():
        raise FileExistsError("A checksum file already exists for this bundle")
    # Exclusive creation prevents replacement of a previous private handoff.
    with output.open("xb") as stream:
        try:
            with tarfile.open(fileobj=stream, mode="w:gz") as archive:
                for relative in paths:
                    archive.add(
                        workspace / relative,
                        arcname=relative,
                        recursive=False,
                        filter=_tar_metadata,
                    )
        except BaseException:
            output.unlink(missing_ok=True)
            raise
    digest = file_hash(output)
    with checksum.open("x") as target:
        target.write(digest + "  " + output.name + "\n")
    return {"set_number": number, "files": len(paths), "sha256": digest}


def receive_bundle(
    bundle: Path, expected_hash: str, number: int, output: Path
) -> dict:
    """Verify and unpack only the allowlisted pilot and original images."""
    if file_hash(bundle) != expected_hash:
        raise ValueError("Bundle checksum does not match the sender's checksum")
    output = private_path(output)
    os.umask(0o077)
    with tarfile.open(bundle, "r:gz") as archive:
        members = []
        total = 0
        for item in archive:
            total += item.size
            if (
                not item.isfile()
                or item.size > MAX_IMAGE_BYTES
                or total > MAX_TOTAL_BYTES
                or len(members) >= MAX_MEMBERS
            ):
                raise ValueError("Unsafe or oversized handoff bundle")
            members.append(item)
        names = [m.name for m in members]
        if (
            len(members) > MAX_MEMBERS
            or len(set(names)) != len(names)
            or any(not m.isfile() for m in members)
            or sum(m.size for m in members) > MAX_TOTAL_BYTES
        ):
            raise ValueError("Unsafe or oversized handoff bundle")
        member = archive.getmember("pilot/manifest.json")
        if member.size > MAX_MANIFEST_BYTES:
            raise ValueError("Oversized assignment manifest")
        source = archive.extractfile(member)
        if source is None:
            raise ValueError("Missing assignment manifest")
        content = source.read(MAX_MANIFEST_BYTES)
        manifest = json.loads(content)
        expected = {"pilot/manifest.json": hashlib.sha256(content).hexdigest()}
        expected.update(
            {"pilot/" + p: h for p, h in _image_paths(manifest).items()}
        )
        if set(names) != set(expected):
            raise ValueError("Bundle contains unexpected or missing artifacts")
        if any(m.size > MAX_IMAGE_BYTES for m in members if m != member):
            raise ValueError("Oversized image in handoff bundle")
        output.parent.mkdir(mode=PRIVATE_MODE, parents=True, exist_ok=True)
        output.mkdir(mode=PRIVATE_MODE)
        try:
            for item in members:
                source = archive.extractfile(item)
                if source is None:
                    raise ValueError("Missing bundle artifact")
                pixels = source.read(MAX_IMAGE_BYTES + 1)
                if (
                    len(pixels) != item.size
                    or hashlib.sha256(pixels).hexdigest() != expected[item.name]
                ):
                    raise ValueError("Bundle image hash or length mismatch")
                destination = output / item.name
                destination.parent.mkdir(
                    mode=PRIVATE_MODE, parents=True, exist_ok=True
                )
                destination.write_bytes(pixels)
            plan = validate_workspace(output, number)
        except BaseException:
            shutil.rmtree(output)
            raise
    return {
        "set_number": number,
        "tasks": len(plan["tasks"]),
        "assignment_id": plan["assignment"]["assignment_id"],
    }
