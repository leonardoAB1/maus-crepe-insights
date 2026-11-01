"""Rectify the private hand masks into the view the crepe models read.

Local experiment. From code/ in the ml environment:

    python models/crepe-pose/hand_masks.py --data-dir DATA
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from maus_export import human_masks as hm
from maus_export import synthetic_crepes as sc
from maus_export.label_deviation import run_directories, warp_mask

# Open labels PR: it refreshes set 1, which main still holds a small part of
LABELS_BRANCH = "origin/chore/3-set-1-mask-update"
SNAPSHOTS = ("labels/set-1/masks-v1.json", "labels/set-2/masks-v1.json")
LABELING = "phase-snapshot/labeling"
TRANSFORM_PILOT = "rectified-experimental-v1"
PILOTS = (
    "pilot-v1",
    "assignments-v1/set-1/pilot",
    "assignments-v1/set-2/pilot",
    "handoff/verified-set-2/pilot",
)
SHEET_ROWS = 6
SHEET_COLUMNS = 8
SHEET_TILE = 200
FILL_SHARE = 0.35


def snapshot_text(repo: Path, path: str) -> str:
    """Read a labels file from the branch, leaving the checkout alone."""
    return subprocess.check_output(
        ["git", "-C", str(repo), "show", f"{LABELS_BRANCH}:{path}"], text=True
    )


def small(mask: np.ndarray) -> np.ndarray:
    resized = cv2.resize(
        mask.astype(np.uint8),
        (sc.SIZE, sc.SIZE),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized > 0


def build(data: Path) -> None:
    labeling = data / LABELING
    repo = data / "maus-crepe-labels"
    records = hm.usable_records(snapshot_text(repo, p) for p in SNAPSHOTS)
    frames = hm.frame_index(labeling / p for p in PILOTS)
    identities = hm.identity_index(labeling / p for p in PILOTS)
    pilot = labeling / TRANSFORM_PILOT
    directories = run_directories(pilot)
    rows, images, material, holes, valid = [], [], [], [], []
    dropped: dict[str, int] = {}

    def drop(reason: str) -> None:
        dropped[reason] = dropped.get(reason, 0) + 1

    for record in records:
        run_id = int(record["run_id"])
        path = frames.get(record["image_sha256"])
        same_bytes = path is not None
        if path is None:
            path = identities.get(hm.frame_identity(record))
        if path is None:
            drop("frame not found")
            continue
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if same_bytes and digest != record["image_sha256"]:
            drop("frame hash differs")
            continue
        name = f"{directories.get(run_id, '')}.json"
        transform_path = pilot / "transforms" / name
        if not transform_path.is_file():
            drop("no transform for the run")
            continue
        transform = json.loads(transform_path.read_text())
        frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        found, hole = hm.record_masks(record)
        warped, support = warp_mask(found, transform)
        warped_hole, _ = warp_mask(hole, transform)
        size = tuple(int(v) for v in transform["target_size"])
        matrix = np.array(transform["matrix"], np.float64)
        image = cv2.warpPerspective(frame, matrix, size)
        rows.append(
            {
                "run_id": run_id,
                "state": record["state"],
                "set": record["assignment_set"],
                "seconds": float(record["requested_seconds"]),
                "same_bytes": same_bytes,
            }
        )
        images.append(sc.shrink(image))
        material.append(small(warped))
        holes.append(small(warped_hole))
        valid.append(small(support))
    out = data / "synthetic"
    table = pd.DataFrame(rows)
    table.to_parquet(out / "hand.parquet")
    np.savez(
        out / "hand.npz",
        images=np.stack(images),
        material=np.stack(material),
        holes=np.stack(holes),
        valid=np.stack(valid),
    )
    print("usable", len(records), "kept", len(table), "dropped", dropped)
    print(table.groupby(["run_id", "state"]).size().unstack(fill_value=0))
    print(table.groupby("set")["same_bytes"].agg(["size", "sum"]))
    sheet(out, table, np.stack(images), np.stack(material))


def sheet(
    out: Path, table: pd.DataFrame, images: np.ndarray, masks: np.ndarray
) -> None:
    """Contact sheet: painted masks over rectified frames, across all runs."""
    # Set 2 was matched by identity, so the sheet shows those frames to check
    chosen = (table["state"] == hm.PRESENT) & ~table["same_bytes"]
    present = table.index[chosen].to_numpy()
    count = min(SHEET_ROWS * SHEET_COLUMNS, len(present))
    picks = present[np.linspace(0, len(present) - 1, count).astype(int)]
    tiles = []
    for index in picks:
        tile = cv2.resize(images[index], (SHEET_TILE, SHEET_TILE))
        mask = cv2.resize(
            masks[index].astype(np.uint8),
            (SHEET_TILE, SHEET_TILE),
            interpolation=cv2.INTER_NEAREST,
        )
        painted = tile[mask > 0] * (1 - FILL_SHARE)
        tile[mask > 0] = painted + np.array([0, 255, 0]) * FILL_SHARE
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        cv2.drawContours(tile, contours, -1, (0, 255, 0), 1)
        tiles.append(tile)
    while len(tiles) % SHEET_COLUMNS:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [
        np.hstack(tiles[i : i + SHEET_COLUMNS])
        for i in range(0, len(tiles), SHEET_COLUMNS)
    ]
    cv2.imwrite(str(out / "hand-masks.jpg"), np.vstack(rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    build(parser.parse_args().data_dir)


if __name__ == "__main__":
    main()
