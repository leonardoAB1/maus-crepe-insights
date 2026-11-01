"""Fine-tune on hand masks, and score it on runs it did not train on.

Local experiment. All hand masks come from one robot and from runs in the shared
test split. Each fold holds out one run. The check-frame scores below leave out
every hand-labelled run, so they stay unseen by every model here.

    python models/crepe-pose/hand_finetune.py --data-dir DATA
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
START_FROM = "tuned"
EPOCHS = 60
LEARNING_RATE = 1e-4
PRESENT = "crepe_present"
LOW_PERCENTILE = 10
MASK_THRESHOLD = 0.5


def load_train():
    """Import train.py, which sits in a folder that is not a package name."""
    spec = importlib.util.spec_from_file_location("train", HERE / "train.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["train"] = module
    spec.loader.exec_module(module)
    return module


t = load_train()
sc = t.sc


def model_from(path: Path) -> t.PoseNet:
    model = t.PoseNet()
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model


def found_masks(model: t.PoseNet, images: np.ndarray) -> np.ndarray:
    chances = t.predict(model, images)[0]
    return np.stack([sc.clean_mask(p > MASK_THRESHOLD) for p in chances])


def hand_scores(
    found: np.ndarray, truth: np.ndarray, valid: np.ndarray, present: np.ndarray
) -> dict[str, float | int]:
    """Agreement on supported pixels, split into crepes and empty plates."""
    ious = np.array(
        [
            sc.iou(found[i] & valid[i], truth[i] & valid[i])
            for i in np.flatnonzero(present)
        ]
    )
    out: dict[str, float | int] = {"crepes": int(present.sum())}
    if len(ious):
        out["iou"] = round(float(np.median(ious)), 3)
        out["iou_low"] = round(float(np.percentile(ious, LOW_PERCENTILE)), 3)
        out["missed"] = round(
            float(
                np.mean([not found[i].any() for i in np.flatnonzero(present)])
            ),
            3,
        )
    empty = np.flatnonzero(~present)
    out["empty"] = len(empty)
    if len(empty):
        out["false_on_empty"] = round(
            float(np.mean([found[i].any() for i in empty])), 3
        )
    return out


def check_frame_scores(
    model: t.PoseNet, data: Path, excluded: set[int]
) -> dict[str, float | int]:
    """Robot-outline agreement on test check frames from other runs."""
    table, images, masks = t.load_real(data)
    keep = (
        (table["split"] == "test")
        & table["check"].isin(t.FOLD_SHAPE)
        & ~table["run_id"].isin(excluded)
    )
    part = table[keep]
    found = found_masks(model, images[part.index])
    ious = np.array(
        [sc.iou(found[i], masks[k]) for i, k in enumerate(part.index)]
    )
    out: dict[str, float | int] = {"frames": len(part)}
    for robot, rows in part.groupby("robot"):
        positions = part.index.get_indexer(rows.index)
        out[str(robot)] = round(float(np.median(ious[positions])), 3)
        out[f"{robot} n"] = len(rows)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    data = parser.parse_args().data_dir
    out = data / "synthetic"
    table = pd.read_parquet(out / "hand.parquet")
    arrays = np.load(out / "hand.npz")
    images, valid = arrays["images"], arrays["valid"]
    truth = arrays["material"] | arrays["holes"]
    present = table["state"].to_numpy() == PRESENT
    runs = table["run_id"].to_numpy()
    start = t.weights_path(data, START_FROM)
    no_direction = np.zeros((len(table), 2), np.float32)

    results: dict[str, object] = {"start_from": START_FROM, "folds": {}}
    before, after = {}, {}
    for run in sorted(set(runs.tolist())):
        held = runs == run
        torch.manual_seed(t.SEED)
        model = model_from(start)
        before[run] = found_masks(model, images[held])
        t.fit(
            model,
            (images[~held], truth[~held], no_direction[~held]),
            EPOCHS,
            LEARNING_RATE,
            True,
        )
        after[run] = found_masks(model, images[held])
        results["folds"][str(run)] = {
            "before": hand_scores(
                before[run], truth[held], valid[held], present[held]
            ),
            "after": hand_scores(
                after[run], truth[held], valid[held], present[held]
            ),
        }
        print(run, json.dumps(results["folds"][str(run)]), flush=True)

    order = np.concatenate([np.flatnonzero(runs == r) for r in before])
    results["all_held_out"] = {
        name: hand_scores(
            np.concatenate(list(groups.values())),
            truth[order],
            valid[order],
            present[order],
        )
        for name, groups in (("before", before), ("after", after))
    }
    excluded = set(runs.tolist())
    results["other_runs_before"] = check_frame_scores(
        model_from(start), data, excluded
    )
    torch.manual_seed(t.SEED)
    final = model_from(start)
    t.fit(final, (images, truth, no_direction), EPOCHS, LEARNING_RATE, True)
    results["other_runs_after"] = check_frame_scores(final, data, excluded)
    torch.save(final.state_dict(), t.weights_path(data, "hand"))
    (out / "scores-hand-finetune.json").write_text(
        json.dumps(results, indent=1)
    )
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
