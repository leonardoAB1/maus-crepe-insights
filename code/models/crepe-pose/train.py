"""Crepe position and fold direction, trained on synthetic images.

Local experiment. Usage, from code/ in the ml environment:

    python models/crepe-pose/train.py --data-dir DATA prepare
    python models/crepe-pose/train.py --data-dir DATA generate
    python models/crepe-pose/train.py --data-dir DATA preview
    python models/crepe-pose/train.py --data-dir DATA train synth
    python models/crepe-pose/train.py --data-dir DATA train real
    python models/crepe-pose/train.py --data-dir DATA train tuned
    python models/crepe-pose/train.py --data-dir DATA score --split train
    python models/crepe-pose/train.py --data-dir DATA score --split test
    python models/crepe-pose/train.py --data-dir DATA export
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from torch import nn

from maus_export import synthetic_crepes as sc

HERE = Path(__file__).resolve().parent
CALIBRATION = HERE.parents[2] / "robot-image-calibration.json"
SEED = 7
SAMPLES = 6000
SHARD = 2000
WORKERS = 8
BATCH = 64
BASE_CHANNELS = 16
LEARNING_RATE = 2e-3
TUNE_LEARNING_RATE = 3e-4
EPOCHS = {"synth": 10, "real": 60, "tuned": 40}
DIRECTION_WEIGHT = 0.5
MASK_THRESHOLD = 0.5
MIN_ALIGNMENT = 0.75
CHECKS = ("plate_clean", "plate_present", "spread", "fold1", "fold2")
EMPTY_CHECKS = ("plate_clean", "plate_present")
FOLD_SHAPE = {"spread": "flat", "fold1": "half", "fold2": "quarter"}
BOOTSTRAP_DRAWS = 1000
PERCENTILES = (2.5, 97.5)
PARITY_TOLERANCE = 1e-3
LATENCY_RUNS = 50
PREVIEW_SAMPLES = 24
PREVIEW_COLUMNS = 8
PREVIEW_SCALE = 240


def synthetic_dir(data: Path) -> Path:
    out = data / "synthetic"
    out.mkdir(parents=True, exist_ok=True)
    return out


def alignment(frame: np.ndarray, overlay: np.ndarray) -> float:
    """Correlation of a rectified frame and its overlay inside the plate.
    A low value means the robot's mean correction misses this frame."""
    disc = sc.zone_disc()
    a = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[disc].astype(float)
    b = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)[disc].astype(float)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def prepare(data: Path) -> None:
    """Rectify every usable frame and read its reference from the overlay."""
    corrections = json.loads(CALIBRATION.read_text())["robot_corrections"]
    frames = pd.read_parquet(data / "training" / "frames.parquet")
    frames = frames[frames["check"].isin(CHECKS) & frames["frame_path"].notna()]
    rows, images, masks, dropped = [], [], [], {}
    for row in frames.itertuples():
        correction = corrections.get(row.robot)
        path = data / row.frame_path
        overlay_path = path.with_name(path.name.replace("_frame", "_overlay"))
        reason = None
        if correction is None or not overlay_path.is_file():
            reason = "no correction or overlay"
        else:
            matrix = np.array(correction["matrix_mean"])
            image = sc.shrink(sc.rectify_frame(cv2.imread(str(path)), matrix))
            overlay = cv2.imread(str(overlay_path))
            score = alignment(image, sc.shrink(overlay))
            if score < MIN_ALIGNMENT:
                reason = "misaligned"
        reference = None
        if reason is None and row.check in EMPTY_CHECKS:
            if not row.ok:
                reason = "plate check failed"
        elif reason is None:
            reference = sc.overlay_reference(overlay, row.check)
            if reference is None:
                reason = "no outline"
        if reason:
            key = f"{row.robot} {row.split} {reason}"
            dropped[key] = dropped.get(key, 0) + 1
            continue
        mask = np.zeros((sc.SIZE, sc.SIZE), bool)
        direction = None
        if reference is not None:
            mask, direction = reference["mask"], reference["direction"]
        rows.append(
            {
                "run_id": row.run_id,
                "check": row.check,
                "robot": row.robot,
                "ok": bool(row.ok),
                "split": row.split,
                "alignment": score,
                "dir_x": float("nan") if direction is None else direction[0],
                "dir_y": float("nan") if direction is None else direction[1],
            }
        )
        images.append(image)
        masks.append(mask)
    out = synthetic_dir(data)
    table = pd.DataFrame(rows)
    table.to_parquet(out / "real.parquet")
    np.savez(out / "real.npz", images=np.stack(images), masks=np.stack(masks))
    print(
        table.groupby(["split", "robot", "check"]).size().unstack(fill_value=0)
    )
    print("dropped:", json.dumps(dropped, indent=1, sort_keys=True))


def load_real(data: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    out = synthetic_dir(data)
    table = pd.read_parquet(out / "real.parquet")
    arrays = np.load(out / "real.npz")
    return table, arrays["images"], arrays["masks"]


def load_sources(data: Path) -> sc.Sources:
    """Plates and textures from the training split only."""
    table, images, masks = load_real(data)
    train = table[table["split"] == "train"]
    empty = train[train["check"].isin(EMPTY_CHECKS)]
    plates = [images[i].astype(np.float32) for i in empty.index]
    textures: dict[str, list[np.ndarray]] = {}
    cutouts: dict[str, list[sc.Cutout]] = {}
    for check, shape in FOLD_SHAPE.items():
        rows = train[(train["check"] == check) & train["ok"]]
        tiles = (sc.texture_from_frame(images[i], masks[i]) for i in rows.index)
        textures[shape] = [t for t in tiles if t is not None]
        cutouts[shape] = [
            sc.Cutout(
                images[i],
                masks[i],
                np.nan_to_num(
                    np.array(
                        [rows.loc[i, "dir_x"], rows.loc[i, "dir_y"]],
                        np.float32,
                    )
                ),
            )
            for i in rows.index
            if masks[i].any()
        ]
    return sc.Sources(plates, textures, cutouts)


def write_shard(args: tuple[Path, int, Path]) -> None:
    data, index, path = args
    sources = load_sources(data)
    rng = np.random.default_rng([SEED, index])
    samples = [sc.generate_sample(rng, sources) for _ in range(SHARD)]
    np.savez(
        path,
        images=np.stack([s.image for s in samples]),
        masks=np.stack([s.mask for s in samples]),
        directions=np.stack([s.direction for s in samples]),
    )


def generate(data: Path) -> None:
    sources = load_sources(data)
    print(
        "plates",
        len(sources.plates),
        {k: len(v) for k, v in sources.textures.items()},
    )
    out = synthetic_dir(data) / "shards"
    out.mkdir(exist_ok=True)
    jobs = [(data, i, out / f"{i:03d}.npz") for i in range(SAMPLES // SHARD)]
    cv2.setNumThreads(1)
    with multiprocessing.get_context("spawn").Pool(WORKERS) as pool:
        pool.map(write_shard, jobs)
    print("wrote", len(jobs), "shards to", out)


def preview(data: Path) -> None:
    sources = load_sources(data)
    rng = np.random.default_rng(SEED)
    tiles = []
    for _ in range(PREVIEW_SAMPLES):
        sample = sc.generate_sample(rng, sources)
        tile = cv2.resize(sample.image, (PREVIEW_SCALE, PREVIEW_SCALE))
        tiles.append(tile)
    rows = [
        np.hstack(tiles[i : i + PREVIEW_COLUMNS])
        for i in range(0, PREVIEW_SAMPLES, PREVIEW_COLUMNS)
    ]
    path = synthetic_dir(data) / "preview.jpg"
    cv2.imwrite(str(path), np.vstack(rows))
    print(path)


class Block(nn.Module):
    def __init__(self, channels_in: int, channels_out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_out, channels_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PoseNet(nn.Module):
    """Small U-Net with a fold direction head on the bottleneck."""

    def __init__(self, base: int = BASE_CHANNELS) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.down1 = Block(3, base)
        self.down2 = Block(base, base * 2)
        self.down3 = Block(base * 2, base * 4)
        self.middle = Block(base * 4, base * 8)
        self.up3 = Block(base * 12, base * 4)
        self.up2 = Block(base * 6, base * 2)
        self.up1 = Block(base * 3, base)
        self.mask = nn.Conv2d(base, 1, 1)
        self.direction = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(base * 8 * 16, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.down1(x)
        b = self.down2(self.pool(a))
        c = self.down3(self.pool(b))
        m = self.middle(self.pool(c))
        d = self.up3(torch.cat([self.up(m), c], 1))
        d = self.up2(torch.cat([self.up(d), b], 1))
        d = self.up1(torch.cat([self.up(d), a], 1))
        return torch.sigmoid(self.mask(d)), self.direction(m)


def to_tensor(images: np.ndarray) -> torch.Tensor:
    array = images.astype(np.float32) / sc.MAX_U8
    return torch.from_numpy(array).permute(0, 3, 1, 2)


def turn_batch(
    images: np.ndarray,
    masks: np.ndarray,
    directions: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate real frames by a random quarter turn, so a small real set also
    sees other fold directions. Image y points down, so one np.rot90 step maps
    (x, y) to (y, -x)."""
    steps = int(rng.integers(4))
    images = np.rot90(images, steps, axes=(1, 2)).copy()
    masks = np.rot90(masks, steps, axes=(1, 2)).copy()
    directions = directions.copy()
    for _ in range(steps):
        directions = np.column_stack([directions[:, 1], -directions[:, 0]])
    return images, masks, directions


def real_train_arrays(
    data: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table, images, masks = load_real(data)
    train = table[table["split"] == "train"]
    directions = train[["dir_x", "dir_y"]].fillna(0).to_numpy(np.float32).copy()
    # A failed fold's magenta shape is what the robot wanted, not what is there
    directions[~train["ok"].to_numpy()] = 0
    return images[train.index], masks[train.index], directions


def synth_arrays(data: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths = sorted((synthetic_dir(data) / "shards").glob("*.npz"))
    shards = [np.load(p) for p in paths]
    return (
        np.concatenate([s["images"] for s in shards]),
        np.concatenate([s["masks"] for s in shards]),
        np.concatenate([s["directions"] for s in shards]),
    )


def loss_terms(
    mask: torch.Tensor,
    direction: torch.Tensor,
    target: torch.Tensor,
    target_direction: torch.Tensor,
) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy(mask, target)
    overlap = (mask * target).sum((1, 2, 3))
    total = mask.sum((1, 2, 3)) + target.sum((1, 2, 3))
    dice = 1 - ((2 * overlap + 1) / (total + 1)).mean()
    has_fold = target_direction.abs().sum(1) > 0
    angle = torch.zeros((), device=mask.device)
    if has_fold.any():
        unit = nn.functional.normalize(direction[has_fold], dim=1)
        angle = (1 - (unit * target_direction[has_fold]).sum(1)).mean()
    return bce + dice + DIRECTION_WEIGHT * angle


def fit(
    model: PoseNet,
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    epochs: int,
    rate: float,
    augment: bool,
) -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device).train()
    images, masks, directions = arrays
    optimiser = torch.optim.AdamW(model.parameters(), lr=rate)
    steps = max(len(images) // BATCH, 1) * epochs
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=rate, total_steps=steps
    )
    rng = np.random.default_rng(SEED)
    for epoch in range(epochs):
        order = rng.permutation(len(images))
        total, started = 0.0, time.time()
        batches = max(len(images) // BATCH, 1)
        for b in range(batches):
            pick = np.sort(order[b * BATCH : (b + 1) * BATCH])
            x, y, z = images[pick], masks[pick], directions[pick]
            if augment:
                x, y, z = turn_batch(x, y, z, rng)
            mask, direction = model(to_tensor(x).to(device))
            target = torch.from_numpy(y.astype(np.float32))[:, None].to(device)
            loss = loss_terms(
                mask, direction, target, torch.from_numpy(z).to(device)
            )
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            schedule.step()
            total += float(loss.detach())
        print(
            f"epoch {epoch + 1}/{epochs} loss {total / batches:.3f} "
            f"{time.time() - started:.0f}s",
            flush=True,
        )
    model.to("cpu").eval()


def weights_path(data: Path, name: str) -> Path:
    return synthetic_dir(data) / f"{name}.pt"


def train(data: Path, name: str) -> None:
    torch.manual_seed(SEED)
    model = PoseNet()
    if name == "synth":
        fit(model, synth_arrays(data), EPOCHS[name], LEARNING_RATE, False)
    elif name == "real":
        fit(model, real_train_arrays(data), EPOCHS[name], LEARNING_RATE, True)
    else:
        state = torch.load(weights_path(data, "synth"), weights_only=True)
        model.load_state_dict(state)
        fit(
            model,
            real_train_arrays(data),
            EPOCHS[name],
            TUNE_LEARNING_RATE,
            True,
        )
    torch.save(model.state_dict(), weights_path(data, name))


def predict(
    model: PoseNet, images: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    masks, directions = [], []
    with torch.no_grad():
        for start in range(0, len(images), BATCH):
            mask, direction = model(to_tensor(images[start : start + BATCH]))
            masks.append(mask[:, 0].numpy())
            directions.append(direction.numpy())
    return np.concatenate(masks), np.concatenate(directions)


def run_interval(values: pd.Series, runs: pd.Series) -> list[float] | None:
    """95% interval of the median, resampling whole runs."""
    if values.empty:
        return None
    rng = np.random.default_rng(SEED)
    groups = [g.to_numpy() for _, g in values.groupby(runs.loc[values.index])]
    draws = [
        float(
            np.median(
                np.concatenate(
                    [
                        groups[i]
                        for i in rng.integers(len(groups), size=len(groups))
                    ]
                )
            )
        )
        for _ in range(BOOTSTRAP_DRAWS)
    ]
    return [round(float(v), 3) for v in np.percentile(draws, PERCENTILES)]


def method_rows(
    table: pd.DataFrame,
    masks: np.ndarray,
    found: np.ndarray,
    directions: np.ndarray | None,
) -> pd.DataFrame:
    rows = []
    for position, index in enumerate(table.index):
        row = table.loc[index]
        centre = sc.centre_error_rel(found[position], masks[index])
        angle = None
        if (
            directions is not None
            and row["check"] in ("fold1", "fold2")
            and row["ok"]
            and not np.isnan(row["dir_x"])
        ):
            angle = sc.angle_error_deg(
                directions[position],
                np.array([row["dir_x"], row["dir_y"]]),
            )
        rows.append(
            {
                "centre": centre,
                "iou": sc.iou(found[position], masks[index]),
                "angle": angle,
                "false_crepe": (
                    bool(found[position].any())
                    if row["check"] in EMPTY_CHECKS
                    else None
                ),
            }
        )
    return pd.DataFrame(rows, index=table.index, dtype=float)


def summarise(
    table: pd.DataFrame, scores: pd.DataFrame
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    groups: dict[str, pd.Index] = {"all": table.index}
    for robot, part in table.groupby("robot"):
        groups[str(robot)] = part.index
    for check, part in table.groupby("check"):
        groups[str(check)] = part.index
    for name, index in groups.items():
        part = scores.loc[index]
        entry: dict[str, Any] = {"n": len(part)}
        for metric in ("centre", "iou", "angle"):
            values = part[metric].dropna()
            if values.empty:
                continue
            entry[metric] = round(float(values.median()), 3)
            entry[f"{metric}_ci"] = run_interval(values, table["run_id"])
            entry[f"{metric}_n"] = len(values)
        empties = part["false_crepe"].dropna()
        if not empties.empty:
            entry["false_crepe_rate"] = round(float(empties.mean()), 3)
        out[name] = entry
    return out


def constant_baseline(
    table: pd.DataFrame, masks: np.ndarray, part: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Per robot and check, always answer the training average: the mean
    mask (thresholded) and the mean direction. A model has to beat this to
    show it sees anything in the frame at all."""
    train = table[table["split"] == "train"]
    found = np.zeros((len(part), sc.SIZE, sc.SIZE), bool)
    directions = np.zeros((len(part), 2))
    for position, index in enumerate(part.index):
        row = part.loc[index]
        rows = train[
            (train["robot"] == row["robot"]) & (train["check"] == row["check"])
        ]
        if len(rows):
            found[position] = masks[rows.index].mean(0) > MASK_THRESHOLD
            good = rows[rows["ok"]][["dir_x", "dir_y"]].dropna().to_numpy()
            if len(good):
                directions[position] = good.mean(0)
    return found, directions


def score(data: Path, split: str) -> None:
    table, images, masks = load_real(data)
    part = table[table["split"] == split]
    results: dict[str, Any] = {}
    otsu = np.stack([sc.otsu_baseline(images[i]) for i in part.index])
    results["otsu"] = summarise(part, method_rows(part, masks, otsu, None))
    found, directions = constant_baseline(table, masks, part)
    results["constant"] = summarise(
        part, method_rows(part, masks, found, directions)
    )
    for name in EPOCHS:
        path = weights_path(data, name)
        if not path.is_file():
            continue
        model = PoseNet()
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()
        probabilities, directions = predict(model, images[part.index])
        found = np.stack(
            [sc.clean_mask(p > MASK_THRESHOLD) for p in probabilities]
        )
        results[name] = summarise(
            part, method_rows(part, masks, found, directions)
        )
    path = synthetic_dir(data) / f"scores-{split}.json"
    path.write_text(json.dumps(results, indent=1))
    show = ["all", *sorted(k for k in results["otsu"] if k != "all")]
    for group in show:
        print(f"\n{group} (n={results['otsu'][group]['n']})")
        for method, summary in results.items():
            item = summary[group]
            print(
                f"  {method:9s} centre {item.get('centre')} "
                f"{item.get('centre_ci')}  iou {item.get('iou')} "
                f"{item.get('iou_ci')}  angle {item.get('angle')} "
                f"{item.get('angle_ci')}  "
                f"false_crepe {item.get('false_crepe_rate')}"
            )
    print("\nwrote", path)


def export(data: Path, name: str = "synth") -> None:
    model = PoseNet()
    model.load_state_dict(
        torch.load(weights_path(data, name), weights_only=True)
    )
    model.eval()
    sample = torch.rand(1, 3, sc.SIZE, sc.SIZE)
    path = synthetic_dir(data) / f"{name}.onnx"
    # A fixed batch of 1 keeps onnxruntime's fast paths: 3.6 times faster on one
    # thread than the same weights exported with a variable batch
    torch.onnx.export(
        model,
        (sample,),
        str(path),
        input_names=["x"],
        output_names=["mask", "direction"],
        opset_version=17,
        dynamo=False,
    )
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(path), options)
    onnx_mask, onnx_direction = session.run(None, {"x": sample.numpy()})
    with torch.no_grad():
        mask, direction = model(sample)
    gap = max(
        float(np.abs(onnx_mask - mask.numpy()).max()),
        float(np.abs(onnx_direction - direction.numpy()).max()),
    )
    assert gap < PARITY_TOLERANCE, f"onnx parity gap {gap}"
    timings = []
    for _ in range(LATENCY_RUNS):
        started = time.perf_counter()
        session.run(None, {"x": sample.numpy()})
        timings.append((time.perf_counter() - started) * 1000)
    parameters = sum(p.numel() for p in model.parameters())
    print(
        f"parity gap {gap:.2e}, {parameters} parameters, "
        f"{path.stat().st_size / 1e6:.1f} MB, single thread on this machine "
        f"median {np.median(timings):.1f} ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    commands.add_parser("generate")
    commands.add_parser("preview")
    train_parser = commands.add_parser("train")
    train_parser.add_argument("name", choices=sorted(EPOCHS))
    score_parser = commands.add_parser("score")
    score_parser.add_argument("--split", choices=("train", "test"))
    export_parser = commands.add_parser("export")
    export_parser.add_argument("--name", default="synth")
    args = parser.parse_args()
    if args.command == "train":
        train(args.data_dir, args.name)
    elif args.command == "score":
        score(args.data_dir, args.split)
    elif args.command == "export":
        export(args.data_dir, args.name)
    else:
        {"prepare": prepare, "generate": generate, "preview": preview}[
            args.command
        ](args.data_dir)


if __name__ == "__main__":
    main()
