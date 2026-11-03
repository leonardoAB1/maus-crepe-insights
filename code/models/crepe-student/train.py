"""Distil teacher crêpe masks into the student and export it to ONNX.

Teacher: binary crêpe masks per raw check frame (SAM 3, prompts "crepe" and
"food", union of instances). Generating them needs SAM 3 with CUDA and is not
part of the repo environments; see README.md. This script only needs the
teacher masks:

    pixi run -e ml python models/crepe-student/train.py \\
        --teacher /abs/path/teacher.parquet

``teacher.parquet`` has one row per frame with ``image`` (raw BGR frame) and
``mask`` (binary PNG, same size), both absolute or relative to the parquet
folder, and ``run_id``. Runs with ``is_test_run`` are held out.

Writes ``model.onnx`` next to this file (or ``--out``) and prints the test
IoU against the teacher and the torch/onnxruntime parity.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

from maus_export.crepe_student import (
    INPUT_NAME,
    INPUT_SIZE,
    is_test_run,
    open_session,
    preprocess,
)

HERE = Path(__file__).resolve().parent
OUTPUT_NAME = "mask"
CLASSES = 2
EPOCHS = 30
BATCH = 16
LEARNING_RATE = 1e-3
EVAL_EVERY = 5
FLIP_CHANCE = 0.5
GAIN_RANGE = (0.8, 1.2)
OFFSET_RANGE = (-0.2, 0.2)
OPSET = 17
WORKERS = 4
SEED = 0
PARITY_MIN_AGREEMENT = 0.999


class Frames(Dataset):
    """Raw frames and teacher masks at the student resolution."""

    def __init__(self, rows: pd.DataFrame, root: Path, train: bool) -> None:
        self.rows, self.root, self.train = rows, root, train

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows.iloc[i]
        image = cv2.imread(str(self.root / row["image"]))
        mask = cv2.imread(str(self.root / row["mask"]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, INPUT_SIZE, interpolation=cv2.INTER_NEAREST)
        x = preprocess(image)[0]
        y = mask > 0
        if self.train:
            if np.random.rand() < FLIP_CHANCE:
                x, y = x[:, :, ::-1], y[:, ::-1]
            x = x * np.random.uniform(*GAIN_RANGE) + np.random.uniform(
                *OFFSET_RANGE
            )
        return (
            torch.from_numpy(np.ascontiguousarray(x)).float(),
            torch.from_numpy(np.ascontiguousarray(y)).long(),
        )


class MaskHead(nn.Module):
    """Wraps the network so the graph returns a uint8 mask."""

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)["out"]
        return logits.argmax(1, keepdim=True).to(torch.uint8)


def build_model(pretrained: bool = True) -> nn.Module:
    """LR-ASPP MobileNetV3-Large with two classes (background, crêpe)."""
    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
    return lraspp_mobilenet_v3_large(
        weights_backbone=weights, num_classes=CLASSES
    )


def test_iou(model: nn.Module, loader: DataLoader, device: str) -> float:
    """Pixel IoU of the crêpe class against the teacher."""
    model.eval()
    inter = union = 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device))["out"].argmax(1).cpu() > 0
            truth = y > 0
            inter += int((pred & truth).sum())
            union += int((pred | truth).sum())
    return inter / max(union, 1)


def export(model: nn.Module, path: Path) -> None:
    """Export the uint8 mask graph with a fixed 1x3x216x384 input."""
    head = MaskHead(model.cpu()).eval()
    sample = torch.zeros(1, 3, INPUT_SIZE[1], INPUT_SIZE[0])
    torch.onnx.export(
        head,
        sample,
        path,
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        opset_version=OPSET,
        dynamo=False,
    )


def parity(model: nn.Module, path: Path, frame: np.ndarray) -> float:
    """Share of mask pixels where torch and onnxruntime agree."""
    head = MaskHead(model.cpu()).eval()
    x = preprocess(frame)
    with torch.no_grad():
        expected = head(torch.from_numpy(x)).numpy()
    got = open_session(path).run(None, {INPUT_NAME: x})[0]
    return float(np.mean(expected == got))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=HERE / "model.onnx")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    root = args.teacher.parent
    table = pd.read_parquet(args.teacher).dropna(subset=["mask"])
    test = table["run_id"].map(lambda run: is_test_run(int(run)))
    train_rows, test_rows = table[~test], table[test]
    print(
        f"train {len(train_rows)} frames / {train_rows.run_id.nunique()} "
        f"runs, test {len(test_rows)} / {test_rows.run_id.nunique()} runs"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model().to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, args.epochs
    )
    loss_fn = nn.CrossEntropyLoss()
    train_loader = DataLoader(
        Frames(train_rows, root, True),
        batch_size=BATCH,
        shuffle=True,
        num_workers=WORKERS,
    )
    test_loader = DataLoader(
        Frames(test_rows, root, False), batch_size=BATCH, num_workers=WORKERS
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x)["out"], y)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += loss.item() * len(x)
        schedule.step()
        if epoch % EVAL_EVERY == 0 or epoch == args.epochs:
            iou = test_iou(model, test_loader, device)
            print(
                f"epoch {epoch}: loss {total / len(train_rows):.4f}, "
                f"test IoU vs teacher {iou:.3f}"
            )
    export(model, args.out)
    sample = cv2.imread(str(root / test_rows.iloc[0]["image"]))
    agreement = parity(model, args.out, sample)
    print(f"torch vs onnxruntime pixel agreement {agreement:.6f}")
    if agreement < PARITY_MIN_AGREEMENT:
        raise SystemExit("onnx export differs from the torch model")


if __name__ == "__main__":
    main()
