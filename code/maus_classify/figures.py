"""Example result grid shared by every frame model.

Each tile is one frame with a caption: the robot's label against the
model's answer, the model's confidence and the robot name. Every model PR
shows one of these (see code/models/README.md). Same palette and JPEG
settings as maus_export.figures.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib
import numpy as np

from maus_export.figures import INK, INK2, JPEG_QUALITY, SURFACE

matplotlib.use("Agg")  # files only, no display
import matplotlib.pyplot as plt  # noqa: E402

MATCH = "#2e7d32"  # model agrees with the robot label
MISS = "#c62828"  # model disagrees
TILE_IN = (3.6, 3.9)  # inches per tile, caption included
CAPTION_SPACE = 0.38  # gap between rows, share of a tile height
CAPTION_IN = 0.75  # room for the first row of captions, inches
TILE_GAP = 0.05
MARGIN = 0.01
DPI = 90
COLS = 3
TITLE_PAD = 0.9  # inches above the grid for title and subtitle


@dataclass
class Example:
    """One tile. `image` is BGR as read by cv2."""

    image: np.ndarray
    truth: str
    predicted: str
    confidence: float
    robot: str
    note: str = ""
    # Whether the model agrees with the label (green caption). None falls
    # back to comparing truth and predicted text, fine for plain labels
    agree: bool | None = None
    # Names the number in the caption, e.g. "P(fail)"; empty means it is
    # the model's confidence in its answer
    score_name: str = ""


def _caption(ex: Example) -> str:
    lines = [
        f"robot: {ex.truth}",
        f"model: {ex.predicted} "
        f"({f'{ex.score_name} ' if ex.score_name else ''}"
        f"{ex.confidence:.2f})",
        ex.robot,
    ]
    if ex.note:
        lines.append(ex.note)
    return "\n".join(lines)


def example_grid(
    examples: Sequence[Example],
    out: Path,
    title: str,
    subtitle: str = "",
    crop: Callable[[np.ndarray], np.ndarray] | None = None,
    cols: int = COLS,
    dpi: int = DPI,
) -> Path:
    """Write a captioned grid of examples to `out` (.jpg or .png).

    `crop` trims each frame before drawing (e.g. to the plate). A caption
    is green when the model agrees with the robot label, red otherwise."""
    if not examples:
        raise ValueError("need at least one example")
    cols = min(cols, len(examples))
    rows = math.ceil(len(examples) / cols)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(
            TILE_IN[0] * cols,
            TILE_IN[1] * rows + TITLE_PAD + CAPTION_IN,
        ),
        facecolor=SURFACE,
        squeeze=False,
        gridspec_kw={"hspace": CAPTION_SPACE, "wspace": TILE_GAP},
    )
    for ax in axes.flat:
        ax.axis("off")
    for ax, ex in zip(axes.flat, examples, strict=False):
        img = crop(ex.image) if crop else ex.image
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        agree = (
            ex.agree
            if ex.agree is not None
            else ex.truth.split()[0] == ex.predicted
        )
        ax.set_title(
            _caption(ex),
            loc="left",
            fontsize=9,
            family="DejaVu Sans Mono",
            color=MATCH if agree else MISS,
        )
    fig.suptitle(
        title, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK
    )
    if subtitle:
        fig.text(
            0.01,
            1 - 0.55 / fig.get_figheight(),
            subtitle,
            fontsize=9,
            color=INK2,
            va="top",
        )
    fig.subplots_adjust(
        left=MARGIN,
        right=1 - MARGIN,
        bottom=MARGIN,
        top=1 - (TITLE_PAD + CAPTION_IN) / fig.get_figheight(),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out, dpi=dpi, facecolor=SURFACE, pil_kwargs={"quality": JPEG_QUALITY}
    )
    plt.close(fig)
    return out
