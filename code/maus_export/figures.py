"""Annotated example figure: how the vision scores are measured.

Draws one spread check and one fold-2 check from the robot's overlays with
our measurements on top. Used for the README image in `assets/`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np

from . import vision_metrics as vm

matplotlib.use("Agg")  # files only, no display
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

DEFAULT_SPREAD_RUN = 3297
DEFAULT_FOLD_RUN = 3282
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
OURS = "#ffffff"  # our marks, distinct from every robot colour
ZONE_CENTRE = "#2a78d6"
BARE_RGB = (245, 200, 0)  # yellow: bare plate we measure
INSIDE_RGB = (255, 255, 255)  # crepe inside the ideal fold
OUTSIDE_RGB = (235, 104, 52)  # crepe outside the ideal fold
MISSED_RGB = (42, 120, 214)  # ideal fold not covered
BARE_ALPHA = 0.85
FOLD_ALPHA = 0.45
FIG_SIZE = (13, 7.4)
DPI = 110
JPEG_QUALITY = 85
PASSING_AREA_RATIO = 0.54  # median over passing fold-2 checks
PASSING_IOU = 0.90
# Halfway between a good quarter fold (0.54) and no second fold (1.0)
STAYED_HALF_RATIO = 0.77
BOX = {"boxstyle": "round,pad=0.5", "fc": SURFACE, "ec": "none", "alpha": 0.92}


def overlay_path(data_dir: Path, run_id: int, check: str) -> Path:
    """Overlay image of the last attempt of a check in a run."""
    run_file = data_dir / "raw" / "runs" / f"{run_id}.json"
    run: dict[str, Any] = json.loads(run_file.read_text(encoding="utf-8"))
    attempts = [
        c for c in run.get("visionChecks") or [] if c.get("phase") == check
    ]
    if not attempts or not attempts[-1].get("overlay"):
        raise ValueError(f"run {run_id} has no {check} overlay")
    path = data_dir / "raw" / "media" / str(run_id) / attempts[-1]["overlay"]
    if not path.is_file():
        raise FileNotFoundError(f"{path} (download with fetch --media)")
    return path


def _tint(rgb: np.ndarray, mask: np.ndarray, colour: tuple, alpha: float):
    rgb[mask] = (1 - alpha) * rgb[mask] + alpha * np.array(colour)


def _text(ax: Axes, x: float, txt: str, ha: str = "left") -> None:
    ax.text(
        x,
        0.02,
        txt,
        transform=ax.transAxes,
        ha=ha,
        va="bottom",
        family="DejaVu Sans Mono",
        fontsize=10,
        color=INK,
        bbox=BOX,
    )


def draw_spread(ax: Axes, path: Path) -> None:
    img = cv2.imread(str(path))
    prep = vm.prepare(img)
    if prep is None:
        raise ValueError(f"no analysis zone found in {path}")
    hsv, zone, inside = prep
    zx, zy, zr = zone
    crepe = vm.filled_outline(vm.outline_mask(hsv) & inside, zone)
    if crepe is None:
        raise ValueError(f"no crepe outline found in {path}")
    m = vm.spread_metrics(img)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(float)
    _tint(rgb, vm.bare_plate(img, hsv, crepe, zr), BARE_RGB, BARE_ALPHA)
    ax.imshow(rgb.astype(np.uint8))
    ax.contour(
        crepe, levels=[0.5], colors=OURS, linewidths=1.6, linestyles="--"
    )
    ys, xs = np.nonzero(crepe)
    cx, cy = xs.mean(), ys.mean()
    r_eq = m["diameter_rel"] * zr
    ax.add_patch(Circle((cx, cy), r_eq, fill=False, ec=OURS, lw=1.2, ls=":"))
    ax.plot([zx], [zy], "+", color=ZONE_CENTRE, ms=16, mew=2.5)
    ax.plot([cx], [cy], "x", color=OURS, ms=11, mew=2.5)
    ax.annotate(
        "",
        xy=(cx + r_eq, cy),
        xytext=(cx, cy),
        arrowprops={"arrowstyle": "<->", "color": OURS, "lw": 1.6},
    )
    _text(
        ax,
        0.02,
        f"diameter      {m['diameter_rel']:.2f} of zone\n"
        f"centre offset {m['offset_rel']:.3f} of radius\n"
        f"circularity   {m['circularity']:.2f}\n"
        f"bare plate    {m['uncovered_frac']:.1%} of crepe (yellow)\n"
        f"  robot red   {m['hole_frac']:.1%} (marks only some gaps)\n"
        f"residue       {m['residue_frac']:.2%} of zone",
    )


def draw_fold(ax: Axes, fold1: Path, fold2: Path) -> None:
    img = cv2.imread(str(fold2))
    prep = vm.prepare(img)
    if prep is None:
        raise ValueError(f"no analysis zone found in {fold2}")
    crepe, ideal = vm.fold_regions(*prep)
    if crepe is None or ideal is None:
        raise ValueError(f"no crepe or ideal fold found in {fold2}")
    m = vm.fold_metrics(img)
    first = vm.fold_metrics(cv2.imread(str(fold1)))
    ratio = m["area_rel"] / first["area_rel"]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(float)
    _tint(rgb, crepe & ~ideal, OUTSIDE_RGB, FOLD_ALPHA)
    _tint(rgb, ideal & ~crepe, MISSED_RGB, FOLD_ALPHA)
    _tint(rgb, crepe & ideal, INSIDE_RGB, FOLD_ALPHA)
    ax.imshow(rgb.astype(np.uint8))
    verdict = (
        "second fold ran, but the crepe stayed a half"
        if ratio > STAYED_HALF_RATIO
        else "second fold halved the crepe"
    )
    _text(
        ax,
        0.02,
        f"{verdict}\n"
        f"area ratio    {ratio:.2f} fold 2 / fold 1"
        f" (passing {PASSING_AREA_RATIO:.2f})\n"
        f"overlap (IoU) {m['iou']:.2f} with ideal"
        f"   (passing {PASSING_IOU:.2f})\n"
        f"area          {m['area_rel']:.2f} of zone\n"
        f"browning L*   {m['l_mean']:.0f} +/- {m['l_std']:.0f}",
    )
    ax.text(
        0.98,
        0.98,
        "white tint: crepe in ideal shape\norange tint: crepe outside it\n"
        "blue tint: ideal shape not covered",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.5,
        color=INK,
        bbox=BOX,
    )


def _check_title(data_dir: Path, run_id: int, check: str) -> str:
    run = json.loads(
        (data_dir / "raw" / "runs" / f"{run_id}.json").read_text("utf-8")
    )
    attempts = [c for c in run["visionChecks"] if c.get("phase") == check]
    errors = attempts[-1].get("errors") or []
    return "flagged: " + ", ".join(errors) if errors else "passed"


def vision_example(
    data_dir: Path,
    out: Path,
    spread_run: int = DEFAULT_SPREAD_RUN,
    fold_run: int = DEFAULT_FOLD_RUN,
) -> Path:
    """Write the annotated spread + fold-2 example figure to `out`."""
    fig, (left, right) = plt.subplots(1, 2, figsize=FIG_SIZE, facecolor=SURFACE)
    draw_spread(left, overlay_path(data_dir, spread_run, "spread"))
    draw_fold(
        right,
        overlay_path(data_dir, fold_run, "fold1"),
        overlay_path(data_dir, fold_run, "fold2"),
    )
    for ax, label, run_id, check in (
        (left, "Spread check", spread_run, "spread"),
        (right, "Fold 2 check", fold_run, "fold2"),
    ):
        flag = _check_title(data_dir, run_id, check)
        ax.set_title(
            f"{label} ({flag})",
            loc="left",
            fontsize=12,
            fontweight="bold",
            color=INK,
        )
        ax.axis("off")
    fig.suptitle(
        "How the vision scores are measured on the robot's check overlays",
        x=0.01,
        ha="left",
        fontsize=14,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.01,
        0.885,
        "Robot drawings: blue zone, green outline, cyan fitted circle, "
        "red holes, magenta ideal fold.\nOurs: yellow = bare plate inside "
        "the crepe; in white, dashed = crepe region, dotted = equal-area "
        "circle, + zone centre, x crepe centre.",
        fontsize=9.5,
        color=INK2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out,
        dpi=DPI,
        facecolor=SURFACE,
        pil_kwargs={"quality": JPEG_QUALITY},
    )
    plt.close(fig)
    print(f"example figure -> {out}")
    return out
