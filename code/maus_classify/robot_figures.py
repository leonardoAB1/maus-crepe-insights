"""Per-robot score chart from harness model cards (aggregates only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .harness import MAJORITY_BA  # noqa: E402
from .split import TEST  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
HOLDOUT = "#2a78d6"  # categorical slot 1
LORO = "#eb6834"  # categorical slot 2
BAR_WIDTH = 0.36
BAR_GAP = 0.02
PANEL_SIZE = (3.6, 3.6)
NA_Y = 0.06
LABEL_PAD = 0.02
TITLE_PAD = 20
DPI = 150
ROBOT_PREFIX = "maus-"


def _score(m: dict[str, Any] | None) -> float | None:
    """Balanced accuracy, or None when not scored or one class only."""
    if not m or "skipped" in m:
        return None
    if sum(1 for c in m["per_class"].values() if c["support"] > 0) < 2:
        return None
    return m["balanced_accuracy"]


def per_robot_figure(cards: dict[str, dict[str, Any]]) -> plt.Figure:
    """One panel per model: holdout and leave-one-robot-out balanced
    accuracy per robot, against the majority baseline."""
    fig, axes = plt.subplots(
        1,
        len(cards),
        figsize=(PANEL_SIZE[0] * len(cards), PANEL_SIZE[1]),
        sharey=True,
        facecolor=SURFACE,
    )
    axes = [axes] if len(cards) == 1 else list(axes)
    for ax, (name, card) in zip(axes, cards.items(), strict=True):
        per = card.get("metrics", {}).get("per_robot", {}).get(TEST, {})
        loro = card.get("leave_one_robot_out", {})
        robots = sorted(set(per) | set(loro))
        ax.set_facecolor(SURFACE)
        for i, robot in enumerate(robots):
            for offset, value, color in (
                (-1, _score(per.get(robot)), HOLDOUT),
                (1, _score(loro.get(robot)), LORO),
            ):
                x = i + offset * (BAR_WIDTH + BAR_GAP) / 2
                if value is None:
                    ax.text(x, NA_Y, "n/a", ha="center", color=MUTED, size=8)
                    continue
                ax.bar(x, value, BAR_WIDTH, color=color, zorder=2)
                ax.text(
                    x,
                    value + LABEL_PAD,
                    f"{value:.2f}",
                    ha="center",
                    color=INK_2,
                    size=8,
                    zorder=4,
                    bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1},
                )
        ax.axhline(MAJORITY_BA, color=INK_2, lw=1, ls="--", zorder=3)
        ax.set_xticks(range(len(robots)))
        ax.set_xticklabels(
            [r.removeprefix(ROBOT_PREFIX) for r in robots], color=INK_2
        )
        verdicts = []
        if "deployable" in card:
            ok = card["deployable"]["deployable"]
            verdicts.append("beats null" if ok else "not above null")
        if "generalization" in card:
            ok = card["generalization"]["generalizes"]
            verdicts.append("generalizes" if ok else "does not generalize")
        ax.set_title(name, color=INK, size=11, loc="left", pad=TITLE_PAD)
        ax.text(
            0,
            1.02,
            ", ".join(verdicts),
            transform=ax.transAxes,
            color=INK_2,
            size=8,
            va="bottom",
        )
        ax.set_ylim(0, 1.08)
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(MUTED)
        ax.tick_params(colors=MUTED, length=0)
    axes[0].set_ylabel("balanced accuracy", color=INK_2)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=HOLDOUT),
        plt.Rectangle((0, 0), 1, 1, color=LORO),
        plt.Line2D([0], [0], color=INK_2, lw=1, ls="--"),
    ]
    fig.legend(
        handles,
        ["holdout of that robot", "robot held out entirely", "majority"],
        loc="lower center",
        ncol=3,
        frameon=False,
        labelcolor=INK_2,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def save_per_robot_figure(cards: dict[str, dict[str, Any]], path: Path) -> None:
    """Render `per_robot_figure` to an image file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = per_robot_figure(cards)
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
