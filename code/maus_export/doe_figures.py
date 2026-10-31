"""Figures for the DoE batch (#26): the design and the results.

Everything is drawn on the coded scale (low / middle / high), so the figures
show no recipe values and can be committed to the public repo.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from .analysis import OUTCOMES
from .doe import (
    CODED_SUFFIX,
    DEFAULT_FACTORS,
    RUN_ID,
    Factor,
    dsd,
    run_sheet,
)
from .figures import DPI, INK, SURFACE

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402
from scipy.stats import t as t_dist  # noqa: E402

from .doe_analyze import (  # noqa: E402
    ALPHA,
    INTERCEPT,
    MAIN,
    SQUARE,
    best_level,
    predict,
)

INK_2 = "#52514e"  # secondary text
GRID = "#e4e3df"
LOW = "#e34948"  # diverging red pole
MID = "#f0efec"  # diverging neutral midpoint
HIGH = "#2a78d6"  # diverging blue pole
# Setting levels are ordered, not good or bad: one blue hue, light to dark
LEVEL_COLOURS = ("#cde2fb", "#6da7ec", "#1c5cab")
FADED = "#c3c2b7"  # effect not clear yet
CI = 0.90
CURVE_POINTS = 50
JITTER = 0.04
MARKER = 30
LINE = 2.0
DESIGN_SIZE = (15, 7.6)
EFFECTS_SIZE = (13, 6.4)
SIM_SEED = 1  # a draw where only the planted effects pass the cut
SIM_OUTCOME = "spread_diameter"
# Simulated batch, coded scale: fan power raises the spread size, y center
# lowers it, the inner radius offset has a sweet spot, the rest do nothing
SIM_BASE, SIM_FAN, SIM_YC, SIM_CURVE, SIM_NOISE = 0.8, 0.05, -0.03, -0.04, 0.01
SIM_FIRST_RUN = 1000
SIM_ROBOT = "sim"

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "text.color": INK,
        "axes.titlecolor": INK,
    }
)


def label(name: str) -> str:
    """Short readable setting name."""
    leaf = name.rsplit(".", 1)[-1].removeprefix("concentric_")
    return leaf.removesuffix("_mm").replace("_", " ")


def simulated_factors() -> list[Factor]:
    """The default settings with made-up levels (0 to 1)."""
    return [Factor(name, 0.0, 1.0) for name in DEFAULT_FACTORS]


def simulated_batch(
    seed: int = SIM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """A filled run sheet with a planted response, plus matching run and
    vision tables. Used by the example figure and the tests."""
    factors = simulated_factors()
    sheet = run_sheet(dsd(len(factors)), factors, SIM_ROBOT, seed=seed)
    sheet[RUN_ID] = np.arange(SIM_FIRST_RUN, SIM_FIRST_RUN + len(sheet))
    x = sheet[[f.name + CODED_SUFFIX for f in factors]].to_numpy(float)
    rng = np.random.default_rng(seed)
    y = (
        SIM_BASE
        + SIM_FAN * x[:, 0]
        + SIM_YC * x[:, 2]
        + SIM_CURVE * x[:, 4] ** 2
        + rng.normal(0, SIM_NOISE, len(x))
    )
    runs = pd.DataFrame(
        {
            "id": sheet[RUN_ID],
            "robot": SIM_ROBOT,
            "topping": "plain",
            "status": "succeeded",
            "check.spread.ok": True,
            **{f.column: sheet[f.name] for f in factors},
        }
    )
    vision = pd.DataFrame(
        {"id": sheet[RUN_ID], OUTCOMES[SIM_OUTCOME].column: y}
    )
    return sheet, runs, vision


def _finish(fig: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


# --- design ----------------------------------------------------------------


def design_figure(
    design: np.ndarray, factors: list[Factor], path: Path
) -> Path:
    """Run sheet grid with the setting values, and the correlation map."""
    x = np.asarray(design, dtype=float)
    n, m = x.shape
    labels = [label(f.name) for f in factors]
    values = np.column_stack([f.decode(x[:, j]) for j, f in enumerate(factors)])
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=DESIGN_SIZE, width_ratios=[1.3, 1.25]
    )
    fig.patch.set_facecolor(SURFACE)

    # Run sheet: one row per run, one column per setting
    ax1.imshow(x, cmap=ListedColormap(LEVEL_COLOURS), vmin=-1.5, vmax=1.5)
    for i in range(n):
        for j in range(m):
            ax1.text(
                j,
                i,
                f"{values[i, j]:g}",
                ha="center",
                va="center",
                fontsize=7.5,
                color=SURFACE if x[i, j] > 0 else INK,
            )
    ax1.set_xticks(range(m), labels, rotation=45, ha="right")
    centre = (x == 0).all(axis=1)
    ax1.set_yticks(
        range(n),
        [f"{i + 1} centre" if centre[i] else str(i + 1) for i in range(n)],
        fontsize=8,
    )
    ax1.set_ylabel("design row (Maus runs them in a shuffled order)")
    ax1.set_title(
        f"The batch: {n} runs, each setting at 3 levels\n"
        "light = low, mid blue = middle, dark = high",
        loc="left",
        fontsize=11,
    )
    ax1.tick_params(length=0)
    for side in ax1.spines.values():
        side.set_visible(False)
    ax1.set_xticks(np.arange(-0.5, m), minor=True)
    ax1.set_yticks(np.arange(-0.5, n), minor=True)
    ax1.grid(which="minor", color=SURFACE, linewidth=2)

    # Correlation between every main effect and every squared term
    terms = np.hstack([x, x**2])
    corr = np.corrcoef(terms, rowvar=False)
    tick = labels + [f"{lb}²" for lb in labels]
    cmap = LinearSegmentedColormap.from_list("div", [LOW, MID, HIGH])
    im = ax2.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
    for i in range(2 * m):
        for j in range(2 * m):
            if i != j and abs(corr[i, j]) > 1e-9:
                ax2.text(
                    j,
                    i,
                    f"{corr[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=INK,
                )
    ax2.set_xticks(range(2 * m), tick, rotation=45, ha="right", fontsize=8)
    ax2.set_yticks(range(2 * m), tick, fontsize=8)
    ax2.axhline(m - 0.5, color=INK_2, linewidth=1)
    ax2.axvline(m - 0.5, color=INK_2, linewidth=1)
    ax2.tick_params(length=0)
    for side in ax2.spines.values():
        side.set_visible(False)
    ax2.set_title(
        "Why it works: gray cells = no overlap (correlation 0).\n"
        "Each setting's effect is measured on its own, apart from every\n"
        "other setting and from any curve (² = sweet-spot term)",
        loc="left",
        fontsize=10,
    )
    cb = fig.colorbar(im, ax=ax2, fraction=0.04, pad=0.02)
    cb.set_label("correlation")
    cb.outline.set_visible(False)
    fig.tight_layout()
    return _finish(fig, path)


# --- results ---------------------------------------------------------------


def effects_figure(
    result: dict[str, Any],
    outcome: str,
    path: Path,
    simulated: bool = False,
) -> Path:
    """Which setting to change and which way, plus the curve per setting."""
    factors: list[Factor] = result["factors"]
    eff = result["effects"]
    sub = eff[eff["outcome"] == outcome]
    better = int(sub["better"].iloc[0])
    df = int(sub["df"].iloc[0])
    tq = t_dist.ppf(0.5 + CI / 2, df) if df > 0 else np.nan

    def coef(kind: str, name: str) -> pd.Series | None:
        row = sub[(sub["kind"] == kind) & (sub["factor"] == name)]
        return None if row.empty else row.iloc[0]

    fig = plt.figure(figsize=EFFECTS_SIZE)
    fig.patch.set_facecolor(SURFACE)
    grid = fig.add_gridspec(2, 4, width_ratios=[1.5, 1, 1, 1], wspace=0.35)
    ax = fig.add_subplot(grid[:, 0])

    # Panel 1: change from low to high, 90% interval, coloured by help/hurt
    rows = []
    for f in factors:
        r = coef(MAIN, f.name)
        sq = coef(SQUARE, f.name)
        curved = sq is not None and sq["p"] < ALPHA
        rows.append(
            (label(f.name), 2 * r["coef"], 2 * tq * r["se"], r["p"], curved)
        )
    rows.sort(key=lambda t: abs(t[1]))
    span = max(abs(t[1]) + t[2] for t in rows)
    for i, (_, d, half, p, curved) in enumerate(rows):
        clear = p < ALPHA
        helps = better * d > 0
        colour = (HIGH if helps else LOW) if clear else FADED
        ax.barh(i, d, height=0.55, color=colour)
        ax.errorbar(d, i, xerr=half, color=INK_2, capsize=3, linewidth=1)
        notes = []
        if clear:
            notes.append("raise" if helps else "lower")
        if curved:
            notes.append("sweet spot in range")
        if notes:
            ax.text(
                max(d + half, 0) + span * 0.04,
                i,
                ", ".join(notes),
                ha="left",
                va="center",
                fontsize=8,
                color=INK,
            )
    ax.axvline(0, color=INK_2, linewidth=1)
    lo = min(min(t[1] - t[2] for t in rows), 0)
    ax.set_xlim(lo - span * 0.1, span * 2.0)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlabel(f"change in {outcome}, low -> high setting")
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=HIGH),
        plt.Rectangle((0, 0), 1, 1, color=LOW),
        plt.Rectangle((0, 0), 1, 1, color=FADED),
    ]
    ax.legend(
        handles,
        ["helps (clear)", "hurts (clear)", "not clear yet"],
        loc="upper center",
        bbox_to_anchor=(0.45, -0.1),
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    ax.set_title("Which setting, which way", loc="left", fontsize=11)

    # Panel 2: outcome against each setting, with the fitted curve
    frame = result["frame"]
    y = pd.to_numeric(frame[OUTCOMES[outcome].column], errors="coerce")
    coded = frame[[f.name + CODED_SUFFIX for f in factors]].to_numpy(float)
    fitted = predict(eff, factors, [outcome], coded)[:, 0]
    intercept = sub.loc[sub["kind"] == INTERCEPT, "coef"].iloc[0]
    rng = np.random.default_rng(SIM_SEED)
    xs = np.linspace(-1, 1, CURVE_POINTS)
    first = None
    for k, f in enumerate(factors):
        a = fig.add_subplot(grid[k // 3, 1 + k % 3], sharey=first)
        first = first or a
        xc = coded[:, k]
        b = coef(MAIN, f.name)["coef"]
        sq = coef(SQUARE, f.name)
        c = 0.0 if sq is None else sq["coef"]
        # Dots: each run with the other settings' effects taken out
        own = intercept + b * xc + c * xc**2
        a.scatter(
            xc + rng.uniform(-JITTER, JITTER, len(xc)),
            y - fitted + own,
            s=MARKER,
            color=INK_2,
            alpha=0.6,
            linewidths=0,
        )
        a.plot(xs, intercept + b * xs + c * xs**2, color=HIGH, lw=LINE)
        best = best_level(b, c, better)
        a.scatter(
            [best],
            [intercept + b * best + c * best**2],
            s=70,
            color=HIGH,
            edgecolors=SURFACE,
            linewidths=2,
            zorder=3,
        )
        ticks = ["low", "mid", "high"]
        if not simulated:
            lv = f.decode(np.array([-1, 0, 1]))
            ticks = [f"{t}\n{v:g}" for t, v in zip(ticks, lv, strict=True)]
        a.set_xticks([-1, 0, 1], ticks)
        a.set_title(label(f.name), fontsize=9, loc="left")
        a.grid(color=GRID)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        if k % 3:
            a.tick_params(labelleft=False)
    fig.text(
        0.42,
        0.98,
        "Runs adjusted for the other settings (dots), fitted trend (line), "
        "best level (big dot)",
        fontsize=10,
        va="top",
    )
    sim = "Simulated batch - example only. " if simulated else ""
    fig.suptitle(
        f"{sim}{outcome}: {len(frame)} runs, "
        f"{'higher' if better > 0 else 'lower'} is better",
        x=0.02,
        ha="left",
        y=1.05,
        fontsize=12,
    )
    return _finish(fig, path)


def example_figures(data_dir: Path, assets: Path) -> list[Path]:
    """Redraw the committed figures: the design with the levels from the run
    history, and a simulated result (no real batch yet)."""
    from .doe import SHORTLIST_FILE, factors_from_history
    from .doe_analyze import analyze_sheet

    runs = pd.read_parquet(data_dir / "runs.parquet")
    path = data_dir / SHORTLIST_FILE
    shortlist = (
        json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    )
    factors = factors_from_history(runs, shortlist)
    out = [design_figure(dsd(len(factors)), factors, assets / "doe-design.png")]
    result = analyze_sheet(*simulated_batch(SIM_SEED))
    out.append(
        effects_figure(
            result, SIM_OUTCOME, assets / "doe-results-example.png", True
        )
    )
    return out
