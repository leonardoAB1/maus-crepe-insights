"""Example result images for model PRs (see code/models/README.md).

- check-verdict: one pass and one fail frame per check from the holdout,
  with the head's fail probability, plus a fold failure the head missed
- frame-embedder: 2-D PCA of every frame embedding, coloured by robot and
  marked by check. Aggregate only, no frames.

Needs the ml environment: `pixi run -e ml model-examples`.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import onnxruntime as ort
import pandas as pd

from maus_export.figures import INK, INK2, JPEG_QUALITY, SURFACE
from maus_export.frame_preprocess import crop_plate, load_frame

from .check_verdict import CHECKS, FAIL, PASS, check_rows, embedding_columns
from .figures import Example, example_grid
from .harness import ONNX_INPUT, ONNX_PROBA
from .split import TEST

matplotlib.use("Agg")  # files only, no display
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

FRAMES = Path("training") / "frames.parquet"
EMBEDDINGS = Path("training") / "frame_embeddings.parquet"
ASSETS = Path("assets")
CHECK_VERDICT_ASSET = ASSETS / "check-verdict-example.jpg"
EMBEDDER_ASSET = ASSETS / "frame-embedder-example.jpg"
FOLD_CHECKS = ("fold1", "fold2")
# Fixed categorical order (validated: colour-blind safe on the surface);
# robots beyond these fall back to grey
ROBOT_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a")
OTHER_COLOUR = "#8a8984"
CHECK_MARKERS = ("o", "s", "^", "v", "D", "P", "X", "*")
SCATTER_SIZE = 22
SCATTER_ALPHA = 0.75
RING = 0.6  # surface ring around markers so overlaps stay readable
FIG_SIZE = (9.5, 6.8)
DPI = 100
PCA_SEED = 0
GRID_DPI = 80  # 3x3 grid stays near 200 KB


def _load(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_parquet(data_dir / FRAMES),
        pd.read_parquet(data_dir / EMBEDDINGS),
    )


def _typical(rows: pd.DataFrame, col: str) -> pd.Series:
    """The row at the median of `col`: typical, not cherry-picked."""
    order = rows.sort_values(col, kind="stable")
    return order.iloc[len(order) // 2]


def _scored_check(
    frames: pd.DataFrame,
    emb: pd.DataFrame,
    check: str,
    model_dir: Path,
) -> pd.DataFrame:
    """Holdout rows of one check with p_fail and the head's answer."""
    rows = check_rows(frames, emb, check)
    rows = rows[rows["split"] == TEST].reset_index(drop=True)
    head = model_dir / check
    card = json.loads((head / "model.json").read_text(encoding="utf-8"))
    classes = card["outputs"][1]["classes"]
    sess = ort.InferenceSession(
        str(head / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    x = rows[embedding_columns(rows)].to_numpy(np.float32)
    proba = sess.run([ONNX_PROBA], {ONNX_INPUT: x})[0]
    return rows.assign(
        p_fail=proba[:, classes.index(FAIL)],
        predicted=np.asarray(classes)[proba.argmax(axis=1)],
    )


def _tile(data_dir: Path, row: pd.Series, note: str = "") -> Example:
    caught = row["predicted"] == row["label"]
    miss = "missed by the head" if row["label"] == FAIL else "false alarm"
    notes = [row["check"], note or ("" if caught else miss)]
    return Example(
        image=load_frame(data_dir / row["frame_path"]),
        truth=str(row["label"]),
        predicted=row["predicted"],
        confidence=float(row["p_fail"]),
        robot=str(row["robot"]),
        note="; ".join(filter(None, notes)),
        agree=bool(caught),
        score_name="P(fail)",
    )


def check_verdict_example(data_dir: Path, model_dir: Path, out: Path) -> Path:
    """One typical pass and fail per check (holdout, median P(fail)) and
    the fold failure the head was most sure was a pass."""
    frames, emb = _load(data_dir)
    tiles: list[Example] = []
    folds = []
    for check in CHECKS:
        rows = _scored_check(frames, emb, check, model_dir)
        for label in (PASS, FAIL):
            group = rows[rows["label"] == label]
            if len(group):
                tiles.append(_tile(data_dir, _typical(group, "p_fail")))
        if check in FOLD_CHECKS:
            folds.append(rows)
    fold = pd.concat(folds)
    missed = fold[(fold["label"] == FAIL) & (fold["predicted"] == PASS)]
    if len(missed):
        row = missed.sort_values("p_fail").iloc[0]
        tiles.append(_tile(data_dir, row, "most confident fold miss"))
    return example_grid(
        tiles,
        out,
        title="check-verdict: robot pass/fail verdict from one frame",
        subtitle=(
            "Robot verdict vs head answer with the head's P(fail). "
            "Holdout frames, typical (median P(fail)) per verdict, plus "
            "the most confident miss. Green = agree, red = disagree."
        ),
        crop=crop_plate,
        dpi=GRID_DPI,
    )


def embedder_example(data_dir: Path, out: Path) -> Path:
    """PCA scatter of all frame embeddings: colour = robot, marker =
    check. Shows how strongly the embedding separates robots."""
    frames, emb = _load(data_dir)
    keys = ["run_id", "check", "attempt"]
    cols = embedding_columns(emb)
    df = emb.merge(frames[[*keys, "robot"]], on=keys, how="inner")
    pca = PCA(n_components=2, random_state=PCA_SEED)
    xy = pca.fit_transform(df[cols].to_numpy(np.float64))
    var = pca.explained_variance_ratio_

    robots = sorted(df["robot"].dropna().unique())
    checks = [c for c in df["check"].value_counts().index]
    colour = {
        r: ROBOT_COLOURS[i] if i < len(ROBOT_COLOURS) else OTHER_COLOUR
        for i, r in enumerate(robots)
    }
    marker = {
        c: CHECK_MARKERS[i % len(CHECK_MARKERS)] for i, c in enumerate(checks)
    }
    fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for robot in robots:
        for check in checks:
            m = ((df["robot"] == robot) & (df["check"] == check)).to_numpy()
            if m.any():
                ax.scatter(
                    xy[m, 0],
                    xy[m, 1],
                    s=SCATTER_SIZE,
                    c=colour[robot],
                    marker=marker[check],
                    alpha=SCATTER_ALPHA,
                    edgecolors=SURFACE,
                    linewidths=RING,
                )
        # direct label at the robot's centroid, in ink, not series colour
        m = (df["robot"] == robot).to_numpy()
        cx, cy = xy[m].mean(axis=0)
        ax.annotate(
            f"{robot} ({m.sum()})",
            (cx, cy),
            fontsize=10,
            fontweight="bold",
            color=INK,
            ha="center",
            bbox={"boxstyle": "round,pad=0.3", "fc": SURFACE, "ec": "none"},
        )
    robot_handles = [
        plt.Line2D([], [], ls="", marker="o", color=colour[r], label=r)
        for r in robots
    ]
    check_handles = [
        plt.Line2D([], [], ls="", marker=marker[c], color=INK2, label=c)
        for c in checks
    ]
    first = ax.legend(
        handles=robot_handles,
        title="robot (colour)",
        loc="upper left",
        frameon=False,
        fontsize=9,
    )
    ax.add_artist(first)
    ax.legend(
        handles=check_handles,
        title="check (marker)",
        loc="upper right",
        frameon=False,
        fontsize=9,
    )
    ax.set_xlabel(f"PC1 ({var[0]:.0%} of variance)", color=INK2)
    ax.set_ylabel(f"PC2 ({var[1]:.0%} of variance)", color=INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.set_title(
        f"frame-embedder: {len(df)} frame embeddings, first two PCA axes",
        loc="left",
        fontsize=13,
        fontweight="bold",
        color=INK,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out, dpi=DPI, facecolor=SURFACE, pil_kwargs={"quality": JPEG_QUALITY}
    )
    plt.close(fig)
    return out
