"""Evaluation reports for the fold segmentation pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .fold_state import HALF_STATE, QUARTER_STATE, STATE_ORDER

PROTOTYPE_AGREEMENT = 0.946
PROTOTYPE_FRAMES = 991


def _rank_auc(scores: pd.Series, labels: pd.Series) -> float | None:
    clean = pd.DataFrame({"score": scores, "label": labels}).dropna()
    if clean.empty or clean["label"].nunique() < 2:
        return None
    ranks = clean["score"].rank(method="average")
    positives = clean["label"].astype(bool)
    n_pos = int(positives.sum())
    n_neg = int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    rank_sum = float(ranks[positives].sum())
    auc = (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(max(auc, 1.0 - auc))


def confusion_matrix(table: pd.DataFrame) -> pd.DataFrame:
    """State confusion matrix for passed checks."""
    passed = table[table["check_ok"]]
    return pd.crosstab(
        passed["expected_state"],
        passed["predicted_state"],
        dropna=False,
    ).reindex(index=STATE_ORDER, columns=STATE_ORDER, fill_value=0)


def _agreement(table: pd.DataFrame) -> float:
    if table.empty:
        return float("nan")
    return float((table["expected_state"] == table["predicted_state"]).mean())


def metric_tables(features: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build evaluation tables for parquet export."""
    passed = features[features["check_ok"]].copy()
    overall = pd.DataFrame(
        [
            {
                "scope": "overall",
                "frames": len(passed),
                "agreement": _agreement(passed),
                "prototype_agreement": PROTOTYPE_AGREEMENT,
                "prototype_frames": PROTOTYPE_FRAMES,
            }
        ]
    )
    per_robot = (
        passed.groupby("machine_name", dropna=False)
        .apply(lambda group: _agreement(group), include_groups=False)
        .reset_index(name="agreement")
    )
    per_robot["frames"] = (
        passed.groupby("machine_name", dropna=False).size().values
    )
    fold_checks = features[
        features["check_name"].isin(["fold1", "fold2"])
    ].copy()
    fold_checks["expected_fold_ok"] = (
        (fold_checks["check_name"] == "fold1")
        & (fold_checks["predicted_state"] == HALF_STATE)
    ) | (
        (fold_checks["check_name"] == "fold2")
        & (fold_checks["predicted_state"] == QUARTER_STATE)
    )
    spread = features[features["check_name"] == "spread"].copy()
    auc_rows: list[dict[str, Any]] = []
    for column in (
        "diameter_plate_share",
        "offset_plate_share",
        "hole_fraction",
        "circularity",
        "solidity",
    ):
        auc_rows.append(
            {
                "metric": column,
                "rank_auc": _rank_auc(spread[column], spread["check_ok"])
                if column in spread
                else None,
            }
        )
    coverage = pd.DataFrame(
        [
            {
                "frames": len(features),
                "with_center": int(features["center_x"].notna().sum()),
                "with_fold_edge_angle": int(
                    features["edge_angle_deg"].notna().sum()
                ),
                "with_quarter_corner": int(features["corner_x"].notna().sum()),
            }
        ]
    )
    return {
        "confusion": confusion_matrix(features).reset_index(
            names="expected_state"
        ),
        "overall": overall,
        "per_robot": per_robot,
        "fold_checks": fold_checks[
            ["run_id", "check_name", "check_ok", "expected_fold_ok"]
        ],
        "spread_auc": pd.DataFrame(auc_rows),
        "coverage": coverage,
    }


def _markdown_table(table: pd.DataFrame) -> str:
    if table.empty:
        return "No rows.\n"
    columns = [str(column) for column in table.columns]
    rows = [[str(value) for value in row] for row in table.to_numpy()]
    widths = [len(column) for column in columns]
    for row in rows:
        widths = [
            max(width, len(value))
            for width, value in zip(widths, row, strict=True)
        ]
    header = (
        "| "
        + " | ".join(
            column.ljust(width)
            for column, width in zip(columns, widths, strict=True)
        )
        + " |"
    )
    divider = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [
        "| "
        + " | ".join(
            value.ljust(width) for value, width in zip(row, widths, strict=True)
        )
        + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body, ""])


def write_report(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Write report.md and metric parquet tables."""
    output = data_dir / "fold_segmentation"
    features = pd.read_parquet(output / "features.parquet")
    tables = metric_tables(features)
    metrics_dir = output / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_parquet(metrics_dir / f"{name}.parquet", index=False)
    overall = tables["overall"].iloc[0].to_dict()
    agreement = overall["agreement"]
    agreement_text = "n/a" if not np.isfinite(agreement) else f"{agreement:.1%}"
    report = [
        "# Fold segmentation report",
        "",
        "This report is generated from cached private exports. It contains "
        "aggregate metrics only.",
        "",
        "## Overall",
        "",
        f"- Passed-check frames: {int(overall['frames'])}",
        f"- Agreement: {agreement_text}",
        f"- Prototype reference: {PROTOTYPE_AGREEMENT:.1%} on "
        f"{PROTOTYPE_FRAMES} frames",
        "",
        "## Confusion matrix",
        "",
        _markdown_table(tables["confusion"]),
        "## Per robot",
        "",
        _markdown_table(tables["per_robot"]),
        "## Spread rank AUC",
        "",
        _markdown_table(tables["spread_auc"]),
        "## Geometry coverage",
        "",
        _markdown_table(tables["coverage"]),
    ]
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    return tables
