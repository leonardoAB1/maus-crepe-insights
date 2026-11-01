"""Offline aggregate evaluation against robot checks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .fold_state import FLAT_PLATE_FRACTION

STATES = ("empty", "flat", "half", "quarter")
SPREAD_METRICS = (
    "diameter_fraction",
    "offset_fraction",
    "hole_fraction",
    "circularity",
    "solidity",
)


def rank_auc(values: pd.Series, verdicts: pd.Series) -> float | None:
    """Rank AUC of a measurement for positive check verdicts, including ties."""
    pairs = pd.DataFrame({"value": values, "verdict": verdicts}).dropna()
    if pairs.empty:
        return None
    positive = pairs["verdict"].astype(bool)
    n_positive = int(positive.sum())
    n_negative = len(pairs) - n_positive
    if not n_positive or not n_negative:
        return None
    ranks = pairs["value"].rank(method="average")
    return float(
        (ranks[positive].sum() - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative)
    )


def evaluate(data_dir: Path) -> dict[str, int]:
    """Write aggregate parquet tables and a Markdown report locally."""
    root = data_dir / "fold_segmentation"
    features = pd.read_parquet(root / "features.parquet")
    report = root / "report.md"
    tables = root / "evaluation"
    if report.exists() or tables.exists():
        raise FileExistsError("Evaluation exists; move it aside first")
    usable = features[features["status"] == "ok"].copy()
    passed = usable[usable["check_ok"] == True].copy()  # noqa: E712
    confusion = (
        passed.groupby(["robot", "expected_state", "predicted_state"])
        .size()
        .rename("frames")
        .reset_index()
    )
    fold = usable[usable["expected_state"].isin(["half", "quarter"])].copy()
    fold["our_match"] = fold["expected_state"] == fold["predicted_state"]
    verdicts = (
        fold.groupby(["robot", "expected_state", "check_ok", "our_match"])
        .size()
        .rename("frames")
        .reset_index()
    )
    spread = usable[usable["expected_state"] == "flat"]
    aucs = pd.DataFrame(
        [
            {
                "metric": name,
                "auc": rank_auc(spread[name], spread["check_ok"]),
                "frames": int(spread[[name, "check_ok"]].dropna().shape[0]),
            }
            for name in SPREAD_METRICS
        ]
    )
    correct = int((passed["expected_state"] == passed["predicted_state"]).sum())
    count = len(passed)
    lines = [
        "# Fold segmentation evaluation",
        "",
        "Private local report. The rectified-frame thresholds were fitted on",
        "this same export. This is not held-out validation.",
        "",
        f"- Frames: {len(features)}; usable: {len(usable)}.",
        f"- Passed checks: {count}; agreement: {correct / count:.1%}."
        if count
        else "- No passed checks could be evaluated.",
        "- Prototype: 94.6% on 991 passed checks. Differences may reflect",
        "  rectification, plate estimation, references and sample selection.",
        "  This comparison does not establish which difference caused the gap.",
        "",
        "## Per-state agreement",
        "",
    ]
    for state in STATES:
        subset = passed[passed["expected_state"] == state]
        matches = int(
            (subset["expected_state"] == subset["predicted_state"]).sum()
        )
        value = (
            f"{matches / len(subset):.1%}" if len(subset) else "not available"
        )
        lines.append(f"- {state}: {value} ({len(subset)} frames)")
    lines.extend(
        [
            "",
            "## Confusion matrix",
            "",
            "| Expected | " + " | ".join(STATES) + " |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for expected in STATES:
        subset = passed[passed["expected_state"] == expected]
        counts = [
            str(int((subset["predicted_state"] == state).sum()))
            for state in STATES
        ]
        lines.append(f"| {expected} | " + " | ".join(counts) + " |")
    if "rule_state" in passed.columns:
        flips = passed[passed["rule_state"] != passed["predicted_state"]]
        flip_correct = int(
            (flips["predicted_state"] == flips["expected_state"]).sum()
        )
        zone_lines = [
            "",
            "## Boundary-zone classifier flips",
            "",
            f"- The model changed the rule's decision on {len(flips)} passed",
            f"  frame(s); {flip_correct} of those changes were correct.",
        ]
        zone = passed[passed["expected_state"].isin(["half", "quarter"])]
        if len(zone):
            rule_correct = int(
                (zone["rule_state"] == zone["expected_state"]).sum()
            )
            final_correct = int(
                (zone["predicted_state"] == zone["expected_state"]).sum()
            )
            zone_lines.append(
                f"- On {len(zone)} half/quarter frames, rule agreement was"
                f" {rule_correct / len(zone):.1%} and final agreement"
                f" {final_correct / len(zone):.1%}."
            )
        lines.extend(zone_lines)
    half = passed[passed["expected_state"] == "half"]
    if len(half):
        as_flat = int((half["predicted_state"] == "flat").sum())
        median_area = half["plate_fraction"].median()
        lines.extend(
            [
                "",
                f"Passed half checks classified flat: {as_flat}/{len(half)}.",
                f"Their median plate fraction is {median_area:.3f}; the",
                f"flat threshold is {FLAT_PLATE_FRACTION}. Validate the",
                "thresholds on held-out data before deployment.",
            ]
        )
    lines.extend(
        [
            "",
            "## Fold verdict counts",
            "",
            "| Robot verdict | Our expected fold | Frames |",
            "| --- | --- | ---: |",
        ]
    )
    for (robot_ok, our_match), count_value in (
        fold.groupby(["check_ok", "our_match"]).size().items()
    ):
        lines.append(
            f"| {bool(robot_ok)} | {bool(our_match)} | {count_value} |"
        )
    lines.extend(["", "## Spread verdict rank AUC", ""])
    for row in aucs.itertuples():
        value = f"{row.auc:.3f}" if pd.notna(row.auc) else "not available"
        lines.append(f"- {row.metric}: {value} ({row.frames} frames)")
    lines.extend(["", "## Geometry coverage", ""])
    for label, column in (
        ("centre", "centre_x"),
        ("edge angle", "edge_angle"),
        ("quarter corner", "corner_x"),
        ("fold rect angle", "rect_angle"),
    ):
        subset = (
            usable[usable["expected_state"] == "quarter"]
            if label == "quarter corner"
            else usable
        )
        coverage = int(subset[column].notna().sum()) if column in subset else 0
        lines.append(f"- {label}: {coverage}/{len(subset)} frames")
    lines.extend(
        [
            "",
            "Confusion and fold-verdict counts by robot are in the",
            "adjacent evaluation tables. Failed segmentations are excluded",
            "from agreement but remain visible in the features table.",
            "",
        ]
    )
    tables.mkdir()
    confusion.to_parquet(tables / "confusion.parquet", index=False)
    verdicts.to_parquet(tables / "fold_verdicts.parquet", index=False)
    aucs.to_parquet(tables / "spread_auc.parquet", index=False)
    report.write_text("\n".join(lines))
    return {
        "frames": len(features),
        "usable": len(usable),
        "passed": count,
        "correct": correct,
    }
