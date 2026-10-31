"""Analyze a finished DSD batch (#26): which knob, which way, any curvature.

Maus fills the run_id column of the run sheet. The runs are joined to the run
table and the vision scores, and each outcome gets a least squares fit on the
coded factors and their squares. A DSD keeps main effects orthogonal to each
other and to the squares, so each main effect stands on its own.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist

from .analysis import OUTCOMES, Outcome
from .doe import CODED_SUFFIX, DOE_DIR, RUN_ID, Factor, sheet_factors

ALPHA = 0.10  # screening: keep terms up to this p-value
MIN_RESID_DF = 1  # residual degrees of freedom needed for p-values
PLAN_TOL = 1e-6  # planned vs actual setting, above this is a mismatch
INTERCEPT, MAIN, SQUARE = "intercept", "main", "square"
ONNX_OPSET = 17
ONNX_IR_VERSION = 8  # matches opset 17, readable by older onnxruntime
ONNX_INPUT, ONNX_OUTPUT = "x", "y"
SUCCEEDED = "succeeded"


# --- data ------------------------------------------------------------------


def load_results(
    sheet: pd.DataFrame, runs: pd.DataFrame, vision: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sheet rows with a run id, joined to their runs and vision scores.

    Also returns the settings that were run differently from the plan.
    """
    done = sheet.dropna(subset=[RUN_ID]).copy()
    done[RUN_ID] = done[RUN_ID].astype("int64")
    missing = set(done[RUN_ID]) - set(runs["id"])
    if missing:
        raise ValueError(
            f"{len(missing)} run ids on the sheet are not in the run "
            "table; fetch and rebuild the table first"
        )
    frame = done.merge(
        runs, left_on=RUN_ID, right_on="id", suffixes=("", "_run")
    )
    if not vision.empty:
        frame = frame.merge(vision, on="id", how="left")
    frame["success"] = frame["status"] == SUCCEEDED
    rows = []
    for f in sheet_factors(sheet):
        if f.column not in frame:
            continue
        actual = pd.to_numeric(frame[f.column], errors="coerce")
        off = (actual - frame[f.name]).abs() > PLAN_TOL
        for _, r in frame[off.fillna(True)].iterrows():
            rows.append(
                {
                    RUN_ID: r[RUN_ID],
                    "factor": f.name,
                    "planned": r[f.name],
                    "actual": r[f.column],
                }
            )
    mismatches = pd.DataFrame(
        rows, columns=[RUN_ID, "factor", "planned", "actual"]
    )
    return frame, mismatches


# --- model -----------------------------------------------------------------


def model_matrix(coded: np.ndarray, squares: bool = True) -> np.ndarray:
    """[1, x, x^2] on the coded scale."""
    x = np.asarray(coded, dtype=float)
    parts = [np.ones((len(x), 1)), x]
    if squares:
        parts.append(x**2)
    return np.hstack(parts)


def fit(
    frame: pd.DataFrame, factors: list[Factor], name: str, outcome: Outcome
) -> pd.DataFrame:
    """Least squares on coded mains and squares, one row per term.

    Falls back to mains only when too few runs are back for the squares.
    Empty when there is too little data for either.
    """
    cols = [f.name + CODED_SUFFIX for f in factors]
    raw = frame[outcome.column].astype(object)
    if outcome.binary:  # bool or None from the run table
        raw = raw.map({True: 1.0, False: 0.0})
    y = pd.to_numeric(raw, errors="coerce").astype(float)
    keep = y.notna().to_numpy()
    x = frame.loc[keep, cols].to_numpy(dtype=float)
    y = y[keep].to_numpy()
    if len(np.unique(y)) < 2:  # nothing to explain
        return pd.DataFrame()
    m = len(factors)
    squares = len(y) - (1 + 2 * m) >= MIN_RESID_DF
    if not squares and len(y) - (1 + m) < MIN_RESID_DF:
        return pd.DataFrame()
    xm = model_matrix(x, squares)
    coef, *_ = np.linalg.lstsq(xm, y, rcond=None)
    resid = y - xm @ coef
    df = len(y) - np.linalg.matrix_rank(xm)
    sigma2 = float(resid @ resid) / df if df > 0 else np.nan
    se = np.sqrt(np.diag(np.linalg.pinv(xm.T @ xm)) * sigma2)
    with np.errstate(divide="ignore", invalid="ignore"):
        tval = coef / se
    p = 2 * t_dist.sf(np.abs(tval), df) if df > 0 else np.full_like(se, 1)
    total = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(resid @ resid) / total if total > 0 else np.nan
    kinds = [INTERCEPT] + [MAIN] * m + ([SQUARE] * m if squares else [])
    names = [""] + [f.name for f in factors] * (2 if squares else 1)
    return pd.DataFrame(
        {
            "outcome": name,
            "term": [
                k if k == INTERCEPT else f"{k}:{n}"
                for k, n in zip(kinds, names, strict=True)
            ],
            "kind": kinds,
            "factor": names,
            "coef": coef,
            "se": se,
            "t": tval,
            "p": p,
            "better": outcome.better,
            "n_runs": len(y),
            "df": df,
            "r2": r2,
        }
    )


def best_level(main: float, square: float, better: int) -> float:
    """Coded level in [-1, 1] that is best for one factor on its own."""
    candidates = [-1.0, 1.0]
    if square:
        vertex = -main / (2 * square)
        if -1 < vertex < 1:
            candidates.append(vertex)
    return max(candidates, key=lambda x: better * (main * x + square * x * x))


def coefficients(
    effects: pd.DataFrame, factors: list[Factor], outcomes: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Weights (2m x k) on [x, x^2] and intercepts (k) for these outcomes.
    Terms that were not fitted count as zero."""
    m = len(factors)
    w = np.zeros((2 * m, len(outcomes)))
    b = np.zeros(len(outcomes))
    for k, name in enumerate(outcomes):
        sub = effects[effects["outcome"] == name]
        for _, r in sub.iterrows():
            if r["kind"] == INTERCEPT:
                b[k] = r["coef"]
                continue
            j = [f.name for f in factors].index(r["factor"])
            w[j + (m if r["kind"] == SQUARE else 0), k] = r["coef"]
    return w, b


def predict(
    effects: pd.DataFrame,
    factors: list[Factor],
    outcomes: list[str],
    coded: np.ndarray,
) -> np.ndarray:
    """Predicted outcomes for coded settings (rows x outcomes)."""
    w, b = coefficients(effects, factors, outcomes)
    x = np.asarray(coded, dtype=float)
    return np.hstack([x, x**2]) @ w + b


def to_onnx(
    effects: pd.DataFrame, factors: list[Factor], outcomes: list[str]
) -> Any:
    """The fitted quadratic as an ONNX graph: y = [x, x^2] @ W + b."""
    from onnx import TensorProto, helper, numpy_helper

    w, b = coefficients(effects, factors, outcomes)
    m, k = len(factors), len(outcomes)
    nodes = [
        helper.make_node("Mul", [ONNX_INPUT, ONNX_INPUT], ["sq"]),
        helper.make_node("Concat", [ONNX_INPUT, "sq"], ["z"], axis=1),
        helper.make_node("MatMul", ["z", "W"], ["zw"]),
        helper.make_node("Add", ["zw", "b"], [ONNX_OUTPUT]),
    ]
    graph = helper.make_graph(
        nodes,
        "doe_response",
        [
            helper.make_tensor_value_info(
                ONNX_INPUT, TensorProto.FLOAT, [None, m]
            )
        ],
        [
            helper.make_tensor_value_info(
                ONNX_OUTPUT, TensorProto.FLOAT, [None, k]
            )
        ],
        initializer=[
            numpy_helper.from_array(w.astype(np.float32), "W"),
            numpy_helper.from_array(b.astype(np.float32), "b"),
        ],
    )
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", ONNX_OPSET)],
        ir_version=ONNX_IR_VERSION,
    )


# --- run -------------------------------------------------------------------


def analyze_sheet(
    sheet: pd.DataFrame, runs: pd.DataFrame, vision: pd.DataFrame
) -> dict[str, Any]:
    factors = sheet_factors(sheet)
    frame, mismatches = load_results(sheet, runs, vision)
    fits = []
    for name, out in OUTCOMES.items():
        if out.column in frame:
            fits.append(fit(frame, factors, name, out))
    effects = pd.concat([f for f in fits if not f.empty], ignore_index=True)
    return {
        "doe_id": str(sheet["doe_id"].iloc[0]),
        "factors": factors,
        "run_ids": [int(i) for i in frame[RUN_ID]],
        "frame": frame,
        "effects": effects,
        "mismatches": mismatches,
    }


def analyze(data_dir: Path, sheet_path: Path) -> dict[str, Any]:
    """Read a filled run sheet, write data/doe/<doe_id>-effects.parquet."""
    sheet = pd.read_parquet(sheet_path)
    runs = pd.read_parquet(data_dir / "runs.parquet")
    vision_path = data_dir / "vision.parquet"
    vision = (
        pd.read_parquet(vision_path)
        if vision_path.is_file()
        else pd.DataFrame()
    )
    result = analyze_sheet(sheet, runs, vision)
    out = data_dir / DOE_DIR
    out.mkdir(parents=True, exist_ok=True)
    result["effects"].to_parquet(
        out / f"{result['doe_id']}-effects.parquet", index=False
    )
    from .doe_figures import effects_figure

    for name in result["effects"]["outcome"].unique():
        effects_figure(result, name, out / f"{result['doe_id']}-{name}.png")
    print(summary(result))
    return result


def summary(result: dict[str, Any]) -> str:
    effects = result["effects"]
    lines = [f"{result['doe_id']}: {len(result['run_ids'])} runs back"]
    if not result["mismatches"].empty:
        lines.append(
            f"  warning: {len(result['mismatches'])} settings differ from "
            "the plan, check the sheet"
        )
    for name, sub in effects.groupby("outcome", sort=False):
        better = int(sub["better"].iloc[0])
        found = []
        for _, r in sub[
            (sub["p"] < ALPHA) & (sub["kind"] != INTERCEPT)
        ].iterrows():
            if r["kind"] == MAIN:
                way = "raise" if better * r["coef"] > 0 else "lower"
                found.append(f"{way} {r['factor']} (p={r['p']:.2f})")
            else:
                found.append(f"curved {r['factor']} (p={r['p']:.2f})")
        r2 = sub["r2"].iloc[0]
        lines.append(
            f"  {name} (n={sub['n_runs'].iloc[0]}, r2={r2:.2f}): "
            + ("; ".join(found) or "no clear effect")
        )
    return "\n".join(lines)


def model_card(result: dict[str, Any], outcomes: list[str]) -> dict[str, Any]:
    """model.json content for code/models/doe-response/."""
    effects = result["effects"]
    return {
        "name": "doe-response",
        "inputs": [
            {
                "name": ONNX_INPUT,
                "dtype": "float32",
                "shape": [None, len(result["factors"])],
                "features": [
                    {
                        "name": f.name,
                        "low": f.low,
                        "high": f.high,
                        "coding": "2 * (value - low) / (high - low) - 1",
                    }
                    for f in result["factors"]
                ],
            }
        ],
        "outputs": [
            {
                "name": ONNX_OUTPUT,
                "dtype": "float32",
                "shape": [None, len(outcomes)],
                "features": outcomes,
            }
        ],
        "training": {
            "doe_id": result["doe_id"],
            "run_ids": result["run_ids"],
        },
        "metrics": {
            name: {
                "r2": float(sub["r2"].iloc[0]),
                "n_runs": int(sub["n_runs"].iloc[0]),
            }
            for name, sub in effects.groupby("outcome", sort=False)
            if name in outcomes
        },
    }


def dump_json(data: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
