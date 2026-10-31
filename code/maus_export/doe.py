"""Definitive Screening Design (DSD) batch for Maus to run (#26).

A DSD (Jones and Nachtsheim 2011) tests every factor at three levels in about
two runs per factor. Main effects are orthogonal to each other and to all
squared terms, so each knob gets a clean direction and a curvature check. The
design is built from a conference matrix C as rows [C; -C; 0].

Levels come from the history analysis (#25): the range where runs succeeded,
checked against everything the robots already ran. We never send anything to
a robot; the run sheet is handed to Maus.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .to_table import PARAM_PREFIX

# Shortlist from #25: fan powers and radius offsets moved together in the
# history, the design varies them independently
DEFAULT_FACTORS = (
    "dispense.batter.fan_power",
    "dispense.batter.concentric_inner_fan_power",
    "machine.y_center",
    "machine.z_scrape_start",
    "dispense.batter.concentric_inner_radius_offset_mm",
    "dispense.batter.concentric_outer_radius_offset_mm",
)
# Two unused (fake) columns make the squared terms better estimable and give
# a 17-run design for 6 factors
DEFAULT_FAKE = 2
DEFAULT_CENTRE = 3  # centre runs in total, to estimate pure noise
DEFAULT_TOPPING = "plain"
DEFAULT_DECIMALS = 3
SEED = 0
CODED_SUFFIX = "__coded"
RUN_ID = "run_id"
DESIGN, CENTRE = "design", "centre"
SHORTLIST_FILE = Path("analysis") / "shortlist.json"
DOE_DIR = "doe"
LEVELS = (-1, 0, 1)
ORTHO_TOL = 1e-9
MAX_PALEY_Q = 47  # largest prime tried for a conference matrix


@dataclass(frozen=True)
class Factor:
    name: str  # recipe key without the param. prefix
    low: float
    high: float
    decimals: int = DEFAULT_DECIMALS

    @property
    def column(self) -> str:
        """Column of this setting in runs.parquet."""
        return PARAM_PREFIX + self.name

    def decode(self, coded: np.ndarray) -> np.ndarray:
        """Coded -1/0/+1 to real setting values."""
        mid = (self.low + self.high) / 2
        half = (self.high - self.low) / 2
        return np.round(mid + np.asarray(coded) * half, self.decimals)

    def encode(self, values: np.ndarray) -> np.ndarray:
        """Real setting values to the coded scale."""
        mid = (self.low + self.high) / 2
        half = (self.high - self.low) / 2
        return (np.asarray(values, dtype=float) - mid) / half


# --- design ----------------------------------------------------------------


def _is_prime(n: int) -> bool:
    return n >= 2 and all(n % d for d in range(2, int(n**0.5) + 1))


def conference_matrix(n: int) -> np.ndarray:
    """Paley conference matrix of order n = q + 1, q an odd prime.

    Zero diagonal, +-1 elsewhere, C.T @ C = (n - 1) I. Symmetric when
    q = 1 mod 4, antisymmetric when q = 3 mod 4; both work for a DSD.
    """
    q = n - 1
    if q < 3 or not _is_prime(q):
        raise ValueError(f"no Paley conference matrix of order {n}")
    residues = {(i * i) % q for i in range(1, q)}
    chi = np.array([0] + [1 if k in residues else -1 for k in range(1, q)])
    idx = np.arange(q)
    jac = chi[(idx[None, :] - idx[:, None]) % q]
    sign = 1 if q % 4 == 1 else -1
    c = np.zeros((n, n), dtype=int)
    c[0, 1:] = 1
    c[1:, 0] = sign
    c[1:, 1:] = jac
    return c


def conference_order(columns: int) -> int:
    """Smallest Paley order that holds this many columns."""
    for q in range(3, MAX_PALEY_Q + 1):
        if _is_prime(q) and q + 1 >= columns:
            return q + 1
    raise ValueError(f"too many factors for a DSD: {columns}")


def dsd(
    n_factors: int,
    n_fake: int = DEFAULT_FAKE,
    n_centre: int = DEFAULT_CENTRE,
) -> np.ndarray:
    """Coded DSD: [C; -C; 0] plus extra centre runs, fake columns dropped."""
    if n_centre < 1:
        raise ValueError("a DSD needs at least one centre run")
    c = conference_order(n_factors + n_fake)
    conf = conference_matrix(c)[:, :n_factors]
    centre = np.zeros((n_centre, n_factors), dtype=int)
    return np.vstack([conf, -conf, centre])


def design_report(design: np.ndarray) -> dict[str, Any]:
    """Balance and correlations of a coded design."""
    x = np.asarray(design, dtype=float)
    sq = x**2
    m = x.shape[1]

    def max_corr(a: np.ndarray, b: np.ndarray, same: bool) -> float:
        a = a - a.mean(axis=0)
        b = b - b.mean(axis=0)
        na = np.linalg.norm(a, axis=0)
        nb = np.linalg.norm(b, axis=0)
        r = (a.T @ b) / np.outer(na, nb)
        if same:
            r = r[~np.eye(m, dtype=bool)]
        return float(np.abs(r).max()) if r.size else 0.0

    return {
        "runs": int(x.shape[0]),
        "factors": m,
        "balance": [
            {str(v): int((x[:, j] == v).sum()) for v in LEVELS}
            for j in range(m)
        ],
        "max_corr_main": max_corr(x, x, True),
        "max_corr_main_square": max_corr(x, sq, False),
        "max_corr_square": max_corr(sq, sq, True),
    }


# --- levels ----------------------------------------------------------------


def factors_from_history(
    runs: pd.DataFrame,
    shortlist: dict[str, Any] | None,
    names: tuple[str, ...] = DEFAULT_FACTORS,
) -> list[Factor]:
    """Levels per factor: the success range from the #25 shortlist, else the
    full range already run. Every level must lie inside what was run."""
    safe: dict[str, list[float]] = {}
    for item in (shortlist or {}).get("shortlist", []):
        safe.update(item.get("safe_range", {}))
    factors = []
    for name in names:
        col = PARAM_PREFIX + name
        if col not in runs:
            raise ValueError(f"{name} never set in the run history")
        seen = pd.to_numeric(runs[col], errors="coerce").dropna()
        lo, hi = safe.get(name, (seen.min(), seen.max()))
        if lo >= hi:
            raise ValueError(f"{name} only ever ran at one value")
        if lo < seen.min() or hi > seen.max():
            raise ValueError(f"{name} range [{lo}, {hi}] was never run")
        factors.append(Factor(name, float(lo), float(hi)))
    return factors


# --- run sheet -------------------------------------------------------------


def run_sheet(
    design: np.ndarray,
    factors: list[Factor],
    robot: str,
    topping: str = DEFAULT_TOPPING,
    seed: int = SEED,
) -> pd.DataFrame:
    """One row per run in randomized order, with an empty run_id column for
    Maus to fill with the dashboard run id."""
    if design.shape[1] != len(factors):
        raise ValueError("design columns and factors differ")
    n = design.shape[0]
    doe_id = f"dsd{n}-{robot}-s{seed}"
    centre = (design == 0).all(axis=1)
    sheet = pd.DataFrame(
        {
            "doe_id": doe_id,
            "std_order": np.arange(1, n + 1),
            "point_type": np.where(centre, CENTRE, DESIGN),
            "robot": robot,
            "topping": topping,
        }
    )
    for j, f in enumerate(factors):
        sheet[f.name + CODED_SUFFIX] = design[:, j]
        sheet[f.name] = f.decode(design[:, j])
    order = np.random.default_rng(seed).permutation(n)
    sheet = sheet.iloc[order].reset_index(drop=True)
    sheet.insert(1, "run_order", np.arange(1, n + 1))
    sheet[RUN_ID] = pd.array([pd.NA] * n, dtype="Int64")
    return sheet


def sheet_factors(sheet: pd.DataFrame) -> list[Factor]:
    """Recover factors and levels from a run sheet."""
    factors = []
    for col in sheet.columns:
        if not col.endswith(CODED_SUFFIX):
            continue
        name = col.removesuffix(CODED_SUFFIX)
        lo = sheet.loc[sheet[col] == -1, name]
        hi = sheet.loc[sheet[col] == 1, name]
        factors.append(Factor(name, float(lo.iloc[0]), float(hi.iloc[0])))
    return factors


def build_run_sheet(
    data_dir: Path,
    robot: str,
    topping: str = DEFAULT_TOPPING,
    seed: int = SEED,
    n_fake: int = DEFAULT_FAKE,
    n_centre: int = DEFAULT_CENTRE,
) -> pd.DataFrame:
    """Write data/doe/<doe_id>.parquet and .csv from the run history."""
    runs = pd.read_parquet(data_dir / "runs.parquet")
    path = data_dir / SHORTLIST_FILE
    shortlist = (
        json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    )
    factors = factors_from_history(runs, shortlist)
    design = dsd(len(factors), n_fake, n_centre)
    sheet = run_sheet(design, factors, robot, topping, seed)
    out = data_dir / DOE_DIR
    out.mkdir(parents=True, exist_ok=True)
    doe_id = sheet["doe_id"].iloc[0]
    sheet.to_parquet(out / f"{doe_id}.parquet", index=False)
    sheet.to_csv(out / f"{doe_id}.csv", index=False)
    print(summary(design, factors, out / f"{doe_id}.csv"))
    return sheet


def summary(design: np.ndarray, factors: list[Factor], path: Path) -> str:
    rep = design_report(design)
    lines = [f"{rep['runs']} runs, {rep['factors']} factors -> {path}"]
    for f in factors:
        mid = f.decode(np.array(0))
        lines.append(f"  {f.name}: {f.low:g} / {mid:g} / {f.high:g}")
    lines.append(
        f"max |corr| main-main {rep['max_corr_main']:.3f}, "
        f"main-square {rep['max_corr_main_square']:.3f}, "
        f"square-square {rep['max_corr_square']:.3f}"
    )
    return "\n".join(lines)
