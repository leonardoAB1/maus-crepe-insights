"""Which recipe settings matter, from the run history (observational).

Three views on the same runs:
- confounding: which settings always moved together and cannot be separated
- one-change experiments: consecutive settings blocks on one robot that
  differ in a single setting (or one inseparable group), before vs after
- a regularised model over all runs, with a bootstrap over whole segments
  (runs inside a segment are repeats of one config, not independent)

The shortlist combines them into settings with a direction and a safe range,
the input for the designed experiment (#26).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

from .to_table import CHECK_PREFIX, PARAM_PREFIX
from .vision_metrics import PREFIX as VISION_PREFIX

HOT_PLATE_MIN_C = 150  # below this the run was a cold test cycle
FACTOR_PREFIXES = ("dispense.batter.", "machine.", "push_off.")
# Recipe switches and their variant settings: a separate recipe, few runs
EXCLUDE_MARK = "gluten_free"
# A setting needs this many runs away from its most common value to say
# anything about it
MIN_MINORITY_RUNS = 10
CONFOUND_R = 0.8
MIN_OVERLAP = 20  # runs where both settings are set, to trust a correlation
WINDOW = 10  # runs compared on each side of a one-setting change
MIN_WINDOW = 3
N_BOOT = 500
CI = 0.90
RIDGE_ALPHA = 1.0
LOGIT_C = 1.0
SHORTLIST_N = 6
TRADE_OFF = "trade-off"
SIGN_SHARE_MIN = 0.8  # bootstrap agreement needed to call a direction
ONE_CHANGE_P = 0.1
RECENT_RUNS = 30  # "current" value = median over the latest full cycles
SEED = 0
COVARIATES_NUM = ("hotplate_temp", "env_temp_start")
COVARIATES_CAT = ("robot", "topping")


@dataclass(frozen=True)
class Outcome:
    column: str
    binary: bool
    better: int  # +1 if higher is better, -1 if lower is better


OUTCOMES: dict[str, Outcome] = {
    "success": Outcome("success", True, 1),
    "spread_ok": Outcome(f"{CHECK_PREFIX}spread.ok", True, 1),
    "fold1_ok": Outcome(f"{CHECK_PREFIX}fold1.ok", True, 1),
    "fold2_ok": Outcome(f"{CHECK_PREFIX}fold2.ok", True, 1),
    "spread_diameter": Outcome(f"{VISION_PREFIX}spread.diameter_rel", False, 1),
    "spread_uncovered": Outcome(
        f"{VISION_PREFIX}spread.uncovered_frac", False, -1
    ),
    "spread_offset": Outcome(f"{VISION_PREFIX}spread.offset_rel", False, -1),
    "fold1_iou": Outcome(f"{VISION_PREFIX}fold1.iou", False, 1),
    # A good quarter fold halves the area; failures sit near 1
    "fold2_area_ratio": Outcome(f"{VISION_PREFIX}fold2.area_ratio", False, -1),
}


DATA_ENV = "MAUS_DATA_DIR"
GITDIR_PREFIX = "gitdir:"


def find_data_dir(start: Path | None = None) -> Path:
    """The export folder holding runs.parquet: $MAUS_DATA_DIR, else a
    `data/` in this folder or a parent, else the main checkout's `data/`
    when running inside a git worktree (worktrees have no data of their
    own)."""
    if os.environ.get(DATA_ENV):
        return Path(os.environ[DATA_ENV])
    here = (start or Path.cwd()).resolve()
    for folder in (here, *here.parents):
        if (folder / "data" / "runs.parquet").is_file():
            return folder / "data"
        git = folder / ".git"
        if git.is_file():  # a worktree: .git points into the main repo
            text = git.read_text(encoding="utf-8").strip()
            if text.startswith(GITDIR_PREFIX):
                gitdir = Path(text.removeprefix(GITDIR_PREFIX).strip())
                main = gitdir.resolve().parents[2]  # <main>/.git/worktrees/x
                if (main / "data" / "runs.parquet").is_file():
                    return main / "data"
    raise FileNotFoundError(
        f"no data/runs.parquet found from {here}; run `pixi run table` "
        f"or set {DATA_ENV}"
    )


def short(factor: str) -> str:
    """Readable factor name without the column prefix."""
    return factor.removeprefix(PARAM_PREFIX)


# --- data ------------------------------------------------------------------


def analysis_frame(runs: pd.DataFrame, vision: pd.DataFrame) -> pd.DataFrame:
    """Hot full cycles with outcomes, joined to the vision scores."""
    hot = runs["hotplate_temp"].fillna(0) >= HOT_PLATE_MIN_C
    frame = runs[runs["is_full_cycle"].fillna(False).astype(bool) & hot]
    if not vision.empty:
        frame = frame.merge(vision, on="id", how="left")
    frame = frame.sort_values("id").reset_index(drop=True)
    frame["success"] = frame["status"] == "succeeded"
    for out in OUTCOMES.values():
        if out.column not in frame:
            frame[out.column] = np.nan
    return frame


def factor_columns(frame: pd.DataFrame) -> list[str]:
    """Numeric recipe settings that vary enough to learn from."""
    cols = []
    for c in frame.columns:
        if not c.startswith(PARAM_PREFIX):
            continue
        if not short(c).startswith(FACTOR_PREFIXES) or EXCLUDE_MARK in c:
            continue
        values = pd.to_numeric(frame[c], errors="coerce")
        if values.notna().sum() == 0 or values.nunique() < 2:
            continue
        top = values.value_counts(dropna=False).iloc[0]
        if len(values) - top >= MIN_MINORITY_RUNS:
            cols.append(c)
    return cols


# --- confounding -----------------------------------------------------------


def confounding(
    frame: pd.DataFrame, factors: list[str]
) -> tuple[pd.DataFrame, list[list[str]]]:
    """Correlation between settings, and groups that moved together.

    Groups are listed most-filled member first; that member stands for the
    group in the model.
    """
    x = frame[factors].apply(pd.to_numeric, errors="coerce")
    corr = x.corr(min_periods=MIN_OVERLAP)
    parent = {f: f for f in factors}

    def root(f: str) -> str:
        while parent[f] != f:
            f = parent[f]
        return f

    for i, a in enumerate(factors):
        for b in factors[i + 1 :]:
            r = corr.at[a, b]
            if pd.notna(r) and abs(r) >= CONFOUND_R:
                parent[root(b)] = root(a)
    groups: dict[str, list[str]] = {}
    for f in factors:
        groups.setdefault(root(f), []).append(f)
    filled = x.notna().sum()
    ordered = [
        sorted(g, key=lambda f: (-filled[f], f)) for g in groups.values()
    ]
    return corr, sorted(ordered, key=lambda g: g[0])


def group_name(group: list[str]) -> str:
    return " + ".join(short(f) for f in group)


# --- one-setting changes ---------------------------------------------------


def _config_blocks(runs: pd.DataFrame, factors: list[str]) -> list[pd.Index]:
    """Consecutive runs (one robot, by id) with identical factor values."""
    vals = runs[factors].apply(pd.to_numeric, errors="coerce")
    key = pd.Series(
        [tuple(r) for r in vals.fillna(np.inf).to_numpy()], index=runs.index
    )
    change = key.ne(key.shift())
    return [idx for _, idx in runs.groupby(change.cumsum()).groups.items()]


def _compare(
    before: pd.DataFrame, after: pd.DataFrame
) -> dict[str, float | None]:
    row: dict[str, float | None] = {}
    for name, out in OUTCOMES.items():
        a = before[out.column].dropna()
        b = after[out.column].dropna()
        if len(a) < MIN_WINDOW or len(b) < MIN_WINDOW:
            continue
        if out.binary:
            a, b = a.astype(bool), b.astype(bool)
            table = [[a.sum(), (~a).sum()], [b.sum(), (~b).sum()]]
            row[f"{name}.before"] = float(a.mean())
            row[f"{name}.after"] = float(b.mean())
            row[f"{name}.p"] = float(fisher_exact(table)[1])
        else:
            row[f"{name}.before"] = float(a.median())
            row[f"{name}.after"] = float(b.median())
            row[f"{name}.p"] = float(mannwhitneyu(a, b).pvalue)
    return row


def _same(x: Any, y: Any) -> bool:
    x, y = pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")
    return (pd.isna(x) and pd.isna(y)) or x == y


def one_change_effects(
    frame: pd.DataFrame, groups: list[list[str]]
) -> pd.DataFrame:
    """Before/after comparisons where exactly one group of settings changed
    between two consecutive config blocks on the same robot."""
    factors = [f for g in groups for f in g]
    of_group = {f: i for i, g in enumerate(groups) for f in g}
    rows = []
    for robot, runs in frame.groupby("robot"):
        runs = runs.sort_values("id")
        blocks = _config_blocks(runs, factors)
        for prev, nxt in zip(blocks, blocks[1:], strict=False):
            a, b = runs.loc[prev], runs.loc[nxt]
            changed = [
                f for f in factors if not _same(a[f].iloc[-1], b[f].iloc[0])
            ]
            touched = {of_group[f] for f in changed}
            if len(touched) != 1:
                continue
            g = groups[touched.pop()]
            rep = g[0]
            old = pd.to_numeric(a[rep].iloc[-1], errors="coerce")
            new = pd.to_numeric(b[rep].iloc[0], errors="coerce")
            before, after = a.tail(WINDOW), b.head(WINDOW)
            if len(before) < MIN_WINDOW or len(after) < MIN_WINDOW:
                continue
            row: dict[str, Any] = {
                "robot": robot,
                "group": group_name(g),
                "factor": short(rep),
                "from": old,
                "to": new,
                "first_id_after": int(b["id"].iloc[0]),
                "n_before": len(before),
                "n_after": len(after),
            }
            row.update(_compare(before, after))
            rows.append(row)
    return pd.DataFrame(rows)


# --- model over all runs ---------------------------------------------------


def design_matrix(
    frame: pd.DataFrame, reps: list[str]
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Standardised factors (unset -> median plus an indicator) and
    covariates. Returns the matrix and each factor's standard deviation,
    to turn per-SD effects back into units."""
    cols: dict[str, pd.Series] = {}
    scale: dict[str, float] = {}
    for f in reps:
        v = pd.to_numeric(frame[f], errors="coerce")
        sd = float(v.std()) or 1.0
        scale[f] = sd
        cols[f] = (v.fillna(v.median()) - v.median()) / sd
        if v.isna().any() and v.notna().any():
            cols[f"{f}.unset"] = v.isna().astype(float)
    for c in COVARIATES_NUM:
        v = pd.to_numeric(frame[c], errors="coerce")
        cols[c] = (v.fillna(v.median()) - v.median()) / (float(v.std()) or 1)
    x = pd.DataFrame(cols, index=frame.index)
    cats = pd.get_dummies(
        frame[list(COVARIATES_CAT)].astype(str), drop_first=True, dtype=float
    )
    x = pd.concat([x, cats], axis=1)
    return x.loc[:, x.nunique() > 1], scale


def _fit(x: np.ndarray, y: np.ndarray, binary: bool) -> np.ndarray | None:
    from sklearn.linear_model import LogisticRegression, Ridge

    if binary:
        if len(np.unique(y)) < 2:
            return None
        model = LogisticRegression(C=LOGIT_C, max_iter=2000)
        return model.fit(x, y).coef_[0]
    return Ridge(alpha=RIDGE_ALPHA).fit(x, y).coef_


def model_effects(
    frame: pd.DataFrame,
    groups: list[list[str]],
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> pd.DataFrame:
    """Per outcome and settings group: effect of +1 SD, with a bootstrap
    interval that resamples whole segments."""
    reps = [g[0] for g in groups]
    names = {g[0]: group_name(g) for g in groups}
    rng = np.random.default_rng(seed)
    rows = []
    for name, out in OUTCOMES.items():
        sub = frame[frame[out.column].notna()]
        if len(sub) < MIN_MINORITY_RUNS:
            continue
        x, scale = design_matrix(sub, reps)
        y = sub[out.column].astype(float if not out.binary else bool)
        y = y.to_numpy()
        point = _fit(x.to_numpy(), y, out.binary)
        if point is None:
            continue
        segs = sub["segment_id"].to_numpy()
        uniq = np.unique(segs)
        where = {s: np.flatnonzero(segs == s) for s in uniq}
        draws = []
        for _ in range(n_boot):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([where[s] for s in pick])
            coef = _fit(x.to_numpy()[idx], y[idx], out.binary)
            if coef is not None:
                draws.append(coef)
        boot = np.array(draws)
        lo_q, hi_q = (1 - CI) / 2, 1 - (1 - CI) / 2
        for j, col in enumerate(x.columns):
            if col not in names:
                continue
            b = boot[:, j] if len(boot) else np.array([np.nan])
            rows.append(
                {
                    "outcome": name,
                    "group": names[col],
                    "factor": short(col),
                    "coef_per_sd": float(point[j]),
                    "lo": float(np.quantile(b, lo_q)),
                    "hi": float(np.quantile(b, hi_q)),
                    "sign_share": float(
                        np.mean(np.sign(b) == np.sign(point[j]))
                    ),
                    "sd": scale[col],
                    "better": out.better,
                    "n_runs": len(sub),
                    "n_segments": len(uniq),
                    "n_boot": len(boot),
                }
            )
    return pd.DataFrame(rows)


# --- shortlist -------------------------------------------------------------


def _support(one_change: pd.DataFrame, group: str) -> list[dict[str, Any]]:
    """One-setting changes of this group with a clear outcome change."""
    if one_change.empty:
        return []
    hits = []
    for _, r in one_change[one_change["group"] == group].iterrows():
        if pd.isna(r["from"]) or pd.isna(r["to"]) or r["from"] == r["to"]:
            continue
        step = np.sign(r["to"] - r["from"])
        for name, out in OUTCOMES.items():
            p = r.get(f"{name}.p")
            if p is None or pd.isna(p) or p >= ONE_CHANGE_P:
                continue
            delta = r[f"{name}.after"] - r[f"{name}.before"]
            hits.append(
                {
                    "outcome": name,
                    "direction": "raise"
                    if step * np.sign(delta) * out.better > 0
                    else "lower",
                    "p": float(p),
                }
            )
    return hits


def shortlist(
    frame: pd.DataFrame,
    groups: list[list[str]],
    effects: pd.DataFrame,
    one_change: pd.DataFrame,
    n: int = SHORTLIST_N,
) -> list[dict[str, Any]]:
    """Settings groups with a consistent direction, ranked by evidence."""
    if effects.empty:
        return []
    clear = effects[
        (effects["sign_share"] >= SIGN_SHARE_MIN)
        & ((effects["lo"] > 0) | (effects["hi"] < 0))
    ].copy()
    clear["good"] = np.sign(clear["coef_per_sd"]) * clear["better"]
    clear["strength"] = clear["coef_per_sd"].abs() / (
        clear["hi"] - clear["lo"]
    ).clip(lower=1e-9)
    by_group = {group_name(g): g for g in groups}
    ok = frame["success"]
    recent = frame.sort_values("id").tail(RECENT_RUNS)
    items = []
    for group, rows in clear.groupby("group"):
        up = rows[rows["good"] > 0]  # outcomes that improve when raised
        down = rows[rows["good"] < 0]
        if len(up) == len(down):
            direction, agree, against = TRADE_OFF, up, down
        elif len(up) > len(down):
            direction, agree, against = "raise", up, down
        else:
            direction, agree, against = "lower", down, up
        support = _support(one_change, group)
        members = by_group[group]
        safe = {
            short(f): [
                float(pd.to_numeric(frame.loc[ok, f]).min()),
                float(pd.to_numeric(frame.loc[ok, f]).max()),
            ]
            for f in members
            if pd.to_numeric(frame.loc[ok, f]).notna().any()
        }
        items.append(
            {
                "group": group,
                "factors": [short(f) for f in members],
                "direction": direction,
                # with a trade-off: helps = raising helps, hurts = lowering
                "helps": sorted(agree["outcome"]),
                "hurts": sorted(against["outcome"]),
                "strength": float(agree["strength"].max())
                if len(agree)
                else 0.0,
                "one_change_support": [
                    s for s in support if s["direction"] == direction
                ],
                "one_change_against": [
                    s for s in support if s["direction"] != direction
                ],
                "safe_range": safe,
                # Successes only at one value leave nothing to explore
                "explore": any(hi > lo for lo, hi in safe.values()),
                "current": {
                    short(f): float(pd.to_numeric(recent[f]).median())
                    for f in members
                    if pd.to_numeric(recent[f]).notna().any()
                },
            }
        )
    items.sort(
        key=lambda i: (
            i["direction"] == TRADE_OFF,
            not i["explore"],
            not i["one_change_support"],
            -len(i["helps"]) + len(i["hurts"]),
            -i["strength"],
        )
    )
    return items[:n]


# --- run everything --------------------------------------------------------


def run_analysis(data_dir: Path, n_boot: int = N_BOOT) -> dict[str, Any]:
    """Read the run and vision tables, write results to data/analysis/."""
    runs = pd.read_parquet(data_dir / "runs.parquet")
    vision_path = data_dir / "vision.parquet"
    vision = (
        pd.read_parquet(vision_path)
        if vision_path.is_file()
        else pd.DataFrame()
    )
    frame = analysis_frame(runs, vision)
    factors = factor_columns(frame)
    corr, groups = confounding(frame, factors)
    one = one_change_effects(frame, groups)
    eff = model_effects(frame, groups, n_boot=n_boot)
    top = shortlist(frame, groups, eff, one)

    out = data_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    corr.rename(index=short, columns=short).to_csv(out / "confounding.csv")
    one.to_parquet(out / "one_change.parquet", index=False)
    eff.to_parquet(out / "effects.parquet", index=False)
    result = {
        "n_runs": len(frame),
        "n_factors": len(factors),
        "groups": [group_name(g) for g in groups if len(g) > 1],
        "shortlist": top,
    }
    (out / "shortlist.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(summary(result))
    return result


def summary(result: dict[str, Any]) -> str:
    lines = [
        f"{result['n_runs']} hot full cycles, "
        f"{result['n_factors']} varying settings",
    ]
    for g in result["groups"]:
        lines.append(f"  moved together (cannot separate): {g}")
    lines.append("shortlist:")
    for i, item in enumerate(result["shortlist"], 1):
        confirm = (
            " [one-change confirmed]" if item["one_change_support"] else ""
        )
        keep = "" if item["explore"] else " [only one value ever succeeded]"
        if item["direction"] == TRADE_OFF:
            what = (
                f"raising helps {', '.join(item['helps'])}; "
                f"lowering helps {', '.join(item['hurts'])}"
            )
        else:
            what = f"helps {', '.join(item['helps'])}" + (
                f"; hurts {', '.join(item['hurts'])}" if item["hurts"] else ""
            )
        lines.append(
            f"  {i}. {item['direction']} {item['group']} - {what}"
            + confirm
            + keep
        )
    return "\n".join(lines)
