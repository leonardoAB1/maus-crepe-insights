"""Offline tests for the history analysis, on synthetic run tables shaped
like data/runs.parquet (no real data)."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")  # model runs in the ml environment

from maus_export.analysis import (  # noqa: E402
    analysis_frame,
    confounding,
    factor_columns,
    model_effects,
    one_change_effects,
    shortlist,
)

FAN = "param.dispense.batter.fan_power"
INNER = "param.dispense.batter.concentric_inner_fan_power"
YC = "param.machine.y_center"
TOPPING = "param.dispense.toppings.apple.serving_size_g"
GF = "param.dispense.batter.gluten_free_krampouz"
SEG_RUNS = 8
FAN_LEVELS = (0.8, 0.9, 1.0)
GOOD_FAN = 0.9  # planted rule: success needs fan power at least this
N_BOOT = 40


def run_rows(segments: list[dict], robot: str = "r1") -> pd.DataFrame:
    """One row per run; each segment dict sets the params for SEG_RUNS runs.
    Success follows the planted fan rule."""
    rows, run_id = [], 1
    for seg, params in enumerate(segments):
        for _ in range(SEG_RUNS):
            ok = params[FAN] >= GOOD_FAN
            rows.append(
                {
                    "id": run_id,
                    "robot": robot,
                    "status": "succeeded" if ok else "not_succeeded",
                    "is_full_cycle": True,
                    "hotplate_temp": 200.0,
                    "env_temp_start": 40.0 + seg % 3,
                    "topping": "plain",
                    "segment_id": seg,
                    "check.spread.ok": ok,
                    **params,
                }
            )
            run_id += 1
    return pd.DataFrame(rows)


def history(n_segments: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    segs = []
    for _ in range(n_segments):
        fan = float(rng.choice(FAN_LEVELS))
        segs.append(
            {
                FAN: fan,
                INNER: round(fan * 0.97, 3),  # always moved with fan
                YC: float(rng.choice([-46.0, -45.0, -44.0])),  # no effect
                TOPPING: float(rng.choice([20.0, 25.0])),
                GF: True,
            }
        )
    return analysis_frame(run_rows(segs), pd.DataFrame())


def test_factor_columns_skip_toppings_and_recipe_switches() -> None:
    cols = factor_columns(history())
    assert FAN in cols and YC in cols
    assert TOPPING not in cols and GF not in cols


def test_parameter_id_is_not_a_model_factor() -> None:
    frame = history()
    frame["parameter_id"] = [f"v1:sha256:{i}" for i in range(len(frame))]
    assert "parameter_id" not in factor_columns(frame)


def test_cold_and_short_runs_are_dropped() -> None:
    runs = run_rows([{FAN: 1.0, INNER: 0.97, YC: -45.0}])
    runs.loc[0, "hotplate_temp"] = 40.0
    runs.loc[1, "is_full_cycle"] = False
    assert len(analysis_frame(runs, pd.DataFrame())) == SEG_RUNS - 2


def test_settings_moved_together_are_one_group() -> None:
    frame = history()
    _, groups = confounding(frame, factor_columns(frame))
    together = [g for g in groups if FAN in g]
    assert together and INNER in together[0]
    assert [YC] in groups


def test_one_change_detected_and_two_changes_skipped() -> None:
    a = {FAN: 0.8, INNER: 0.776, YC: -45.0}
    b = {FAN: 1.0, INNER: 0.97, YC: -45.0}  # only the fan group changed
    c = {FAN: 0.8, INNER: 0.776, YC: -44.0}  # fan group and y changed
    frame = analysis_frame(run_rows([a, b, c]), pd.DataFrame())
    _, groups = confounding(frame, [FAN, INNER, YC])
    one = one_change_effects(frame, groups)
    assert len(one) == 1
    row = one.iloc[0]
    assert "dispense.batter.fan_power" in row["group"]
    assert row["success.before"] == 0 and row["success.after"] == 1


def test_planted_effect_is_found_with_the_right_direction() -> None:
    frame = history()
    _, groups = confounding(frame, factor_columns(frame))
    eff = model_effects(frame, groups, n_boot=N_BOOT)
    fan = eff[(eff["outcome"] == "success") & eff["group"].str.contains("fan")]
    assert fan["coef_per_sd"].iloc[0] > 0 and fan["lo"].iloc[0] > 0
    # The bootstrap resamples whole segments, not single runs
    assert fan["n_segments"].iloc[0] == frame["segment_id"].nunique()

    top = shortlist(frame, groups, eff, one_change_effects(frame, groups))
    assert "fan_power" in top[0]["group"]
    assert top[0]["direction"] == "raise"
    # Safe range spans only values that produced successes
    lo, hi = top[0]["safe_range"]["dispense.batter.fan_power"]
    assert lo >= GOOD_FAN and hi == max(FAN_LEVELS)


def test_find_data_dir_walks_up_and_follows_worktrees(
    tmp_path, monkeypatch
) -> None:
    from maus_export.analysis import DATA_ENV, find_data_dir

    monkeypatch.delenv(DATA_ENV, raising=False)
    main = tmp_path / "repo"
    (main / "data").mkdir(parents=True)
    (main / "data" / "runs.parquet").write_bytes(b"")
    nested = main / "code" / "notebooks"
    nested.mkdir(parents=True)
    assert find_data_dir(nested) == main / "data"

    # A worktree has a .git file pointing at <main>/.git/worktrees/<name>
    wt = main / "worktrees" / "feat" / "x"
    (wt / "code" / "notebooks").mkdir(parents=True)
    (main / ".git" / "worktrees" / "x").mkdir(parents=True)
    gitdir = main / ".git" / "worktrees" / "x"
    (wt / ".git").write_text(f"gitdir: {gitdir}\n", encoding="utf-8")
    assert find_data_dir(wt / "code" / "notebooks") == main / "data"

    monkeypatch.setenv(DATA_ENV, str(tmp_path / "elsewhere"))
    assert find_data_dir(nested) == tmp_path / "elsewhere"
