# Which settings matter (history analysis)

The question: from the runs Maus already made, which recipe settings go with
better crêpes, and in which direction? The answer feeds the designed experiment
(#26).

The code lives in `code/maus_export/analysis.py` (tested). The notebook
`code/notebooks/25-history-effects.ipynb` holds the plots and the story.

```bash
cd code
pixi run -e ml analyze     # writes data/analysis/, prints the shortlist
pixi run -e ml notebook    # opens JupyterLab on code/notebooks/
```

## Data used

- **Runs:** hot full cycles only. That means `is_full_cycle` (the run reached
  `cook`) with the hotplate at or above 150 °C. Short test cycles and cold
  runs are left out.
- **Outcomes:**
  - pass/fail: run success, plus the spread, fold 1 and fold 2 checks
  - continuous: the vision scores (spread diameter, bare plate, centre
    offset, fold 1 overlap, fold 2 area ratio)
- **Factors:** numeric batter, machine and push-off settings that vary
  enough. A setting needs at least 10 runs away from its most common value.
  - Excluded: toppings (they only affect their own topping) and the
    gluten-free recipe variants (a separate recipe with few runs).
  - A setting that is unset in older runs gets an "unset" indicator, not
    an invented value.
- **Where the data comes from:** `$MAUS_DATA_DIR` if set, else a `data/` folder
  in the current folder or a parent, else the main checkout's `data/` when
  running inside a worktree (`find_data_dir`).
- **Covariates:** robot, topping, hotplate setpoint and room temperature.
  Humidity is left out because it tracks room temperature (r = -0.87).

## Three views

1. **What moved together.** Settings correlated at |r| ≥ 0.8 form one group.
   The data cannot separate the members of a group, so effects are reported
   per group.
2. **One-setting changes.** These are consecutive settings blocks on one robot
   where only one group changed. The last 10 runs before are compared with
   the first 10 after: Fisher exact test for pass/fail, Mann-Whitney for
   scores. This is the cleanest evidence, but rare, because most tuning steps
   lasted only 1-2 runs.
3. **Model over all runs.**
   - L2 logistic regression for pass/fail outcomes and ridge regression for
     scores, on standardised settings.
   - Intervals come from a bootstrap that resamples whole segments. Runs in
     one segment repeat one config and are not independent.

## Shortlist (`data/analysis/shortlist.json`)

Each item is one settings group with:

| Field | Meaning |
|---|---|
| `direction` | `raise`, `lower`, or `trade-off` when raising helps some outcomes and lowering others |
| `helps` / `hurts` | outcomes with a clear effect (90% interval excludes 0, ≥ 80% of bootstrap draws agree). For a trade-off, `helps` improve when raised and `hurts` when lowered |
| `one_change_support` | one-setting changes that agree, with their p-value |
| `safe_range` | min and max of each member over runs that succeeded. The experiment stays inside it |
| `explore` | false when only one value ever succeeded, so there is nothing to vary |
| `current` | median over the latest 30 full cycles |

The ranking is: clear direction first, then something to explore, then
one-change support, then more outcomes helped than hurt, then effect size.

## Caveats

- **This is observational.** The settings were chosen by people reacting to
  results, so a setting can look good just because it was in use when other
  things improved. Robot and topping are covariates, but time trends
  (batter batches, wear) are not modelled.
- **Grouped settings stay grouped.** #26 must vary them independently, or the
  experiment repeats the history's blind spot.
- **The vision scores only exist for recent runs,** about 130 of 400. Score
  effects rest on fewer segments than the success effects.
- **Committed notebooks keep no outputs** (`code/tests/test_notebooks.py`
  enforces it). Clear them before committing:
  `jupyter nbconvert --clear-output --inplace notebooks/*.ipynb`.
