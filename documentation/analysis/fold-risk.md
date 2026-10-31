# Fold risk from the spread check

**Pitch takeaway:** in pooled data, a failed spread check looks like a warning
that the fold will fail. Once you compare runs on the same robot, neither the
spread verdict nor any spread score predicts fold failure with today's 106
runs. About 140 runs on the same robots would be enough to detect an odds ratio
of 2 per SD.

This page answers issue #94: can the spread check tell us early that the fold
after it will fail? Figures are aggregates from one private, account-limited
export as of 2026-10-31. A refresh that day found no new runs. The model,
card and code are in [`code/models/fold-risk/`](../../code/models/fold-risk/).

## Data

- **Rows:** one per run with a spread check, overlay scores for it in
  `vision.parquet`, and at least one fold check. There are 113 such runs.
- **Features:** these come from the last spread attempt and were fixed before
  the adjusted analysis (pre-registered in #94):
  - the robot's verdict on the spread
  - diameter, offset, circularity, holes, uncovered area and residue
- **Targets:** these are the robot's own fold verdicts, so they are weak
  labels.
  - **any:** a fold1 or fold2 attempt failed
  - **residue:** a fold error "batter left on the plate"
  - **shape:** "not a half fold", "not a quarter fold" or "wrong fold shape"
  - Runs where a fold check saw no crepe are left out of residue and shape.

| Robot | Window (UTC) | Runs | Spread failed | Any fold fail | Residue | Shape |
|---|---|---:|---:|---:|---:|---:|
| maus-quin | Oct 13-20 | 73 | 6 | 15 | 2 | 11 |
| maus-otto | Oct 22 | 7 | 6 | 6 | 5 | 2 |
| maus-pixi | Oct 27-30 | 33 | 9 | 20 | 7 | 17 |

maus-nilo has no runs with vision checks in this export.

## Why robot comes first

Each robot ran in its own time window, so "robot" also stands for era, recipe
and software version. Fold failure differs a lot by robot: 21% on maus-quin,
61% on maus-pixi and 6 of 7 runs on maus-otto. Anything that differs between
robots therefore looks predictive when the robots are pooled.

- Pooled over robots, folds fail after 13 of 21 failed spreads (62%), against
  28 of 92 passed spreads (30%).
- Most failed spreads come from maus-otto and maus-pixi, the robots that fail
  folds anyway. Within one robot the gap mostly disappears. Adjusted for
  robot, the odds ratio for a failed spread is 1.68, with a 95% interval of
  0.49 to 5.74.
- Brightness shows the same trap. Robot explains 25% of the variance in mean
  lightness (pixi images are brighter), and pooled, brighter spreads look
  riskier. Adjusted for robot, the odds ratio per SD is 0.77 [0.49, 1.22].
  Lightness is a robot proxy and is used only as this check.

A single time-ordered holdout cannot fix this, because the latest runs are all
one robot (see [frames](../frames.md)).

## Robot-adjusted effects (primary result)

The fits are logistic regressions with a robot fixed effect, one feature at a
time, on maus-quin and maus-pixi. maus-otto has too few runs to fit and is used
only as an out-of-sample robot below. Continuous features are scaled by their
within-robot SD. Holm corrects across the seven features of each mechanism.

![Robot-adjusted odds ratios per feature and mechanism](../../assets/fold-risk-effects.png)

| Feature | Any (35 of 106) | Residue (9 of 104) | Shape (28 of 104) |
|---|---|---|---|
| Spread failed (yes vs no) | 1.68 [0.49, 5.74] | 0.96 [0.16, 5.72] | 1.98 [0.56, 7.01] |
| Diameter | 0.74 [0.47, 1.16] | 0.79 [0.43, 1.45] | 0.90 [0.57, 1.41] |
| Offset | 1.36 [0.86, 2.15] | 1.54 [0.90, 2.64] | 1.14 [0.73, 1.76] |
| Circularity | 0.94 [0.61, 1.45] | 0.55 [0.29, 1.02] | 0.88 [0.56, 1.38] |
| Holes | 1.16 [0.67, 2.00] | 1.50 [0.72, 3.11] | 1.22 [0.68, 2.17] |
| Uncovered area | 1.68 [0.76, 3.70] | 2.66 [1.15, 6.20] | 1.26 [0.74, 2.14] |
| Residue | 1.24 [0.66, 2.34] | 1.62 [0.65, 4.04] | 1.36 [0.65, 2.86] |

Each cell is the odds ratio per within-robot SD with its 95% interval
(unadjusted). Number of fold failures and runs in the headers.

- **No effect survives Holm correction** in any mechanism (smallest Holm
  p = 0.16).
- The strongest lead is **uncovered area on residue failures**, with OR 2.66
  per SD and unadjusted p = 0.023. It rests on 9 residue failures, so treat it
  as a hypothesis to test with new runs, not a finding. The direction is
  plausible: batter that did not cover the plate at spread time is still there
  at fold time.

## Power analysis

How big an effect could this data detect, and how many runs would be needed?
This uses the Wald approximation for a logistic slope (Hsieh 1989), with 80%
power, alpha 0.05 Bonferroni-split over 7 features (the strictest Holm step),
and within-robot variance only.

| Mechanism | Runs now | Smallest detectable OR per SD | Runs for OR 2 per SD | Runs for OR 2, spread verdict |
|---|---:|---:|---:|---:|
| Any | 106 | 2.21 | 140 | 1,223 |
| Residue | 104 | 3.64 | 362 | 3,184 |
| Shape | 104 | 2.32 | 154 | 1,354 |

- **Continuous scores:** about 140 to 150 runs on these robots would detect a
  moderate effect (OR 2 per SD) on any or shape failures. Residue failures
  are rare, so that target needs about 360 runs.
- **The robot's spread verdict:** it is rarely "fail" within a robot (6 of 73
  on maus-quin). An OR of 2 would therefore take over 1,200 runs, and only a
  large effect is detectable soon.
- A test in `tests/test_fold_effects.py` checks the formula against
  simulation.

## Model (secondary)

The model is the same seven features, scaled, with L2 logistic regression.
Every check compares against a robot-only baseline (each robot's fold-failure
rate), and the intervals bootstrap runs within each robot.

| Check (any fold failure) | Result [95% CI] |
|---|---|
| Within-robot delta AUC over robot-only, pooled CV (20x5) | -0.18 [-0.29, -0.06] |
| Brier skill over robot-only | -0.11 [-0.21, -0.03] |
| CV inside maus-quin only, AUC | 0.41 [0.27, 0.56] |
| CV inside maus-pixi only, AUC | 0.45 [0.24, 0.65] |
| Leave maus-quin out, AUC | 0.42 [0.27, 0.59] |
| Leave maus-pixi out, AUC | 0.47 [0.27, 0.68] |
| Leave maus-otto out, AUC | 1.00 (only 1 run without a fold failure, not informative) |
| Shared holdout, one robot (maus-pixi, 20 runs) | 0.40 [0.10, 0.75] |

- Inside a robot the model ranks no better than chance. With labels shuffled
  within robot, delta AUC falls between -0.18 and +0.10 (5% to 95%, 100
  shuffles), so the observed -0.18 is at the low edge of noise. That means no
  signal, not a reverse signal.
- Residue and shape give the same picture (full numbers in `model.json`).
- **Card verdict: research only.** The rule is "deployable" only if the delta
  AUC interval is above 0. The ONNX export exists so the analysis can be
  reproduced and rerun on more data, not to run on the robot.

## What would change the answer

- **More runs per robot:** about 150 with spread and fold checks, ideally
  across all four robots and mixed in time, so robot and era stop coinciding.
- **More spread failures inside each robot:** today almost all spread
  failures come from the robots that fail folds anyway.
- **Rerun:** `pixi run -e ml python models/fold-risk/train.py` rebuilds the
  card, the plot and these numbers.
