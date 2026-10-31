# fold-risk

Predicts from the spread check whether fold1 or fold2 will fail. **Verdict:
research only.** Within a robot, the spread check does not predict fold
failure in the current data. The analysis, plot and power numbers are in
[fold risk](../../../documentation/analysis/fold-risk.md).

- `model.onnx`: input `x` float32 `[N, 7]` (feature order in `model.json`),
  outputs `label` (`fail` / `ok`) and `probabilities` float32 `[N, 2]`
- `model.json`: model card. The `fold_risk` section holds:
  - the verdict and the rule behind it
  - per-robot counts
  - robot-adjusted Firth odds ratios with profile intervals, Holm
    corrected within each mechanism (Wald kept for comparison)
  - the brightness confound check
  - the power analysis
  - within-robot CV, leave-one-robot-out and a 1,000-shuffle permutation
    test per mechanism
  - the shared holdout, labelled as one robot
- `train.py`: rebuilds both files and `assets/fold-risk-effects.png`

## Design

- **Rows:** one per run with a spread check, overlay scores and at least one
  fold check (`maus_classify.fold_risk.fold_risk_table`).
- **Features:** fixed in issue #94, taken from the last spread attempt:
  - `spread_fail`, the robot's spread verdict
  - `diameter_rel`, `offset_rel`, `circularity`, `hole_frac`,
    `uncovered_frac` and `residue_frac` from `vision.parquet`

  Lightness is left out because it proxies the robot.
- **Targets:** any, residue or shape fold failure, all robot verdicts (weak
  labels).
- **Robot first:** each robot ran in its own time window, so every effect and
  every model check is computed within robot.
  - Effects come from `maus_classify.fold_effects`: Firth logistic
    regression (`maus_classify.firth`) with a robot fixed effect.
  - Model checks come from `maus_classify.fold_cv`: robot-only baseline,
    within-robot delta AUC, leave-one-robot-out.
  - maus-otto (7 runs) is left out of the fits and scored as an out-of-sample
    robot.
- **Shipped model:** `StandardScaler` and L2 logistic regression, trained and
  exported through `maus_classify.harness.train_and_evaluate` (onnxruntime
  parity check included).
- **Verdict rule:** "deployable" only if within-robot delta AUC over
  robot-only beats labels shuffled within robot (one-sided permutation
  p < 0.05).

## Rebuild

From `code/`, after `pixi run frames` and `pixi run vision`:

```bash
pixi run -e ml python models/fold-risk/train.py
```
