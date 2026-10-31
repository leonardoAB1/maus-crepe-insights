# fold-risk

Predicts from the spread check whether fold1 or fold2 will fail, so a bad
crepe could be caught early. **Result: no usable signal in this data.** The
model is exported so the result is reproducible and the pipeline is ready
for more runs, not for use on the robot.

- `model.onnx`: input `x` float32 `[N, 10]` (feature order in `model.json`),
  outputs `label` (`fail` / `ok`) and `probabilities` float32 `[N, 2]`
- `model.json`: model card, holdout lift with 95% intervals, comparisons,
  robot breakdown, feature importance
- `train.py`: rebuilds both files

## Data

- **Rows:** one per run with a spread check, spread overlay scores and at
  least one fold check (`maus_classify.fold_risk.fold_risk_table`).
- **Target:** any fold1 or fold2 attempt in the run not ok. This is the
  robot's verdict, a weak label.
- **Features:** these are known when the spread check ends, taken from its
  last attempt:
  - nine overlay scores from `vision.parquet`: area, diameter, offset,
    circularity, holes, uncovered area, lightness mean and SD, residue
  - the robot's spread verdict (`spread_ok`)
- **Split:** the shared holdout from `data/training/frames.parquet` (see
  [frames](../../../documentation/frames.md)), one side per run.
- **Model:** `StandardScaler` and L2 logistic regression with default `C`,
  untuned. Trained through `maus_classify.harness.train_and_evaluate`, which
  also runs the onnxruntime parity check.

## Result (local export, 2026-10-31)

113 runs, 41 with a fold failure. Train has 93 runs, test 20 runs with 8
failures, so every holdout interval is wide.

| Holdout (20 runs) | ROC AUC [95% CI] | AP lift | Top 25% lift |
|---|---|---|---|
| Scores + verdict (shipped) | 0.33 [0.07, 0.65] | 0.94 | 0.50 |
| Spread verdict only | 0.50 [0.31, 0.70] | 1.00 | 1.00 |
| Scores + verdict + embedding PCA(8) | 0.45 [0.18, 0.72] | 1.16 | 0.50 |

Lift is relative to the holdout base rate of 0.40. A lift of 1 is chance.
All intervals are in `model.json`.

The training window looks better (cross-validated AUC 0.78) because of a
robot confound:

- In train, robot B failed a fold in 12 of 13 runs, while robot A (73 runs)
  failed in 21%. Robot B's spread images are also brighter, so lightness
  stood in for "robot B". The lightness features carry the largest
  coefficients.
- All 20 holdout runs are robot B, with a 40% fail rate, so that shortcut
  does not transfer.
- Within robot A alone, cross-validated AUC is 0.51 with all features, 0.47
  with the verdict only and 0.46 without lightness. That is chance.

So the spread check, as scored here, does not tell whether the fold will
fail. Fold failures likely depend on what happens after the spread, or on
robot state the overlay does not capture. A useful model needs more runs per
robot, and robot-balanced train and test sets, before this is worth
revisiting.

## Rebuild

From `code/`, after `pixi run frames`, `pixi run vision` and optionally
`pixi run -e ml embed`:

```bash
pixi run -e ml python models/fold-risk/train.py
```
