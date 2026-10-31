# check-verdict

One small classifier per vision check (`spread`, `fold1`, `fold2`,
`push_off`) that predicts the robot's own pass or fail verdict for a single
camera frame. These are the predictions that Level 1 scores against the
robot.

## How it works

1. `frame-embedder` turns the frame into a 576-value embedding.
2. The head for that check is a scaled logistic regression with balanced
   class weights. It reads the embedding and returns `label` (`pass` or
   `fail`) and `probabilities`.

Each check has its own folder: `<check>/model.onnx` and `<check>/model.json`.
The ONNX input is `x` (float32, `[None, 576]`), in the feature order listed in
`model.json`.

## Data and labels

- Rows: every attempt of the check in `training/frames.parquet` that has a
  verdict and an embedding. push_off fails once by design and is then
  retried, so every attempt counts as its own frame.
- Label: `fail` when the robot marked the check not ok. These are weak
  labels. The heads learn to agree with the robot, not to judge crepe
  quality.
- Split: the fixed per-robot chronological, run-grouped holdout from
  `frames.split`. Train rows are fit, test rows are only scored.

## Report

Each `model.json` holds:

- the holdout metrics: balanced accuracy, macro F1, agreement with the robot,
  the confusion matrix and calibration
- the same metrics on last attempts only
- two baselines:
  - majority class
  - a logistic regression on the overlay scores in `vision.parquet`,
    for spread, fold1 and fold2 (push_off has none)
- a one-line verdict on whether the head beats the majority baseline

The overlay scores come from the robot's own overlay image. That makes the
overlay baseline close to an upper bound, not a fair competitor.

## Holdout results

Trained on the 1,112 cached frame embeddings with the per-robot split
(`robot-chrono-v2`): each robot's latest 20% of runs are test, so the
holdout mixes maus-quin, maus-otto and maus-pixi. maus-nilo has no runs with
vision checks. Intervals are 95% bootstraps over runs.

| check | test frames | test fails | balanced acc [95% CI] | fail recall [95% CI] | majority BA | overlay BA | generalizes |
|---|---|---|---|---|---|---|---|
| spread | 34 | 12 | 0.70 [0.55, 0.85] | 0.50 [0.25, 0.75] | 0.50 | 0.60 | no |
| fold1 | 32 | 4 | 0.75 [0.50, 1.00] | 0.50 [0.00, 1.00] | 0.50 | 0.74 | yes, weakly |
| fold2 | 31 | 5 | 0.67 [0.37, 0.89] | 0.60 [0.00, 1.00] | 0.50 | 0.88 | no |
| push_off | 61 | 32 | 0.92 [0.85, 0.97] | 0.94 [0.86, 1.00] | 0.50 | - | no |

- **push_off** is the only strong head: high on every robot in the holdout
  (0.75 to 0.97). It drops to majority when maus-otto is held out entirely.
  otto has only 11 runs.
- **spread** fell from 0.91 on the old pixi-only holdout to 0.70. It works
  on pixi (0.85) but sits at majority on the quin and otto holdouts. The
  earlier score mostly described one robot.
- **fold1** passes the generalization rule, but quin and otto have no test
  failures, so their holdouts cannot be scored for it. Leave-one-robot-out
  gives 0.73 to 0.81.
- **fold2** is at or below majority for otto and quin in both views. It is
  weaker than the overlay baseline.
- **Weights:** robot-balanced weights won cross validation for spread
  (0.88 vs 0.85) and tied for push_off. They lost for fold1 and fold2, so
  those use class weights. The choice is in each card under `weighting`.
- With 4 to 5 fold failures in the holdout, fold recall intervals span
  almost 0 to 1. More pixi and otto runs are the main way to tighten them.
  Fold recall work continues in #103.

## Train

```
pixi run -e ml python models/check-verdict/train.py [--data-dir ../data]
```

You need `training/frames.parquet` (`pixi run frames`) and
`training/frame_embeddings.parquet` (`pixi run embed`). Training prints the
holdout summary table and each head's verdict.
