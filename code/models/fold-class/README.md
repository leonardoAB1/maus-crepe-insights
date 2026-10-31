# fold-class

Challenge Task 3 from one frame: is the crepe **flat**, folded in **half**
or folded in **quarters**, with a probability for each.

A logistic-regression head (standardized inputs) on the 576-d output of
the frozen [frame-embedder](../frame-embedder/README.md). Trained, scored and
exported with the shared harness in `maus_classify.harness`.

- `model.onnx` - input `x` float32 `[N, 576]` (embeddings `e0..e575`),
  outputs `label` (string) and `probabilities` float32 `[N, 3]` in the
  order `flat, half, quarter`
- `model.json` - model card: holdout metrics, per-class calibration, the
  confidence check, sanity checks and caveats
- `train.py` - rebuilds both files

## Pipeline

```
raw frame -> frame_preprocess.preprocess -> frame-embedder -> fold-class
```

## Labels

Weak labels from the `fold_class` column of the frame table: a passed spread
check is flat, a passed fold1 is half, a passed fold2 is quarter. Failed
checks have no label and are left out. The holdout is the chronological,
run-grouped split fixed in the frame table.

## Results (holdout, latest 20% of runs)

| class | frames | accuracy |
|---|---|---|
| flat | 16 | 0.94 |
| half | 13 | 0.92 |
| quarter | 14 | 1.00 |

Overall 41 of 43 correct (0.95). The mean top-class confidence is 0.98, so
the model is slightly overconfident. Per-class expected calibration error
is 0.02 to 0.03. With 43 frames, each error moves accuracy by over two
points.

## Caveats

- **Triangle folds are unsupported.** No frame shows one, so the head only
  knows flat, half and quarter. Human fold labels from the labelling
  workstream (#87) are the follow-up path, but its visual taxonomy
  (crepe-visual-v2) has no triangle label yet. The follow-up (#106) depends
  on adding one.
- **Label leakage.** The class equals the check that took the frame, so the
  head could learn camera timing or plate state instead of shape. As a sanity
  check, passed push_off frames in the holdout show a quarter fold from a
  check the head never trained on. All 14 come out as quarter. The stronger
  check against human labels on video frames is tracked in #104.
- **No "no crepe" class.** Empty pans (plate_clean) are called quarter with
  near-certain confidence. Only run the head on frames known to show a
  crepe.
- The edge angle for plate alignment is out of scope.

## Rebuild

From `code/`:

```bash
pixi run frames            # data/training/frames.parquet
pixi run -e ml embed       # data/training/frame_embeddings.parquet
pixi run -e ml fold-class  # model.onnx + model.json here
```
