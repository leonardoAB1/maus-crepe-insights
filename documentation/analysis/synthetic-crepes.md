# Crepe position from synthetic images

> **EXPERIMENT - NOT USED.** Nothing in the project, the robot or the team's
> results depends on this work. The team's crepe segmentation model is
> [crepe-student](../vision/crepe-student.md). This branch is not proposed for
> merging and is kept for reference only.

Experiment for challenge task 2: where is the crepe, and at what angle? It asks
whether generated images can replace hand labels. This is a local experiment,
not a shipped model.

**Verdict:** for position, use the model pretrained on synthetic images, then
fine-tuned on robot outlines and hand masks. It is the best balance measured
here, but with five hand-labelled runs from one robot this is a lean, not
proof.
Do not claim a fold angle result.

## Method

- A small U-Net (619,267 parameters) reads a rectified 160 x 160 frame. It
  returns a crepe mask and a fold direction. The centre is the mask centroid.
- Plates, textures and cut-outs come from training-split frames only.
- The reference on real frames comes from the robot's overlays: the green
  outline for the mask and the magenta ideal fold for the direction.
- Scores use the shared per-robot, run-grouped holdout, with medians and 95%
  intervals from a bootstrap of runs. Settings were fixed before the test split
  was scored.

Code: `code/maus_export/synthetic_crepes.py` and
`code/models/crepe-pose/train.py`. Outputs stay in `data/synthetic/`.

## Results

Test split: 131 frames from 34 runs, of which 97 show a crepe and 34 are empty
plates. Centre error is a share of the plate radius. IoU includes the empty
plates. Angle is scored on the 54 passed fold frames with a reference.

| Method | Centre error | IoU | Angle error | False crepe on empty plate |
|---|---|---|---|---|
| Training average for that robot and check | 0.048 [0.041, 0.058] | 0.871 [0.841, 0.910] | 2.4 [1.8, 2.8] | 0% |
| Otsu threshold | 0.018 [0.013, 0.025] | 0.923 [0.894, 0.944] | - | 100% |
| Synthetic only | 0.008 [0.007, 0.011] | 0.968 [0.961, 0.971] | 2.6 [1.8, 3.1] | 0% |
| Real overlay labels only | 0.009 [0.006, 0.011] | 0.972 [0.969, 0.976] | 2.1 [1.4, 3.5] | 0% |
| Synthetic, fine-tuned on real | 0.008 [0.006, 0.010] | 0.972 [0.969, 0.976] | 2.0 [1.3, 2.4] | 0% |

The ONNX export of the real-label model matches PyTorch within 5.1e-07. It is
2.5 MB and takes a median 24.8 ms per frame on one thread of an Apple M4 Pro.
That is not a Raspberry Pi 5 number, and it leaves out rectification and mask
cleanup.

That export used a variable batch size. Exporting the final model with a fixed
batch of 1 gives the same outputs but takes about 6.7 ms on one thread, and
warping frames to 320 pixels first cuts preparation from 7.8 ms to 0.4 ms. The
full path is now about 7.3 ms on one thread of this laptop. See the
[model card](../../code/models/crepe-pose/README.md).

## Against hand masks

The table above measures agreement with the robot. This one uses 193 reviewed
hand masks (113 crepes, 80 empty plates) from five `maus-pixi` runs, rectified
into the same view. Only records marked `ok` are used. Masks include their holes
and are compared on supported pixels. No model trained on these masks.

| Model | Crepe IoU | 10th percentile IoU | Centre error | Missed crepes | False crepe on empty plate |
|---|---|---|---|---|---|
| Otsu threshold | 0.892 | 0.533 | 0.048 | 0% | 100% |
| Synthetic only | 0.966 | 0.914 | 0.012 | 0% | 23% |
| Real overlay labels only | 0.953 | 0.835 | 0.017 | 3% | 3% |
| Synthetic, fine-tuned on real | 0.960 | 0.903 | 0.015 | 3% | 0% |

Overall crepe IoU intervals overlap for the three models. This reverses the
earlier choice of the real-label model, which agreed best with the robot.

## Fine-tuning on hand masks

Starting from the fine-tuned model, each fold trains on four runs and scores the
held-out fifth, so every number below comes from frames that fold did not train
on.

| Held-out hand frames | Crepe IoU | 10th percentile IoU | Missed crepes | False crepe on empty plate |
|---|---|---|---|---|
| Before | 0.960 | 0.903 | 3 of 113 | 0 of 80 |
| After | 0.965 | 0.922 | 0 of 113 | 4 of 80 |

Crepe IoU rose in all five folds, but the five runs have no intervals here, and
one run holds only 2 crepes. The cost is false crepes on empty plates, in two
runs.

A final model trained on all five runs was checked against robot outlines on 85
test check frames from runs without hand masks. It did not visibly harm the
other robots:

| Robot | Frames | Median IoU before | Median IoU after |
|---|---|---|---|
| `maus-otto` | 6 | 0.944 | 0.946 |
| `maus-pixi` | 25 | 0.972 | 0.975 |
| `maus-quin` | 54 | 0.965 | 0.963 |

The final model has no held-out hand frames of its own, so its hand score is
unknown. The fold results are the estimate.

## Against crepe-student

Both models were scored the same way on the same frames, with the student's mask
warped into the rectified view. Crepe-pose is shown before hand fine-tuning
here, because the final version trained on the hand masks. The final file is the
one timed. Times run from a frame to a cleaned mask on one Apple M4 Pro laptop.

| | crepe-pose | crepe-student |
|---|---|---|
| File size | 2.5 MB | 12.9 MB |
| Time, 1 thread | 7.3 ms | 18.8 ms |
| Time, 4 threads | 4.1 ms | 8.7 ms |
| Hand masks, crepe IoU (113 crepes) | 0.961 [0.942, 0.974] | 0.955 [0.934, 0.970] |
| Hand masks, missed crepes | 3 of 113 | 8 of 113 |
| Hand masks, false crepes | 0 of 80 | 0 of 80 |
| Robot outlines, crepe IoU (79 frames) | 0.966 [0.959, 0.970] | 0.951 [0.940, 0.960] |
| Fold state | no | yes |

The hand-mask intervals overlap, so this does not show crepe-pose is more
accurate. Robot outlines favour crepe-pose, which trained on them, while the
student trained on SAM 3 masks. All hand-labelled runs are on the training side
of the student's split, and all come from one robot. Crepe-student stays the
team's model because it also gives the fold state.

## How the first version failed

Version 1 drew every crepe, with a hard pasted edge. It reached a median IoU of
1.00 on synthetic images and 0.00 on real training-split frames. Removing cues
one at a time on 150 images showed why:

| Images shown to version 1 | Median IoU | Images above IoU 0.5 |
|---|---|---|
| Synthetic, as trained | 1.00 | 100% |
| Dark ring removed | 1.00 | 98% |
| Edge softened | 0.06 | 17% |
| Edge softened and ring removed | 0.00 | 10% |
| Real cut-outs on real plates | 0.00 | 2% |
| Real frames | 0.00 | 0% |

The model had learned the hard edge, not the crepe. Version 2 uses real cut-outs
for 75% of crepes, feathered edges and random blur.

## Limitations

- **Hand masks are limited.** They come from one robot and may start from pixel
  suggestions, so they are reviewed masks, not independent manual ground truth.
  Set 2 frames are matched by run, frame and timestamp, because their PNG bytes
  differ from the local copies. Set 1 comes from a labels update not yet merged
  in the private labels repo.
- **Test split overlap.** All hand-labelled runs are in the shared test split.
  The hand-tuned model must be scored without them, so its first table does not
  apply to it.
- **The reference is not ground truth.** The scores measure agreement with the
  robot's outline. Those outlines can be wrong, for example when a spatula or
  hand crosses the frame, and a model can be marked down for being right.
- **Not label-free.** Version 2 cut-outs use the same robot outlines, so its
  agreement with them is not independent.
- **Fold angle is unproven.** No model clearly beats always answering the
  training average, and the reference is the fold the robot intended rather
  than the fold it got.
- **Hands and sleeves.** On one `maus-otto` fold 2 frame, both the synthetic and
  real-label models mark a sleeve as crepe, despite synthetic hands in training.
- **Small crepes.** On the worst test frame, a very small spread, IoU is 0.29
  for synthetic and 0.00 for real labels, which misses it entirely.
- **Thin coverage.** `maus-otto` has only 8 test frames.
- **Robustness is untested.** Steam and light changes were not measured on real
  frames, so any benefit from synthetic disturbances is unknown.
- **Otsu needs an empty-plate check.** It sees a crepe on every empty plate, so
  its score is only fair on frames known to hold a crepe.
