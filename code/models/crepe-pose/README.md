# crepe-pose

> **EXPERIMENT - NOT USED.** Nothing in the project, the robot or the team's
> results depends on this model. The team's crepe segmentation model is
> [crepe-student](../crepe-student/README.md). This branch is not proposed for
> merging and is kept for reference only.

Finds the crepe in one rectified frame, for challenge task 2: a mask, and its
centre from the mask centroid. **Verdict: a local experiment, not deployable.**
The full reasoning and every table are in
[Crepe position from synthetic images](../../../documentation/analysis/synthetic-crepes.md).

![crepe-pose examples](../../../assets/crepe-pose-example.jpg)

Red is the model and green is the robot's outline, on test frames from runs the
model never trained on. The top row is typical. The bottom row is the three
worst, including a sleeve marked as crepe. In the middle of that row, the
robot's outline is the one at fault.

## Size and speed

- 619,267 parameters and a 2.5 MB ONNX file
- The full path from a video frame to a cleaned mask, on an Apple M4 Pro laptop:

| Threads | Prepare frame | Model | Total, median | Total, p95 |
|---|---|---|---|---|
| 1 | 0.4 ms | 6.7 ms | 7.3 ms | 7.5 ms |
| 4 | 0.4 ms | 3.7 ms | 4.3 ms | 5.0 ms |

**Not measured on a Raspberry Pi 5.** A rough estimate is 29-58 ms per frame,
against the 300 ms target. Keep the fixed batch: the same weights exported with
a variable batch take 24.7 ms instead of about 6.8 ms on one thread.

## How well it works

| Check | Reference | Result |
|---|---|---|
| Held-out runs, 113 crepes and 80 empty plates | Reviewed hand masks | IoU 0.965, 0 missed, 4 false crepes |
| Unseen test check frames, 85 from 27 runs | Robot outline | IoU 0.964 |
| The same frames, `maus-pixi` (25) | Robot outline | IoU 0.975 |
| The same frames, `maus-quin` (54) | Robot outline | IoU 0.963 |
| The same frames, `maus-otto` (6) | Robot outline | IoU 0.946 |

The hand-mask row comes from models that each held out one run. The final model
trained on all five, so it has no hand score of its own. The hand masks all come
from `maus-pixi`, and some may have started from pixel suggestions.

## Do not use it for

- **Fold angle.** The `direction` output never beat the training average, and
  hand fine-tuning did not train it.
- **Fold state or holes.** It names no state, and holes are filled in the mask.

## Known problems and gaps

- It can mark a hand or sleeve as crepe, and can find a crepe on an empty plate
- It needs a rectification transform for the run before it can read a frame
- There are no bootstrap intervals for the final model
- There is no leave-one-robot-out check, which [the model rules](../README.md)
  require
- Real steam and light changes are untested

## Files

- `model.onnx`: input `x` float32 `[1, 3, 160, 160]`, BGR, scaled 0 to 1.
  Outputs `mask` `[1, 1, 160, 160]`, threshold 0.5, and `direction` `[1, 2]`
- `model.json`: the model card
- `train.py`: synthetic images and the first two training stages
- `hand_masks.py`, `hand_finetune.py`: rectify the hand masks, then fine-tune
- `card.py`: rebuilds the card and the example image
