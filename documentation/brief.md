# Challenge brief

The BaselHack 2026 challenge from Maus Robotics, restated, with what this
repository built for each part of it and a plain verdict per part: delivered,
partial, or not delivered. Every number is restated from
[project status](status.md), which names the command or file behind each one.

## The challenge

Maus's brief, "Recognize Crêpes": build a vision model that turns one camera
frame of the hotplate into decisions for the spread, fold 1, fold 2 and
push-off checks.

| # | Task | Question | Output |
|---|------|----------|--------|
| 1 | Spread quality | Is the batter even and round? Find thin spots, holes and ragged edges | Crêpe mask, diameter, quality score |
| 2 | Position and orientation | Where exactly is the crêpe on the rotating hotplate? | Centre (x, y) and angle, precise enough for the spatula |
| 3 | Fold detection | Flat, folded in half, in quarters or into a triangle? | Fold class + confidence, edge angle to align with the plate |

What makes it hard: steam, varying browning, shiny steel, a rotating plate,
overlapping edges, changing room light, and it must run in real time.

## How it is scored

- **Level 1, works on Maus data.** A model plus notebook scored on frames from
  the API, compared with the robot's own verdicts.
- **Level 2, works in real time.** Under about 300 ms per frame on a Raspberry
  Pi 5 CPU, robust to steam and light changes.
- **Bonus.** Auto-labelling from run videos, a spread score that predicts a
  failed fold, and detecting the crêpe while the plate rotates.

## What this repository built

| Challenge item | In this repository | Verdict |
| --- | --- | --- |
| 1, spread quality | Per-run vision metrics over 138 runs (diameter, centre offset, circularity, holes, uncovered fraction) and a classical pixel-mask suggester in the labeling tool. No spread model | partial |
| 2, position and orientation | Centre offset is a recorded metric; `code/maus_export/fold_geometry.py` computes an edge and corner angle, but the straight-edge gate accepted none of the 854 frames with a centre, and no fold edge angle or quarter corner was accepted | not delivered in practice |
| 3, fold detection | `code/models/fold-class` ONNX, four classes (empty, flat, half, quarter), 110-frame holdout, balanced accuracy 0.99 on weak labels. Triangle is unsupported, and the edge angle lives only in the unmerged fold work | partial, triangle missing |
| Level 1, works on Maus data | The check-verdict heads scored against the robot's verdicts, on holdouts of 31 to 61 frames. Three of the four model cards say they do not generalize | partial |
| Level 2, real time on a Pi 5 | 4 ms per frame on a desktop AMD CPU, a 16 ms unmeasured x4 extrapolation, and the arm64 environment only in a draft PR | not demonstrated |
| Bonus, auto-labelling from video | The fold segmentation pipeline in the draft stack | partial, unmerged |
| Bonus, spread score predicts a failed fold | `code/models/fold-risk`, balanced accuracy 0.29, and a permutation test that does not beat the null | attempted, negative result disclosed |
| Bonus, rotating-plate detection | No code tracks the rotating plate | not delivered |

The gap between asked and delivered is the point of this page. Each verdict
above has the weak number stated, and the fuller inventory, including what is
not working, not finished or not verified, is in [project status](status.md).
The shipped models and their holdout numbers are in
[code/models](../code/models/README.md).
