# Crêpe student

A distilled segmentation model that gives the crêpe mask on the raw camera
frame, and a fold state that needs no plate detection. Model files and card:
[`code/models/crepe-student`](../../code/models/crepe-student/README.md).
Code: `maus_export.crepe_student`.

## Why a student

SAM 3 segments the crêpe well from a text prompt ("crepe" or "food") but
needs about a second per frame on a GPU. The student learned SAM 3's masks
on 957 check frames and runs in about 6 ms on four CPU threads. On held-out
runs its masks overlap SAM 3's with a pixel IoU of 0.966.

## Fold state without the plate

Absolute plate shares turned out to differ between robots and camera setups:
the mean robot correction does not fit every run, and a flat crêpe covered
0.48 to 0.83 of the calibrated plate depending on the robot. The area
relative to the flat crêpe of the same run does not have that problem. At the
check moments half a crêpe covers about 0.39 of its flat area and a quarter
about 0.21, on every robot.

`classify_area(area, flat_area)`:

| Condition | State |
|---|---|
| largest mask part below 0.02 of the frame | empty |
| no flat area known yet | flat |
| area / flat area at least 0.60 | flat |
| area / flat area at least 0.30 | half |
| otherwise | quarter |

"No flat area yet means flat" matters for small spreads: the first crêpe on
an empty plate is always a spread, however small. The classical rule read a
too small spread as half, and the state machine then refused to leave empty.

## Live state machine

`StudentStateMachine` runs at 10 analysed frames per second:

- a 9-frame majority vote smooths single frames
- the state only moves forward: empty, flat, half, quarter
- empty must hold 30 frames, so a spatula hiding the crêpe does not reset it;
  then the flat area is forgotten for the next crêpe
- the flat area is the largest area while the crêpe reads as flat
- a fold is only accepted after the flat area has not grown by more than 2
  percent for 50 frames: while batter is poured and spread, the arm hiding
  part of it looks like a fold

`pixi run -e ml fold-student` writes `annotated.mp4`, `timeline.csv`,
`timeline.png` and `summary.json` per run under
`data/fold_segmentation/student/live/`, like `fold-live`.

## Results

`pixi run -e ml fold-student-eval` on the 2026-10-31 export, at the robot
check moments with a passed check (plate_clean = empty, spread = flat,
fold1 = half, fold2 = quarter):

| Group | Frames | Agreement |
|---|---|---|
| all | 991 | 99.5 % |
| test runs of the student | 191 | 99.5 % |
| empty / flat / half / quarter | 298 / 246 / 240 / 207 | 99.7 / 99.6 / 99.2 / 99.5 % |
| maus-quin / maus-pixi / maus-otto | 626 / 353 / 12 | 99.5 / 99.4 / 100 % |
| classical pipeline, same frames | 991 | 88.5 % |

The rule thresholds were read off the same frames, so these numbers are
optimistic. The remaining errors are mostly hands or tools over the crêpe at
the check moment.

Whole videos (live): a run with a too small spread now goes empty, flat,
half, quarter instead of staying empty; a run whose folds failed stays flat
through the fold phase, which is what happened on the plate.

## Limits

- Trained only on check-moment frames: small fresh batter at the start of
  spreading, very small failed spreads, a crêpe being pushed off and hands
  over the plate are outside its training data
- No hole class: holes count as crêpe, so it does not measure spread quality
- No position or angle yet; the mask is in the raw view, not on the plate
- No leave-one-robot-out check; maus-otto has only 6 training runs
- Teacher masks need SAM 3 on a GPU, which no repo environment provides yet
