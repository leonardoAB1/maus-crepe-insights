# Challenge brief

The BaselHack 2026 challenge from Maus Robotics, restated, with a short
summary of what this repository built for it. The verdict per part, and the
command or file behind each number, is in [project status](status.md).

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

The verdict for each task and level, with the number behind it, is kept in
one place so it stays current:
[brief coverage in project status](status.md#brief-coverage). In short:

- Tasks 1, 2 and 3 each have a working answer, scored against the robot's own
  verdicts and overlays. The triangle fold has no frames to learn from.
- Level 2 is measured on a Raspberry Pi 5; the numbers and the method are on
  the status page.
- The shipped models and their holdout numbers are in
  [code/models](../code/models/README.md).
