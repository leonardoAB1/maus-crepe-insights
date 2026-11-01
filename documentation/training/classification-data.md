# Classification data coverage

**Why do the frame models cite about 100 runs when the export has 828?**
Camera checks only exist from 13 October. Every robot ran in its own few
days, and a model needs the right checks on the right robot. This page
follows the data from all runs down to each model's sample and explains
every drop.

Figures are aggregates from one private, account-limited export as of
2026-10-31 and come from one command. From `code/`:

```bash
pixi run coverage
# or another export (global option before the subcommand):
pixi run maus-classify --data-dir /path/to/data coverage
```

It reads `raw/runs`, `raw/robots.json`, `training/frames.parquet` and
`vision.parquet`, prints Markdown tables and writes nothing. From a worktree
it uses the main checkout's `data/`. Rerun it after a new fetch or split
change and update the tables below.

## The funnel

| Step | Runs | Why runs drop out |
|---|---:|---|
| All runs | 828 | |
| With any camera check | 160 | 668 runs have no vision check. 591 are older than the first run with one: camera checks with images start on 13 October. The other 77 are later test cycles (74 of 77) that run only part of the sequence, for example ending at oiling. |
| With a spread check | 139 | 21 stopped before the spread, for example at the plate check. |
| With a fold check after the spread | 114 | 25 ended after the spread, all canceled. |
| With spread overlay scores | 113 | 1 has no usable spread overlay. |
| Fold-risk fit | 106 | Fitted robots need at least 20 runs. maus-otto has 7, so it is scored only as an unseen robot. |

Each model starts from a different step:

- **Frame heads** (check verdict, fold class, failure reason) use frames from
  the 160 runs with camera checks.
- **Fold risk** ([analysis](../analysis/fold-risk.md)) needs a spread, a
  later fold and overlay scores, which gives 113 runs, or 106 in the fit.

Counts per model are in each model card.

## Robot is also era

| Robot | All runs | All runs window (UTC) | Runs with camera checks | Camera window (UTC) | Test | Production |
|---|---:|---|---:|---|---:|---:|
| maus-quin | 384 | Oct 05 to Oct 22 | 83 | Oct 13 to Oct 20 | 22 | 61 |
| maus-otto | 34 | Oct 16 to Oct 22 | 11 | Oct 22 | 11 | 0 |
| maus-pixi | 410 | Sep 02 to Oct 30 | 66 | Oct 27 to Oct 30 | 57 | 9 |
| maus-nilo | 0 | - | 0 | - | 0 | 0 |

The robots with camera checks ran one after another: maus-quin first, then
maus-otto for one day, then maus-pixi. Recipe parameters and robot software
also changed over that time (see [DoE](../analysis/doe.md)). A difference
between robots is therefore also a difference in era, recipe and software,
and no table can separate them. maus-nilo is registered but has no runs in
this export. 90 of the 160 runs with camera checks are test runs.

## Frames per check and robot

| Check | maus-quin | maus-otto | maus-pixi | Total |
|---|---:|---:|---:|---:|
| plate_present | 4 | 11 | 64 | 79 |
| spread | 99 | 7 | 57 | 163 |
| fold1 | 98 | 7 | 33 | 138 |
| fold2 | 97 | 7 | 31 | 135 |
| push_off | 187 | 14 | 58 | 259 |
| plate_clean | 99 | 10 | 73 | 182 |
| plate_loaded | 113 | 5 | 54 | 172 |
| **Total** | **697** | **61** | **370** | **1,128** |

A frame is one check attempt. Checks repeat within a run (push_off retries
by design), which is why push_off has the most frames. See
[frames](../vision/frames.md) for the table and its labels.

## Label support

**Fold class** comes from passed spread and fold checks:

| Class | maus-quin | maus-otto | maus-pixi |
|---|---:|---:|---:|
| flat (spread passed) | 92 | 1 | 32 |
| half (fold1 passed) | 94 | 6 | 19 |
| quarter (fold2 passed) | 83 | 1 | 17 |

**Failure reasons** come from failed frames. A frame with several errors
counts once per error:

| Check | Robot error | Train | Test |
|---|---|---:|---:|
| spread | crepe too small | 14 | 4 |
| spread | large hole in the centre | 7 | 2 |
| spread | large hole off-center | 8 | 2 |
| spread | batter left on the plate | 4 | 5 |
| spread | no crepe in the plate | 1 | 1 |
| fold1 | not a half fold | 10 | 4 |
| fold1 | wrong fold shape | 8 | 2 |
| fold1 | batter left on the plate | 6 | 0 |
| fold1 | no crepe detected | 1 | 0 |
| fold2 | not a quarter fold | 11 | 3 |
| fold2 | wrong fold shape | 19 | 4 |
| fold2 | batter left on the plate | 7 | 2 |
| fold2 | no crepe detected | 2 | 0 |
| push_off | crepe not aligned with the plate | 99 | 32 |
| push_off | serving plate not found | 6 | 0 |
| push_off | no crepe detected | 5 | 0 |

Most reasons have 5 to 20 examples. 131 of 140 failed push_off frames are
"crepe not aligned", so a push_off reason model mostly learns the majority.

## What the split covers

The split is `robot-chrono-v2` (#102): within each robot, the latest 20% of
its runs with camera checks are held out, and tied start times stay on one
side. Every robot with camera checks is therefore in both train and test.

| Robot | Train runs | Test runs | Train frames | Test frames |
|---|---:|---:|---:|---:|
| maus-quin | 66 | 17 | 570 | 127 |
| maus-otto | 8 | 3 | 46 | 15 |
| maus-pixi | 52 | 14 | 274 | 96 |

- **A test score now covers all three robots.** Each robot is tested on its
  own latest runs. Per-robot test scores are still small: 3 maus-otto runs
  tell you little on their own.
- **The previous split** held out the latest 20% of all runs with camera
  checks. Robot follows time, so its whole test set was maus-pixi (32 runs,
  192 frames), and a test score said nothing about maus-quin or maus-otto.
  Model cards trained before #102 cite that split.
- **Still not a fix for robot = era.** The split mixes robots, but each
  robot's runs still come from its own few days. Leave-one-robot-out checks
  (in the shared harness) test transfer to a robot the model has not seen.
- **Small reasons get thinner in test.** Some failure reasons now have 0 test
  frames (fold1 residue, push_off "serving plate not found"), so the table
  above decides which reasons can be scored at all.

## Where more data could come from

| Robot | Runs without camera checks | Window (UTC) | With a video on the dashboard |
|---|---:|---|---:|
| maus-quin | 301 | Oct 05 to Oct 22 | 301 |
| maus-otto | 23 | Oct 16 to Oct 22 | 23 |
| maus-pixi | 344 | Sep 02 to Oct 30 | 344 |
| **Total** | **668** | | **668** |

- **Video:** every run without camera checks still has a video on the
  dashboard. This export did not download them (`fetch --video`). An
  earlier fetch on another machine downloaded 815 of 828, see the
  [handoff](data-preparation-handoff.md).
- **Phases and labels:** frames at the spread and fold moments could be cut
  from these videos using the phase timings. The labels would then have to
  come from people, because these runs have no robot verdicts. The
  [phase labeling pilot](../labeling/phase-labeling-pilot.md) is the starting point.
- **New runs:** runs on all robots, interleaved in time, would separate robot
  from era. That is the biggest single improvement for every model here.
