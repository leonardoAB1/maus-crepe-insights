# Frame-level check dataset

`data/training/frames.parquet` has one row per vision check attempt (one
camera frame), with weak labels taken from the robot's own verdicts and a fixed
holdout split. Every frame classifier trains and is scored on these rows and
this split, through the shared harness in `code/maus_classify/harness.py`.

The labels are the robot's recorded check results, not human ground truth. A
`fail` can be a robot-side false alarm, and a `pass` can hide a defect the check
does not look for.

How many runs and frames each robot and model really has, and why most of the
828 runs have no frames, is on
[classification data coverage](../training/classification-data.md).

## Build

From `code/`, after a fetch with `--media`:

```bash
pixi run frames
pixi run maus-classify --data-dir /path/to/data frames
```

The command reads `raw/runs/*.json` and `raw/media/`, and writes
`training/frames.parquet` atomically, so a failed build keeps the old file. It
prints counts only.

## Columns

| Column | Type | Meaning |
|---|---|---|
| `run_id` | int64 | Dashboard run id |
| `check` | string | Check name (`visionChecks[].phase`), e.g. `spread`, `push_off` |
| `check_index` | int32 | Position in `visionChecks`; the `NN` in `NN_<check>_frame.jpg` |
| `attempt` | int32 | 1-based count of this check within the run |
| `ts_utc` | timestamp UTC | Check time, converted from robot-local time |
| `robot` | string | `machineName` |
| `ok` | bool | Robot verdict |
| `message` | string | Robot message, as recorded |
| `errors` | list of string | Robot error strings, as recorded |
| `frame_path` | string | `raw/media/<run_id>/<frame>` relative to the data dir; null if the file is missing |
| `is_last_attempt` | bool | Last attempt of this check in the run (the rule `runs.parquet` uses) |
| `fold_class` | string | Shape label, see below |
| `reason` | string | Normalized failure reason, see below |
| `split` | string | `train` or `test`, see below |
| `split_version` | string | Split rule the row was built with (`robot-chrono-v2`) |

`(run_id, check, attempt)` is unique and is the join key for derived tables,
such as frame embeddings. Numbering follows the raw `visionChecks` order, so it
is stable for a given raw run file, and missing images still count.

## Label rules

- **`fold_class`**: a passed `spread` check means `flat`, a passed `fold1` means
  `half` and a passed `fold2` means `quarter`. Every other row is null. A failed
  fold is not given a shape, because the robot only says the shape is wrong.
- **`reason`**: null when `ok`. Otherwise each error is lowercased, trimmed and
  whitespace-collapsed; duplicates are dropped and the rest are sorted and
  joined with `; `. A failed check with no errors uses its normalized
  `message`.
- **`ts_utc`**: check timestamps are naive robot-local time. They are converted
  with `maus_export.to_mcap.Clock`, which measures the offset from the run's
  first log line against the UTC `tsStart`. If a run has no logs, it treats them
  as Europe/Zurich time.

## Holdout split

The split is fixed in the table before any modelling, and models never choose
their own. Models must generalize across robots, so both sides mix every
robot (`maus_classify.split.robot_split`, version `robot-chrono-v2`).

- It is per run, so all frames of a run share one side.
- It is chronological within each robot, by run `tsStart` in UTC. Each
  robot's latest 20% of runs with any vision check form its `test` part
  (`maus_classify.split.TEST_FRAC`).
- A robot needs at least `MIN_RUNS_FOR_TEST` (5) runs to give any to test.
  Smaller robots stay in train only.
- The cutoff is set over runs with checks, whatever their verdicts, labels or
  features.
- Runs that start exactly at a robot's cutoff all go to `test`, so tied
  timestamps never straddle the split.

Each robot ran in its own time window, and robot also stands for era, recipe
and software version. The first version of the split (`v1`) held out the
latest 20% of all runs, which made the test part one robot only. The
per-robot split keeps "later data" within each robot while every robot is
scored.

## Harness

`train_and_evaluate(estimator, X, y, split, groups, *, name, feature_names,
out_dir, timestamps=None, robots=None, positive=None, sample_weight=None)`
(ml environment) takes any feature matrix, so it works the same for hand-made
image features and embeddings:

1. It checks that no run (`groups`) spans both splits and that features are
   finite.
2. It fits a clone of the estimator on `train` rows and scores `test` rows:
   balanced accuracy, macro F1, per-class precision, recall and support,
   confusion matrix, and one-vs-rest calibration (Brier score, ECE and
   reliability bins).
3. It exports with skl2onnx (opset 17, input `x`, outputs `label` and
   `probabilities`, zipmap off) and checks that onnxruntime matches the fitted
   model on the test rows.
4. It writes `model.onnx` and the `model.json` card (inputs and feature order,
   outputs and classes, train and test run id and date ranges, metrics,
   commit), following [models/README.md](../../code/models/README.md).
5. It adds `intervals`: 95% bootstrap intervals for balanced accuracy, macro
   F1 and, with `positive`, that class's recall. Whole runs are resampled,
   within each robot when robots are given.
6. With `robots` it also names the robots each side covers, adds per-robot
   holdout metrics and a leave-one-robot-out check (train on the other
   robots, score the held-out one), and a `generalization` verdict. A model
   generalizes only if every robot beats the majority baseline in both
   views.

For model and threshold selection, `cross_val_proba` returns out-of-fold
probabilities on the train part from time-ordered, run-grouped folds, cut
within each robot. `run_bootstrap` and `leave_one_robot_out` are generic, so
other models (fold risk) reuse them.

## Counts in the local export

These are aggregate counts from one private, account-limited local export as of
2026-10-31. Other exports will differ. In total there are 1,128 frames from 160
runs, 799 of which are last attempts, and 16 frames whose image file is
missing. Per-robot breakdowns are on
[classification data coverage](../training/classification-data.md).

| Check | pass | fail | train | test |
|---|---:|---:|---:|---:|
| `fold1` | 119 | 19 | 106 | 32 |
| `fold2` | 101 | 34 | 104 | 31 |
| `plate_clean` | 162 | 20 | 148 | 34 |
| `plate_loaded` | 134 | 38 | 143 | 29 |
| `plate_present` | 49 | 30 | 62 | 17 |
| `push_off` | 119 | 140 | 198 | 61 |
| `spread` | 125 | 38 | 129 | 34 |
| **total** | 809 | 319 | 890 | 238 |

Runs per robot (train / test): maus-quin 66 / 17 (2026-10-13 to 10-20),
maus-otto 8 / 3 (2026-10-22), maus-pixi 52 / 14 (2026-10-27 to 10-30).
maus-nilo has no runs with vision checks, so the data covers 3 robots. `push_off` fails by design while it retries
alignment, so most of its `fail` rows are retries, not bad crepes.

`fold_class`: 125 `flat`, 119 `half`, 101 `quarter`.

Failure reasons:

| Check | Reason | Frames |
|---|---|---:|
| `fold1` | not a half fold; wrong fold shape | 8 |
| `fold1` | not a half fold | 4 |
| `fold1` | batter left on the plate | 4 |
| `fold1` | batter left on the plate; not a half fold; wrong fold shape | 2 |
| `fold1` | no crepe detected | 1 |
| `fold2` | not a quarter fold; wrong fold shape | 11 |
| `fold2` | wrong fold shape | 10 |
| `fold2` | batter left on the plate | 7 |
| `fold2` | not a quarter fold | 2 |
| `fold2` | no crepe detected | 2 |
| `fold2` | batter left on the plate; wrong fold shape | 1 |
| `fold2` | batter left on the plate; not a quarter fold; wrong fold shape | 1 |
| `plate_clean` | batter left on the plate; substantial leftover | 17 |
| `plate_clean` | batter left on the plate | 3 |
| `plate_loaded` | plate not loaded | 38 |
| `plate_present` | no plate in the slot | 30 |
| `push_off` | crepe not aligned with the plate | 131 |
| `push_off` | serving plate not found | 4 |
| `push_off` | no crepe detected | 3 |
| `push_off` | no crepe detected; serving plate not found | 2 |
| `spread` | crepe too small | 13 |
| `spread` | batter left on the plate | 7 |
| `spread` | large hole in the centre; large hole off-center | 5 |
| `spread` | large hole off-center | 4 |
| `spread` | batter left on the plate; crepe too small | 2 |
| `spread` | crepe too small; large hole in the centre | 2 |
| `spread` | large hole in the centre | 2 |
| `spread` | no crepe in the plate | 2 |
| `spread` | crepe too small; large hole off-center | 1 |

Small cells are too few to train on alone. Merge rare reasons, or use them for
inspection only.
