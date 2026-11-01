# Project status: what works and what does not

As of 2026-11-01, against `main` at `d1c3701` plus the open pull requests
listed at the end. Every number below names the command or file that produced
it. Run data lives in a private checkout, so a fresh clone reproduces the code
and the offline tests, not the counts.

## Working today

Every row below is built and runs. Where the result is weak or the work is
incomplete, the row says so, and the section after it names what is missing.

| Capability | Entry point | Verified result |
| --- | --- | --- |
| Run export from the dashboard | `pixi run export` | 828 run JSONs and 828 MCAP files in the private data directory |
| Runs table | `pixi run table` | `data/runs.parquet`, 828 rows, 177 columns |
| Run outcomes in that table | `data/runs.parquet` | 470 succeeded, 148 not_succeeded, 129 aborted, 81 canceled |
| Check records in that table | `data/runs.parquet` | 160 of 828 runs carry a check at all, and 117 of the 148 `not_succeeded` runs carry none, so most failures have no recorded reason |
| Vision scoring of check overlays | `pixi run vision` | `data/vision.parquet`, 138 rows, 22 columns, all from October 2026 |
| Run-start training dataset | `pixi run export training` | `data/training/run_start.parquet`, 828 rows, 18 columns |
| Vision-target dataset | `pixi run export vision-targets` | `data/training/vision_targets.parquet`, 828 rows, 37 columns |
| Coverage audit | `pixi run export audit` | `data/training/audit.json`: 828 runs, 633 test runs, vision coverage 138 of 828 |
| One frame preprocessing contract | `code/maus_export/frame_preprocess.py` | one crop and resize, shared by training, export and inference |
| Frame table and embeddings | `pixi run frames`, then `pixi run -e ml embed` | `data/training/frames.parquet` and `data/training/frame_embeddings.parquet`, the inputs the frame models train on. Both are rebuilt locally, so a checkout without them can score a model but cannot retrain one |
| Per-check verdict model | `code/models/check-verdict` | holdout 31 to 61 frames per check, balanced accuracy: push_off 0.92, fold1 0.75, spread 0.70, fold2 0.67. Three of the four cards report `generalizes: false`, and a logistic regression on the robot's own overlay scores beats the fold2 head, 0.88 against 0.67 |
| Fold-state model | `code/models/fold-class` | holdout 110 frames, balanced accuracy 0.99 over empty, flat, half, quarter. The labels are the check that took the frame, the card's own leakage probe fails, and holding maus-quin out drops quarter recall to 0.02 of 82 frames |
| Failure-reason model | `code/models/failure-reason` | one deployable head, the spread `too_small` class on 13 holdout frames. The fold2 heads are research only: their holdout holds a single class, where the trivial rule scores set-F1 1.00 against the 0.73 the card reports |
| Labeling projects, one app per person | `code/labeling` helpers | set 2: 525 phase tasks with 186 labels, 525 mask tasks with 113 submissions. Set 1 still holds zero labels |
| Mask prefill suggestions | mask-assist helper | 301 editable predictions on set 2, applied only to untouched tasks |
| One state vocabulary for labels and models | `code/maus_export/crepe_states.py` | six states, legacy labels and model labels mapped on |
| Video overlay renderer | `pixi run maus-export overlay` | draws masks, states, checks and deviation scores onto a recorded video or a camera index, with a JSONL sidecar of exactly what was drawn, and never overwrites an output. Only the records sidecar feeds it so far |
| Local run viewer | `pixi run maus-export view` | a loopback web app that plays a run with the mask suggestions and the frame record beside it. It names the producers it does not have, and no model answers yet |
| Fold segmentation from video | `pixi run fold-frames`, `fold-segment`, `fold-eval` | 1,183 frames from 286 runs, 88.5 percent agreement on the passed checks. Thresholds were tuned on the same export it is evaluated on |
| Deviation and fine-tuning pipeline | `code/maus_export/label_deviation.py` | deviation scoring and a weighted fine-tuning dataset, 27 offline tests. It has never compared a real prediction with a real reviewed mask |

## Not working, not finished or not verified

- **Vision measurements cover 138 of 828 runs, all from October 2026.** Any
  measurement built on them inherits that selection. The 690 runs without
  measurements are not evidence of anything.
- **Video to run phase alignment is unverified.** The recorded robot phase
  sequence is shown during labeling only as explicitly unaligned context. No
  per-frame phase label exists anywhere, so phase-level answers are not
  available from this repository.
- **The fold-risk model is not usable.** Balanced accuracy 0.29 on 20 holdout
  frames, which is worse than guessing. Treat it as an experiment, not a
  result.
- **The per-check verdict models miss failures.** Fail recall is 0.5 for fold1
  and 0.6 for fold2, on holdouts of 32 and 31 frames. A missed failure is the
  expensive error, and the small holdout means the intervals are wide.
- **The failure-reason model abstains often.** On the spread check it abstained
  on 10 of 13 holdout cases, and the fold2 card rests on 5 cases.
- **No reviewed material-mask export exists yet.** The mask exporter refuses
  any frame whose hole brush overlaps the material brush, because the classes
  must stay disjoint. The first export attempt was rejected for nine such
  frames and labeling has continued since, so the count has to be measured
  again before the next attempt.
- **The material-mask model does not exist.** The deviation pipeline in PR #144
  is complete and tested, but it has never compared a real prediction with a
  real reviewed mask. Its end-to-end check used stand-in masks and a shifted
  copy of the same mask.
- **Set 1 has never been opened.** The split and the pilot for set 1 exist; the
  app was never started and it holds zero labels.
- **Two state vocabularies are in use.** The labeling UI says `spreading` and
  `circle`, analysis and the deviation pipeline say `flat` and `cooking`, and
  the alias mapping bridges them. The migration is deliberate and pending.
- **Rectification is experimental upstream alignment, not camera
  calibration.** Lens distortion is not corrected, a planar homography cannot
  remove parallax, and metric pixel distances inherit that.
- **The overlay renderer exists but nothing is wired to it.**
  `code/maus_export/overlay.py` draws masks, states, checks and deviation
  scores onto a video and is covered by offline tests, but no producer feeds
  it yet and no live stream has been tested. See
  [live overlay](vision/live-overlay.md).
- **The run viewer shows what exists and nothing more.**
  `pixi run maus-export view` serves a video with the pixel-mask suggestion
  drawn on it and the record fields in the side panel, and it names the
  producers it does not have. No model answers yet. See
  [run viewer](vision/review-app.md).
- **No model has been trained on human labels.** Every model metric above comes
  from labels derived from robot check verdicts, not from the labeling work.
- **A fresh clone produces no artifacts.** Fetching runs needs dashboard access,
  and every frame, mask, label and table stays out of Git by design.

## Brief coverage

The challenge tasks, and what this repository has for each of them.

| Task | Asked output | In this repository | Verdict |
| --- | --- | --- | --- |
| 1, spread quality | crepe mask, diameter, quality score | per-run vision metrics over 138 runs (diameter, centre offset, circularity, holes, uncovered fraction), and a classical pixel-mask suggester inside the labeling tool. No spread model | partial |
| 2, position and orientation | centre (x, y) and angle, precise enough for the spatula | centre offset is a recorded metric; `code/maus_export/fold_geometry.py` computes an edge angle, but the straight-edge gate accepted none of the 854 measured frames, and the fold-class card lists the edge angle as out of scope | not delivered |
| 3, fold detection | flat, half, quarter or triangle, with confidence and edge angle | `code/models/fold-class`, four classes, weak labels. The triangle fold is unsupported by design | partial, triangle missing |
| Level 1, works on Maus data | model scored against the robot verdicts | the check-verdict heads, with agreement metrics on holdouts of 31 to 61 frames | partial |
| Level 2, real time on a Pi 5 | under about 300 ms per frame | 4 ms per frame on a desktop CPU; the Pi figure is an unmeasured extrapolation, and the arm64 environment lives in PR #162 | not demonstrated |
| Bonus, auto-labelling from video | | fold segmentation on `main` from PR #133, with the follow-ups in the draft stack below | in draft |
| Bonus, a spread score that predicts a failed fold | | `code/models/fold-risk`, balanced accuracy 0.29, and a permutation test that does not beat the null | attempted, negative result reported |
| Bonus, detect the crepe while the plate rotates | | no code tracks the rotating plate | not delivered |

## In flight

- PRs #137, #156, #158, #160 and #162, the classical fold-state segmentation
  stack and a minimal Raspberry Pi runtime, all in draft and 45 commits and 63
  files ahead of `main`. Landing them is the largest single item left.
- Issue #128, retrain fold risk and failure reason on the per-robot harness.
- Issue #126, enforce LF line endings.
- Issue #95, one inference entry point that runs every frame classifier.

## Reproducing a number

```bash
cd code
pixi run test          # offline suite, no dashboard needed
pixi run lint
pixi run docs-check
pixi run export        # needs dashboard access; writes into the private data/
pixi run table
pixi run vision
pixi run export training
pixi run export vision-targets
pixi run export audit
pixi run frames        # frame table, built from the cached runs and media
pixi run -e ml embed   # frame embeddings, needs the ml environment
pixi run -e ml maus-classify coverage   # needs frames.parquet from above
```

The first three need no data at all. Every `pixi run export ...` step needs
dashboard access and writes into the private data directory, and the two ml
commands need `pixi install -e ml` once. `pixi run coverage` fails with
"missing frames.parquet" until `pixi run frames` has run.

The table above is from this checkout's private data directory. A reader
without the data or a dashboard account can run the tests and the lint, and
can read every model card in `code/models/`, but cannot recompute the counts.

## Limits of this page

- It is tied to one date and one revision, and it will drift. When a number
  here disagrees with a command's output, trust the command and correct this
  page.
- It records what is measured, not what is promising. Items are listed as done
  only when a command or file shows the result.
