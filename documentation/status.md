# Project status: what works and what does not

As of 2026-10-31, against `main` at `48dd68b` plus the open pull requests
listed at the end. Every number below names the command or file that produced
it. Run data lives in a private checkout, so a fresh clone reproduces the code
and the offline tests, not the counts.

## Working today

| Capability | Entry point | Verified result |
| --- | --- | --- |
| Run export from the dashboard | `pixi run export` | 828 run JSONs and 828 MCAP files in the private data directory |
| Runs table | `pixi run table` | `data/runs.parquet`, 828 rows, 177 columns |
| Run outcomes in that table | `data/runs.parquet` | 470 succeeded, 148 not_succeeded, 129 aborted, 81 canceled |
| Vision scoring of check overlays | `pixi run vision` | `data/vision.parquet`, 138 rows, 22 columns |
| Run-start training dataset | `pixi run training` | `data/training/run_start.parquet`, 828 rows, 18 columns |
| Vision-target dataset | `pixi run vision-targets` | `data/training/vision_targets.parquet`, 828 rows, 37 columns |
| Coverage audit | `pixi run audit` | `data/training/audit.json`: 828 runs, 633 test runs, vision coverage 138 of 828 |
| One frame preprocessing contract | `code/maus_export/frame_preprocess.py` | one crop and resize, shared by training, export and inference |
| Frame embeddings | `pixi run embed` in the ml environment | `data/training/frame_embeddings.parquet`, input to the fold-class head |
| Per-check verdict model | `code/models/check-verdict` | holdout 31 to 61 frames per check, balanced accuracy: push_off 0.92, fold1 0.75, spread 0.70, fold2 0.67 |
| Fold-state model | `code/models/fold-class` | holdout 110 frames, balanced accuracy 0.99 over empty, flat, half, quarter |
| Failure-reason model | `code/models/failure-reason` | spread set-F1 0.69, fold2 set-F1 0.73 |
| Labeling projects, one app per person | `code/labeling` helpers | set 2: 525 phase tasks with 186 labels, 525 mask tasks with 97 submissions |
| Mask prefill suggestions | mask-assist helper | 301 editable predictions on set 2, applied only to untouched tasks |
| One state vocabulary for labels and models | `code/maus_export/crepe_states.py` | six states, legacy labels and model labels mapped on |
| Deviation and fine-tuning pipeline | PR #144 | 37 offline tests; wired end to end on real frames |

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

## In flight

- PR #144, unified state vocabulary plus the deviation and fine-tuning
  pipeline. Open for review.
- PR #137 and PR #133, classical fold-state segmentation from run videos.
- Issue #128, retrain fold risk and failure reason on the per-robot harness.
- Issue #126, enforce LF line endings.

## Reproducing a number

```bash
cd code
pixi run test          # offline suite, no dashboard needed
pixi run lint
pixi run docs-check
pixi run export        # needs dashboard access; writes into the private data/
pixi run table
pixi run vision
pixi run training
pixi run vision-targets
pixi run audit
pixi run -e ml maus-classify coverage
```

The table above is from this checkout's private data directory. A reader
without the data or a dashboard account can run the tests and the lint, and
can read every model card in `code/models/`, but cannot recompute the counts.

## Limits of this page

- It is tied to one date and one revision, and it will drift. When a number
  here disagrees with a command's output, trust the command and correct this
  page.
- It records what is measured, not what is promising. Items are listed as done
  only when a command or file shows the result.
