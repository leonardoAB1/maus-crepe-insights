# Documentation

Start with [what works and what does not](status.md). It lists the capabilities,
the gaps and the numbers behind each claim. Everything below goes deeper, one
page per question.

## Data

- [Dashboard data](data/data.md): what the Maus dashboard records, seen with a
  hackathon account
- [MCAP format](data/mcap.md): one file per run, ROS 2 profile, topics and
  schemas
- [Pipeline and data model](data/pipeline.md): dashboard to JSON to MCAP and
  tables, plus the vision score columns

## Vision and frames

- [Frame-level check dataset](vision/frames.md): one row per camera check frame
  with weak labels and the fixed split
- [Frame embeddings](vision/frame-embeddings.md): the cached embedder output the
  classifier heads train on
- [Camera rectification](vision/rectification.md): the experimental upstream
  alignment, and its acceptance thresholds
- [Per-robot correction](vision/robot-correction.md): mapping the robots into one
  consistent image view
- [Live overlay](vision/live-overlay.md): drawing every layer onto a video or
  a stream, and what each source has to emit
- [Fold segmentation](fold-segmentation.md): the classical video pipeline for
  fold-state features, evaluation and a local review dashboard
- [Crêpe student](vision/crepe-student.md): the distilled crêpe mask model and
  the plate-free fold state, live and at the check moments
- [Run viewer](vision/review-app.md): a local web app that plays a run with the
  pipeline beside it, and how a model plugs in

## Labeling

- [Private phase labeling](labeling/phase-labeling-pilot.md): the local Label
  Studio pilot and how its frames are prepared
- [Two-person handoff](labeling/labeling-collaboration.md): splitting whole runs
  into set 1 and set 2, and starting an assigned set
- [Material and hole masks](labeling/mask-labeling.md): the brush contract for
  crepe material and enclosed holes
- [Mask assistance](labeling/mask-assist.md): preview, prepare and apply pixel
  suggestions to untouched tasks
- [Pixel mask suggestions](labeling/pixel-masks.md): what the suggestion pass does
  and which thresholds it exposes

## Training

- [Run-start dataset](training/training-dataset.md): candidates for predicting a
  run before it starts
- [Training data audit](training/training-audit.md): coverage of the training and
  vision data, aggregate only
- [Vision-target dataset](training/vision-targets.md): the vision measurements
  joined to per-run records
- [Classification data coverage](training/classification-data.md): why the frame
  models cite about 100 runs when the export has 828
- [Crepe state vocabulary](training/crepe-states.md): the six states used by
  labels, models and analysis
- [Deviation and fine-tuning](training/deviation-pipeline.md): comparing labels
  with predictions, and weighting a fine-tuning dataset
- [Data preparation handoff](training/data-preparation-handoff.md): what was
  prepared in the account-limited export, and how to reproduce it

## Analysis

- [Historical settings analysis](analysis/analysis.md): which recipe settings go
  with better crepes in the runs already made
- [Designed experiment](analysis/doe.md): which settings to change next, on a
  designed batch
- [Fold risk from the spread check](analysis/fold-risk.md): whether the spread
  frame predicts a later fold failure
- [Crepe position from synthetic images](analysis/synthetic-crepes.md): an
  unused experiment on whether generated images can replace hand labels

## System

- [Robot system](system/system.md): boards, camera, kiosk and cloud, and who
  decides what
- [Crepe stages](system/crepe-stages.md): the stages of a run, the checks between
  them, and how runs end

## Diagrams

Shared Mermaid sources are in [diagrams](diagrams/). Edit the `.mmd` file and run
`pixi run docs`; the copies inside pages are generated, and CI checks them.
