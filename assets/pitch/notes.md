# Maus Crepe Insights pitch: metrics and sources

BaselHack 2026, Maus Robotics challenge. Three slides, about three minutes.

## Talk track

1. Maus robots record run phases, checks, recipe settings and video. Maus Crepe
Insights turns those records into measurable signals for studying crêpe runs.

2. The private export contains 828 run records, each available as JSON and ROS 2
MCAP, plus a 177-field analysis table. Vision scores cover 138 runs. The robot's
check overlays give us spread diameter, area, centre offset, circularity,
holes, uncovered plate, residue and browning. Fold checks add ideal-shape
overlap, folded area and area ratio. Overlay pose extraction also writes centre
and axis-angle labels for each check attempt. On the local export, 257 push-off
overlays were processed and 248 passed the line-quality gate.

3. The model cards quantify several kinds of agreement. The segmentation
student's mask IoU against SAM 3 teacher masks is 0.966 on 225 held-out frames.
Against human masks, median IoU is 0.944 on 62 valid masks. Fold-state agreement
is 99.5% on 991 passed-check moments. Per-check verdict heads range from 0.67
to 0.92 balanced accuracy on small holdouts. These scores have different
references and sample scopes, so read the source and caveats alongside each.

Close: Explore the public code and bring authorized run data to compare batches
and plan the next experiment. Private Maus run data is not included.

## Numbers and sources

| Number | Value | Source |
|---|---|---|
| Run records | 828 | private `data/runs.parquet`, row count |
| Analysis fields | 177 | private `data/runs.parquet`, column count |
| Vision rows | 138 | private `data/vision.parquet`, row count |
| Push-off overlays | 257; 248 `lines_ok` | local `overlay_pose.parquet`, documented in `documentation/data/pipeline.md` |
| Student IoU vs teacher | 0.966 on 225 frames | `code/models/crepe-student/model.json` |
| Student median IoU vs human | 0.944 on 62 valid masks | student model card; five-run, one-robot pilot |
| Fold-state agreement | 99.5% on 991 passed-check moments | student model card; thresholds read from these frames |
| Check-verdict balanced accuracy | 0.67-0.92 on 31-61 frames | `code/models/check-verdict/*/model.json`; robot-verdict labels |

Metric definitions and commands: `documentation/data/pipeline.md`.
The private aggregate tables and human annotations are not shipped in Git.
