# Fold segmentation

This page documents the classical hotplate-camera pipeline for issue #130. It
turns cached run videos into fold-state features and aggregate reports. All
frames, masks, videos, contact sheets and per-run outputs stay under the local
`data/fold_segmentation/` directory and are not committed.

## Commands

Run from `code/` in the feature worktree. Use the shared private export from
the main checkout when rebuilding locally:

```bash
pixi run fold-frames --data-dir /absolute/path/to/main/data
pixi run fold-segment --data-dir /absolute/path/to/main/data
pixi run fold-eval --data-dir /absolute/path/to/main/data
pixi run fold-live --data-dir /absolute/path/to/main/data --ids 123 456
pixi run fold-dashboard --data-dir /absolute/path/to/main/data
```

The first three commands rebuild the check-frame dataset, features and report.
`fold-live` writes reviewed whole-video artifacts per run. `fold-dashboard`
creates a local `dashboard.html` that embeds the generated videos and lets a
reviewer jump in a video by clicking the timeline image.

## Method

### Perspective correction

Each decoded video frame is resized to the calibration source size and warped
with the per-robot homography in `robot-image-calibration.json`. The output is a
rectangular corrected image with the heating plate centered. If a robot has no
calibration entry, the code falls back to a centered 900 by 900 resize so tests
and new robots fail softly.

### Check-frame sampling

The frame extractor parses log pairs for `check_plate_clean`, `check_spread`,
`check_fold1` and `check_fold2`. A check passes when the following result line
starts with the expected robot verdict text. Failed checks are kept.

Phase timestamps bound the robot-local run-start clock to
`[timestamp - elapsed_sec, timestamp - elapsed_sec + 1 s)`. The extractor
intersects all phase bounds and uses the midpoint. Video time is then the check
log time minus that offset, plus 0.1 s. Runs whose video duration differs from
`tsEnd - tsStart` by more than 2 s are skipped.

### Plate ellipse

A Hough circle is found on the grey image with median blur 7, `dp=1.5`, Canny
threshold 100, vote threshold 40 and radius bounds from 0.28 to 0.60 of image
height. Inside that circle grown by 5 percent, dark pixels with HSV value below
70 are opened with a 7 px ellipse kernel. The largest component is converted to
a convex hull and fitted as an ellipse. The fit is accepted only when its centre
moves less than 0.3 circle radii and both axes are within 0.6 to 1.15 of the
circle diameter. Accepted axes are shrunk to 94 percent. The per-run plate is
the median of detected empty, half and quarter frames.

### Crêpe mask

Inside the plate ellipse, bright pixels use a threshold of
`max(Otsu threshold, 90)`, then an open and close with the 7 px ellipse kernel.
The preferred reference is a passed empty frame from the same run. If unavailable
at runtime, the batch segmenter searches a nearby run id within three ids that
has the same robot and frame width.

Exposure is matched by the median Lab-L difference over pixels darker than 80 in
both images. A pixel is changed when the exposure-corrected Lab-L delta is above
25. A candidate component is kept when at least 50 percent of its pixels changed.
Without a reference, dim yellow glare-like components with mean L below 180 and
mean b below 134 are dropped. Components smaller than 5 percent of the largest
component are removed. The result is empty if its area is below 3 percent of the
plate ellipse.

### Geometry

The largest external contour fills holes. The feature table stores centre,
equivalent diameter, diameter and offset relative to the plate, solidity,
circularity and hole fraction. Straight edges come from
`approxPolyDP(contour, 0.015 * perimeter)`. An edge counts if it is at least
0.18 plate diameters long, its contour points have RMS distance at most 2.5
percent of chord length and its median distance to the plate rim is at least 3
percent of the plate diameter. The table stores the leading edge angle in image
degrees from 0 to 180, plus the intersection and angle for two edges when found.

### State rule

The rule is synchronized to the known robot check moments:

1. No mask area means `empty`.
2. If a separate flat reference for the run exists, area at least 0.75 of that
   reference means `flat`.
3. Without a flat reference, area at least 0.24 of the plate means `flat`.
4. Without a flat reference, no straight edge and area at least 0.20 of the
   plate means `flat`.
5. With a separate flat reference, relative area at least 0.38 means `half`,
   otherwise `quarter`.
6. Without a flat reference, area at least 0.11 of the plate means `half`,
   otherwise `quarter`.

A flat frame is not used as its own flat reference for offline evaluation.

### Live inference

Whole-video inference reads frames sequentially with `grab` and `retrieve`, not
per-frame seeking. It analyses 10 frames per second, downscaled to a long side
near 900 px. A 9-frame majority vote feeds a state machine:
`empty -> flat -> half -> quarter`, with skipped folds allowed. Empty resets can
happen from any state, but empty must hold for 30 analysed frames. Only flat may
follow empty.

## Dataset schema

`frames.parquet` has one row per sampled check frame:

| Column | Meaning |
|---|---|
| `run_id` | Dashboard run id. |
| `check_name` | `plate_clean`, `spread`, `fold1` or `fold2`. |
| `expected_state` | `empty`, `flat`, `half` or `quarter` from the check name. |
| `check_ok` | Whether the robot result line matched the passed verdict. |
| `log_time` | Naive robot-local check start time. |
| `video_t` | Seconds from video start after clock synchronization. |
| `frame_path` | Local path under `data/` for the corrected JPEG frame. |
| `machine_name` | Robot name from the run metadata. |
| `terminal_status` | Run terminal status. |
| `width`, `height` | Corrected frame size. |
| `decoded_frame_index` | Source video frame index decoded by OpenCV. |

`features.parquet` joins the frame columns with plate, mask, geometry and state
columns such as `predicted_state`, `plate_cx`, `area_plate_share`,
`diameter_plate_share`, `offset_plate_share`, `solidity`, `circularity`,
`hole_fraction`, `edge_count`, `edge_angle_deg`, `corner_x` and
`corner_angle_deg`.

Metric tables and `report.md` are written to `data/fold_segmentation/metrics/`.

## Metrics snapshot

On the 2026-10-31 cached export, frame extraction produced 1,183 frames from 286
runs, with 991 passed-check frames. The classical pipeline reached 88.5 percent
agreement on those passed checks: empty 298 of 298, flat 246 of 246, half 155 of
240 and quarter 178 of 207. The spread rank AUC was 0.64 for diameter, 0.58 for
offset, 0.58 for hole fraction, 0.58 for circularity and 0.58 for solidity.
Geometry coverage was 854 frames with a centre, no accepted fold edge angle and
no quarter corner.

The prototype reference was 94.6 percent agreement on 991 passed-check frames,
with class agreements around 96 percent empty, 98 percent flat, 90 percent half
and 94 percent quarter. The lower fold agreement here is mostly from the
classical mask overestimating folded crêpe area on some oblique corrected views
and from the current straight-edge gate rejecting all fold edges. The repository
documentation only stores aggregate descriptions, not private per-run tables.

## Known limits

- Browned crêpes can break into several patches.
- The spatula and robot arm can look like crêpe for single frames.
- Angles are measured in the corrected image and are not yet a physical top-down
  plate coordinate.
- Thresholds are tuned on the same cached export used for evaluation.
- People can be visible in source frames, so generated visual artifacts stay
  local under `data/`.
