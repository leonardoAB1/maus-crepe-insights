# Experimental fold-state segmentation

## Website model switch

Prepare both segmentation backends and open the review site:

```bash
cd code
pixi run -e ml fold-watch 1234 --model=both --data-dir "$DATA"
```

The **Segmentation** selector switches between **Classical CV** and **ONNX
student**. It switches the video, timeline and check estimates together,
keeping the same run and playback position when both analyses exist. Missing
analyses are explicitly unavailable; the website never silently substitutes
another backend or runs expensive inference in an HTTP request.

Classical remains the default and writes `fold_segmentation/live/<run>/`.
ONNX writes `fold_segmentation/live-onnx/<run>/`; `--replace` affects only the
requested backend, or both when `--model both` is selected. Use
`fold-experimental-live 1234 --model onnx` to prepare just the ONNX result.
ONNX inference requires the `ml` environment; classical review does not.

The student sees the original raw BGR camera frame with its existing model
preprocessing. Its low-resolution binary mask is resized to the raw frame,
then the calibrated source size, using nearest-neighbour interpolation.
The same camera homography used for the classical image maps that mask to
the corrected view, again using nearest-neighbour interpolation. Unsupported
pixels and pixels outside the detected plate are excluded. Geometry,
causal mask smoothing and fold voting use those corrected coordinates.

This toggle compares segmentation backends, not two different fold state
machines. Both use the existing experimental geometry and fold rules; it
does not inherit the separate `fold-student` command's plate-free state
machine or its reported agreement. Plate detection is still required.
Confidence remains vote support, not calibrated ONNX confidence. ONNX masks
do not use the classical warmth or arm-occlusion gates; a tool hiding food
can change the detected area and fold estimate. No accuracy gain is implied
by changing the backend.

## Integration status

This offline analysis lives in `maus_export.experimental_fold`. It does not
replace the existing `maus_export.fold_*` pipeline or the run viewer.
Commands that overlapped now use the `fold-experimental-*` prefix;
`fold-watch`, `fold-live-eval` and `fold-review` remain unique review helpers.
Keep experimental outputs in a separate data directory because both
pipelines use `fold_segmentation/` for private artifacts. A symlink to the
existing `raw/` directory can share cached inputs without copying them.

This is an experimental inspection tool, not reliable detection or a robot
control component. Synthetic mask regressions and private video smoke tests
are not held-out accuracy. No independently labelled real-video benchmark
is available yet. Known limitations below remain release limitations.

This local, classical vision pipeline examines a hotplate video and compares
its measurements with robot checks. It is experimental and must not be used
for robot control or to decide whether food is safe. It does not communicate
with a robot or call dashboard write tools.

## Privacy

Video frames can show people. Frames, masks, contact sheets, per-run tables,
timelines and annotated videos stay in the private, gitignored `data/`
directory. The dashboard binds to `127.0.0.1` and has no authentication.
Do not expose it through a proxy or share its output. Close it when finished.

## Rebuild

From `code/` in the feature worktree, after `pixi install`, point these commands
at the main checkout's cached export:

```bash
DATA=/absolute/path/to/main/data
pixi run fold-experimental-frames --data-dir "$DATA"
pixi run fold-experimental-segment --data-dir "$DATA"
pixi run fold-experimental-eval --data-dir "$DATA"
pixi run fold-review --data-dir "$DATA"
pixi run fold-experimental-live 1234 --data-dir "$DATA"   # default 10 fps (200 ms/frame)
pixi run fold-experimental-live 1234 --fps 5 --data-dir "$DATA"
pixi run fold-live-eval --data-dir "$DATA"
pixi run fold-experimental-dashboard --data-dir "$DATA"

# One offline command: analyze cached videos, then open local review.
pixi run fold-watch 1234 --data-dir "$DATA"
pixi run fold-watch 1234 1235 --data-dir "$DATA" --replace --no-browser
```

`fold-watch` uses port 8766 by default (`--port` changes it), reuses
completed outputs unless `--replace` is supplied, and never downloads data
or authenticates. Ctrl-C stops review. Rebuild old outputs explicitly to
obtain the new foreground policy, verdicts and confidence fields.

The example run ID must be replaced by a locally cached run with a video. The
live evaluation processes 20 whole videos, so it can take a long time and use
substantial disk space. Open the local URL printed by the dashboard and select
an analysed run. Both the graph and check list seek the embedded video.
Each run in the side list also shows its dough-dispensing verdict
(`Dough: stable`, `Dough: flagged`, or `Dough: not measured` for runs
without a fresh `summary.json`).
The chart's top strip shows the logged cooking phases — dough
(`spread_batter`), cooking (`cook`), folding (`fold_carefully`), cleaning
(`wipe_spatula`) and handover (`plate_handover`) — so a check dot can be
read against the phase the robot was in. The chart's bottom band traces plate coverage throughout the video, with
missing or occluded measurements shown as gaps rather than zero. The dashed
40% line is a dispensing target only, not a fold threshold. The private
`timeline.png` uses the same full-duration coverage and check verdict labels.
The check list and graph show success/failure/unknown and vote confidence;
the run overview separates visual failures from robot-reported failures.

All outputs are under `$DATA/fold_segmentation/`. To avoid accidental loss,
commands refuse to overwrite their outputs. Move old generated files or use a
fresh private data directory before rebuilding. Frame extraction and mask
creation stage their output until complete. If an evaluation is interrupted,
remove only its incomplete generated evaluation files before rerunning it.

## Parallel batch processing and live progress

From `code/`, process every cached run/video pair with configurable concurrency:

```bash
pixi run fold-batch --threads 4 --data-dir /absolute/path/to/data
# Optional: reprocess completed runs at a different sampling rate.
pixi run fold-batch --threads 1 --fps 5 --replace --data-dir /absolute/path/to/data
```

Discovery includes numeric `raw/runs/<id>.json` files with a matching
`raw/media/<id>/video.mp4`. No download or sampled-evaluation restriction applies.
`--threads` defaults to 1 and controls concurrent videos, not frames within a
video. Each worker has its own decoder/encoder and frame buffers, so increase
concurrency cautiously. Timeline plotting is serialized for Matplotlib safety.

Runs with all five artifacts (summary, timeline CSV, observations CSV, annotated
video and timeline PNG) are skipped by default. Partial outputs are retried;
`--replace` recomputes completed outputs. Existing output is replaced only after
successful processing. Individual failures do not stop the batch; final totals
report completed/skipped/failed runs, and any failure gives exit code 1. Run only
one batch per data directory and avoid other workers processing the same run ID.

Keep `pixi run fold-experimental-dashboard --data-dir /absolute/path/to/data`
running in another terminal. The sidebar shows a loading circle on active runs.
Select a run, even before its output exists, to see its progress bar below the
video: sampled frame counts, completion percentage, frames/s, seconds/frame,
elapsed time and approximate ETA. Speed is end-to-end analysed-frame throughput,
not source FPS or inference-only timing. Final artifact writing is included in
elapsed time but not predicted by the ETA. Finished videos load automatically
without interrupting playback of other selected runs.

Processing failures move into the initially collapsed **Failed runs** section
at the bottom, with exception type and message. Older snapshots lacking a
message explicitly say so. Robot-reported failed checks do not count as runner
failures. Failed runs remain selectable; retried runs return to the normal list.

The read-only `/api/progress` endpoint polls atomic snapshots in private
`data/fold_segmentation/progress/<id>.json`. Writes are throttled to twice per
second, and the UI polls once per second. Snapshots survive exit and retries
replace them. Active updates older than 30 seconds are flagged as potentially
stopped; forced termination cannot write a terminal status.

Implementation validation: offline tests exercise bounded concurrent workers,
resume/retry/replace, failure isolation, clock-based throughput and ETA, HTTP
polling, and a 500-run sidebar with selection, failure grouping and completion
loading. The progress changes were ported onto the `experimental_fold` layout
without replacing the main fold pipeline or its viewer.

## Method

1. Resize frames to the source dimensions in `robot-image-calibration.json`
   and apply its per-robot mean homography through the existing rectifier. The
   aspect ratio must agree within 1%. These corrections are experimental mean
   image alignments, not physical calibration, and may drift between setups.
2. Parse `plate_clean`, `spread`, `fold1` and `fold2` log checks, retaining
   pass and fail verdicts and keeping check starts whose result message is
   unknown or missing. Intersect phase bounds `[timestamp - elapsed,
   timestamp - elapsed + 1 second)` on the naive Zurich local clock. Use the
   midpoint and add 0.1 seconds to each check start for video alignment. UTC
   run timestamps are used only for duration validation. Skip videos whose
   duration differs by more than two seconds. The robot verdict is a report,
   never ground truth, and the logs do not record when a fold movement starts
   or finishes.
3. Detect a Hough circle on the median-blurred grey image. Within a circle
   grown by 5%, fit a convex-hull ellipse to dark pixels after morphological
   opening. Reject implausible fits and fall back to the circle; shrink its
   axes to 94%. Use median ellipses from all detectable frames within each run,
   without selecting them by logged check state. Missing plates are reported, not silently counted as correct.
4. Inside the plate use grey > max(Otsu, 90), 7-pixel elliptical opening and
   closing, and connected components. Compare Lab lightness with another empty
   frame identified as empty by the initial visual mask, from the run or
   a compatible robot and frame width within three run IDs. Robot verdicts
   and expected states never select reference images. Match exposure using the median lightness
   difference in pixels below 80 in both images. Keep components with at least
   50% of pixels changed by more than 25. Without a reference, remove low-L,
   low-b glare. Remove components under 5% of the largest and masks under 3%
   of the plate. Missing dark pixels for reference matching are a failure.
5. Fill external contours to measure area, centre, equivalent diameter, plate
   offset, hole fraction, solidity and circularity. Approximate the largest
   contour at 1.5% of perimeter. Accept straight chords at least 18% of plate
   diameter, with RMS distance at most 2.5% of chord length and median rim
   distance at least 3% of plate diameter. Report edge angle and the
   intersection and angle of the two longest edges where defined.
6. Measure the minimum-area rectangle of the filled mask: its aspect ratio
   and long-axis angle. A half disc is about 2:1, a quarter fold and a full
   crepe are close to square, and the long-axis angle tracks fold rotation
   on the rectified top-down view with coverage on every non-empty mask.
7. Classify in order: no crepe is empty; plate fraction >= 0.49 or
   flat-reference fraction >= 0.75 is flat. Otherwise, flat-reference
   fraction >= 0.32 or plate fraction >= 0.21 is half, but only when the
   minimum-rectangle aspect is at least 1.4: a quarter fold folded into a
   narrow wedge can match a half fold area without being elongated.
   Anything smaller is quarter. The flat reference is the largest filled
   area among the run's other visually flat frames, so an occluded check frame cannot
   shrink it. A flat frame cannot use itself as its own reference. Straight
   edges do not change the decision: on rectified frames flat crepes often
   expose an edge and half folds often expose none. These thresholds
   describe the top-down view and were fitted on the local export.
8. In the boundary zone between half and quarter (relative filled area
   between 0.24 and 0.42, where the thresholds above overlap), a small
   standardised logistic model over nine geometry features (relative area,
   plate fraction, minimum-rectangle aspect, diameter and offset fractions,
   hole fraction, solidity, circularity, edge count) may flip the rule's
   half/quarter decision, and only when the model output is at least 0.25
   away from an even split: on the local export every correct flip clears
   that margin, so it costs nothing offline while rejecting low-confidence
   flips on noisy live frames. Outside the zone, on any non-fold state, or
   when a feature is missing, the rule's decision stands. The model is numpy-only
   (a few multiplications per frame) so it fits the 200 ms budget and can be
   swapped for an ONNX model later; `fold_classifier.fit_logistic` retrains
   the shipped weights. Cross validation on the 397 half/quarter check
   frames: rule 96.5% pair agreement, model 98.5%, with quarter recall
   94.6% rising to 97.9%. The rule decision is kept in the `rule_state`
   column for comparison.
9. Whole-video analysis sequentially grabs frames at 10 Hz by default (a 200
   ms/frame budget, `--fps` for other rates) and rectifies to 900 by 900
   pixels. The live adapter uses pixel-level diffuse foreground evidence
   rather than accepting entire bright components (see below). Bright narrow components connected to the
   image border and extending into the inner plate cause an occlusion
   abstention, not an empty prediction. This is conservative classical
   filtering, not general robot-arm recognition. Empty-reference learning
   uses at least three low-warmth frames before the dispensing stage, never
   predicted state labels. The flat-area reference is the median of up to
   nine recent visually flat frames, requiring three samples. It is updated
   after inference for subsequent frames and never consults log check names,
   expected states or robot verdicts. Log phases supply stage timing only.
   Each check collects visual predictions until one second after its logged
   start. A strict majority is the unverified estimate for comparison with
   the expected check state. Ties, fewer than three valid predictions, less
   than 60% valid samples or folds without a visual flat reference are unknown.
   The independent comparison verdict is success when visual and expected
   states agree, failure when they differ, and unknown on abstention.
   Confidence is winning votes divided by all samples, including invalid
   frames. It is vote support, not a calibrated correctness probability.
   Predictions and geometry do not consume these comparison verdicts.
   This is delayed, experimental video inference, not a control system.
   Each timeline row also carries the active logged phase, and the dough
   dispensing samples (every ``spread_batter*`` phase variant, e.g.
   ``spread_batter_concentric``) feed a coverage supervisor:
   first, peak and settled-final (median of the last second) plate coverage,
   plus an ``ok`` flag requiring at least 40% final coverage that keeps 70%
   of the peak, so a dough that collapses after spreading is flagged. The
   dough blob is rotation-invariant by construction because only its plate
   coverage is used. These limits are provisional supervisor values, not
   fitted ground truth.

## Outputs

- `frames.parquet`: one row per extracted check, with frame ID, run ID, robot,
  check, expected state, verdict, video seconds, source size and relative path.
- `frames/*.jpg`: corrected check frames. `masks/*.png`: binary 0/255 masks.
- `features.parquet`: frame columns, processing status, selected reference,
  mask path, plate and crepe areas, geometric measurements, edges, corner,
  flat-reference ID and predicted state. Failed processing has a status and is
  excluded from agreement; inspect the denominator in the report.
- `report.md` and `evaluation/*.parquet`: state confusion counts by robot,
  fold verdict counts, spread rank AUC, boundary-zone classifier flips and
  geometry coverage.
- `review/contact-sheet.jpg`: private representative check frames, prioritizing
  mismatches. It is not an exhaustive dataset review.
- `live/<run>/annotated.mp4`, `timeline.csv`, `observations.csv`,
  `timeline.png` and, for newly processed videos, `summary.json`: private
  playback, per-frame processing status and per-check visual estimates with
  their status. `timeline.csv` includes the active logged phase per frame;
  `summary.json` includes the dough-dispensing coverage block (first, peak,
  settled final fraction and the ``ok`` flag).
- `live_checks.parquet` and `live_report.md`: fixed-seed, setup-balanced sample
  of 10 successful runs with all checks passed and 10 runs with a failed spread
  or fold check. Compares each check's visual estimate with the expected state
  and reports the robot verdict separately; checks without an estimate are
  excluded from agreement. It also reports how many sampled runs end the
  dough-dispensing phase with a stable, adequately covered plate.

Rank AUC uses passed checks as positive, average ranks for ties and the stated
metric direction; a value below 0.5 indicates the opposite direction. No
threshold is chosen using the AUC.

## Pixel-level reflection rejection

The live pipeline no longer retains an entire bright connected component
merely because some of its pixels resemble food. `fold_diffuse.py` evaluates
warm diffuse evidence per pixel inside the plate: Lab L > 100, b >= 134 and
a <= 145 on OpenCV's byte scale. Saturated near-neutral pixels (L >= 245,
b < 140) are removed even when attached to dough.

With an empty reference, warm pixels need lightness more than 22 above the
exposure-matched highlight envelope, or a yellow-channel increase over the
median reference greater than 10 with L > 125. Pale pixels with
100 < L < 245, 120 <= a <= 145 and b >= 120 can also qualify through the
positive lightness test. Without an empty reference the warm gate remains.

Food can make a known reflection darker rather than brighter. Pixels with
L > 125 that pass the warm or pale gate can therefore qualify through a
negative exposure-corrected lightness change greater than 22. A retained
component must have at least 50% direct positive-lightness or warm-colour
evidence, so standalone fading glare cannot create a food component.
Saturated near-neutral pixels remain excluded. Static warm rims are not
changed food. Exposure correction requires jointly dark pixels; missing
support causes abstention instead of inventing a shift. This uses a fixed
camera reference, not a plate-rotation estimate.

A 3-pixel opening and 7-pixel closing stabilize texture; saturated neutral
pixels are excluded again after closing. Components smaller than 3% of the
plate or 5% of the largest remaining component are removed. The existing
causal mask average and occlusion handling operate on this new mask.

This change applies to live video inference. The offline check-frame
`fold-experimental-segment` procedure remains a historical baseline and must not be used
to claim accuracy for the new foreground policy. Saturated or blue-lit dough
can still be rejected, and neutral or warm tools can resemble food. Dark
brown food may fragment. Camera white balance, novel glints, shadows and
highlight-envelope contamination remain limitations. No held-out accuracy
or Raspberry Pi timing improvement is assumed from these synthetic tests.

On 2026-11-01, two private videos produced 2,161 processed frames: 1,904
measured, 171 occluded and 86 temporal warm-up frames. Median desktop
processing times were 44.9 and 42.7 ms/frame. Initial empty-plate coverage
was already zero in the earlier comparison output and remained zero.
Cooking coverage changed under pixel pruning, but without annotated masks
that is not evidence of improved accuracy. Synthetic regression tests verify
connected neutral glare rejection and recovery of warm food over known glare.

## Initial cleaning/preparation highlight

If a video starts with an `init`, `prepare`, `check_serving_plate` prefix,
show a cyan **initial cleaning/prep (inferred)** interval from video start
until the first production phase, commonly `oil` or `spread_batter*`.
This appears in the event graph, clickable event list, video overlay and
private timeline plot. The API preserves the original logged phases and
provides a separate `highlights` list with `inferred_preparation` provenance.
No cleaning action is fabricated or counted as a passed visual check.
Videos beginning in cooking, folding or explicit cleaning get no inferred
preparation interval. No interval is invented without a known end time.

The timeline adds `initial_cleaning_prep`; summary fields include
`initial_cleaning_prep_end` and `foreground_method`. These are display and
provenance fields, never visual classifier inputs.

## Temporal masks and toppings markers

`add_topping*` phases appear as pink **toppings (log)** intervals, clickable
stage events and video labels. This marker comes only from stage timing. It
is not a visual food state, ingredient classifier or confirmation that topping
application succeeded. Fold predictions remain empty, flat, half or quarter.
Changing a cooking phase to a toppings phase does not change classification.

Live masks use a strict majority of the most recent five valid binary masks.
At least three frames are required at startup and after occlusions or failed
measurements. No future frames are read. At 10 fps this suppresses isolated
one-frame flashes but adds approximately 0.2 seconds of transition delay.
Occlusion resets history instead of carrying the last mask across a missing
measurement. `temporal_warming` means unknown, not an empty plate.

The live empty-reference dataset now retains up to 15 low-warmth, unoccluded
pre-dispensing frames at 225 by 225 pixels. At the first three samples and
every five accepted samples thereafter it rebuilds a median colour image
and a per-pixel 90th-percentile Lab lightness envelope. Reference matching
uses the greater of median and envelope lightness, so previously observed
bright glare is less likely to count as changed food. This remains causal,
with no robot verdict or expected-state input. No reference is invented when
the video has no usable pre-dispensing footage.

Each new `live/<run>/reference/` directory contains the exact small sample set
used, `median.png`, `highlight-l.png` and `manifest.json`. The manifest records
observed and used sample counts, readiness, percentile and sample size.
These private reference datasets can be inspected or rebuilt from the cached
video; they must never be committed or used as public test fixtures.

`timeline.csv` adds `raw_plate_fraction` (unaveraged foreground pixel coverage)
and `structural_prediction`. Existing `plate_fraction` remains filled contour
coverage from the averaged mask, so the two fractions differ both because of
smoothing and hole filling. The summary records the averaging window and
`toppings_stage_logged`, not toppings detection or topping success.

A bright reflection outside the sampled envelope can still pass; suppressing
highlights can hide pale dough over a bright plate patch without enough
connected direct evidence. Narrow-arm detection is unchanged. Averaging can
smear moving masks, and
three-frame reacquisition can reduce usable votes near a check. Inspect the
unknown denominator, not only agreement or confidence. Raspberry Pi runtime
and held-out detection quality have not been validated.

On 2026-11-01 a private 1,607-frame smoke test retained 1,435 measurements,
109 occlusions and 63 temporal warm-up frames. It accepted 68 empty-reference
frames and saved the 15 samples used by the last completed reference update.
Median desktop processing was 50.1 ms/frame with another batch active. The
local dashboard API exposed the toppings interval and full timeline. This
checks execution and output wiring, not held-out accuracy improvement.

## Historical metrics snapshot (before cleanup)

These results describe the original reference selection, which consulted
log labels offline. They are not validation of the independent cleanup
pipeline. Rebuild and re-evaluate before comparing aggregate agreement.

On 2026-11-01, using the cached export and the method above:

- 285 runs, 1,182 extracted frames; 158 videos failed the duration check and
  one check fell outside its video. Eight frames lacked a plate estimate.
- On 988 usable passed checks, agreement was 97.8%, versus the prototype's
  94.6% on 991 frames. Empty 96.0%, flat 99.6%, half 97.9%, quarter 98.1%.
  The thresholds were refit for the rectified top-down view, where a full
  crepe covers about 0.7 of the plate, a half fold about 0.3 and a quarter
  fold about 0.17. The old oblique-frame flat threshold of 0.24 sat below the
  median half-fold fraction of 0.318 and mislabeled 201 of 240 half frames.
  Requiring an elongated minimum-rectangle aspect (>= 1.4) for half folds
  separated narrow-wedge quarter folds from true half folds, raising quarter
  agreement from 87.9% to 94.2%. The boundary-zone classifier then reduced
  the quarter/half confusion from 17 mislabels to 8 (quarter 94.2% to
  98.1%, half 97.5% to 97.9%, overall 96.9% to 97.8%). The remaining errors
  and the 12 empty frames with visible residue (4 to 19% of the plate) are
  area overlaps and robot misverdicts, not threshold gaps. Agreement stayed
  within a point when the flat, half and relative thresholds moved by 0.04,
  0.02 and 0.04 respectively, and per-robot agreement was 97.7% (maus-pixi),
  97.8% (maus-quin) and 100% on the small maus-otto sample. All numbers are
  tuned on this same export and need held-out validation before deployment.
- Spread-verdict rank AUC was 0.653 for diameter fraction and 0.651 for offset
  fraction, versus the prototype best of 0.66. Centre coverage was 853/1,174,
  edge-angle coverage 531/1,174 and quarter-corner coverage 109/276.
- In the sampled check-synchronized evaluation, 80/82 logged checks across 20
  runs produced an estimate; agreement with the expected state was 91.2%
  (empty 81.8%, flat 95.0%, half 100.0%, quarter 89.5%), with 97.6% on the
  ten all-passed runs and 84.6% on the ten runs with a failed check. Fold
  checks without a flat reference are reported unknown rather than guessed.
  The boundary-zone classifier changed no live estimate on this sample: its
  one would-be low-confidence flip (a quarter crepe at 20.9% plate fraction)
  is rejected by the 0.25 margin, so live agreement is unchanged while the
  single-frame offline evaluation improves. On the same host the median of
  per-run median processing times was 8.3 ms/frame at 10 fps, far below the
  200 ms/frame budget; this timing excludes video decoding, encoding and
  plotting. Raspberry Pi timing was not measured. The dispensing supervisor found 19 of 20 sampled runs ending
  with a stable dough coverage (median final 70% of the plate); the single
  flagged run is a genuine fault, logged ``not_succeeded``, whose dough never
  left a small blob and then disappeared.

## Cleanup validation (2026-11-01)

Synthetic tests alter check names, expected states and robot verdicts while
holding images fixed. Live predictions and offline masks, geometry and
references remain unchanged. Logs only supply timing/stages and independent
comparison labels. Visual comparison verdicts never feed back into inference.

A private 554-frame smoke test produced 492 usable measurements and 62
occlusion abstentions, with median processing time 41.3 ms/frame on the
local desktop. This is not a held-out quality metric or Raspberry Pi timing.
The original generated artifacts were preserved in a separate data directory.

## Limits

The live foreground gate can miss saturated or heavily browned dough. A coloured tool
can still resemble food, and an arm merged into a broad foreground component
may evade the narrow-object test. The adapter must be validated on held-out
videos and replaced if these assumptions do not hold. Never treat unknown
measurements as an empty plate or a successful check.


Browned crepes can break into patches. A spatula or arm can look like a crepe
in individual frames. Angles are in the corrected image, not calibrated robot
coordinates. An incorrect flat reference can bias subsequent classifications.
Camera movement or bad mean homographies can invalidate plate estimates.
Contact sheets and videos may show people. Thresholds and the boundary-zone
classifier weights were tuned on the same export used for evaluation, so
these numbers are not independent validation and may not generalize. Neither the robot verdict nor the visual estimate is
guaranteed correct; the two are independent and may disagree.
