# Experimental fold-state segmentation

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
pixi run fold-frames --data-dir "$DATA"
pixi run fold-segment --data-dir "$DATA"
pixi run fold-eval --data-dir "$DATA"
pixi run fold-review --data-dir "$DATA"
pixi run fold-live 1234 --data-dir "$DATA"   # default 10 fps (200 ms/frame)
pixi run fold-live 1234 --fps 5 --data-dir "$DATA"
pixi run fold-live-eval --data-dir "$DATA"
pixi run fold-dashboard --data-dir "$DATA"
```

The example run ID must be replaced by a locally cached run with a video. The
live evaluation processes 20 whole videos, so it can take a long time and use
substantial disk space. Open the local URL printed by the dashboard and select
an analysed run. Both the graph and check list seek the embedded video.
The chart's top strip shows the logged cooking phases — dough
(`spread_batter`), cooking (`cook`), folding (`fold_carefully`), cleaning
(`wipe_spatula`) and handover (`plate_handover`) — so a check dot can be
read against the phase the robot was in. The per-run `timeline.png` plot
marks the same phase bands.

All outputs are under `$DATA/fold_segmentation/`. To avoid accidental loss,
commands refuse to overwrite their outputs. Move old generated files or use a
fresh private data directory before rebuilding. Frame extraction and mask
creation stage their output until complete. If an evaluation is interrupted,
remove only its incomplete generated evaluation files before rerunning it.

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
   axes to 94%. Use median ellipses from empty, half and quarter frames within
   each run. Missing plates are reported, not silently counted as correct.
4. Inside the plate use grey > max(Otsu, 90), 7-pixel elliptical opening and
   closing, and connected components. Compare Lab lightness with another empty
   frame from the run, preferring a passed check, or a compatible robot and
   frame width within three run IDs. Match exposure using the median lightness
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
   area among the run's other flat frames, so an occluded check frame cannot
   shrink it. A flat frame cannot use itself as its own reference. Straight
   edges do not change the decision: on rectified frames flat crepes often
   expose an edge and half folds often expose none. These thresholds
   describe the top-down view and were fitted on the local export.
8. Whole-video analysis sequentially grabs frames at 10 Hz by default (a 200
   ms/frame budget, `--fps` for other rates) and rectifies to 900 by 900
   pixels. It learns the plate, first empty reference and flat area from
   frames already observed. Classification is still a per-frame estimate, but
   no state is reported before a logged check starts. Each check collects the
   per-frame predictions until one second after its logged start; a strict
   majority is the unverified visual estimate for that check. Ties, fewer than
   three predictions, or fold checks with no flat reference yet are unknown.
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
  fold verdict counts, spread rank AUC, and geometry coverage.
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

## Local metrics snapshot

On 2026-11-01, using the cached export and the method above:

- 285 runs, 1,182 extracted frames; 158 videos failed the duration check and
  one check fell outside its video. Eight frames lacked a plate estimate.
- On 988 usable passed checks, agreement was 96.9%, versus the prototype's
  94.6% on 991 frames. Empty 96.0%, flat 99.6%, half 97.5%, quarter 94.2%.
  The thresholds were refit for the rectified top-down view, where a full
  crepe covers about 0.7 of the plate, a half fold about 0.3 and a quarter
  fold about 0.17. The old oblique-frame flat threshold of 0.24 sat below the
  median half-fold fraction of 0.318 and mislabeled 201 of 240 half frames.
  Requiring an elongated minimum-rectangle aspect (>= 1.4) for half folds
  separated narrow-wedge quarter folds from true half folds, raising quarter
  agreement from 87.9% to 94.2%. The remaining errors and the 12 empty frames
  with visible residue (4 to 19% of the plate) are area overlaps and robot
  misverdicts, not threshold gaps. Agreement stayed within a point when the
  flat, half and relative thresholds moved by 0.04, 0.02 and 0.04
  respectively, and per-robot agreement was 96.3% (maus-pixi), 97.1%
  (maus-quin) and 100% on the small maus-otto sample. All numbers are tuned
  on this same export and need held-out validation before deployment.
- Spread-verdict rank AUC was 0.653 for diameter fraction and 0.651 for offset
  fraction, versus the prototype best of 0.66. Centre coverage was 853/1,174,
  edge-angle coverage 531/1,174 and quarter-corner coverage 109/276.
- In the sampled check-synchronized evaluation, 80/82 logged checks across 20
  runs produced an estimate; agreement with the expected state was 91.2%
  (empty 81.8%, flat 95.0%, half 100.0%, quarter 89.5%), with 97.6% on the
  ten all-passed runs and 84.6% on the ten runs with a failed check. Fold
  checks without a flat reference are reported unknown rather than guessed.
  On the same host the median of per-run median processing times was 8.2
  ms/frame at 10 fps, far below the 200 ms/frame budget; this timing excludes
  video decoding, encoding and plotting. Raspberry Pi timing was not
  measured. The dispensing supervisor found 19 of 20 sampled runs ending
  with a stable dough coverage (median final 70% of the plate); the single
  flagged run is a genuine fault, logged ``not_succeeded``, whose dough never
  left a small blob and then disappeared.

## Limits

Browned crepes can break into patches. A spatula or arm can look like a crepe
in individual frames. Angles are in the corrected image, not calibrated robot
coordinates. An incorrect flat reference can bias subsequent classifications.
Camera movement or bad mean homographies can invalidate plate estimates.
Contact sheets and videos may show people. Thresholds were tuned on the same
export used for evaluation, so these numbers are not independent validation
and may not generalize. Neither the robot verdict nor the visual estimate is
guaranteed correct; the two are independent and may disagree.
