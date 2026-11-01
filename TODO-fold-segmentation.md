# Fold-state segmentation TODO

- [x] Apply the known camera perspective correction to extracted frames.
  - [x] Implement a checked video-frame rectification primitive.
- [x] Extract check frames using phase-bound clock alignment.
  - [x] Implement phase clock bounds and check-event parsing.
- [x] Estimate a per-run plate ellipse.
  - [x] Implement ellipse detection and median aggregation primitives.
- [x] Segment crepes using empty-plate references.
  - [x] Implement reference selection and component-based segmentation.
- [x] Measure geometry and classify fold state.
  - [x] Implement geometry and ordered state-rule primitives.
- [x] Evaluate against check verdicts and produce aggregate metrics.
- [x] Analyze whole videos and evaluate sampled runs.
- [x] Build the video and synchronized timeline dashboard.
- [x] Document the method, limitations, schema and commands.
- [x] Validate with offline tests and the cached private export.
- [x] Run lint, test and docs-check, then update the PR.

- [x] Supervise dough dispensing from the video timeline.
  - [x] Each timeline row carries the active logged phase.
  - [x] `fold_dispense` summarizes ``spread_batter`` coverage (first, peak,
    settled final) and flags runs whose final coverage is below 40% of the
    plate or collapses below 70% of the measured peak.
  - [x] Per-run `summary.json` and the sampled live report carry the verdict.
  - [x] The dashboard run list shows each run's dough verdict.

- [x] Highlight all logged cooking phases, not just the fold.
  - [x] The API exposes every timed phase (dough, cooking, folding,
    cleaning, handover, discard, ...) instead of only `fold_carefully`.
  - [x] The dashboard chart draws a labelled phase strip with a legend so a
    check dot reads against the phase the robot was in.
  - [x] The per-run `timeline.png` marks the same phase bands.

- [x] Re-fit the state-rule thresholds for rectified top-down frames.
  - [x] Flat threshold 0.49 (median full crepe covers ~0.7 of the plate);
    the old 0.24 oblique threshold mislabeled 201 of 240 half frames.
  - [x] Drop the no-edge flat band-aid that hid half folds.
  - [x] Use the largest flat area of the run as the flat reference so an
    occluded check frame cannot shrink it; half at relative 0.32 or plate
    fraction 0.21.
  - [x] Add minimum-rectangle aspect and long-axis angle to the geometry.
  - [x] A half fold must be elongated (aspect >= 1.4) so a narrow-wedge
    quarter fold with a half-fold area is not called half.
  - [x] The long-axis angle is a rotation estimate present on every
    non-empty mask, complementing the sparse straight-edge and corner angles.
- [x] Result: 96.9% agreement on 988 passed checks (empty 96.0, flat 99.6,
    half 97.5, quarter 94.2), above the prototype's 94.6% on oblique frames.

- [x] Refine the half/quarter boundary zone with a small classifier.
  - [x] `fold_classifier`: numpy-only standardised logistic model over nine
    geometry features, shipped weights plus `fit_logistic` to retrain.
  - [x] Only acts in the zone (relative area 0.24-0.42) and only when the
    model is at least 0.25 from an even split; rule decision kept as
    `rule_state` for comparison.
  - [x] Applied in both the offline segmentation and the live video path.
  - [x] 10-fold cross validation on the 397 half/quarter check frames: rule
    96.5% pair agreement, model 98.5%.
- [x] Result: 97.8% agreement on 988 passed checks (empty 96.0, flat 99.6,
    half 97.9, quarter 98.1); quarter/half confusion fell from 17 to 8
    mislabels.

## Known follow-up

- The rectified-frame thresholds were fitted on the same export they are
  evaluated on. Validate them on a separate held-out dataset before any
  deployment.
- The boundary-zone classifier recovered most of the remaining quarter/half
  overlap (17 to 8 mislabels). The 4 remaining quarter and 4 half mislabels,
  the flip margin and the shipped weights were all chosen on the same export
  as the evaluation; validate them on held-out data before deployment.
- 12 passed plate-clean frames show 4-19% plate residue; the robot verdict
  disagrees with the visible mask. Keep as a known robot-misverdict case.
