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

- [x] Re-fit the state-rule thresholds for rectified top-down frames.
  - [x] Flat threshold 0.49 (median full crepe covers ~0.7 of the plate);
    the old 0.24 oblique threshold mislabeled 201 of 240 half frames.
  - [x] Drop the no-edge flat band-aid that hid half folds.
  - [x] Use the largest flat area of the run as the flat reference so an
    occluded check frame cannot shrink it; half at relative 0.32 or plate
    fraction 0.21.
  - [x] Result: 95.9% agreement on 988 passed checks (empty 96.0, flat 99.6,
    half 98.8, quarter 87.9), above the prototype's 94.6% on oblique frames.

## Known follow-up

- The rectified-frame thresholds were fitted on the same export they are
  evaluated on. Validate them on a separate held-out dataset before any
  deployment.
- Quarter/half area overlap: 25 quarter frames read as half. Consider a small
  classifier over the existing geometry features for the boundary zone.
- 12 passed plate-clean frames show 4-19% plate residue; the robot verdict
  disagrees with the visible mask. Keep as a known robot-misverdict case.
