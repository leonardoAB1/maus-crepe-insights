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
- [ ] Run lint, test and docs-check, then update the PR.

## Known follow-up

- The reference thresholds classify many rectified half-fold frames as flat.
  Validate thresholds on a separate held-out dataset before any deployment.
