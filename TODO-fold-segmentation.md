# Fold-state segmentation TODO

- [ ] Apply the known camera perspective correction to extracted frames.
- [ ] Extract check frames using phase-bound clock alignment.
  - [x] Implement phase clock bounds and check-event parsing.
- [ ] Estimate a per-run plate ellipse.
  - [x] Implement ellipse detection and median aggregation primitives.
- [ ] Segment crepes using empty-plate references.
  - [x] Implement reference selection and component-based segmentation.
- [ ] Measure geometry and classify fold state.
  - [x] Implement geometry and ordered state-rule primitives.
- [ ] Evaluate against check verdicts and produce aggregate metrics.
- [ ] Analyze whole videos and evaluate sampled runs.
- [ ] Build the video and synchronized timeline dashboard.
- [ ] Document the method, limitations, schema and commands.
- [ ] Validate with offline tests and the cached private export.
- [ ] Run lint, test and docs-check, then open the PR.
