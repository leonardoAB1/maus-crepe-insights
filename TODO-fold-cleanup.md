# Fold review cleanup (#155)

- [x] Create an isolated branch stacked on branch 136.
- [ ] Reject neutral reflections and boundary-connected arm occlusions.
- [ ] Learn an empty reference only before dispensing, not from predictions.
- [ ] Plot measured coverage for the whole video, with missing data gaps.
- [ ] Add independent check verdicts and uncalibrated vote confidence.
- [ ] Add one offline command for analysis and dashboard playback.
- [ ] Run synthetic tests, lint, docs checks and a private video smoke test.
- [ ] Document limitations and aggregate experiment results.
- [ ] Commit logical steps and open a stacked PR referencing #155.

## Experiments

All generated artifacts and detailed measurements stay in the shared private
`data/` directory. No Raspberry Pi hardware is available, so desktop timings
must not be presented as Raspberry Pi measurements.
