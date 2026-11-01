# Fold review cleanup (#155)

- [x] Create an isolated branch stacked on branch 136 (PR #137).
- [x] Reject neutral reflections and narrow boundary-connected arm occlusions.
- [x] Learn empty references before dispensing, not from predicted states.
- [x] Remove log-label reference selection from live and offline inference.
- [x] Plot measured coverage for the whole video, with missing data gaps.
- [x] Add independent check verdicts and uncalibrated vote confidence.
- [x] Add `pixi run fold-watch` for offline analysis and dashboard playback.
- [x] Run synthetic tests, lint, format and docs checks.
- [x] Run a private video smoke test without replacing existing results.
- [x] Document limitations and aggregate experiment results.
- [ ] Resolve the inherited full-suite vision-target subprocess stall.
- [ ] Complete full held-out evaluation and Raspberry Pi timing.
- [ ] Merge after the parent PR and required CI checks pass.

## Experiments (2026-11-01)

- 108 fold tests passed, 3 skipped; lint, formatting and docs checks passed.
- Wider suite excluding `test_vision_targets.py`: 433 passed, 57 skipped
  before the last additional foreground regression test was added.
- Full-suite attempts stall in vision-target CLI subprocesses. Faulthandler
  traces and test logs stay in the shared private `data/` directory.
- A too-broad edge-connected occlusion rule rejected a complete video.
  Restricting it to narrow components prevents background false abstentions;
  synthetic regression coverage includes a broad bright background.
- Final private smoke test: 554 frames, 492 measured, 62 occluded,
  median desktop processing 41.3 ms/frame. No Raspberry Pi is available.
- Warm food can be missed when very pale or heavily browned. A coloured tool
  or an arm merged into a broad component may still evade rejection.
- Live and offline tests scramble check labels and verdicts while holding
  images fixed; predictions, geometry and references remain unchanged.
- A separate localhost dashboard serves the smoke test on port 8768.
  Earlier review on port 8766 and original generated results were preserved.

All generated artifacts and detailed measurements stay under the shared
private `data/` directory. No real images or per-run data are committed.
