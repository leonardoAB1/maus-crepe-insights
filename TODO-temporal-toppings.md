# Temporal masks and toppings markers (#157)

- [x] Add causal five-frame averaging with occlusion resets.
- [x] Add a log-only toppings stage marker, not a visual toppings classifier.
- [x] Keep fold classification independent of topping labels and recipes.
- [x] Expand empty references to 15 samples with a highlight envelope.
- [x] Save a private reference dataset and manifest with each analyzed run.
- [x] Validate synthetic tests, formatting, lint and docs checks.
- [x] Run isolated cached-video validation without changing the active rebuild.
- [x] Document limits and aggregate experiment results.
- [x] Commit logical steps for references, averaging and stage markers.
- [ ] Merge after #156 and required CI checks pass.

## Validation (2026-11-01)

- 113 fold tests passed, 3 skipped; lint, format and docs checks passed.
- Private smoke test: 1,607 frames; 1,435 valid, 109 occluded and 63 warming.
- Reference dataset: 68 accepted frames observed, 15 retained samples from
  the most recent completed update at accepted sample 65.
- Median desktop processing was 50.1 ms/frame while another batch was active.
  Raspberry Pi performance and held-out quality remain unmeasured.
- API verified a logged toppings interval and full timeline on port 8769.
- The inherited full-suite vision-target subprocess stall is not resolved
  by this change; see the parent cleanup TODO.
- A pixel-coverage smoothing comparison is not an accuracy result, because
  the reported filled-contour coverage also differs from raw mask coverage.
  No held-out improvement is claimed.

## Decisions

- Per clarification, no appearance-change or ingredient classifier is added.
  `add_topping*` is labelled "toppings (log)" in the phase strip, video and
  clickable event list. It is not evidence of successful dispensing.
- Mask averaging is causal: five valid frames with strict majority, requiring
  three frames after startup or an occlusion. At 10 fps transition lag is
  about 0.2 seconds; timestamps are not shifted.
- Reference frames are selected before dispensing by low warm-pixel coverage
  and an unoccluded plate. They never use robot result or expected-state labels.
- Reference images are reduced to 225 px, retaining at most 15 samples.
  Every five accepted frames, rebuild the median image and the 90th-percentile
  Lab lightness envelope. Save the exact sample set used, not later samples.
- All images, reference datasets, videos and measurements stay in `data/`.
