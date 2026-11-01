# Temporal masks and toppings markers (#157)

- [x] Add causal five-frame averaging with occlusion resets.
- [x] Add a log-only toppings stage marker, not a visual toppings classifier.
- [x] Keep fold classification independent of topping labels and recipes.
- [x] Expand empty references to 15 samples with a highlight envelope.
- [x] Save a private reference dataset and manifest with each analyzed run.
- [ ] Validate synthetic tests, formatting, lint and docs checks.
- [ ] Run isolated cached-video validation without changing the active rebuild.
- [ ] Document limits and aggregate experiment results.
- [ ] Commit logical steps and open a stacked PR on #156.

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
