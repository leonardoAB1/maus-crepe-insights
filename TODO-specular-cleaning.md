# Specular rejection and initial cleaning (#159)

- [x] Inspect private early empty-plate frames and false foreground.
- [x] Replace whole-component retention with pixel-level diffuse evidence.
- [x] Reject neutral highlights even when connected to food.
- [x] Preserve warm food over known glare using chromatic change.
- [x] Highlight initial cleaning/preparation with inferred provenance.
- [x] Keep original logged phases and visual predictions independent.
- [x] Test synthetic glare, connected glare, exposure and food regressions.
- [x] Compare private cached-video outputs and processing cost.
- [x] Document limits and commit logical steps.
- [ ] Merge after #158 and required CI checks pass.

## Validation (2026-11-01)

- 121 fold tests passed, 3 skipped; lint, format and docs checks passed.
- Wider suite excluding the inherited stalled vision-target tests:
  447 passed, 57 skipped.
- Two private cached videos: 2,161 frames, 1,904 valid measurements,
  171 occluded and 86 temporal warm-up frames.
- Median desktop processing times were 44.9 and 42.7 ms/frame.
- Dashboard JSON and video byte ranges passed on port 8770; inferred
  preparation highlights preserve the original logged phases.
- Early empty-plate measurements were already zero in the comparison
  output and remain zero. Reduced cooking coverage is not an accuracy
  metric; no held-out reflection-rejection improvement is claimed.
- Synthetic tests demonstrate the targeted connected-glare failure is
  removed while warm food over a known highlight remains detectable.
- Raspberry Pi timing and held-out quality remain unmeasured.
- All contact sheets and comparison artifacts remain private under data/.

## Findings

The inspected empty plate has both neutral bright glints and a warm rim.
Whole-component retention can keep glare connected to dough, even when only
part of the component is food. Initial log phases are preparation stages,
not an explicit hotplate-cleaning phase. The UI must not claim the robot
performed an unlogged cleaning action: label the inferred initial interval
as cleaning/preparation and keep its provenance visible.
