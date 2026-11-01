# Specular rejection and initial cleaning (#159)

- [ ] Inspect private early empty-plate frames and false foreground.
- [ ] Replace whole-component retention with pixel-level diffuse evidence.
- [ ] Reject neutral highlights even when connected to food.
- [ ] Preserve warm food over known glare using chromatic change.
- [ ] Highlight initial cleaning/preparation with explicit inferred provenance.
- [ ] Keep original logged phases and visual predictions independent.
- [ ] Test synthetic glare, connected glare, exposure and food regressions.
- [ ] Compare private cached-video outputs and processing cost.
- [ ] Document limits, commit steps and open a stacked PR on #158.

## Findings

The inspected empty plate has both neutral bright glints and a warm rim.
Whole-component retention can keep glare connected to dough, even when only
part of the component is food. Initial log phases are preparation stages,
not an explicit hotplate-cleaning phase. The UI must not claim the robot
performed an unlogged cleaning action: label the inferred initial interval
as cleaning/preparation and keep its provenance visible.
