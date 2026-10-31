# Private crêpe and hole masks

The mask project is separate from the phase project, in the same local Label
Studio app. Both reference the **same immutable rectified PNG files**, not
regenerated or resized copies. Setup does not read, copy or change phase
annotations. The image hashes join the two label sets, with run/frame provenance
to disambiguate identical images. Raw pilots are not accepted for mask projects.

## Set up

Use the existing app, pilot and owner account from the
[phase-labeling guide](phase-labeling-pilot.md). Keep the same absolute paths.
No new account, server, broader file root or storage sync is needed:

```bash
pixi run --locked --manifest-path "$SOURCE/code/labeling/pixi.toml" \
  setup --kind masks --pilot "$PILOT" --app-data "$APP_DATA" \
  --owner-email "$OWNER_EMAIL"
```

The result contains the new project ID. Refresh the project list in the app.
The project title starts with `Crepe masks v1`. Rerunning setup verifies the
existing project and preserves annotations. It refuses configuration/task drift.
Do not rename or reconfigure managed projects and do not click storage **Sync**.

## Annotation contract: `crepe-material-holes-v1`

- **`crepe_material`**: paint visible batter/crêpe pixels. Exclude holes,
  utensils, toppings that hide the crêpe, and other occlusions. Do not invent
  hidden material behind folds or tools. This is visible, not amodal segmentation.
- **`hole`**: paint visible bare hotplate enclosed inside the crêpe silhouette.
  Open gaps at a ragged outside edge are background, not enclosed holes. Shadows,
  glare, browning and toppings are not holes. Do not infer hidden holes.
- Select **`crepe_present`** when visible crêpe material is annotated. At least
  one material pixel must be present in the exported mask. Holes are optional.
- Select **`no_crepe`** only after reviewing an image with no visible crêpe.
  Leave both brush classes empty. This is an explicit reviewed negative, never
  inferred from a skipped or unlabelled task.

Use the brush and eraser, changing brush size and zoom as needed. Material and
hole masks must not overlap: erase holes out of the material brush region before
submitting. Choose the review state, then **Submit**. Use **Skip** for uncertain
boundaries, heavy steam, unclear occlusion or bad rectification. Review a small
shared sample before annotating a large batch. No phase or robot verdict is
displayed as a mask suggestion, and no predictions are created.

Experimental rectification stays visibly marked. Inspect the geometry before
painting. Validity masks represent unsupported warp pixels, **not** human hole
labels. Export ignores unsupported pixels rather than training on black borders.
No physical diameter, angle or area accuracy is implied by these masks.

## Private export for training

Use the **mask project's** Export action, choose **JSON** and do not download
resources. Save the file privately under the labeling directory. Then:

```bash
pixi run --locked --manifest-path "$SOURCE/code/labeling/pixi.toml" \
  export-masks --pilot "$PILOT" \
  --annotations "$PRIVATE/labeling/human-masks.json" \
  --output "$PRIVATE/labeling/exports/masks-v1"
```

Use a new output name for each export. Only submitted human annotations are
converted. Drafts, predictions, skipped and unlabelled tasks are not negatives.
Multiple submitted annotations require review before export. Preparation fields
such as `human_label=null` and `review_status=unreviewed` are provenance, not
current human labels.

The exporter verifies task provenance against the immutable pilot and rejects
rotated/resized regions, unexpected controls, malformed RLE, overlapping classes,
and incomplete review states. It never follows image URLs or changes images,
projects or the original annotation JSON.

For each reviewed frame it writes native-size, lossless PNGs:

- `*-material.png`: visible material, 0 or 255.
- `*-holes.png`: enclosed holes, 0 or 255.
- `*-labels.png`: 0 = other scene, 1 = material, 2 = hole, 255 = unsupported.

The private `manifest.json` records annotation IDs, image and transform hashes,
original frame timing, mask taxonomy and preprocessing version. Use the same
stored input PNG for training. Exclude label 255 from losses and measurements;
keep train/test splits grouped by run. If resizing is needed for a model, apply
one recorded transform to image and masks, with nearest-neighbour mask sampling.

Keep annotation JSON, PNGs and manifests private under `data/`. Do not commit
them, publish them in PRs or share app databases. The privacy and backup rules
in the phase-labeling guide still apply.
