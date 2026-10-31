# Per-robot image correction

This experiment estimates one mean image correction per robot from local check
images. It is meant to make downstream image datasets consistent enough for
quick model training. It is not physical camera calibration.

## Estimate one correction

Run from `code/` in the worktree. The data directory is the shared private
`data/` folder, not committed repo content.

```bash
pixi run estimate-correction \
  --data-dir /workspace/data \
  --robot maus-pixi \
  --out /workspace/data/rectification/corrections/maus-pixi.json
```

`--robot` accepts either the machine name (`maus-pixi`) or the numeric
`machineId`. Repeat for the local robots:

- `maus-otto`
- `maus-pixi`
- `maus-quin`

The current local export has no `maus-nilo` runs. Estimate it when local raw
runs and media for that robot exist.

By default the estimator tries the `spread`, `fold1`, `fold2` and `push_off`
vision checks. Use repeated `--phase` flags to restrict this, for example:

```bash
pixi run estimate-correction \
  --data-dir /workspace/data \
  --robot maus-pixi \
  --phase spread \
  --out /workspace/data/rectification/corrections/maus-pixi-spread.json
```

The script matches raw check frames to their overlays using the validated
feature-matching homography estimator. Every accepted pair contributes one
normalized 3x3 matrix. The output JSON stores:

- `matrix_mean`: element-wise mean homography to apply.
- `matrix_stddev`: element-wise standard deviation across accepted pairs.
- `corner_stddev_px`: standard deviation of the four transformed source-image
  corners in output pixels.
- `corner_displacement_mean_px` and `corner_displacement_max_px`: how far each
  accepted estimate moves from the group mean at each corner.
- `estimates`: accepted per-image transforms and validation metrics.
- `rejections`: attempted pairs that failed validation.

Use the standard deviations as a camera-stability warning. High spread means
that one robot may have multiple camera positions, crops, resolutions, or bad
matches. In that case split by time, resolution, or camera setup instead of
using one mean correction.

## Apply a correction

Apply the mean correction to one image or a directory of images:

```bash
pixi run apply-correction \
  --data-dir /workspace/data \
  --correction /workspace/data/rectification/corrections/maus-pixi.json \
  --input /workspace/data/raw/media/3123 \
  --out /workspace/data/rectification/corrected/maus-pixi/3123 \
  --write-valid-mask
```

The output directory must be inside the selected private `data/` directory and
outside `data/raw/`. Images are written as PNG files. With `--write-valid-mask`,
each corrected image also receives a sibling `*_valid.png` mask showing pixels
supported by the source image.

The apply script refuses to overwrite existing files. Use a new output folder
for a new experiment.

## Review best and worst cases

Create contact sheets for visual inspection:

```bash
pixi run review-correction \
  --data-dir /workspace/data \
  --correction /workspace/data/rectification/corrections/maus-pixi.json \
  --out /workspace/data/rectification/review/maus-pixi
```

The review script writes `best.jpg` and `worst.jpg`. Each row shows the raw
frame, the mean-corrected frame, the robot overlay target and an absolute
pixel-difference heatmap. Best and worst are ranked by maximum corner
displacement from the mean transform, which highlights camera drift or bad
matches.

## Limits

Averaging homographies is an engineering shortcut. It can be useful when one
robot has a stable camera and repeated estimates agree, but it does not correct
lens distortion, parallax, plate height differences, or camera moves. Inspect
corrected images and validity masks before using them for model training.
