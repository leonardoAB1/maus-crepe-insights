# Label versus prediction: deviation score and fine-tuning dataset

This pipeline measures how far a model's crepe-material prediction is from the
reviewed human mask, on the same pixels, and turns that measurement into a
weighted dataset for fine-tuning. It runs offline, on private data, and never
touches the dashboard or a labeling database.

Only the **material** class is compared. The hole class is deliberately left
out until the material class is stable.

## Why rectification

Human masks are painted on the frames of a labeling pilot. A model may see a
different camera view, a different crop or a rectified frame. Comparing a mask
and a prediction without a shared coordinate space measures the viewpoint, not
the model.

The pipeline therefore rectifies both sides through the repository's own
rectification transform (`maus_export.rectification`, schema version 1, the
`upstream-rectified-v1` contract used to prepare the labeling pilots), and
restricts every metric to pixels the warp fully supports. A warp that leaves a
border unsupported simply excludes that border from the comparison instead of
comparing invented pixels.

## Pipeline

```
labeling mask export ---\
                        >-- rectify (one transform per run) --> compare --> report
model predictions  ----/                                              \--> deviation.parquet
                                                                       \--> finetune-manifest.json
                                                                       \--> finetune.parquet
```

## Inputs

**Human side**: an export directory produced by the labeling mask export, with
`manifest.json` and one `<task>-material.png` per reviewed frame. Each record
carries `run_id`, `decoded_frame_index`, `requested_seconds` and the image hash,
which together identify a frame across machines. Local task and annotation ids
are not used as identities.

**Model side**: a prediction index JSON next to the mask files:

```json
{
  "model": "crepe-material-unet-v0",
  "source": "local onnx run, rectified frames",
  "predictions": [
    {
      "run_id": 3350,
      "decoded_frame_index": 5,
      "requested_seconds": 25.0,
      "mask": "run-3350/000005.png",
      "state": "flat",
      "confidence": 0.83
    }
  ]
}
```

`state` is optional and is read in the unified vocabulary or in the model's own
labels. A missing `state` costs nothing in the mask comparison; it only leaves
that frame out of the state agreement.

**Transforms**: a directory of `<frame-directory>.json` files as written by the
rectified pilot preparation, plus the pilot itself so a run id can be mapped to
its frame directory.

## Metrics

| Metric | Meaning |
| --- | --- |
| `dice`, `iou` | pixel overlap of prediction and human mask |
| `precision`, `recall` | share of predicted pixels that are real, and of real pixels found |
| `area_ratio` | predicted area over human area, capped at 4 |
| `centroid_offset_px` | distance between the two mask centres |
| `boundary_mean_px`, `boundary_p95_px` | symmetric boundary distance in pixels |

## The deviation score

One number in `[0, 1]`, where `0` is a perfect match:

```
deviation = 0.50 * (1 - dice)
          + 0.25 * min(boundary_p95_px / 20, 1)
          + 0.15 * min(centroid_offset_px / 60, 1)
          + 0.10 * min(|area_ratio - 1| / 0.5, 1)
```

Every term is capped before weighting, so one extreme term cannot swamp the
score. Because it is a weighted mean, a total miss by a same-area mask lands
high but below `1.0`. A frame the model did not predict at all is scored `1.0`
by definition and counted separately, so a model cannot look good by silently
skipping the difficult frames.

The weights and scales are the module constants in
`code/maus_export/label_deviation.py`. Change them there, in one place, and say
so in the report: every report records the weights it used.

## Outputs

- `report.json`: counts, aggregate metrics, per-state breakdown, state
  agreement and confusion, worst frames, taxonomy version and the weights.
- `deviation.parquet`: one row per labelled frame with every metric, plus the
  private paths of the human mask and the prediction mask.
- `finetune-manifest.json`: the frames at or above the score threshold,
  with the identity and hashes needed to reproduce the comparison.

## From deviation to fine-tuning

The dataset builder turns the report into training material:

- **Weighted samples.** `weight = 1 + 2 * deviation`, capped at `3`, so a frame
  the model gets wrong counts up to three times as much as one it gets right.
- **Run-grouped split.** Whole runs go to `train` or to `eval`, assigned by a
  deterministic hash order with a fixed seed. Frames from one run are strongly
  correlated, so a random frame split would put near-duplicates on both sides
  and overstate accuracy.
- **Review queue.** Frames above the review threshold are marked, because a
  large deviation can mean the model is wrong or the human mask is.
- **Provenance.** The manifest records the counts, the runs in each split, the
  weight settings and the hash of the deviation table it read.

`finetune.parquet` holds private, machine-specific paths. The manifest is the
portable summary. Neither is committed to the public repository.

## Commands

```bash
pixi run maus-export label-states

pixi run maus-export label-deviation \
  --labels "$WORKSPACE/exports/masks-v1" \
  --predictions "$PREDICTIONS/predictions.json" \
  --pilot "$WORKSPACE/pilot" \
  --transforms "$RECTIFIED/transforms" \
  --out "$PRIVATE/deviation/v1"

pixi run maus-export finetune-dataset \
  --deviation "$PRIVATE/deviation/v1" \
  --labels "$WORKSPACE/exports/masks-v1" \
  --predictions "$PREDICTIONS" \
  --out "$PRIVATE/finetune/v1"
```

Use `--space native` for both sides when a prediction was made on the very same
frames the labels were painted on, which skips the warp. Using native space
across different viewpoints compares the camera, not the model.

## Limits worth remembering

- A mask comparison cannot see a wrong fold; it sees where material was
  painted. Keep state agreement in the report for that.
- The human mask is the reference, not the truth. A disagreement can be a
  labeling mistake, which is what the review queue is for.
- Rectification here is the repository's experimental upstream alignment, not
  physical camera calibration. Metric pixel distances inherit that.
- Nothing in this pipeline proves a model will improve. It tells you which
  frames disagree, by how much, and by what shape.
