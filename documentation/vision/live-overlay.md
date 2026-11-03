# Live overlay: video or stream in, annotated video out

One program that takes a recorded run or a live camera stream and returns the
same video with everything the pipeline knows drawn on top. Today the same
knowledge sits in a table, a mask, a model card and an untouched video file.

The renderer exists in `code/maus_export/overlay.py` and is tested offline. It
draws nothing on its own: layers arrive as records. As of today the human masks,
the labeling export and the model predictions are the parts still to be wired in,
and nothing is wired to a live stream yet.

Live crêpe masks on video already exist elsewhere: the fold video analysis
draws the classical or student mask on every sampled frame into
`annotated.mp4`, and both review sites play it (see
[vision pipeline](vision-pipeline.md#6-video-timeline)). This renderer is
the generic path for drawing every layer at once.

## Run it

```bash
# a recorded run, with or without records
pixi run maus-export overlay \
  --input "$PRIVATE/run.mp4" \
  --records "$PRIVATE/run.records.jsonl" \
  --masks-dir "$PRIVATE/masks" \
  --run-id 3350 \
  --output "$PRIVATE/run-overlay.mp4"

# a camera or stream index, `device:0`
pixi run maus-export overlay --input device:0 --output "$PRIVATE/live-overlay.mp4"
```

The output is an `.mp4` plus a `<name>.records.jsonl` sidecar holding exactly the
records that were drawn, so a rendered clip can be audited without the source
data. Existing outputs are never overwritten, and masks are only read from inside
`--masks-dir`.

## The record

One JSON object per frame:

```json
{
  "version": 1,
  "run_id": 3350,
  "decoded_frame_index": 145,
  "requested_seconds": 25.0,
  "state": "flat",
  "state_source": "human",
  "material_mask": "run-000/000005-material.png",
  "hole_mask": null,
  "prediction_state": "flat",
  "prediction_mask": null,
  "deviation": 0.14,
  "check": "spread",
  "verdict": "pass",
  "phase_context": "init -> oil -> spread"
}
```

Every field except the frame identity is optional, and states are read in the
[unified vocabulary](../training/crepe-states.md): the unified
`emptyplate`, `flat`, `cooking`, `halffold`, `quarter fold`, `ugly`. An older
name such as `circle` is normalised, and an unknown one is rejected rather than
drawn as text nobody can interpret.

## Layers and draw order

1. Material masks as a translucent green tint with an outline, the same colour
   the labeling project uses, so a reviewer reads both the same way.
2. Hole masks in red, the same convention.
3. Prediction outlines in blue, drawn over the human tint so the two can be
   compared in one frame.
4. A text banner: frame identity and offset, the human state with its source,
   the model state and the deviation score, the check and its recorded verdict,
   and the recorded phase sequence marked `UNALIGNED`.
5. Notes carried by the record, for example a review-queue marker.

A frame with no record still renders, with a banner saying so. A model that
skipped the hard frames must not produce a silently shorter video.

## What each source has to emit

| Source | Emits | Status |
| --- | --- | --- |
| Labeling mask export | `state`, `material_mask`, `hole_mask` | to wire |
| Model heads | `prediction_state`, `prediction_mask` | to wire, no mask model yet |
| `maus_export.label_deviation` | `deviation` per frame | to wire, pipeline merged |
| Fold segmentation work | masks and states from video | to wire, in a draft PR |
| Robot checks and phases | `check`, `verdict`, `phase_context` | to wire |

A producer only writes records; the renderer never changes when a source lands.

## Not in this step

- No new model and no new labels.
- No robot commands, no dashboard writes.
- No publishing: input frames and rendered video stay in the private data
  directory.

## Verified

`code/tests/test_overlay.py` covers the record round trip, rejected states and
scores, JSONL parsing with its error messages, the sidecar, mask tinting against
the pixels inside and outside a mask, refusal to read a mask outside its
directory, and an end-to-end render of a synthetic clip read back frame by frame.
