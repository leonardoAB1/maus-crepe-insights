# Run viewer: a video with the pipeline beside it

A local web app that plays a run video and shows what the pipeline knows about
the frame under the playhead. It is the place where the models and the visual
recognition work plug in.

```bash
cd code
pixi run maus-export view \
  --video "$PRIVATE/run-3349.mp4" \
  --records "$PRIVATE/run-3349.records.jsonl" \
  --masks-dir "$PRIVATE/masks" \
  --producer pixel_masks \
  --run-id 3349 --port 8087
```

It binds to loopback and prints its address. Nothing is uploaded, nothing is
written back, and the video, masks and records are read from where they already
are.

## The page

One screen, nothing scrolls: a title bar, the video, and a control deck in one
line under it.

- **The title bar**: the fold-state colours as a legend, and the frame counter
  in large type.
- **The video** in the middle, in a framed viewport, with a canvas overlay. Only
  the masks are drawn on it: material in a translucent green, holes in red, the
  colours the labeling project uses.
- **The control deck**, one line, left to right: the deviation score per frame
  as a strip with the review threshold, the fold states as a track with the
  playhead, then the readouts (state, model, deviation, check, recorded), then
  the producer and the mask switches.
- The track is clickable: click anywhere to jump there. The title bar and the
  deck name what the colours and the numbers are, so nothing needs a lookup.

The page is re-read on every request, so an edit to `viewer.html` shows up on a
refresh without restarting the app.

## What is drawn today

| Producer | What it gives | State of it |
| --- | --- | --- |
| `pixel_masks` | material and hole masks from the merged colour and connectivity pass, plus any record fields | working, and it says in the panel that it names no fold state |
| `records` | the fields and masks from a records sidecar, as written by `maus-export overlay` | working when a sidecar exists |
| `check_verdict` | pass or fail from the ONNX check heads | placeholder: it needs `onnxruntime`, and it says so instead of showing nothing |

Live crêpe masks on video already exist elsewhere: the fold video analysis
draws the classical or student mask on every sampled frame into
`annotated.mp4`, and both review sites play it (see
[vision pipeline](vision-pipeline.md#6-video-timeline)). This viewer is
where further producers plug in.

A frame the model never saw, or a frame with no record, is shown as such: the
panel says which producer failed and why, and the video keeps playing.

## Plugging a model in

A producer is one method:

```python
class MyModel:
    name = "my_model"
    describes = "what it does, in one line"

    def layers(self, frame, context):
        # frame is a BGR array, context carries the frame index and any record
        return LayerSet(state=..., prediction_state=..., material=..., holes=...)
```

Add it to `default_producers` in `code/maus_export/producers.py` and it appears
in the selector. The viewer, the timeline and the panel need no change. A
producer that cannot run raises `ProducerError` with a readable reason, which is
what the panel shows.

The expected arrivals:

- the material-mask model, giving `material` and `holes` plus a state;
- the ONNX check and fold heads, giving `check`, `verdict` and `prediction_state`;
- the fold segmentation from video, giving `prediction_state` with its masks.

## What is not there yet

- **Live streams.** The app serves a video file. Rendering a live source is
  `maus-export overlay --input device:N` today, and the viewer would need a
  frame source that pushes instead of a file it can seek.
- **Outlines.** The browser draws tints; the pre-rendered overlay video from
  `maus-export overlay` also draws outlines. Move that drawing into the browser
  when it matters.
- **No accuracy claim.** Nothing here validates a model. It shows what the
  producer says about the frame it is given.
- **Local only.** Loopback, no authentication, one viewer at a time. Do not
  expose it.
