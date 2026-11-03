# vision-pipeline

The frame embedder and every embedding head merged into one ONNX graph. One
raw check frame goes in (through `maus_export.frame_preprocess.preprocess`),
and every probability comes out in a single onnxruntime call:

| Output | Classes | Source |
|---|---|---|
| `embedding` | 576 floats | `frame-embedder` |
| `check_verdict.<check>` | fail, pass | `check-verdict/<check>` |
| `fold_class` | empty, flat, half, quarter | `fold-class` |
| `failure_reason.<check>.<reason>` | not_reason, reason | `failure-reason/...` |

Nothing is retrained. The merge only rewires the graphs, and `build.py`
checks that every output matches the separate models. Metrics and caveats
are those of each source card. The check verdicts and failure reasons learned
the robot's own verdicts (weak labels), and not all of them generalize across
robots.

`maus_export.vision_pipeline.VisionPipeline` runs this graph and, beside it,
the crepe-student mask (different input size). `classify_run` writes one row
per check still to `pipeline.csv`, which the fold website shows.

Rebuild after any source model changes:

```
pixi run -e ml vision-pipeline
```
