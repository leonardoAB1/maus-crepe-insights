# Frame embeddings

`data/training/frame_embeddings.parquet` caches one frozen-CNN embedding per
raw check frame, so classifier heads can train in seconds on a CPU. The model
is [`code/models/frame-embedder/`](../code/models/frame-embedder/README.md).

## Build

From `code/`, with frames downloaded by `fetch --media`:

```bash
pixi run -e ml embed
# or point at another private export (input AND output):
pixi run -e ml export --data-dir /absolute/path/to/data embed
```

It embeds every `raw/media/<run_id>/<NN>_<check>_frame.jpg`. Overlays are
never embedded. Unreadable images are logged and skipped. Rebuild after a new
fetch or a new `model.onnx`.

## Schema

| column | type | meaning |
|---|---|---|
| `run_id` | int64 | dashboard run id |
| `check` | string | `visionChecks` phase (`spread`, `fold1`, ...) |
| `attempt` | int32 | 1-based occurrence of the check in the run |
| `check_index` | int32 | index in `visionChecks` (the `NN` in the name) |
| `frame_path` | string | path relative to the data dir, `/` separators |
| `e0` ... `e575` | float32 | embedding, one column per dimension |
| `model_sha256` | string | hash of the `model.onnx` that produced it |

`attempt` is counted over the run JSON's `visionChecks`, so a frame missing
locally still takes its number. The key `(run_id, check, attempt)` joins the
frame table (`data/training/frames.parquet`). Read the dimension from the
`e<j>` columns or from `embedding_dim` in `model.json`, rather than
hardcoding it. Compare `model_sha256` with the current `model.onnx` to detect
a stale cache.

## Latency

`model.json` holds the measured CPU latency at batch 1 and a Raspberry Pi 5
estimate (single-thread median times a rough factor, not a measurement). On
the export laptop the model plus preprocessing takes a few milliseconds per
frame, far under the 300 ms target.

The table holds private run data. It stays in the gitignored `data/` folder
and is never committed.
