# doe-response

Quadratic response model from the designed experiment (#26):
`y = [x, x^2] @ W + b`, one output per outcome.

- **Input `x`:** float32, shape `[batch, 6]`, the settings on the coded
  scale, `2 * (value - low) / (high - low) - 1`. Feature order and levels are
  in `model.json`.
- **Output `y`:** float32, shape `[batch, outcomes]`, outcome names in
  `model.json`.

`model.onnx` and `model.json` do not exist yet. `train.py` creates them from a
run sheet that Maus has filled with run ids, and checks onnxruntime against
the numpy prediction:

```bash
cd code
pixi run -e ml python models/doe-response/train.py data/doe/<doe_id>.parquet
```

See [documentation/analysis/doe.md](../../../documentation/analysis/doe.md).
