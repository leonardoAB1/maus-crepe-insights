# Models

**Every model we ship is exported to [ONNX](https://onnx.ai/).** Train with whatever fits (scikit-learn,
PyTorch, XGBoost, ...), but the artifact the rest of the team (and the robot) consumes is a `.onnx` file
run with `onnxruntime`.

## Conventions
- One folder per model: `models/<name>/`
  - `model.onnx` – the exported graph (opset ≥ 17)
  - `model.json` – model card: inputs (names, dtypes, shapes, feature order), outputs, training data range
    (run ids / dates), metrics, and the commit it was trained at
  - `train.py` – reproducible training + export script
- Inputs are named and documented; feature order must match `model.json`.
- Large weights (> 50 MB) are not committed; attach them to a GitHub release and link them in `model.json`.
- Every export is checked by running the same sample through the original model and `onnxruntime`
  and comparing the outputs.

## Export snippets
```python
# scikit-learn
from skl2onnx import to_onnx
onx = to_onnx(clf, X_train[:1].astype("float32"), target_opset=17)
open("models/<name>/model.onnx", "wb").write(onx.SerializeToString())

# PyTorch
torch.onnx.export(net, sample, "models/<name>/model.onnx", input_names=["x"], output_names=["y"],
                  dynamic_axes={"x": {0: "batch"}}, opset_version=17)
```

```python
# Inference
import onnxruntime as ort
sess = ort.InferenceSession("models/<name>/model.onnx")
pred = sess.run(None, {"x": X.astype("float32")})[0]
```

Install the ML extras with `uv sync --extra ml`.
