# Maus Crêpe Insights – BaselHack 2026

Our BaselHack 2026 project on the **Maus Robotics** challenge. Maus builds autonomous crêpe robots.
Every run is logged with phases, machine logs, on-robot vision checks (with images), recipe parameters and video.
We turn that data into insight: why runs fail, which settings and conditions matter, and how to catch problems early.

## Team
Leonardo Acha (leonardoAB1) · Qbyte248 · AetherDim · salarehman0 · Timon Tscheulin

## Repo layout (BaselHack template)
```
code/            Python tooling (maus_export) + models
  maus_export/   dashboard → raw JSON → ROS 2 MCAP / analysis tables
  models/        trained models, always exported as ONNX
  tests/
documentation/   data description, MCAP format
assets/          slides, videos, images for the pitch
data/            local exports (gitignored, never committed)
```

## Quickstart
Requires [uv](https://docs.astral.sh/uv/) and a Maus dashboard account.

```bash
cd code
uv sync

# 1. Download runs (opens a browser to sign in the first time; tokens are cached in ~/.maus-export/)
uv run maus-export fetch --limit 10 --media     # drop --limit for all runs; add --video for mp4s

# 2. Convert to ROS 2 MCAP (one file per run in data/mcap/) and open them in Foxglove
uv run maus-export mcap

# 3. Flat table for analysis: data/runs.csv + data/runs.parquet
uv run maus-export table

# or everything at once
uv run maus-export all --media
```

Set `MAUS_TOKEN=<bearer token>` to skip the browser sign-in.

Tests: `uv run pytest`

## Standards we follow
- **Data → [MCAP](https://mcap.dev) (ROS 2 profile)**, the standard container for multimodal robot logs.
  See [documentation/mcap.md](documentation/mcap.md).
- **Models → [ONNX](https://onnx.ai)**. Train with any framework, ship `.onnx` and run it with `onnxruntime`.
  See [code/models/README.md](code/models/README.md). Install with `uv sync --extra ml`.

## Documentation
- [What data the dashboard has](documentation/data.md)
- [MCAP topics & schemas](documentation/mcap.md)

## License
MIT – see [license.txt](license.txt).
