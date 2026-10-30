# Maus Crêpe Insights - BaselHack 2026

Our BaselHack 2026 project on the **Maus Robotics** challenge. Maus builds autonomous crêpe robots.
Every run is logged with phases, machine logs, on-robot vision checks (with images), recipe parameters and video.
We turn that data into insight: why runs fail, which settings and conditions matter, and how to catch problems early.

## Team
Leonardo Acha (leonardoAB1) · Qbyte248 · AetherDim · salarehman · Timon Tscheulin (timonTscheulin) · WhyAge36

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
Requires [pixi](https://pixi.sh) and a Maus dashboard account.

```bash
cd code
pixi install

# 1. Download runs (opens a browser to sign in the first time; tokens are cached in ~/.maus-export/)
pixi run fetch --limit 10 --media    # drop --limit for all runs; add --video for mp4s

# 2. Convert to ROS 2 MCAP (one file per run in data/mcap/) and open them in Foxglove
pixi run mcap

# 3. Flat table for analysis: data/runs.csv + data/runs.parquet
pixi run table

# or everything at once
pixi run export all --media
```

Set `MAUS_TOKEN=<bearer token>` to skip the browser sign-in.

Tests and lint: `pixi run test`, `pixi run lint`, `pixi run fmt`

## Standards we follow
- **Data → [MCAP](https://mcap.dev) (ROS 2 profile)**, the standard container for multimodal robot logs.
  See [documentation/mcap.md](documentation/mcap.md).
- **Models → [ONNX](https://onnx.ai)**. Train with any framework, ship `.onnx` and run it with `onnxruntime`.
  See [code/models/README.md](code/models/README.md). Use the pixi `ml` environment: `pixi run -e ml ...`.

## Documentation
- [What data the dashboard has](documentation/data.md)
- [MCAP topics & schemas](documentation/mcap.md)

## License
MIT - see [license.txt](license.txt).
