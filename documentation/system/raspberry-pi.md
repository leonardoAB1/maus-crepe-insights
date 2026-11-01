# Raspberry Pi detection runtime

The `pi` pixi environment targets `linux-aarch64`, with Python 3.12,
headless OpenCV, NumPy and non-interactive Matplotlib. It excludes the
project's default feature, export/auth clients, pandas, Parquet tools and
ML training packages. Native codec and numerical libraries are still needed.
The web server uses Python's standard library and binds to localhost only.

From `code/` on a 64-bit Raspberry Pi, with pixi installed:

```bash
pixi install -e pi --locked
pixi run -e pi detect --data-dir /path/to/private/data 3344
```

The input directory needs cached `raw/runs/3344.json` and
`raw/media/3344/video.mp4`. These files are private and are not bundled.
The matching `robot-image-calibration.json` must remain at the repo root.
There is no dashboard login, network export or robot-control connection.

Useful options:

```bash
# Analyze without starting the website; prints timing and writes summary.json.
pixi run -e pi detect --data-dir /path/to/private/data 3344 --analyze-only

# Website only, showing existing analyses.
pixi run -e pi detect --data-dir /path/to/private/data --port 8766

# Explicitly rebuild an existing analysis.
pixi run -e pi detect --data-dir /path/to/private/data 3344 --replace
```

The default is 10 analyzed frames per second and one OpenCV thread.
`--threads` changes the latter. BLAS/OpenMP are limited to one thread by
activation settings, appropriate as an initial configuration on a 1 GB Pi.
No source-camera frames or videos are transmitted automatically.

For browser access from another machine, keep private data off the LAN:

```bash
ssh -L 8766:127.0.0.1:8766 raspi@PI_ADDRESS
```

Then open `http://127.0.0.1:8766/` locally. Do not bind the unauthenticated
website to all network interfaces or expose it through a public proxy.

## Performance interpretation

Compare `median_ms` (rectification and inference), `median_render_ms`,
`median_processing_ms` (sample retrieval through annotated-frame writing),
and `over_budget_frames` in each run's private `summary.json`. The processing
measurement excludes decoding work done by skipped-frame `grab()` calls,
final plot generation and HTTP serving. Record elapsed whole-command time
separately if end-to-end throughput matters. Codec, resolution, CPU
throttling, memory pressure and thread count can all affect results.

The 200 ms budget is not proven until measured on the target Pi. Desktop
measurements and a cross-platform lock solve are not native benchmarks.
