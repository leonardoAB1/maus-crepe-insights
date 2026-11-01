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

Then open `http://127.0.0.1:8766/` locally. Keep this private default unless
the operator explicitly authorizes unauthenticated network access.

## Explicit LAN access

`--host 0.0.0.0 --port 80` exposes the website on IPv4 network interfaces.
It serves private videos and run data with no authentication or encryption.
Use only on a trusted network with the data owner's explicit authorization;
no router port forwarding or internet exposure is configured.

The authorized Pi deployment uses a `maus-fold-web.service` drop-in with
`AmbientCapabilities=CAP_NET_BIND_SERVICE` and
`CapabilityBoundingSet=CAP_NET_BIND_SERVICE`. This permits port 80 while
keeping the service under its unprivileged user rather than running Python
as root. The service stays enabled across reboots. Localhost remains the
program's default whenever `--host` is omitted.

Remove `/etc/systemd/system/maus-fold-web.service.d/lan.conf`, reload systemd
and restart the service to restore the original localhost port 8766 setup.

## Performance interpretation

Compare `median_ms` (rectification and inference), `median_render_ms`,
`median_processing_ms` (sample retrieval through annotated-frame writing),
and `over_budget_frames` in each run's private `summary.json`. The processing
measurement excludes decoding work done by skipped-frame `grab()` calls,
final plot generation and HTTP serving. Record elapsed whole-command time
separately if end-to-end throughput matters. Codec, resolution, CPU
throttling, memory pressure and thread count can all affect results.

## Native smoke benchmark (2026-11-01)

On a Raspberry Pi 3 Model B with approximately 1 GB RAM, console-mode boot,
one OpenCV/BLAS thread and OpenCV 5.0.0, a private 554-frame proxy at
672x380 input and 10 fps produced:

| Measurement | Result |
| --- | ---: |
| Median inference | 413.2 ms/frame |
| Median rendering | 57.0 ms/frame |
| Median sampled-frame processing | 527.5 ms/frame |
| Whole-command wall time | 302 seconds |
| Frames over 200 ms processing budget | 554 of 554 |

Analysis still rectified frames to approximately 900 px. This is a native
algorithm smoke test, not a full-resolution decoding benchmark or a quality
evaluation. Initial font-cache generation was included in wall time. Power
and throttling flags were zero in checked readings. The **200 ms target is
not met** on this configuration.

The deployed website runs as an unprivileged system service named
`maus-fold-web.service`, enabled for console-mode boots and still bound to
localhost port 8766. The desktop was stopped and the default system target
set to `multi-user.target`. Run-data and video-range endpoints were verified
through an SSH tunnel. Source, environment and private outputs live under
`~/maus-fold-runtime/`; `~/.local/bin/maus-detect` is the convenience command.
The installation does not configure dashboard authentication or robot control.
