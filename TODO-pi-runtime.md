# Minimal Raspberry Pi runtime (#161)

- [x] Confirm target is Debian on linux-aarch64, approximately 1 GB RAM.
- [x] Add a no-default-feature pixi environment without export/ML packages.
- [x] Lock native ARM64 dependencies from conda-forge.
- [x] Add an offline detection/website entry point with one OpenCV thread.
- [x] Test imports without pandas, pyarrow, MCP and HTTP clients.
- [x] Document tunnel-only private website access and timing definitions.
- [x] Copy minimal source and launch a background installer on the Pi.
- [x] Install on the Pi and validate native performance.
- [x] Disable the desktop and set persistent console-mode boot.
- [x] Enable a localhost website service for console-mode boots.
- [x] Verify HTML, run JSON and video ranges through an SSH tunnel.

## Deployment status

Installation completed on 2026-11-01 after recovery from network stalls
and a user-requested power restart. Two concurrent downloads were stable;
the native installation took approximately 31 minutes after resumption.

The target is a Raspberry Pi 3 Model B with approximately 1 GB RAM. It now
boots to `multi-user.target`; the display manager is stopped. The
`maus-fold-web.service` system service runs as the unprivileged runtime user,
is enabled for boot, and binds only to localhost port 8766.

125 fold tests passed (3 skipped) using `pixi run python -m pytest`; lint,
format and docs checks passed. Native OpenCV 5.0 imports were verified.
One private cached-video proxy and its metadata were transferred explicitly
for the benchmark. No dashboard credentials or tokens were transferred.

The 554-frame proxy used 672x380 input at 10 fps and approximately 900 px
analysis. Median inference: 413.2 ms; rendering: 57.0 ms; full sampled-frame
processing: 527.5 ms; wall time: 302 s. Every frame exceeded the 200 ms
processing budget. The target is not met on this configuration.

HTML, the full 554-row run API and a 1,024-byte video range were verified
through an SSH tunnel. Power/throttling flags remained zero in checked
readings. Full-resolution decoding performance was not measured.
No credentials or private robot data are included in this branch.
