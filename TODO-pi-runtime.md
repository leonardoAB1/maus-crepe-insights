# Minimal Raspberry Pi runtime (#161)

- [x] Confirm target is Debian on linux-aarch64, approximately 1 GB RAM.
- [x] Add a no-default-feature pixi environment without export/ML packages.
- [x] Lock native ARM64 dependencies from conda-forge.
- [x] Add an offline detection/website entry point with one OpenCV thread.
- [x] Test imports without pandas, pyarrow, MCP and HTTP clients.
- [x] Document tunnel-only private website access and timing definitions.
- [x] Copy minimal source and launch a background installer on the Pi.
- [ ] Install on the Pi and validate native performance when SSH recovers.

## Deployment status

SSH initially stalled. A fresh connection later recovered and the minimal
source was copied successfully. A background installer is running under
`~/maus-fold-runtime/install.sh`, logging to `install.log`. It resumes the
native pixi download, then installs the locked `pi` environment and verifies
imports. Completion and native timing have not yet been verified.

125 fold tests passed (3 skipped) using `pixi run python -m pytest`; lint,
format and docs checks passed. The native ARM64 lock was resolved locally.
No test video or robot credentials were transferred.
No credentials or private robot data are included in this branch.
