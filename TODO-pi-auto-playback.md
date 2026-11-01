# Pi-specific automatic playback (#197)

- [x] Add a separate Pi entry point without changing desktop defaults.
- [x] Automatically prepare compact playback after Pi analysis.
- [x] Scan completed analyses in a background website worker every 15 seconds.
- [x] Reuse valid proxies and regenerate missing or stale ones.
- [x] Serialize automatic encoders using a process-safe POSIX lock.
- [x] Keep full-quality originals and recorded inference timings unchanged.
- [x] Support `--no-compress-playback` for explicit opt-out.
- [x] Deploy the profile to the Pi's existing port-80 system service.
- [x] Verify native automatic compression with synthetic data and clean it up.
- [x] Validate 134 fold tests, lint, format and docs checks.
- [ ] Merge after #194 and required CI checks.

## Validation (2026-11-01)

The native website service runs `python -m maus_export.fold_pi`. Without a
manual compression command, its scanner detected a newly published synthetic
completed analysis and produced matching H.264 playback and provenance files.
The synthetic test directory was removed after verification. Existing real
analyses and full-quality originals were preserved.

134 fold tests passed, 3 skipped. Tests cover Pi defaults, opt-out, worker
lifecycle, incomplete/staging directories, proxy reuse and lock exclusion.
Lint, format, docs checks passed; the lock file remains up to date because
no dependencies changed.

The preview remains 480 px at approximately 120 kbps, matching the current
slow-network deployment. Compression is separate from inference and can add
time to the whole analysis command. Fine detail is lost in the preview;
original-video downloads remain available. No new accuracy or native frame
performance result is claimed.
