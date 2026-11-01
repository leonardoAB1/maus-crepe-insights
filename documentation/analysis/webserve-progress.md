# Web dashboard runner progress — implementation log

- Based on locally available video pipeline branch `70c51cf`; no remote fetch
  was possible in the offline workspace.
- Added atomic, throttled per-run status snapshots shared across processes.
  Existing `fold-live` and batch evaluation runners publish these automatically.
- Added a read-only progress endpoint and one-second dashboard polling with
  completion, wall-clock throughput, ETA, terminal state and stale-run warnings.
- Kept output staging intact: progress does not expose partially rendered video.
- Offline validation: synthetic 20-frame video completes with matching frame
  counts and positive throughput; deterministic-clock tests cover rate and ETA;
  HTTP tests cover progress polling; failure and stale/corrupt snapshots are
  covered. Embedded JavaScript passes Node's syntax checker.
- No production footage was processed or committed for this change.

## Parallel batch runner follow-up

- Added `pixi run fold-batch --threads N` to discover every cached run/video
  pair, skip completed artifacts, retry partial outputs and optionally replace
  all results. Submission is bounded to the requested number of workers.
- Each worker uses the existing runner and its dashboard progress snapshots.
  Failures are isolated, counted and reflected in the command's exit status.
- Serialized Matplotlib output creation because pyplot state is not thread-safe;
  frame processing and video encoding still run concurrently across videos.
- Offline tests cover numeric discovery, missing videos, two simultaneous
  workers, continuing after failure, skip/retry/replace, option validation and
  CLI argument forwarding / nonzero failure exit. No new dependencies.
- No production batch was launched: full annotated exports can consume large
  amounts of disk space; users choose the worker count when running locally.
