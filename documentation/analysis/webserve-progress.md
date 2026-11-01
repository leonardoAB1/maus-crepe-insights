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
