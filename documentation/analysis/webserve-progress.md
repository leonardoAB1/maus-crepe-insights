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
- No production batch was launched during implementation: full annotated
  exports can consume large amounts of disk space. Subsequently, at the user's
  request, the dashboard and batch were started with one worker.

## Selected-run progress layout

- Removed the all-jobs progress panel, which grew with hundreds of snapshots.
  Processing runs now have sidebar spinners; failed/stale jobs have warnings.
- Placed a single progress bar and speed/ETA details below the selected video.
  In-progress runs are selectable before output exists and load their finished
  video automatically. Other completions do not interrupt selected playback.
- Made the sidebar independently scrollable and kept the main view top-aligned.
- Offline Node DOM-stub test exercises 500 sidebar entries, a single detail bar,
  preserved button identity, active-run selection, stale warnings and automatic
  selected-video loading on completion. Syntax and HTTP route tests also pass.

## Collapsible processing failures

- Moved processing failures into a native, initially collapsed `Failed runs`
  section at the bottom of the sidebar, with a count and inline exception.
  Expansion survives polling; retries move runs back to the normal list.
- New runner failures record the exception message as well as its class.
  Existing type-only snapshots explicitly say no detailed message was recorded.
  Failure details use text nodes rather than HTML interpolation.
- Extended the offline DOM test for failure grouping, messages, fallback for old
  snapshots, preserved expansion and retries; runner tests assert captured
  file-not-found messages. All 20 targeted tests passed.
- The live dashboard serves updated HTML without restarting the active batch.
  The already-running runner retains its old exception-type-only reporting
  until restarted; new invocations record full messages.
