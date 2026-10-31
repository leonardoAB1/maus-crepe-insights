# Data preparation handoff

This page records what was prepared, not a claim about all robot history. The
figures below describe **one private, account-limited local export as of
2026-10-31**. Another account, date window, cache, code revision or later fetch
can produce different results. The public repository does not contain the data.

## What was done

1. **Fetch.** The dashboard was read for run JSON and the robot registry,
   along with referenced check images and videos. The local snapshot contained
   828 raw run JSON files, four robot registry entries, 815 videos and 2,225
   check images. In that fetch, 13 referenced videos and seven check images
   returned HTTP 404. These counts do not prove media or historical coverage.
2. **Convert.** The 828 raw runs were converted to 828 ROS 2 MCAP files without
   embedding videos. An initial flat analysis export produced `runs.parquet`
   and `runs.csv`, each with 828 rows and 177 columns. The local table predates
   [PR #48] and lacks the later `parameter_id`; regenerate it from raw runs to
   obtain that field. The current column count can differ. `parameter_id` is
   a per-run parameter fingerprint, not a feature; `config_id` and `segment_id`
   are export-local. See [table identity][identity] and [MCAP topics](../data/mcap.md).
3. **Measure overlays.** OpenCV read the robot's existing annotations in check
   overlays, rather than running a newly trained vision model. Measurements
   include spread geometry, uncovered area, fold overlap and lightness. The
   private `vision.parquet` had 138 scored run rows. The scorer uses the last
   listed check attempt; missing or unusable overlays limit coverage. See the
   [vision metrics](../data/pipeline.md#vision-score-columns) for definitions.
4. **Prepare training candidates and audit.** The raw-run export
   `training/run_start.parquet` had 828 rows, 15 candidate input features and
   the recorded `terminal_status` as an operational proxy target. The separate
   aggregate `training/audit.json` validates input schemas and join IDs and
   reports feature and vision coverage, including breakdowns. It does not prove
   input timing, media completeness or common source provenance. See the
   [run-start schema](training-dataset.md) and [audit guide](training-audit.md).
   This historical snapshot used the earlier feature schema. The current
   schema has 16 candidate features; regenerate rather than reuse this export.
5. **Join vision targets.** `training/vision_targets.parquet` retained all 828
   run-start rows with 20 nullable post-run target columns. Of these runs, 138
   had at least one target and 690 had none. A target-bearing row need not have
   every metric. The join validates raw run-start values and IDs, and records
   input hashes. The 138 target rows were independently checked against the
   current overlay computation and source hashes. Matching IDs or hashes still
   do not establish a common original vision snapshot or input capture time.
   See [vision targets](vision-targets.md) for validation and provenance limits.

The analysis export preceded the later parameter-identity change. The
run-start exporter, audit and vision-target exporter were added in that order
in [PR #52], [PR #67] and [PR #79], respectively. Their tests use synthetic
records; they test exporter and validation behavior, not real-world label
quality or proof of completeness. Earlier table fixes include [PR #44],
[PR #47] and [PR #48].

## Meaning and limits

- No new robot runs were requested or robot commands sent as part of this
  preparation. No human ground truth was created and no model was trained by
  this preparation. A prepared table is not a validated model or a label set.
- `terminal_status` and robot checks are recorded operational outcomes, not
  independently verified quality labels. Canceled and aborted runs are not
  necessarily bad products. Earlier samples had empty manual labels. In the
  2026-10-31, 828-run local snapshot, each of the four manual success fields
  had only one or two non-null values; these are not established ground truth.
- Vision scores quantify the robot's own markings and image measurements, not
  human assessments. Missing scores mean unknown, not zero or failure. The
  scored subset may differ by time, robot and test/customer population.
- The fixed run-start features are candidates, not verified start-time
  snapshots. Raw values and matching hashes do not prove acquisition time or
  absence of later updates. Exclude post-run checks, logs, durations, vision
  values, identifiers and outcomes from run-start inputs.
- Reserve a later chronological holdout **before filtering for targets** or
  choosing features. Keep tied timestamps together. Fit imputation, encoding,
  scaling and feature selection on training data only; assess coverage, repeat
  configurations, robot changes and time shifts before interpreting results.

## Reproduce privately

A collaborator must fetch the runs their own dashboard account is authorized
for. In the chosen worktree, from `code/`, choose a **new, empty** private
subdirectory of that worktree's gitignored `data/` and check disk space first:

```bash
pixi install
umask 077
DATA_DIR="$(cd .. && pwd)/data/collaborator-export"
pixi run export --data-dir "$DATA_DIR" fetch --media --video
pixi run export --data-dir "$DATA_DIR" mcap
pixi run export --data-dir "$DATA_DIR" table
pixi run export --data-dir "$DATA_DIR" vision
pixi run export --data-dir "$DATA_DIR" training
pixi run export --data-dir "$DATA_DIR" audit
pixi run export --data-dir "$DATA_DIR" vision-targets
```

The first sign-in opens a browser; use your own account and credential cache.
`--data-dir` must precede the subcommand and sets both inputs and outputs.
Video is optional for the vision and training tables. Omitting `--video` from
`mcap` prevents embedding it. `export all --video` would also embed video;
`all` does not build either training dataset or the audit.

The current fetch defaults are 2026-08-30 through 2026-12-30, subject to
account permissions; use `fetch --from YYYY-MM-DD --to YYYY-MM-DD` after the
same global option to choose a window. Removing `--limit` is not a claim of
complete history. Record the actual window, access scope, revision and media
coverage privately. Cached JSON and media can survive earlier fetches; builders
read cached runs, not just the last selection. A fetch can log failed downloads
and still succeed, and logs may contain private media URLs. Inspect failures
and compare expected against downloaded media locally. On a failed training,
audit or vision-target build, any previous output is stale, not a fresh result.
See the [collaborator runbook][runbook] and
[audit checklist](training-audit.md#handoff-checklist).

Raw JSON, media, MCAP, tables, audit reports and credentials stay private and
out of git. Do not paste rows, run IDs, URLs or tokens into public issues or
PRs. Even aggregates can disclose small groups; review before sharing them.

[identity]: ../data/pipeline.md#parameter-identity-and-compatibility
[runbook]: vision-targets.md#reproduce-the-dataset-privately
[PR #44]: https://github.com/leonardoAB1/maus-crepe-insights/pull/44
[PR #47]: https://github.com/leonardoAB1/maus-crepe-insights/pull/47
[PR #48]: https://github.com/leonardoAB1/maus-crepe-insights/pull/48
[PR #52]: https://github.com/leonardoAB1/maus-crepe-insights/pull/52
[PR #67]: https://github.com/leonardoAB1/maus-crepe-insights/pull/67
[PR #79]: https://github.com/leonardoAB1/maus-crepe-insights/pull/79
