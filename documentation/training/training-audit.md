# Training data audit

This is an **aggregate-only local audit**, not a training set, model, validation
score, or proof that recorded outcomes are ground truth. It compares the
existing run-start candidate table with the existing vision-score table. It
does not fetch, rebuild, or change either input. See
[run-start dataset](training-dataset.md) and
[vision metrics](../data/pipeline.md).

## Reproduce

From this worktree's `code/` directory, using the installed default pixi
environment and a private data directory containing the two inputs:

```bash
pixi run export --data-dir /absolute/path/to/data audit
```

`--data-dir` is a global option and **must precede** `audit`. Replace the
example path with the intended private data directory.

The command reads `<data-dir>/training/run_start.parquet` and
`<data-dir>/vision.parquet`, then atomically replaces only
`<data-dir>/training/audit.json`. It does not read raw JSON, access the
network, or regenerate inputs. Default `--data-dir` is the current worktree's
`data/`, not the shared checkout. `pixi run export all` does not build the
run-start table or this audit. Generate missing input separately, only with
permission to write those artifacts, following the linked documentation.

The recorded training table is built from **all cached** `raw/runs/*.json`, not
necessarily just the latest fetch. Vision comes from runs with downloaded,
readable and scoreable check overlays. Fetch defaults in this revision are
2026-08-30 through 2026-12-30, subject to dashboard account access and fetch
options; `--from`, `--to`, `--limit`, `--hide-tests`, cached files and media
downloads can change the effective population. Neither input records reliable
complete fetch provenance, and this audit cannot establish the actual selection
window, completeness, or that both files were built from the same snapshot.
The report's UTC generation timestamp is not an input fingerprint. Record
fetch dates, options, access scope, input freshness and source versions
separately in private notes; do not infer historical coverage from the report.

## Report schema

JSON schema version `1` has these top-level fields:

- `schema_version`: integer; `generated_at_utc`: UTC ISO timestamp; `sources`:
  input paths relative to the data directory.
- `join`: `training_rows`, `training_unique_ids`, `vision_rows`,
  `vision_unique_ids`, `matched_training_runs`,
  `training_runs_without_vision_row`, and
  `vision_rows_without_training_run`. The two inputs are validated as unique
  positive integer IDs before a one-to-one **left** join. Unmatched training
  runs remain in every training denominator. Extra vision rows are counted,
  not added to the training population.
- `overall`: one population summary, defined below. Top-level `features` and
  `vision_metrics` repeat its feature and metric coverage for convenience.
- `unavailable_vision_metrics`: known score column names absent from the vision
  input. Their absence is not interpreted as zero coverage or bad quality.
- `breakdowns`: `month` (UTC `YYYY-MM` of `ts_start`), `robot` (recorded
  machine name or `unknown`), `test` (`true`, `false`, `unknown`) and
  `status` (the same categories as `status_counts`). Each group contains the
  same summary as `overall`. A group appears only when at least one training
  run belongs to it.

A summary's `runs` is its training-run count. `status_counts` enumerates
`succeeded`, `not_succeeded`, `aborted`, `canceled`, `other`, and `unknown`;
`other` means a non-null string outside the four known categories, while
`unknown` means null. `test_counts` enumerates `true`, `false`, and `unknown`
(null). Both distributions use the same `runs` denominator. The test flag is
the recorded candidate field; even if its type is valid, its provenance and
reliability have not been verified. Do not assume unknown means customer.

Every entry in `features`, `vision_metrics`, and `vision_row_coverage` has:

- `denominator_training_runs`: all training runs in that population or group;
- `non_null_runs`: runs with a non-null feature/metric, or a matched vision row
  for `vision_row_coverage`;
- `null_runs`: the denominator minus `non_null_runs`;
- `coverage`: `non_null_runs / denominator_training_runs` (null if the
  denominator is zero).

Vision metric coverage is computed after the left join and includes both runs
with no vision row and matched rows whose particular metric is null. Those two
causes are **not** equivalent: use `join` and `vision_row_coverage` to quantify
missing vision rows separately. A present vision row does not guarantee every
metric was computed. Missing values mean unknown or unmeasured, not failure,
zero, or an actual quality score. Only metric columns present in the source
have per-metric coverage objects; unrepresented known columns are listed in
`unavailable_vision_metrics`. See the pipeline documentation for the individual
metric definitions and overlay-derived limitations.

## Interpretation and privacy

Recorded `terminal_status` is an operational proxy, not independently verified
crêpe quality. Canceled and aborted runs are distinct outcomes and are not
necessarily bad products. Null status is unknown, never an automatic failure.
Vision scores are post-run measurements, not run-start input features. The
candidate inputs, including the robot, test flag, temperatures, and recipe
parameters, have **unverified run-start availability**. Establish their source
timing before using them for prediction. Manual labels are not in these two
inputs and this report does not assert their coverage.

Vision scores depend on overlay availability and selection. Compare coverage
across UTC month, robot, test flag, and recorded status before drawing any
conclusions. A measured-score subset can be heavily concentrated in a recent
month or a particular robot/test population and have different score
distributions from other populations. Coverage alone does not establish whether
measured scores are representative. There are observational, time, robot,
test/customer, fetch-window and access biases; do not generalize from the
measured subset to unmeasured runs or claim causation. The audit does not report
metric values or model performance.

The artifact contains no run IDs, rows, individual scores, media URLs or raw
feature values. It does contain robot names and potentially very small group
counts; it does **not** apply disclosure suppression. Aggregates can still be
sensitive or re-identifying when combined with other information. Keep the
report in the private, gitignored data directory; do not publish or paste it
without privacy review. Standard CLI messages do not echo input values.

## Failures and troubleshooting

Missing or unreadable input, empty training data, duplicate/null/non-positive
or non-integer IDs, unsupported or duplicate columns, invalid field types,
non-finite numeric values, missing/non-UTC timestamps, or unexpected robot
names fail the audit. For privacy, recorded robot names must match
`maus-[a-z]{1,24}` or be null. Supported vision columns are the known
spread/fold metrics implemented by this revision; an unknown column requires
schema review and a code/test update, not silent admission. A vision table
must contain `id` and at least one supported metric column. A missing vision
file is an error, not an all-null vision population. Non-null candidate
parameters must be canonical JSON scalars (strings, finite numbers, or
booleans). Encoded null, arrays, objects, invalid JSON and noncanonical
encodings fail validation; missing parameters must be Parquet nulls.

On ordinary validation or write failure before replacement, the CLI exits
nonzero without printing offending values; any previous report is unchanged
and must be considered stale. Temporary files are cleaned up on ordinary
exceptions. A successful audit writes no status message or supplied path to
stdout after replacement; check the exit status and report locally. Abrupt
process or system termination can occur before or after replacement and may
leave a temporary file, so the exit status alone cannot establish which report
is present in that case. Inspect inputs locally and check directory permissions,
input schema and freshness; never paste rows or media URLs into issues. Fix or
regenerate the input only with appropriate permission, then rerun the command.

## Handoff checklist

- [ ] Record, privately, input provenance, fetch options, date/access scope and
      freshness; ensure both inputs refer to the intended snapshot.
- [ ] Run the audit and check its exit status, timestamp, join counts, missing
      vision rows, and unavailable columns. Investigate unmatched populations.
- [ ] Review per-feature missingness and each target metric's coverage,
      including month, robot and test strata; treat small cells as sensitive.
- [ ] Confirm the prediction time and verify every selected feature is actually
      available then. Exclude IDs, timestamps, terminal status and all post-run
      vision/check/log/duration fields from run-start features.
- [ ] Define targets, null handling, and operational status semantics
      explicitly. Do not turn missing vision or unknown outcomes into failures.
- [ ] Reserve chronological holdouts **before** feature/model selection,
      keeping equal timestamps together. Check time and robot shifts and any
      repeated settings. Fit imputation, encoding, feature selection, and all
      learned transformations on the training split only.
- [ ] Evaluate selection bias and scored-versus-unscored populations; do not
      claim representative quality or causality. Keep private artifacts local.
