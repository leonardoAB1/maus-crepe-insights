# Candidate run-start training dataset

This export is an observational dataset, not a trained model. Its inputs are
**candidates** for run-start prediction: the dashboard snapshot does not prove
that any field or parameter was available at run start, or that it was not
updated later. Verify provenance and timing of each selected input before any
real-time use. Unverified fields can leak future information.

## Build

From the chosen worktree's `code/` directory, after placing raw dashboard run
JSONs in that worktree's gitignored `data/raw/runs/`:

```bash
pixi install
pixi run export training
# Or use an explicit private data directory, which supplies input AND output:
pixi run export --data-dir /absolute/path/to/data training
```

The default data directory is the current worktree's `data/`, not the main
checkout's data. Each worktree has its own `data/`. An explicit `--data-dir`
can point to an existing private export, but writes its output there too; make
sure you have permission. Keep the data directory gitignored and private. Do
not commit, share, or paste per-run rows. The command reads only
`<data-dir>/raw/runs/*.json` and atomically replaces the single artifact
`<data-dir>/training/run_start.parquet`. It does not modify raw input or
`runs.parquet`. Input files are processed in filename order; do not use row
order to split data.

Empty input, duplicate IDs or JSON keys, malformed JSON, invalid required fields,
and incompatible field types fail the command. A prior artifact is left intact
on failure and is **stale**, not a successful new export. For a bad input file
the CLI error names the file and a fixed reason; it never prints field values.
Inspect that file locally. The file and its manifest are written
together with an atomic replacement; incomplete temporary files are cleaned up
on ordinary write failures.

## Schema and manifest

One Parquet row per raw run, with these disjoint roles:

| Role | Output column | Raw source | Parquet type |
| --- | --- | --- | --- |
| Metadata | `id` | `id` | int64, required unique positive integer |
| Metadata | `ts_start` | `tsStart` | timestamp in UTC, microseconds |
| Target | `terminal_status` | `terminal_status` | nullable string |
| Candidate input | `robot` | `machineName` | nullable string |
| Candidate input | `topping` | `topping` | nullable string |
| Candidate input | `is_test` | `isTest` | nullable boolean |
| Candidate input | `hotplate_temp` | `hotplateTemp` | nullable float64 |
| Candidate input | `env_temp_start` | `envTempStart` | nullable float64 |
| Candidate input | `env_humidity_start` | `envHumidityStart` | nullable float64 |
| Candidate input | `param.<key>` | `parameters[<key>]` | nullable string |

`tsStart` must have an explicit timezone and is converted to UTC. The target
is the recorded terminal status, including `succeeded`, `not_succeeded`,
`canceled`, `aborted`, unknown strings and null. It is not a quality label:
cancellations and aborts are not automatically bad crêpes, and missing status
is not failure. Do not train with `id`, `ts_start`, or `terminal_status` as
features. The timestamp is for chronological splitting and auditing.

The fixed parameter allowlist is:

- `dispense.batter.cook_time`
- `dispense.batter.centre_dispense_time`
- `dispense.batter.centre_flowrate`
- `dispense.batter.fan_power`
- `dispense.batter.concentric_outer_radius_offset_mm`
- `dispense.batter.concentric_inner_radius_offset_mm`
- `dispense.batter.concentric_ring_count`
- `dispense.batter.concentric_rings_enabled`
- `dispense.oil.grams`

Only these keys receive `param.<key>` columns, regardless of what is present
in any run. All other parameter keys are ignored, including nested or scalar
post-run values. Allowlisted values must be JSON numbers, booleans, strings,
or null. Objects and arrays for these keys fail the export before replacement.
Non-null values are compact JSON strings: number `1` becomes `1` and string
`1` becomes `"1"`. JSON null, an absent key, and an absent/null parameter
mapping all become Parquet null; they are not distinguishable. Decode with
`json.loads` when appropriate and preserve missingness. No imputation, scaling,
category encoding, feature selection, or other fitted transformation is
performed. Incompatible context types are rejected rather than silently
coerced. The fixed schema does not depend on parameter keys in later runs.

The Parquet Arrow schema carries JSON metadata under
`maus.training.manifest`, with `schema_version`, `metadata`, `target`,
`candidate_features`, `parameter_encoding`, `target_semantics`, and
`availability`. For programmatic access without printing private rows:

```python
import json
import pyarrow.parquet as pq

schema = pq.read_schema("../data/training/run_start.parquet")
manifest = json.loads(schema.metadata[b"maus.training.manifest"])
features = manifest["candidate_features"]
```

Durations, end times, phase-derived fields, checks, logs, vision metrics,
manual labels, configuration and segment identifiers are not selected as
features. The export is built from raw runs, not the analysis table, whose
post-run fields and dataset-wide feature selection are unsuitable as-is.

## Evaluation and limits

Sort by `ts_start` and reserve a later chronological holdout before model or
feature decisions; keep runs with the same timestamp on the same side of the
boundary. Fit imputers, encoders, feature selection, and every other learned
transformation on training data only, then apply them to the holdout. Decide
how to handle null targets explicitly; never silently label them as failures.
Preserve all status categories or document any purpose-specific filtering and
its bias.

These are observational records, not randomized experiments; correlations do
not establish causality. Test and customer populations, robot changes, selection
and dashboard access windows can bias results. Evaluate temporal and robot
shift, class imbalance, coverage, and missingness before interpreting scores.
No guarantee of quality ground truth or deployment readiness is implied.
