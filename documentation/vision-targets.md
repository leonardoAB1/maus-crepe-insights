# Vision-target training dataset

This is an observational dataset, not a trained model or verified quality
labels. Its intended prediction time is **before a run starts**. The inputs
are candidates only: the dashboard snapshot does not prove that settings and
context were available at that time or unchanged afterward. Verify each field's
provenance and timing before use; otherwise it may leak future information.

## Build

From this worktree's `code/` directory, with pixi installed and private inputs
already present:

```bash
# Build the two inputs separately if needed. Vision needs downloaded overlays.
pixi run export training
pixi run export vision
pixi run export vision-targets

# Or use an existing private data directory for both inputs and the output:
pixi run export --data-dir /absolute/path/to/data vision-targets
```

`vision-targets` reads `training/run_start.parquet`, `vision.parquet`, and the
cached `raw/runs/*.json` from the selected data directory, and replaces only
`training/vision_targets.parquet` there. The default is this worktree's
`data/`, not the main checkout's. `export all --media` builds vision but does
not build either training artifact. Run `training` and `vision-targets`
separately after it. The command does not fetch data, score images, modify the
inputs or update an audit. Both upstream builders read all cached raw runs,
which can include earlier fetches beyond the latest fetch selection. Review
the selected data directory and its cache before building. Keep all raw and
derived data private and out of git; do not share rows or IDs. A successful
`vision-targets` command is silent.

The exporter rejects missing or malformed inputs, unexpected columns or types,
incompatible run-start schema or manifest, nonpositive, null or duplicate IDs,
and vision IDs absent from run-start. The set of unique cached raw IDs must
equal the run-start ID set exactly, with neither missing nor additional IDs.
Mismatch errors report aggregate counts only; duplicate IDs fail closed. Each
run-start row is also checked against its corresponding raw snapshot, including
timestamp, context, terminal status, and allowlisted parameters and nulls.
Missing or invalid raw input fails closed. The exporter validates recipe
encodings, rejects infinities and negative scores, and restricts IoU and hole
and uncovered fractions to [0, 1]. CIE L* means must be in [0, 100]; their
population standard deviations must be in [0, 50]. When both are present,
standard deviation squared must not exceed mean * (100 - mean), allowing an
absolute tolerance of 1e-9 in squared L* units for floating-point roundoff.
This tolerance does not relax the 50 upper bound. It does not clip values or
impose upper bounds on other ratios, offsets or discretized circularity. Parquet
nulls and NaNs are retained as missing, with NaNs
normalized to null; they are never converted to zero. An empty vision table is
valid and yields null targets for all runs. Missing mean and standard deviation
values remain independently unknown, including when a metric column is absent.
The `training` command rejects absent raw runs; `vision-targets` rejects
an empty run-start table. A failure leaves any previous output intact but
**stale**. Ordinary failed writes remove temporary
files. Parquet inputs and all raw files are hashed before use and checked again
before replacement, including raw additions and removals. This is not a lock
against changes after that check, nor proof of acquisition timing.

## Schema and provenance

The result has one row per run-start row, in source order, via a one-to-one
left join on `id`. `id` (int64) and `ts_start` (UTC microsecond timestamp) are
metadata for joining and chronological evaluation, not model inputs. The 15
candidate features are the six context fields and nine fixed `param.<key>`
columns documented in [the run-start dataset](training-dataset.md). Parameter
values are canonical JSON scalar strings, or null. The run-start source's
`terminal_status` is excluded entirely. No durations, end times, phases,
checks, logs, labels, image metrics, or identifier-derived fields are features.

The 20 nullable float64 targets have these fixed names, even when missing from
vision input:

- `vision.spread.area_rel`, `vision.spread.diameter_rel`,
  `vision.spread.offset_rel`, `vision.spread.circularity`
- `vision.spread.hole_frac`, `vision.spread.uncovered_frac`,
  `vision.spread.l_mean`, `vision.spread.l_std`
- `vision.spread.circle_diameter_rel`, `vision.spread.circle_offset_rel`,
  `vision.spread.residue_frac`
- `vision.fold1.area_rel`, `vision.fold1.l_mean`, `vision.fold1.l_std`,
  `vision.fold1.iou`
- `vision.fold2.area_rel`, `vision.fold2.l_mean`, `vision.fold2.l_std`,
  `vision.fold2.iou`, `vision.fold2.area_ratio`

`vision.spread.zone_r_px` is accepted in the source but excluded from the
output. See [vision metric definitions](pipeline.md#vision-score-columns)
for units and interpretation. Targets are computed after the run by reading
robot overlay annotations, not by independent human quality assessment. The
vision exporter chooses the **last listed** attempt per check, not necessarily
the last successful attempt, and missing or unusable overlays cause missing
measurements. Missing targets are unknown, not failures or zeros.

The Arrow schema metadata key `maus.training.manifest` contains JSON with
`schema_version`, `metadata`, `candidate_features`, `targets`,
`parameter_encoding`, `availability`, `target_semantics`, `coverage`, and
`provenance`. Coverage records the total rows and the number with at least one
non-null allowlisted target, not the number of matching vision IDs. An ID-only
vision row has no target coverage.
Provenance records the relative source paths and SHA-256 hashes of the exact
Parquet bytes read. The raw snapshots are the trust boundary for run-start
values: their contents and the comparison do not prove field availability at
run start or acquisition timing. Hashes do not establish vision provenance,
media, exporter version, code, environment, or a common snapshot. Matching IDs
and hashes do not prove that vision came from the same raw snapshot. Save
the source files privately together with code revision, environment/lockfile,
fetch window and media coverage for reproducibility; rebuild both inputs from
consistent snapshots where possible.

## Evaluation

Sort by `ts_start` and reserve a later chronological holdout **before** feature
selection, target filtering, model selection or fitting. Keep equal timestamps
on the same side of the split. Fit imputation, encoding, scaling and selection
on training data only. Examine coverage and missingness by time, robot,
population and target, including changes across the holdout boundary. Vision
availability depends on downloaded overlays, processing and run history, so the
scored subset may be biased. Repeat configurations, temporal drift and robot
changes can inflate apparent performance. These observational measurements do
not establish causation or deployment readiness.
