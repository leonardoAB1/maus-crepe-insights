# Vision-target training dataset

This is an observational dataset, not a trained model or verified quality
labels. Its intended prediction time is **before a run starts**. The inputs
are candidates only: the dashboard snapshot does not prove that settings and
context were available at that time or unchanged afterward. Verify each field's
provenance and timing before use; otherwise it may leak future information.

## Reproduce the dataset privately

A collaborator needs their own Maus dashboard account and access to the desired
runs. From this checkout's `code/` directory, install the pixi environment and
choose an absolute private data directory. This example uses a separate
subdirectory under this worktree's gitignored `data/`:

```bash
pixi install
umask 077
DATA_DIR="$(cd .. && pwd)/data/collaborator-export"
```

Choose a new, empty subdirectory for a clean export, or reconcile any existing
cache before proceeding. If choosing another location, check that it is private
and gitignored. Raw records, images, tables, audits and other derivatives must
stay private and out of git. Never commit them, share links, or paste logs,
run IDs, tokens or rows into issues, pull requests or other public places.

Fetch with the collaborator's own account. The first OAuth sign-in opens a
browser; sign in there with that account. Credentials are cached in
`~/.maus-export/`, so use a local user account with the collaborator's own
credentials, not a shared cache or someone else's token. An existing
`MAUS_TOKEN` bypasses browser sign-in; make sure it is not another account's.

```bash
pixi run export --data-dir "$DATA_DIR" fetch --media
```

`--data-dir` is a global option and must precede the subcommand. `--media`
downloads vision-check images needed by the scorer; video is not needed. Without
`--limit`, the fetch is still restricted by the default dates in
`code/maus_export/fetch.py` and by the account's permissions. Inspect those
current defaults and choose a window appropriate for the account. To override
both dates, set `FROM_DATE` and `TO_DATE` to the desired dates in `YYYY-MM-DD`
format, then use this instead of the fetch command above:

```bash
pixi run export --data-dir "$DATA_DIR" fetch --media \
  --from "$FROM_DATE" --to "$TO_DATE"
```

A cached data directory can contain runs from earlier fetches outside the
latest selection. Both downstream builders read all cached raw runs, so
reconcile the cache against the intended window and account scope before
building. A fetch can log failed downloads and still exit successfully. Logs can
contain private media URLs, so inspect them locally and never paste or share
them. Privately compare expected check images in the raw records with local
media and verify that the required overlays are readable. Missing or unusable
images reduce score coverage; successful exit does not prove complete media.

Then build both inputs and the joined output from the same data directory:

```bash
pixi run export --data-dir "$DATA_DIR" training
pixi run export --data-dir "$DATA_DIR" vision
pixi run export --data-dir "$DATA_DIR" vision-targets

# Optional: audit training and vision coverage locally.
pixi run export --data-dir "$DATA_DIR" audit
```

Check coverage and missingness privately, including per-target coverage and
join counts in the [training audit](training-audit.md). The audit does not
prove that media is complete. Record the fetch window, account access scope,
media coverage, code revision and environment privately; separate accounts or
windows may produce different datasets. Candidate input timing and vision
provenance remain unverified even if the commands succeed. See the provenance
and evaluation sections below before training or interpreting scores.

`vision-targets` reads `training/run_start.parquet`, `vision.parquet`, and the
cached `raw/runs/*.json` from the selected data directory, and replaces only
`training/vision_targets.parquet` there. The default is this worktree's
`data/`, not the main checkout's. `export all --media` builds vision but does
not build either training artifact. Run `training` and `vision-targets`
separately after it. The command does not fetch data, score images, modify the
inputs or update an audit. A successful `vision-targets` command is silent.

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
metadata for joining and chronological evaluation, not model inputs. The 16
candidate features are the six context fields and ten fixed `param.<key>`
columns documented in [the run-start dataset](training-dataset.md) (schema
version 2). Parameter
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
