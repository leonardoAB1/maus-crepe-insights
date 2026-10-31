# Private visual-phase labeling pilot

Clone the public code, fetch data with your own Maus account, then annotate
locally in Label Studio Community **1.23.2**. No private dataset, database,
account, password or human labels ship with the source repository.

The current human taxonomy is **`crepe-visual-v2`**, exactly:

- `emptyplate`
- `spreading`
- `circle`
- `halffold`
- `quarter fold`
- `ugly`

Spelling and spaces are intentional. There is no `unclear` choice. These are
human visual labels, **not** operational robot phases. No automatic mapping is
made. The owner still needs to agree annotation criteria, particularly `ugly`,
before using labels as a training target. This workflow invents neither those
criteria nor geometry semantics. Geometry, outlines, holes and masks are future
work.

## Supported environment

The isolated `code/labeling/pixi.toml` and `pixi.lock` target **Linux x86-64**
and **Apple Silicon macOS** (`osx-arm64`, M-series Macs).
They pin Label Studio 1.23.2 and FFmpeg/ffprobe 7.1.1, with Python 3.12 and locked
transitive dependencies. They do not change the exporter's dependency graph.
Linux x86-64 with Pixi 0.83.0 was installed and exercised. Apple Silicon with
Pixi 0.82.0 is covered by synthetic video and installed-app integration tests.
CI runs the isolated labeling tests on Linux and Apple Silicon macOS.
Intel Macs, Windows and Linux ARM are not supported by this standalone lock.
On Apple Silicon, use a native arm64 terminal, not a shell running under
Rosetta. Check with `uname -m`: it should print `arm64`. This is a local
single-user workflow, not a shared deployment.

## 1. Clone and install

Install Git and curl using your OS package manager. Install Pixi following its
[installation guide](https://pixi.sh/latest/installation/), then open a new
shell. The official macOS/Linux installer can be downloaded and reviewed first:

```bash
curl -fsSL https://pixi.sh/install.sh -o "$HOME/pixi-install.sh"
# Review the downloaded script before running it.
bash "$HOME/pixi-install.sh"
export PATH="$HOME/.pixi/bin:$PATH"

git clone https://github.com/leonardoAB1/maus-crepe-insights.git
cd maus-crepe-insights
export SOURCE="$(pwd -P)"
pixi install --locked --manifest-path "$SOURCE/code/pyproject.toml"
pixi install --locked --manifest-path "$SOURCE/code/labeling/pixi.toml"
```

For an existing checkout, pull the latest `main` instead of cloning again,
then set `SOURCE` to its absolute path and run both install commands. Run
the commands below in the same Bash or zsh shell; both are supported. Do not
use an unrelated private machine's runtime or copy its database.

## 2. Fetch a new private snapshot

Keep this shell open so its variables remain available. Use a **new** private
directory for each snapshot. The example is gitignored; an external private
directory is also allowed. Never put data under `code/` or `documentation/`.

```bash
umask 077
mkdir -p "$SOURCE/data"
export PRIVATE="$SOURCE/data/phase-snapshot"
mkdir -m 700 "$PRIVATE"
df -h "$PRIVATE"
printf 'First date visible to your account (YYYY-MM-DD): '
read -r FROM
printf 'Last requested date (YYYY-MM-DD): '
read -r TO
pixi run --locked --manifest-path "$SOURCE/code/pyproject.toml" \
  export --data-dir "$PRIVATE" fetch --from "$FROM" --to "$TO" --media --video
```

Authenticate through the exporter's normal OAuth browser flow using **your own
Maus account**. Tokens remain in `~/.maus-export/`; never paste them into chat,
a public issue or Git. Maus access is independent of GitHub repository access.
See [data access](data.md) for the exporter's broader workflow.

Choose a date window with at least 50 accessible runs, including tests. Fetch
above downloads all visible runs/videos in that window; video storage can be
large. It does not claim all historical or inaccessible data. The exporter's
current omitted-date defaults are 2026-08-30 through 2026-12-30, so supply your
own dates explicitly. `--limit 50` limits the server-returned list, rather than
independently sorting the full visible history; omit it for an independently
sorted local selection. Do not use `--hide-tests` for this pilot.

Existing cached JSON is reused unless `--refresh` is passed. Even `--refresh`
does not delete older cached runs or replace nonempty downloaded videos.
Preparation reads **all** cached run JSON, not just the last fetch selection.
A new snapshot directory avoids silently mixing date windows. Failed media
requests can be logged even when fetch exits successfully: inspect the private
fetch output and the preparation counts below. Never publish those logs.

## 3. Prepare the 50 newest locally cached runs at 5 seconds

```bash
export PILOT="$PRIVATE/labeling/pilot-v1"
export APP_DATA="$PRIVATE/labeling/app"
pixi run --locked --manifest-path "$SOURCE/code/labeling/pixi.toml" \
  prepare --data-dir "$PRIVATE" --output "$PILOT"
```

Selection sorts by `tsStart`, then numeric run ID descending. Missing or
undecodable videos retain their place in the selected 50 with explicit status;
**older videos are never substituted**. Check `selected`, `ok`, `missing_video`,
`invalid_video` and `frames` in the aggregate output. For a complete 50-video
pilot, require `selected=50`, `ok=50` and both error counts zero. Otherwise fix
the source download and prepare a new output directory, or deliberately review
the recorded missingness before importing a partial pilot. Setup rejects a
pilot with no usable frames.

The output must be a new direct child of `$PRIVATE/labeling`. Existing pilots
are never overwritten. Preparation reserves 2 GiB plus a per-video frame
allowance. Unsupported or ambiguous timestamps abort. Ordinary failures remove
the unfinished output; forced termination may leave a directory without
`manifest.json`. Absence of that final manifest means incomplete, not importable.

### Shared rectified images for all labels

To use the exact same pixels for phase labels, segmentation masks and training,
rectify the extracted images **before** importing any project. Do not rectify
or resize images in place after annotation. Keep the raw pilot unchanged:

```bash
export RAW_PILOT="$PILOT"
export PILOT="$PRIVATE/labeling/rectified-v1"
pixi run --locked --manifest-path "$SOURCE/code/pyproject.toml" \
  python -m maus_export.labeling_rectification \
  --data-dir "$PRIVATE" --pilot "$RAW_PILOT" --output "$PILOT"
```

This requires check images from `fetch --media` in the same private snapshot.
The existing experimental alignment estimator requires two independently
accepted temporal samples to agree against a same-run square overlay.
It estimates a video-resolution transform rather than scaling a check-image
transform. Only raw video pixels are warped; overlay paint is never copied.
Videos are not re-encoded, and original selected frame indices, PTS and time
bases stay unchanged.

Check the aggregate `selected`, `ok`, `rejected` and `frames` counts. Rejected
runs have no output frames and an explicit reason in the private manifest.
They never fall back to raw images. A pilot with zero accepted frames cannot
be imported. A partial pilot requires a deliberate review of that missingness.
Existing output directories are never overwritten; choose a new name to retry.

The pilot stores rectified PNGs, source-image hashes, the original manifest
hash, per-run transforms with reference/video hashes, and binary validity
masks. Startup verifies image, transform and mask hashes and image dimensions.
Validity masks are preprocessing artifacts, not human segmentation labels.
Exclude their unsupported pixels from downstream losses and measurements.
Both phase and mask projects must reference this same pilot and image hashes,
not independently regenerated or resized copies. Exports carry the frame
coordinate space and preprocessing version.

**Visually review accepted outputs before labeling.** Temporal consistency is
an experimental safeguard, not physical camera calibration. Lens distortion,
parallax and camera movement can still invalidate a transform. Review static
plate/background alignment across the run, not just the food silhouette.
Keep these limitations when reproducing preprocessing for live inference.

## 4. Start, sign up, configure and import

First launch permits local signup. No account is created by the helper:

```bash
pixi run --locked --manifest-path "$SOURCE/code/labeling/pixi.toml" \
  start --pilot "$PILOT" --app-data "$APP_DATA" --port 8080 --allow-signup
```

Open `http://127.0.0.1:8080`, create your own Label Studio account, then stop the
server with Ctrl-C. Label Studio login is separate from Maus OAuth. In the same
shell, enter the email you just registered; this is not a password prompt:

```bash
printf 'Your existing local Label Studio email: '
read -r OWNER_EMAIL
pixi run --locked --manifest-path "$SOURCE/code/labeling/pixi.toml" \
  setup --pilot "$PILOT" --app-data "$APP_DATA" --owner-email "$OWNER_EMAIL"
# Rerunning the same command verifies the project without duplicating tasks.
pixi run --locked --manifest-path "$SOURCE/code/labeling/pixi.toml" \
  setup --pilot "$PILOT" --app-data "$APP_DATA" --owner-email "$OWNER_EMAIL"
pixi run --locked --manifest-path "$SOURCE/code/labeling/pixi.toml" \
  start --pilot "$PILOT" --app-data "$APP_DATA" --port 8080
```

Keep the same pilot and app-data pair on each launch. Ordinary startup disables
open signup. Do not distribute invitation links or add untrusted accounts:
Community is a single-organization app, **not a per-user isolation boundary**.
Use separate local apps and reviewed private exports for collaboration. There
is no automatic push, invite, shared-server deployment or annotation merging.

The helper binds **127.0.0.1**, not `0.0.0.0`. Label Studio's `--host` is not a
bind-address setting. If 8080 is busy, Label Studio can choose another port;
read its startup output or choose a free `--port`. Do not expose the listener
through a public tunnel or broaden the local-file root to fix image loading.

Setup uses the installed app's management shell and validated project, task
and storage serializers. It requires an existing active account/organization,
validates frame hashes and PNG decoding, and registers each `run-*` image
folder as **per-project LocalFilesImportStorage**. Setting local-file environment
variables alone is insufficient in this release. It does not sync storage or
create predictions, annotations, drafts or model backends. **Do not click Sync**
on those source storages: tasks already contain the manifest provenance.

A successful result prints project ID, task count, storage-root count and
whether it was created. Creation is transactional; reruns read back the config,
owner, every task and storage root, preserve human labels and fail on drift.
Do not rename/reconfigure managed projects or edit imported task data. Failed
setup details stay in `$APP_DATA/phase-setup.log`, never public logs.

## 5. Annotate and review

Open the project returned by setup. Confirm a real image loads before labeling.
Choose exactly one of the five visual labels and **Submit**. Required single
choice prevents submitting an empty ordinary annotation; **Skip** remains
available when you cannot label confidently. No extra uncertainty class is
added. Never use `ugly` as an invented synonym for missing or uncertain.

The recorded robot phase sequence is displayed as **UNALIGNED** context. It is
not a frame-specific suggestion and must not be translated automatically into
the human visual taxonomy. Context can still bias reviewers. The imported task
records `phase_context_shown=true` (configured to be displayed),
`suggestion_shown=false`, `phase_suggestion=null` and
`human_taxonomy_version=crepe-visual-v2`. This is display configuration, not proof
that a particular reviewer read the context.

Agree class criteria with the dataset owner, review a shared sample, record
disagreements and resolve them before using labels as independent ground truth.
Record annotation-guide and taxonomy versions with each export. Changing the
five labels requires an explicit versioned migration, never a silent rewrite.

## 6. Export, privately share and back up

Use the project's **Export** action and choose **JSON**, without downloading
resources. Save it in a new `$PRIVATE/labeling/exports/` directory, not the public
source tree or a synced public downloads folder. Include all tasks when auditing
coverage. Label Studio JSON is the annotation source of truth:

- A submitted human label is in `annotations[].result[]` with
  `from_name=visual_label`, `type=choices`, and exactly one of the five values.
- `was_cancelled=true` is skipped, not a visual label.
- No submitted annotation is unlabelled; drafts are not reviewed ground truth.
- Imported `data.human_label=null` and `review_status=unreviewed` are immutable
  preparation provenance, **not** the current annotation state. Read annotations
  rather than treating those fields as live labels.
- Predictions and recorded robot context are never human labels.

The user-approved sharing destination is the separate **private** repository
`salarehman/maus-crepe-labels`. Verify that destination is still private and that
recipients are authorized before any manual sharing. This setup does not create
it, grant access or push anything. Review exports locally first. Share only
reviewed human annotation JSON and the minimal private provenance/taxonomy
metadata needed to identify its frames (for example image hash, requested
seconds and source manifest hash). Check for unexpected URLs, account details,
comments and private source fields. Never include raw images, videos, SQLite
DBs, app settings, secret keys, OAuth caches or backups in either source Git or
the labels repository. Labels and hashes remain private even without media.

For a complete local backup, first stop Label Studio, then copy the entire app
directory and immutable pilot to a restricted backup destination. Keep the raw
snapshot separately if you need to regenerate frames. Example, while stopped:

```bash
mkdir -m 700 "$PRIVATE/labeling/backups"
cp -a "$APP_DATA" "$PRIVATE/labeling/backups/app-v1"
cp -a "$PILOT" "$PRIVATE/labeling/backups/pilot-v1"
```

Use new backup names each time. App backups contain secrets and must never be
committed, even to the labels repo. Restoring at the same absolute paths keeps
LocalFilesImportStorage references valid. Relocating a DB is not supported by
this helper: preserve it and plan a reviewed path/annotation migration instead
of editing SQLite or reimporting blindly. Keep exports alongside backups as a
portable review format; JSON alone is not a full app restore.

## Sampling and source provenance

Requested video offsets are 0, 5, 10, ... seconds strictly before stream end.
The selected image is the last decoded frame whose PTS is at or before the
offset from the video's start PTS; it can precede the requested time. The
preparer checks frame PTS equals best-effort PTS, increasing timestamps and
stream boundaries, then compares selected indices, PTS and time base from
`ffprobe` against FFmpeg `showinfo`. It hashes source JSON/video before and
after extraction, and records PNG hashes in the final private manifest.

The manifest stores run IDs/start times, relative source paths, hashes, stream
start PTS, rational time base, duration ticks, frame index, decoded PTS and
requested seconds. Multiply ticks by time base to obtain seconds. Its raw phase
sequence timestamps remain robot-local naive values; they are not UTC frame
labels. Alignment between video start and robot run start is unverified.

Legacy manifest `classes` is the recorded operational phase union plus
`unclear`. It remains source provenance and is validated against the recorded
sequence. **It does not define the current human choices.** Setup leaves the
original manifest byte-for-byte unchanged. The fixed human taxonomy is
versioned separately in `workflow.py`, task data and project description.

Setup verifies the manifest's internal recorded-phase consistency and every
image hash/decodability. It does not re-fetch Maus or re-hash all original
videos: retain the immutable source snapshot and preparation manifest for
source audits. FFmpeg/platform changes can change PNG bytes even for equivalent
frames, so retain the lock and original images when sharing label identities.

## Privacy boundaries and verification

The app database, uploads, logs and generated secret key are under the supplied
private `--app-data` directory (0700; new files use umask 077). Only the pilot is
the local-file document root, and only its registered frame subdirectories are
served to authenticated users. The manifest, raw export, repository and DB are
outside those registered folders. No symlinks or extra files are allowed in
frame folders. Keep them immutable while the server runs: LocalFiles storage
is a directory permission boundary, not a per-file allowlist.

Analytics, latest-version checks, backend Sentry and frontend Sentry are
explicitly disabled. Ambient Label Studio settings/credentials are not inherited.
This is **not an egress sandbox**; do not connect webhooks, ML backends, remote
storage or remote task URLs. An OS firewall is needed for a hard offline policy.

Run synthetic tests without Maus access or private frames:

```bash
pixi run --locked --manifest-path "$SOURCE/code/labeling/pixi.toml" test
cd "$SOURCE/code"
pixi run test tests/test_phase_pilot.py tests/test_labeling_workflow.py
pixi run lint
pixi run docs-check
```

The isolated test exercises a fresh DB, supported management shell/serializers,
per-project storage registration, authenticated image delivery with exact byte
comparison, unauthenticated denial, file-root restrictions, empty initial
annotations/predictions/drafts, idempotent import, human JSON export retention,
unsafe-pilot rejection and actual loopback startup/shutdown. Default exporter
tests skip installed-app integration when Label Studio is absent. Login-page
startup alone is not evidence that imported images or annotations work.
