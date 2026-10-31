# Two-person labeling: set 1 and set 2

For optional pre-masking in either set, use the
[preview-first mask-assistance guide](mask-assist.md). It includes the newer
feature checkout and exact collaborator/agent commands; existing annotation
workspaces are reused, not replaced.

This is the complete handoff runbook for the dataset owner, the collaborator
and an agent helping either person. Read the repository's [AGENTS.md](../AGENTS.md)
first. Those remain the project rules; this guide explains the labeling workflow.

## Read this before doing anything

- **Set 1 belongs to the owner. Set 2 belongs to the collaborator.**
- These are original video frames. Do not rectify, crop, resize, redraw or
  regenerate the input PNGs. The bundle preserves their exact bytes and hashes.
- Entire robot runs belong to one set. No run or frame belongs to both sets.
  The split balances frame counts subject to keeping runs whole. Exact 50/50
  frame counts are not guaranteed when run lengths differ.
- Each set has its own local app/database and two projects: phase labels and
  material/hole masks. Both projects reference the same assigned image files.
- These are annotation assignments, **not** automatically train/test splits.
  Later model evaluation still needs an explicit run-grouped split.
- Existing full-dataset and rectified projects are not edited, deleted or
  migrated. Stop annotating those queues when switching to an assigned set.
  Existing annotations stay there; splitting does not copy human labels.
- The code is public. Frames, bundles, manifests, annotations and databases
  are private. Never commit them or upload them to a public issue or PR.

## What the collaborator receives

The owner supplies, using an approved private channel:

1. `set-2.tar.gz`, containing only set 2's original PNGs and minimal manifest.
2. `set-2.tar.gz.sha256`, or its expected checksum confirmed with the owner.
3. A link to this guide and the code revision/branch to use.

The collaborator does **not** receive the owner's app database, passwords,
session cookies, tokens, `.env`, raw dashboard JSON, videos or set-1 images.
No Maus dashboard account or OAuth login is needed for this handoff.
The helper does not upload or send the bundle anywhere.

The checksum detects corruption; by itself it does not authenticate a sender.
Confirm the expected checksum through a trusted channel. Do not simply compute
a checksum from an unknown download and treat that as the owner's approval.

## Supported machines

The standalone lock supports Apple Silicon macOS (`osx-arm64`) and Linux
x86-64 (`linux-64`). Native Windows, Intel macOS and Linux ARM are not covered.
On an M-series Mac, `uname -m` should print `arm64`, not a Rosetta architecture.
Use a supported machine rather than silently rewriting the lock or installing
packages with pip, conda, uv or a system Python.

Install Git and [Pixi](https://pixi.sh/latest/installation/) first. The first
Pixi install downloads the locked environment, including Label Studio 1.23.2.
Afterward annotation uses local files. Only the standalone labeling environment
is needed, not the exporter, ML environment, dashboard connection or videos.

## Collaborator/agent: get the correct code

Until this feature is merged, use the published `feat/119-labeling-sets` branch.
Do not assume an older `main` checkout has the helper. After merge, the owner
may provide the corresponding merged revision instead.

For a new clone:

```bash
git clone https://github.com/leonardoAB1/maus-crepe-insights.git
cd maus-crepe-insights
export REPO="$(pwd -P)"
git fetch origin feat/119-labeling-sets
git worktree add -b feat/119-collaborator-set-2 \
  worktrees/feat/119-collaborator-set-2 origin/feat/119-labeling-sets
export SOURCE="$REPO/worktrees/feat/119-collaborator-set-2"
pixi install --locked --manifest-path "$SOURCE/code/labeling/pixi.toml"
```

If the repository or worktree already exists, inspect its status and revision
first. Do not clone over it, switch somebody else's branch, discard changes or
reuse another person's working directory. No source edits or new rule files are
needed to run labeling. Read-only runtime use does not require a new PR.

Keep this terminal open so the variables below remain available. Commands work
in Bash and zsh. Use permanent, canonical absolute paths: `pwd -P`, not symlinked
paths. Moving an app database after its first launch is not supported.

## Collaborator/agent: receive set 2 safely

Put the two private handoff files in `$REPO/data/handoff/`. Do not put them in
`code/`, `documentation/`, a public shared folder or an automatically public
cloud-sync destination. Review the checksum with the owner, then run:

```bash
umask 077
export BUNDLE="$REPO/data/handoff/set-2.tar.gz"
export EXPECTED_SHA="$(cut -d ' ' -f 1 "$BUNDLE.sha256")"
export WORKSPACE="$REPO/data/labeling/set-2"

pixi run --locked --manifest-path "$SOURCE/code/labeling/pixi.toml" \
  label-sets receive --set 2 --bundle "$BUNDLE" \
  --sha256 "$EXPECTED_SHA" --output "$WORKSPACE"
```

`WORKSPACE` must be a **new** directory. The receiver checks the outer checksum,
assignment number, every image hash and PNG, and a strict file allowlist. It
rejects traversal, links, devices, duplicate members, unexpected files, missing
images and oversized archives. It never uses unrestricted tar extraction.

A successful result prints set number, task count and assignment identifier.
If it says set 1 or the owner-provided assignment differs, stop. Do not rename
directories, edit manifests or pass a different set number to bypass checks.
On a later session, reuse the received workspace; do not receive again over it.

Expected private layout:

```text
data/labeling/set-2/
  pilot/
    manifest.json
    run-.../
      ....png
  app/                 # created at first startup, never shared
  workspace.json       # project IDs and immutable workspace identity
  exports/             # private human annotation exports
```

## Collaborator: start annotating with one command

```bash
pixi run --locked --manifest-path "$SOURCE/code/labeling/pixi.toml" \
  label-set --set 2 --workspace "$WORKSPACE"
```

Keep the command running. It starts a loopback-only server, normally at
**http://localhost:8083**. Use the URL it prints.

### First launch only

1. Open the printed local URL and create your own local Label Studio account.
   It is separate from GitHub, Maus OAuth and the owner's Label Studio account.
   Do not send your password to an agent or another person.
2. Leave the command running. It detects the completed local signup and imports
   the two assigned projects automatically. No email/ID copying or second setup
   command is needed.
3. The helper briefly restarts the app with open signup disabled. Wait until it
   prints `phase_url` and `mask_url`, then refresh the browser if needed.
4. Verify the titles say **Set 2 phase** and **Set 2 masks**, with the assigned
   task count in each. The queues contain the same images, not twice as many
   independently assigned frames.

Use only trusted local accounts. Community Edition is not a per-user isolation
boundary for a shared server. This workflow is one person per local app, not a
public collaborative deployment. The helper refuses multiple active owners.

### Every later launch

Run the exact same command with the same workspace. Existing projects are
verified and reused; human annotations are not overwritten or duplicated.
Do not delete `app/`, receive another bundle over it, edit `workspace.json`,
rename managed projects or change their labeling configuration.

Submit or save current work before stopping with Ctrl-C. The app stops; the
database and annotations remain. Do not close the terminal while expecting
the server to keep running. On restart, sign in to your existing local account.

Use the printed `localhost` hostname consistently. Cookies are scoped by host,
not by port. If reviewing several independent apps on one machine, use separate
browser profiles. Do not assume another port reuses your account safely.

## Which project to label

Follow the owner's priority. The helper creates both queues so phase labeling
can start immediately and masks can be added without another service setup.
Submitting a phase label does not submit a mask annotation, or vice versa.

### Set 2 phase

Open its printed project URL and click **Label All**. Choose exactly one of:

`emptyplate`, `spreading`, `circle`, `halffold`, `quarter fold`, `ugly`.

Then click **Submit**. Use **Skip** when unsure. These names and spaces are the
versioned `crepe-visual-v2` taxonomy. Do not add `unclear`, rename classes or
introduce a triangle label without a reviewed taxonomy change.

Label the visible image. The robot's sequence is explicitly **UNALIGNED**
context, not a per-frame suggestion or human label. A utensil in view alone
does not establish that spreading is happening. There are no automatic labels.
The owner must agree borderline examples, especially `ugly`, holes in an
otherwise circular crêpe and occluded folds. Do not invent quality thresholds;
skip ambiguous cases until the owner supplies criteria.

### Set 2 masks

See the full [mask annotation contract](mask-labeling.md). In short:

- Green `crepe_material`: visible batter/crêpe pixels, excluding holes and
  occluding tools/toppings. Do not invent hidden material behind folds.
- Red `hole`: visible bare hotplate enclosed inside the crêpe silhouette.
  Ragged open outside edges, shadows, glare and toppings are not holes.
- Keep the masks disjoint. Use the eraser to remove overlaps.
- Select `crepe_present` when material is painted; `no_crepe` only for an
  explicitly reviewed image with no visible crêpe and no foreground masks.
- Click **Submit**, or **Skip** for uncertain boundaries, steam or occlusion.

Viewer zoom is fine; changing the underlying image file or coordinate system
is not. These assignments use original images with no rectification. Do not
interpret camera pixels as physically calibrated diameter, angle or area.

## Save and return the work

Create a private export directory:

```bash
umask 077
mkdir -p "$WORKSPACE/exports"
```

In each project's **Export** action choose **JSON**, include all tasks when
auditing coverage, and do **not** download resources. Save separately as:

- `$WORKSPACE/exports/set-2-phase.json`
- `$WORKSPACE/exports/set-2-masks.json`

The Label Studio annotation JSON is the source of truth. Drafts, skipped tasks,
predictions and unlabelled tasks are not reviewed negatives. Preparation fields
such as `human_label=null` and `review_status=unreviewed` are immutable provenance,
not current annotation status. Read `annotations[].result[]` for human labels.

To produce native-size training PNGs from reviewed masks:

```bash
pixi run --locked --manifest-path "$SOURCE/code/labeling/pixi.toml" \
  export-masks --pilot "$WORKSPACE/pilot" \
  --annotations "$WORKSPACE/exports/set-2-masks.json" \
  --output "$WORKSPACE/exports/set-2-masks-v1"
```

Choose a new output name on each export. The exporter refuses source-provenance
drift, wrong dimensions, rotated regions, overlapping classes, missing review
states and conflicting submitted annotations. Fix the annotation, not the
validator or image hash. Original-frame exports have material/hole binary PNGs
and class PNGs: 0 = other scene, 1 = material, 2 = hole.

Return the human JSON exports through the owner's approved private channel,
with the assignment ID and set number. Keep untouched local originals. Before
any repository sharing, review a separate handoff copy for usernames, email
fields, comments and unnecessary private metadata. The approved private labels
repository is for reviewed annotation JSON and minimal provenance only, not
this frame bundle, images, app databases, settings, cookies, tokens or backups.
Do not push exports anywhere without the owner's approval.

### Owner: reconcile both sets

Verify matching assignment IDs and parent-pilot hashes, the expected set numbers
and membership, then reconcile by `(assignment_id, assignment_set, run_id,
decoded_frame_index, requested_seconds)`, checking image hashes as well.
Local numeric task IDs and annotation IDs can collide between machines and must
not be used as global identifiers. Identical image hashes can also occur in
several frames; keep frame provenance. Join phase and mask labels by this frame
identity, never by local task ID. Do not merge SQLite databases.

## Owner: prepare and start set 1

From a checkout containing this feature, set `SOURCE` to its canonical root and
`PRIVATE` to the existing private snapshot directory. Do not re-fetch or
regenerate the pilot while splitting it.

```bash
export ASSIGNMENTS="$PRIVATE/labeling/assignments-v1"
pixi run --locked --manifest-path "$SOURCE/code/labeling/pixi.toml" \
  label-sets split --pilot "$PRIVATE/labeling/pilot-v1" --output "$ASSIGNMENTS"

pixi run --locked --manifest-path "$SOURCE/code/labeling/pixi.toml" \
  label-set --set 1 --workspace "$ASSIGNMENTS/set-1"
```

The split copies original PNG bytes into two new private workspaces, strips
unneeded source fields and assigns whole runs with a deterministic largest-first
balance. It writes a private `assignment.json` audit record. Set 1 normally runs
at **http://localhost:8082**, with the same first-signup automation as set 2.
It is a new local app; previous full-dataset accounts are not copied.

Create the private set-2 handoff separately:

```bash
pixi run --locked --manifest-path "$SOURCE/code/labeling/pixi.toml" \
  label-sets bundle --set 2 --workspace "$ASSIGNMENTS/set-2" \
  --output "$PRIVATE/labeling/handoff/set-2.tar.gz"
```

Send only that bundle, its `.sha256` sidecar and this guide to the authorized
collaborator using a private channel. No upload happens automatically. Set 1 is
not included. Do not send the entire snapshot directory or app database.
Never rerun `split` over existing output to rebalance work in progress. Use a new
assignment version only after agreeing how completed labels will be reconciled.

## Troubleshooting and agent checklist

| Symptom | Safe next action |
| --- | --- |
| `label-set` task is missing | Verify the published feature revision and correct standalone manifest |
| Unsupported platform | Use Apple Silicon macOS or Linux x86-64; do not improvise pip installs |
| Checksum/set mismatch | Ask the owner for the correct bundle and confirmed checksum |
| Workspace already exists | Reuse it for startup; never overwrite or delete it to make receive succeed |
| Port busy | Keep existing work running; use an explicit free `--port`, not an unknown process kill |
| Launcher lock busy | Use the existing app, or stop its helper cleanly before restarting |
| Waiting for signup | Open the printed local URL and complete one local account registration |
| Signup/import fails | Inspect private `app/server.log` and `app/assignment-setup.log`; redact before sharing |
| Images do not load | Verify the bundle, hashes and managed storage; do not broaden the file root or click Sync |
| Workspace moved/config changed | Preserve the DB and old paths; plan a reviewed migration rather than editing SQLite |
| Login disappears between apps | Use the printed hostname consistently and separate browser profiles |
| Export rejects masks | Review the named condition and fix the human annotation, not provenance metadata |

Agents should verify the set number, image count, original coordinate space and
project titles before reporting success. Wait for the final project URLs, not
just a login page. Test one real image loads. Keep server output and diagnostics
private. Do not report annotations as completed merely because tasks imported.

Do not connect remote ML backends, webhooks, cloud storage, public tunnels or
robot command tools. The server binds to loopback; analytics and error-reporting
settings are disabled by the existing launcher. This is not a hard egress
sandbox; use an OS firewall if an absolute offline policy is required.

Stop the app before a complete restricted local backup. Preserve `pilot/`,
`app/`, `workspace.json` and exports together at their original absolute paths
for restore. Backups contain secrets and must never be committed or handed to
the other annotator. If unsure, stop and ask the owner instead of resetting data.
