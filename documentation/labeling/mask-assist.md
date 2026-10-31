# Pixel-mask assistance: collaborator and agent quickstart

Use this guide to add editable mask suggestions to **your own set 1 or set 2**.
This is ordinary graphics processing: fixed colours, neighbouring pixels and
small opening/closing filters. No AI, GrabCut, clustering, learned weights,
calibration or rectification is used.

**The workflow is: status -> sample preview -> full job -> check -> apply.**
Nothing becomes a human annotation until you review it and click Submit.

## Safety contract

- Choose `--set 1` or `--set 2` explicitly. It must match the workspace, pilot
  and job. Project IDs are read from your workspace, not copied from the owner.
- Only the selected mask project is affected. Phase projects stay unchanged.
- Annotations, skipped annotations, drafts, locks and existing predictions are
  protected. Protection is checked again immediately before each insert.
- Existing image files, task data and human work are never overwritten.
- Empty suggestions stay unlabelled, not automatic `no_crepe` examples.
- A sample preview cannot be applied. Prepare a full job from the reviewed
  preview first. Application requires the explicit `--yes` flag.
- Reruns skip existing predictions. There is deliberately no force-overwrite
  or automatic lock/draft deletion option.

Read [AGENTS.md](../../AGENTS.md) first. This guide supplements the operating
workflow; it is not a second agent rulebook.

## 1. Get the feature without disturbing your data

The helper is on `feat/141-mask-assist` until its PR is merged. An older `main`
or the earlier set-1-only prefill branch is not enough. After merge, the owner
may provide the corresponding main revision instead.

Use your existing repository and **existing set-2 workspace**. Do not download
the bundle again over an existing app or copy the owner's database. From the
main repository directory, create a fresh source worktree:

```bash
export REPO="$(pwd -P)"
git fetch origin feat/141-mask-assist
git worktree add -b feat/141-collaborator-mask-assist \
  worktrees/feat/141-collaborator-mask-assist origin/feat/141-mask-assist
export SOURCE="$REPO/worktrees/feat/141-collaborator-mask-assist"

pixi install --locked --manifest-path "$SOURCE/code/pyproject.toml"
pixi install --locked --manifest-path "$SOURCE/code/labeling/pixi.toml"
```

If that worktree already exists, inspect its status/revision and use it only if
it is yours and contains this feature. Never reset another person's worktree.
Only the default runtime and standalone labeling environments are needed.
Do not install the ML environment, pip packages or model weights.

Supported platforms remain Apple Silicon macOS and Linux x86-64. See the
[two-person labeling runbook](labeling-collaboration.md) for the original bundle
handoff, supported platforms and first local signup if not already completed.
No Maus login, video download or remote ML backend is needed here.

## 2. Set paths once

In the terminal used for assistance commands:

```bash
export WORKSPACE="$REPO/data/labeling/set-2"
export ASSIST="$REPO/data/mask-assist/set-2"

assist() {
  pixi run --locked --manifest-path "$SOURCE/code/pyproject.toml" \
    mask-assist "$@"
}
```

Change `WORKSPACE` if your existing set-2 app is at a different location. It
should contain `pilot/manifest.json`, `workspace.json` and
`app/label_studio.sqlite3`. Keep its existing canonical absolute path. Moving a
database is not a migration. `ASSIST` must be private under `data/`, outside the
workspace/pilot; it stores masks and previews, not live annotations.

These examples use Bash/zsh and set 2. The owner can use the same commands with
`--set 1` and their existing set-1 workspace. Never merely rename a set-1 folder
to set-2 or edit assignment metadata to make a check pass.

## 3. Start/reuse the local app

Save your current drawing first. If an older launcher is running, stop it
cleanly with Ctrl-C in its own terminal, then start from the new checkout:

```bash
assist start --set 2 --workspace "$WORKSPACE"
```

Keep that command running in a separate terminal. Normal set-2 access is
`http://localhost:8083`; use the URL printed by the helper. Existing accounts,
projects and annotations are reused. A new workspace needs one local signup;
the helper then imports the projects automatically.

In a second terminal, set `REPO`, `SOURCE`, `WORKSPACE`,
`ASSIST` and the `assist` function there too. The server must remain running for
browser annotation, but preview generation itself does not need a remote server.
Use the printed hostname consistently; separate apps can otherwise collide in
browser cookies. Do not expose the server publicly to share suggestions.

## 4. Inspect status and image dimensions

```bash
assist status --set 2 --workspace "$WORKSPACE"
```

This verifies workspace/pilot identity and reports original image sizes,
eligible untouched tasks, protection counts and the correct mask-project URL.
Counts for annotations, drafts, locks and predictions overlap; do not add them
as if they were disjoint categories. Zero eligible tasks can be a healthy result
when tasks are already protected or prefilled.

If `workspace.json` is missing, complete startup/signup/import first. If a set,
hash, app path or provenance check fails, stop and investigate. Do not rewrite
hashes, task data, project settings or source images to bypass the validator.

## 5. Preview a small sample first

```bash
export PREVIEW="$ASSIST/preview-v1"
assist preview --set 2 --workspace "$WORKSPACE" \
  --output "$PREVIEW" --limit 12 --full-image
```

Open `$PREVIEW/preview.html` locally in a browser, or open
`$PREVIEW/contact-sheet.jpg`. Each tile shows the original on the left and the
suggestion on the right. Green is material; red is a hole candidate. Empty
suggestions are shown too. Samples are spread through eligible task order.
The page records the ROI and threshold settings. It does not change the app.
Thumbnails are resized for display only; masks retain native input dimensions.
If no tasks are eligible, the HTML reports that instead of showing a sheet.

**Inspect quality, not just successful execution.** Warm reflections and tools
can look like batter; dark browning can look like holes. All-blank output on
images with a crêpe means the settings need attention, not reviewed negatives.
There is no automatic semantic confidence/accuracy score.

### Restrict processing to the plate when useful

If status shows 1344x760 original frames, this is one example to try:

```bash
export PREVIEW="$ASSIST/preview-v2"
assist preview --set 2 --workspace "$WORKSPACE" \
  --output "$PREVIEW" --limit 12 \
  --roi 360 100 640 620 --min-area 10000
```

`--roi X Y W H` is in the original pixels. Those coordinates are not universal:
choose a rectangle that fits your actual camera/image sizes. Do not resize the
images or scale a calibration to make a rectangle fit. Without `--roi`, the
whole image is processed. `--full-image` makes that choice explicit.

All prototype controls are available on `preview`; see
[pixel-mask parameters](pixel-masks.md#tune-the-few-thresholds):

- Raise `--min-area` to reject small warm noise; lower it to keep small batter.
- Lower `--min-lightness` for dark browning, or `--min-yellow` for pale batter.
- Use `--keep-all` for disconnected fragments, at the cost of more distractors.
- Use `--opening 1 --closing 1` to disable cleanup when thin batter or holes
  vanish.
- `--max-hole-lightness` changes which enclosed dark regions are candidates;
  it cannot semantically distinguish real holes from browning.

Use a **new output name** when tuning. Never overwrite an old job or existing
label. Previews contain real frame pixels and are private, even though the
algorithm/code is public. Do not commit or publish contact sheets.

## 6. Prepare the full job using the reviewed settings

After inspecting your chosen preview:

```bash
export JOB="$ASSIST/job-v1"
assist prepare --set 2 --workspace "$WORKSPACE" \
  --from-preview "$PREVIEW" --output "$JOB"
```

This copies the preview's exact ROI/thresholds, checks its set/pilot identity,
and recomputes eligibility against the live app. It processes all currently
untouched tasks and writes native-size binary masks and a private job manifest.
A review sheet for up to 12 frames is included. Source pixels are not changed.
Tasks edited since the preview are protected rather than forced into the job.
Full preparation prints aggregate progress periodically.

The final `manifest.json` is the completion marker. Do not apply a partially
written job. An existing output directory is refused; use `job-v2` for a later
generation. Keep job folders for provenance: they record settings and source
hashes. They must not go into the labels repository.

## 7. Check, then explicitly apply

```bash
assist check --set 2 --workspace "$WORKSPACE" --suggestions "$JOB"

# Only after checking the counts and previews:
assist apply --set 2 --workspace "$WORKSPACE" --suggestions "$JOB" --yes
```

`check` is a dry run: no predictions or human annotations are created.
`apply --yes` creates a private consistent backup under `app/backups/`, then
adds editable predictions. Tasks are rechecked immediately before every insert
so a newly saved annotation/draft or lock wins over the proposed pixels.

Read the result: `created`, `protected`, `existing`, `busy` and
`empty_not_imported`. A concurrent edit can reduce the final created count.
Busy tasks can be retried later with the same job; existing predictions are
not duplicated. Do not delete protected work or use SQLite to force coverage.
The operation does not touch the other set or phase project.

An image with an empty suggestion is deliberately left without a prediction.
It is not assigned `no_crepe`. Sample previews cannot be applied, even with
`--yes`; use `prepare` first. Wrong-set jobs are rejected rather than remapped.

## 8. Refine in Label Studio

Save any current drawing before refreshing the mask project. Open a new
untouched task with a suggestion. Previously opened/locked tasks, saved work
and drafts are intentionally left alone, so some tasks will still be blank.

1. Review the green material and red hole regions against the original image.
2. Correct them with the brush/eraser. Clear a wrong suggestion if necessary.
3. Keep material/hole masks disjoint. Do not invent occluded material.
4. Choose `crepe_present` or `no_crepe` yourself; no review state is
   preselected.
5. Click **Submit**, or **Skip** when uncertain.

Label Studio calls stored suggestions "predictions" and may show a model-version
selector. The selected version is `pixel-colour-components-v1`; that name is an
algorithm tag, not an AI model. An imported prediction alone is not a completed
annotation or a training target.

## 9. Export and restart safely

Use the normal
[mask export workflow](mask-labeling.md#private-export-for-training).
Only submitted, reviewed annotations are eligible. Preserve assisted-source
provenance, including prediction origin/parent links where available, in
reviewed handoffs. Do not relabel an assisted annotation as independent manual
ground truth or push raw predictions as human labels. If an older publishing
script requires every region to have manual origin, it needs a reviewed update,
not forged origin values.

Restart with the same new checkout's `assist start` command. Older launchers
predating explicit pixel-suggestion opt-in may refuse the project settings.
Do not delete the database, project marker or predictions to satisfy old code.
App backups contain credentials/session data: keep them private and never send
them to the collaborator or labels repo. No upload happens in this workflow.

## Agent checklist and troubleshooting

Before reporting success, verify the chosen set, original image sizes, mask
project, preview quality and final import counts. Do not edit the other set or
assume numeric task/project IDs are global. Use only the approved workspace.
Existing set-1 predictions are not delivered by Git or a frame bundle.

| Problem | Safe response |
| --- | --- |
| Task missing | Use the feature checkout and root Pixi manifest |
| Missing workspace state | Finish signup/import via `start` |
| Wrong set/pilot | Find the correct workspace/job; never edit hashes |
| ROI outside image | Check sizes; adjust ROI or use `--full-image` |
| Poor suggestions | Tune a new preview; never accept blindly |
| No untouched tasks | Check protection counts; don't clear saved work |
| Output directory exists | Use a new versioned output directory |
| Sample rejected | Use `prepare --from-preview` for a full job |
| Port/lock busy | Use or cleanly stop the existing launcher |
| No visible suggestion | Save, refresh/open a new task; check version |
| Setup refuses opt-in | Use this feature's new launcher |
| Unexpected predictions | Stop; don't replace unrelated work |
| Import failure | Inspect private `app/pixel-import.log`; redact logs |

No public tunnels, remote ML services, raw data uploads or robot commands are
needed. Code is public; previews, masks, job manifests and databases are not.
