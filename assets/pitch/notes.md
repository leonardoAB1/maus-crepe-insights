# CrêpeFlow pitch: talk track, numbers and sources

BaselHack 2026, Maus Robotics challenge. Five minutes. 14 slides.
One rule for the whole talk: say the weak number, then say what would fix it.

## Timing budget (300 seconds)

| # | Slide | Seconds | Running |
|---|---|---|---|
| 1 | Title | 10 | 10 |
| 2 | The problem, and what Maus asked for | 28 | 38 |
| 3 | What we had to work with | 28 | 66 |
| 4 | What we built | 24 | 90 |
| 5 | Where runs fail | 24 | 114 |
| 6 | Reading the check overlays | 22 | 136 |
| 7 | Models: what the numbers actually say | 34 | 170 |
| 8 | Cross-robot reality check | 18 | 188 |
| 9 | Human labels, kept private | 18 | 206 |
| 10 | Brief coverage: three tasks, two levels | 30 | 236 |
| 11 | What does not work yet | 26 | 262 |
| 12 | What would change the answer | 22 | 284 |
| 13 | Check it yourself in a minute | 12 | 296 |
| 14 | Where to look (close) | 4 | 300 |

## Talk track, slide by slide

**1, Title, 10s.** Maus robots make crêpes on their own. Every run is logged. We
turned those logs into measurements, models and next experiments. Everything I
show comes from the repository's own docs, model cards or private export. The
code is public; the Maus data is not.

**2, The problem and what Maus asked for, 28s.** The robot already sees a lot: a
Pi 5, ROS 2, two Duet boards, a camera over the hotplate, about 110 recipe
settings per run. But the outcome is a pass or fail verdict, not a measurement.
Maus asked for three things: spread quality, position and orientation, fold
detection. Two scoring levels: Level 1 on Maus data against the robot's own
checks, Level 2 real time under about 300 ms per frame on a Pi 5.

**3, What we had, 28s.** One account, 828 runs from 2 September to 30 October.
Three robots: pixi 410, quin 384, otto 34. Mostly tests: 633 test, 195 customer.
Outcomes: 470 succeeded, 148 not_succeeded, 129 aborted, 81 canceled. Two cuts
that explain everything later: camera checks only exist from 13 October, 160 of
828 runs, so 690 runs have no measurement at all; and 117 of the 148 failed runs
have no check data. The 690 are not evidence of anything.

**4, What we built, 24s.** A read-only dashboard client over MCP and OAuth. 828
MCAP files, one ROS 2 profile per run. Tables: runs.parquet 828 by 177,
vision.parquet 138 scored runs. The robot's own overlays read back as numbers.
ONNX models with a model card each, one shared per-robot harness. A private
Label Studio for human labels. A local run viewer. All of it stays in gitignored
data; only the code and a few curated images are public.

**5, Where runs fail, 24s.** Over runs with a check: plate_present 30 of 79,
spread 38 of 163, fold1 19 of 138, fold2 34 of 135, plate_clean 20 of 182,
plate_loaded 38 of 172. push_off 140 of 259, but that one usually fails once by
design while the plate is rotated and rechecked. Key point: a failed check does
not stop the run, and every not_succeeded run ends at plate_load, so failure
analysis has to come from the checks and logs, not from where the run stopped.

**6, Reading the check overlays, 22s.** The robot's check overlay is already a
top-down view with its own marks. We read those marks back as diameter, area,
centre offset, circularity, holes, residue, fold overlap, browning. That is what
turns a verdict into a measurement. Sizes are relative to the robot's analysis
zone; no millimetre scale is published, so we never claim one.

**7, Models: what the numbers actually say, 34s.** This is the honest slide.
check-verdict: holdouts of 31 to 61 frames, balanced accuracy 0.67 to 0.92, but
three of four cards say generalizes: false, and a plain logistic regression on
the robot's own overlay scores beats the fold2 head, 0.88 to 0.67. fold-class:
0.99 on 110 frames, but on weak labels, the card's own leakage probe fails, and
holding out quin drops quarter recall to 0.02. failure-reason: one deployable
head, the rest research only, where a trivial rule scores set-F1 1.00 against
the 0.73 we quote. fold-risk: 0.29, worse than chance, permutation test does not
beat the null. We would rather show this than a flattering number.

**8, Cross-robot reality check, 18s.** Left: balanced accuracy per check, a
robot's own holdout versus that robot held out entirely. push_off transfers
well; spread and fold2 do not. Right: a deliberate failure of the pixel-mask
suggester, which calls a dark stain a hole. Every suggestion is an editable
draft a human confirms, never an accepted label.

**9, Human labels, kept private, 18s.** Local Label Studio 1.23.2, one app per
annotator, nothing leaves the machine. Whole runs split into two sets. Set 2 has
186 phase labels and 113 mask submissions; set 1 is prepared but was never
opened. 301 editable prefill suggestions. No model has been trained on these yet,
which is the biggest gap in the modelling story.

**10, Brief coverage, 30s.** Here is the map a jury wants. Task 1 partial: vision
metrics over 138 runs and a classical mask suggester, no spread model. Task 2 not
delivered: the edge-angle gate accepted 0 of 854 frames. Task 3 partial:
fold-class is four classes, the triangle fold is unsupported by design. Level 1
partial: three of four heads do not generalize. Level 2 not demonstrated: 4 ms
is a desktop number, the Pi figure is an estimate, the arm64 environment is in a
draft PR.

**11, What does not work yet, 26s.** No single entry point turns one frame into
all the decisions; issue #95 is open. The newest work is not on main: five draft
PRs, 45 commits and 63 files ahead, one self-described as not fit for use. The
models are weak. Coverage is thin. The material-mask model does not exist. A
fresh clone reproduces the code and the tests, not the numbers.

**12, What would change the answer, 22s.** More check records, because 117 of
148 failures are unmeasured. About 140 to 150 runs per robot, interleaved, to
separate robot from era. A held-out evaluation on human labels. A reviewed mask
export plus a material-mask model. Real Pi timing. And a designed batch, 19 runs
varying six settings, run sheet ready for Maus.

**13, Check it yourself in a minute, 12s.** Repo is public. Offline:
`cd code && pixi install && pixi run test && pixi run lint`. Honest note, it is
about four minutes, 625 passed, 66 skipped. Model cards in code/models/, status
page in documentation/status.md.

**14, Close, 4s.** The value for Maus is a measured, documented pipeline and a
clear list of what the next runs would settle. The models are experiments; the
next runs are the fix.

## Every number on the deck, and where it comes from

Repo data (private checkout, `data/runs.parquet`, read with pandas; the audit of
2026-11-01 reproduced the same counts):

| Number | Value | Source |
|---|---|---|
| Runs | 828 | runs.parquet rows |
| Runs table shape | 828 x 177 | runs.parquet |
| Recipe settings per run | about 110 | runs.parquet, `param.*` column count |
| Date range | 2 Sep to 30 Oct 2026 UTC | runs.parquet `ts_start` min/max |
| Robots | pixi 410, quin 384, otto 34 | runs.parquet `robot` value counts |
| Test vs customer | 633 / 195 | runs.parquet `is_test` |
| Outcomes | 470 succeeded, 148 not_succeeded, 129 aborted, 81 canceled | runs.parquet `status` |
| Runs with any check | 160 of 828 | runs.parquet `n_checks > 0` |
| Runs without a measurement | 690 | 828 - 160 |
| First camera check | 13 October 2026 | runs.parquet, earliest run with a check |
| Failed runs with no check | 117 of 148 | runs.parquet, status + `n_checks` |
| Failed runs ending at plate_load | 148 of 148 | runs.parquet `last_phase` |
| Vision coverage | 138 of 828 | vision.parquet rows; audit |
| MCAP files | 828 | documentation/status.md |

Repository documentation and model cards:

| Number | Value | Source |
|---|---|---|
| Check fail rates | plate_present 30/79, spread 38/163, fold1 19/138, fold2 34/135, plate_clean 20/182, plate_loaded 38/172, push_off 140/259 | documentation/system/crepe-stages.md |
| Cancels in cook | 34 of 66 | crepe-stages.md |
| check-verdict holdout | 31 to 61 frames, balanced accuracy 0.67 to 0.92 | code/models/check-verdict/README.md |
| check-verdict generalization | three of four cards `generalizes: false` | check-verdict model.json cards; audit |
| Overlay baseline vs fold2 head | 0.88 vs 0.67 | code/models/check-verdict/README.md; audit |
| fold-class | 110 frames, balanced accuracy 0.99 | code/models/fold-class/model.json |
| fold-class leakage probe | verdict `fail` | fold-class/model.json `sanity_checks.geometry.fold2_no_change` |
| fold-class quarter recall, quin held out | 0.024 of 82 frames | fold-class/model.json `leave_one_robot_out.maus-quin` |
| failure-reason deployable head | spread `too_small`, 13 holdout frames | code/models/failure-reason/summary.json |
| failure-reason fold2 | research only; set-F1 0.7333 vs majority 1.0 | failure-reason/fold2/model.json; summary.json |
| fold-risk | balanced accuracy 0.2917, `beats_null: false` | code/models/fold-risk/model.json |
| frame-embedder latency | about 4 ms per frame, desktop CPU | documentation/status.md; audit |
| Human labels | 186 phase labels, 113 mask submissions | documentation/status.md (branch); audit |
| Mask prefill | 301 editable suggestions | status.md |
| Label Studio version | 1.23.2 | documentation/labeling/labeling-collaboration.md |
| Task 2 edge-angle gate | 0 of 854 frames | documentation/status.md (branch) |
| Draft PRs | 5 (#137, #156, #158, #160, #162), 45 commits, 63 files | audit; status.md (branch); GitHub API |
| Runs needed to separate robot from era | about 140 to 150 | documentation/analysis/fold-risk.md |
| Designed batch | 19 runs, six settings | documentation/analysis/doe.md |
| Failed runs unmeasured | about 79 percent | audit (117 of 148) |
| Test suite | 625 passed, 66 skipped, exit 0 | audit, section 7 |
| Brief and levels | 3 tasks, 2 levels, about 300 ms | audit, section 1; repo challenge table (PR #86) |
| Licence | PolyForm Noncommercial 1.0.0 | audit; AGENTS.md |

Two sources disagree on the run window: crepe-stages.md writes "2026-08-30 to
2026-10-31", while runs.parquet `ts_start` runs 2026-09-02 to 2026-10-30. The
deck uses the parquet range, which is what a command produces. The check fail
rates are quoted exactly as crepe-stages.md states them; they are attempt-level
figures from the raw run JSON and are not reproducible from runs.parquet alone.

## If a judge asks

**"Is any model trustworthy?"** One is usable and we say why: check-verdict
push_off, balanced accuracy 0.92, fail recall 0.94 on 61 frames. But even that
drops to the majority baseline when maus-otto is held out entirely, and its
labels come from a retry loop, so a high score there partly means "recognizes
the retry pose". The other three check heads do not beat the shuffled-label null
or do not generalize. fold-risk is worse than chance. fold-class scores 0.99 but
on weak labels, its own leakage probe fails, and holding out one robot collapses
the class the task exists to detect (quarter recall 0.02 of 82). Honest answer:
treat them all as experiments, trust none for a decision on the line yet, and
the fix is human labels and more runs, not a bigger model.

**"What did you not finish?"** The end-to-end deliverable. The brief asks for
one frame in, decisions out; there is no single entry point for that (issue #95
is open) and nothing is wired to the viewer or the overlay renderer. Task 2
(position and orientation) is not delivered: the edge-angle gate accepted 0 of
854 frames. Level 2 is not demonstrated: the Pi number is an estimate, not a
board reading. And the newest work is not on main: five draft PRs, 45 commits
ahead, one open since Saturday and self-described as not fit for use. Say it
plainly: a judge reading main does not see the best work.

**"What would Maus get on Monday?"** Three things they can use now, and one
thing to run. To use: the export and MCAP pipeline, the runs table and the vision
scores, and the label tooling (all documented, with a status page that names its
own weak numbers). To run: the 19-run designed batch that varies six settings
independently, with a run sheet ready. And the clearest ask back to Maus: more
check records (117 of 148 failures have none) and about 140 to 150 runs per robot
interleaved in time, which is what would turn every "we cannot tell yet" on this
deck into an answer.
