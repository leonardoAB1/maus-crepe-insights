# Maus Crepe Insights pitch: 3 slides, talk track and sources

BaselHack 2026, Maus Robotics challenge. Three minutes, three slides.
One rule for the whole talk: say the weak number, then say what would fix it.

## Talk track, slide by slide

**1, Title, 20s.** Maus robots make crepes on their own, every run is logged,
and we turned those logs into measurements, models and next experiments.
Everything I show comes from the repository's own docs, model cards or a private
export. The code is public; the Maus data is not.

**2, What we built, 90s.** Four headline numbers, then the honest read.
828 runs logged, 2 September to 30 October 2026. 138 of them have vision scores.
Four check-verdict ONNX heads, one per check, plus a fold-class model for
empty / flat / half / quarter. Now the caveats, said plainly: three of the four
check heads do not generalize, with balanced accuracy 0.67 to 0.92 on 31 to 61
frame holdouts; fold-class scores 0.99 on 110 frames but on weak labels, its own
leakage probe fails and quarter recall drops to 0.02 of 82 frames when maus-quin
is held out; and 690 of the 828 runs have no camera measurement at all, because
checks only exist from 13 October. The 690 are not evidence of anything; they
are simply unmeasured.

**3, See it live, 60s.** The repository is public at
github.com/leonardoAB1/maus-crepe-insights, and there is a live site (the
placeholder link in the deck still needs the real URL dropped in). The ask is
one line: more check records, because 117 of the 148 failed runs have none, and
about 140 to 150 interleaved runs per robot, which is what turns every "we
cannot tell yet" into an answer. We never send anything to a robot; the next
batch is a run sheet a person runs by hand.

## Numbers on the deck, and where they come from

| Number | Value | Source |
|---|---|---|
| Runs | 828 | data/runs.parquet row count |
| Vision-scored runs | 138 | data/vision.parquet row count |
| check-verdict heads | 4 | code/models/check-verdict/ |
| fold-class | 1 model, 4 classes | code/models/fold-class/model.json |
| check-verdict holdout | 31-61 frames, balanced accuracy 0.67-0.92 | check-verdict model cards |
| Heads that generalize | 1 of 4, three say generalizes: false | check-verdict model.json |
| fold-class holdout | 0.99 on 110 frames, weak labels, leakage probe fails | fold-class model.json |
| Quarter recall, quin held out | 0.02 of 82 frames | fold-class model.json |
| Runs without camera check | 690 of 828, checks from 13 October | runs.parquet |
| Failed runs without check data | 117 of 148 | runs.parquet |
| Runs to separate robot from era | about 140 to 150 | documentation/analysis/fold-risk.md |

Two sources disagree on the run window: crepe-stages.md writes 2026-08-30 to
2026-10-31, while runs.parquet runs 2026-09-02 to 2026-10-30. The deck uses the
parquet range, which is what a command produces.

## If a judge asks

**"Is any model trustworthy?"** Treat them all as experiments; trust none for a
decision on the line yet. The one usable head is check-verdict push_off, but its
labels come from a retry loop and it drops to the majority baseline when
maus-otto is held out entirely. fold-risk is worse than chance. The fix is human
labels and more runs, not a bigger model.

**"What would Maus get on Monday?"** The export and MCAP pipeline, the runs and
vision tables, the label tooling, and the 19-run designed batch with a run sheet
ready. The clearest ask back is more check records and interleaved runs per
robot.
