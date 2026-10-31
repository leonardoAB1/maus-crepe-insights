# One crepe-state vocabulary

This is the single vocabulary for "what is on the plate". Human labels, model
heads and analysis all report in these names, so a human label and a model
prediction can be compared directly instead of through a mapping table nobody
maintains.

The code lives in `code/maus_export/crepe_states.py`. The version is
`crepe-state-v1`.

## The six states

| State | Meaning |
| --- | --- |
| `emptyplate` | bare plate, no crepe material in view |
| `flat` | unfolded batter or crepe spread flat on the plate |
| `cooking` | flat crepe past spreading, browning or bubbling, not folded |
| `halffold` | crepe folded once, into a half |
| `quarter fold` | crepe folded twice, into quarters |
| `ugly` | visible crepe whose shape or condition failed the run |

The vocabulary names the plate, not the robot. A recorded robot phase such as
`spread_batter_concentric` is a command the machine was running; `flat` is what
a viewer sees. Keeping the two apart means a model that reads a single frame is
never asked to guess robot state that the pixels do not show.

## Names that changed

The phase labeling pilot used two names that this vocabulary replaces. The
meaning is unchanged, and the old names stay readable:

| Old label | Unified name |
| --- | --- |
| `spreading` | `flat` |
| `circle` | `cooking` |

`normalise` accepts a unified name, an old label or a model label and returns
the unified one, so stored exports keep working without being rewritten. An
unknown name raises rather than being guessed.

## Model labels

The `fold-class` head reports `empty`, `flat`, `half`, `quarter`. Those map onto
`emptyplate`, `flat`, `halffold` and `quarter fold`. A model that predicts only
those four cannot express `cooking` or `ugly`, and the comparison pipeline
records that as a state disagreement rather than inventing a label.

## Reading the vocabulary from code

```python
from maus_export.crepe_states import normalise, state_table

normalise("spreading")  # "flat"
state_table()  # version, states, meanings, aliases, model labels
```

```bash
pixi run maus-export label-states
```

## Why the labeling UI still shows the old names

The live labeling project holds annotations stored under `circle` and
`spreading`. Changing its label configuration would make the running project
fail its own drift check on the next start and leave stored annotations
pointing at labels that no longer exist. The UI names therefore change in a
deliberate, separate migration, after the current labeling round, with the
alias mapping keeping the meaning intact. Analysis and comparison already run
in the unified vocabulary today.
