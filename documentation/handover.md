# Handover to Maus Robotics

BaselHack 2026 ended on 1 November 2026. This page is for the Maus team
picking the project up: what exists, how to run it on your own data, and
where to continue. Every number here comes from
[project status](status.md), which names the command or file behind it.

## What you get

- **Data pipeline.** Dashboard export to raw JSON, one ROS 2 MCAP per run
  (open in Foxglove) and a flat runs table with 177 columns:
  [pipeline](data/pipeline.md), [MCAP topics](data/mcap.md).
- **Run-history analysis.** Which settings and conditions go with failure,
  and where runs end: [settings analysis](analysis/analysis.md),
  [crêpe stages](system/crepe-stages.md),
  [fold risk](analysis/fold-risk.md).
- **A designed experiment.** A 19-run Definitive Screening Design over six
  recipe and machine settings, with the analysis ready for the results:
  [DoE](analysis/doe.md).
- **Vision.** Reading the robot's check overlays as numbers, a crêpe
  segmentation model distilled from SAM 3, fold state and pose from the mask,
  and frame models for check verdicts, fold class and failure reasons, all
  shipped as ONNX with model cards: [vision pipeline](vision/vision-pipeline.md),
  [crêpe student](vision/crepe-student.md),
  [models](../code/models/README.md).
- **A review website on a Raspberry Pi.** Every run's video with the mask and
  fold state, each robot check next to the model's verdict. It deploys from
  `main` on a Pi 3 or Pi 5, with example systemd units in `code/deploy/pi/`:
  [Raspberry Pi](system/raspberry-pi.md#deploy-from-main).
- **Labeling setup.** Private Label Studio projects for phases and
  material and hole masks: [phase labeling](labeling/phase-labeling-pilot.md),
  [mask labeling](labeling/mask-labeling.md).
- **The final presentation.** [assets/pitch](../assets/pitch/README.md),
  with its [PDF](../assets/pitch/crepeflow-pitch.pdf).

## Main results

- 116 of 147 failed runs that started a crêpe carry no check data, so their
  cause cannot be read from the export.
- Fold failure follows the robot (maus-quin 21%, maus-pixi 61%), not the
  spread. The run history points at lowering fan inset and raising fan
  power; the DoE batch is designed to confirm or reject that.
- The crêpe segmentation model reaches IoU 0.966 against SAM 3 and a median
  0.944 against human masks; its fold state agrees with the robot on 99.5% of
  991 checks.
- On a Raspberry Pi 5 the video analysis runs at 0.15 s per frame end to end,
  under the brief's 300 ms target. A Raspberry Pi 3 takes 854 ms per frame.

The limits behind each of these are on the [status page](status.md).

## Getting started on your own data

1. Install [pixi](https://pixi.sh), then `cd code && pixi install`.
2. Check the code without any data: `pixi run test` and `pixi run lint`.
3. Export runs with your own dashboard account. The first sign-in opens a
   browser; `MAUS_TOKEN` skips it. The hackathon account saw 828 runs; a Maus
   account with wider access will see more:
   `pixi run export all --media` (see the [README quickstart](../readme.md#quickstart)).
4. Rebuild the analysis tables and scores: `pixi run table`,
   `pixi run vision`, `pixi run -e ml analyze`.
5. Retrain or rescore a model: each folder in `code/models/` has a
   `train.py` and a model card with the command.

Exported data stays in `data/`, which is gitignored. The repository rules,
including what may be committed, are in [AGENTS.md](../AGENTS.md).

## Running the designed experiment

1. Confirm whether `machine.y_center` and `machine.z_scrape_start` may change
   from run to run. If not, drop them from the design; it then varies four
   settings.
2. Generate the run sheet for one robot: `pixi run doe --robot <robot>`.
   It writes `data/doe/<doe_id>.csv` with the run order and the values.
3. Run the 19 crêpes by hand in the given order, plain topping, and fill in
   each dashboard run id. Nothing here sends commands to a robot.
4. Export the runs, then `pixi run -e ml doe-analyze data/doe/<doe_id>.parquet`
   says which setting to change, which way, and whether it has a sweet spot.

## Where to continue

In order of value, from what the project found:

1. **Record a check with an image on every run.** Most failures are
   unexplained only because nothing was recorded.
2. **Run the DoE batch** above, and the quick test it suggests: fan power up
   from 0.85, fan inset down from 20, inner flowrate kept at 13 g/s.
3. **Train on human labels.** 186 phase labels and 113 masks exist; no model
   learns from them yet, so every model score is agreement with the robot.
4. **Close the loop on the robot.** Time the pass/fail check heads inside the
   Pi 5 loop, then decide which checks the model can take over.
5. **Open issues** on GitHub, among them #207 (retrain fold risk and failure
   reason per robot) and #106 (the triangle fold).

## Repository and access

- Changes go through pull requests with green CI and squash merge; the
  workflow is in [AGENTS.md](../AGENTS.md).
- The "protect main" ruleset is active. Two freeze rulesets from the
  submission deadline are kept but disabled.
- The code is under PolyForm Noncommercial 1.0.0 ([license](../license.txt));
  commercial use needs a separate license from the authors. The Maus run data
  is Maus's and stays private, except the two demo videos of the final deck.
- The Raspberry Pi addresses in the deck were on the venue network and are
  not reachable from elsewhere.
