# Maus dashboard data

Source: the Maus Robotics dashboard (`https://dash.mausrobotics.com`, MCP endpoint `/mcp`).
This is a review as of 2026-10-30, seen with our hackathon account.

## Access window
Our account only sees runs from **2026-08-30 to 2026-12-30**. Earlier runs exist (ids go back much further)
but are not returned.

## What is there

| Data | Volume | Notes |
|---|---|---|
| Crêpe runs | 828 (195 non-test, 633 test) | the core dataset, see below |
| Robots | 4: `maus-nilo` (1), `maus-otto` (2), `maus-pixi` (3), `maus-quin` (4) | nilo offline since Jan 2026; pixi produces most recent runs |
| Suppliers | ~110 | ~60 categorised, ~50 imported from sheets/invoices without category |
| Contacts | 494 | name / org / email |
| Meetings | 0 visible | our account has no meetings grant |

## A crêpe run (`crepe_get`)
- **Header:** `id`, `machineId`/`machineName`, `topping` (plain, apple, chocolate, raspberry), `isTest`,
  `orderId` (non-empty for customer orders), `tsStart`/`tsEnd` (UTC), `terminal_status`
  (`succeeded` | `not_succeeded` | `canceled`), `prevId`/`nextId`.
- **Sensors:** `hotplateTemp` (setpoint, °C), `envTempStart` (°C), `envHumidityStart` (%), `crepeDiameter`
  (mostly null).
- **Phases:** `init → prepare → check_serving_plate → oil → spread_batter[_concentric] → cook → add_topping →
  fold_carefully → push_off → plate_handover → wipe_spatula → discard_crepe → plate_load`, each with
  `elapsed_sec` and `progress`. Stage machine, timings and failure split:
  [crepe-stages.md](../system/crepe-stages.md).
- **Vision checks:** about 8 per run, in order (`plate_present`, `spread`, `fold1`, `fold2`, `push_off`, `plate_clean`,
  `plate_loaded`, ...), each with `ok`, `message`, `errors[]`, plus a raw `frame` JPEG and an annotated `overlay` JPEG
  (`checkImageUrls`, which need the bearer token).
- **Parameters:** about 150 recipe/machine settings (batter spread geometry, flow rates, cook time, topping pumps,
  gluten-free variants, ...).
- **Logs:** a few hundred lines of ROS-node logs (`make_crepe_action`, `image_recognition`, `duet.duet_comm` G-code,
  `hotplate_controller`, `inventory_node`), with Python logging levels 10–50.
- **Video:** `videoUrl` (mp4) for every run.

For video-derived fold-state analysis, see the experimental
[fold segmentation guide](../fold-segmentation.md).

## Quirks to know
- **Time zones:** `tsStart`/`tsEnd` are UTC, while log/phase/check timestamps are naive robot-local time
  (Europe/Zurich, so +1 h in winter and +2 h in summer). The exporter measures the offset per run.
- **Manual labels:** `successSpread/Topping/Fold/Handover`, `comments` and
  `batch` were null in earlier sampled runs. In one private 2026-10-31 snapshot
  of 828 runs, each of the four success fields had only one or two non-null
  values. `terminal_status` and robot vision checks are operational proxies,
  not independently verified quality ground truth.
- **Many cancels:** customer runs are often `canceled`, and test runs make up about 76% of the data.
- **Log warnings** are retrospective diagnostics, not run-start inputs unless
  their timing and availability are verified. Examples include `hotplate
  demand is ON but duet reports the SSR OFF`, `rpm_from_gps: target_gps ...
  greater than gps_max`, and `fold.spatula_gate_enabled is false`.
- **Robot settings** (`robot_list`): topping→pump-port map, prices (6–7 per crêpe), and inventory capacity /
  grams per crêpe (batter 113 g, apple 20 g, chocolate 13 g, raspberry 15 g, 60 plates).

## More diagrams
- [system.md](../system/system.md): robot hardware and control architecture,
  ROS 2 nodes, customer flow
- [crepe-stages.md](../system/crepe-stages.md): stage machine, stage timings,
  where runs end
- [pipeline.md](pipeline.md): our data pipeline and the run data model

## Handling
The repo is public. Raw exports (`data/`, `*.mcap`, images, videos) are gitignored and must stay local.
