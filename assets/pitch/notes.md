# Maus Crepe Insights pitch: three-slide talk track

BaselHack 2026, Maus Robotics challenge. Three minutes, three slides.

## Talk track

1. Make every run more informative. Maus robots produce logs, camera checks
and recipe settings. Maus Crepe Insights turns those records into a structured
view that teams can use to study crêpe quality and plan follow-up experiments.

2. The workflow brings 828 run records into JSON and ROS 2 MCAP, with a
177-field analysis table. Vision measurements are available for 138 runs, where
the robot's check overlays can quantify diameter, centre offset, holes and fold
overlap. The pipeline covers four camera checks: spread, fold1, fold2 and
push-off. A local viewer and labeling tools help people inspect frames beside
run records.

3. Explore the public project at github.com/leonardoAB1/maus-crepe-insights.
Bring authorized run records and check images into the workflow, compare
batches, and use the measurements to shape the next experiment. The code is
public; private Maus run data is not included.

## Numbers and sources

| Number | Value | Source |
|---|---|---|
| Run records | 828 | private `data/runs.parquet` row count |
| Analysis fields | 177 | `data/runs.parquet` column count |
| Runs with vision metrics | 138 | private `data/vision.parquet` row count |
| Camera-check stages | 4 | `documentation/system/crepe-stages.md` |

Project entry points and schemas: `documentation/data/pipeline.md`.
The exported run and vision counts are documented in
`documentation/status.md`; data files are private and are not shipped in Git.

## Suggested close

"If you work with crêpe robots, bring us the runs you already record. We can
help turn those logs and checks into a clearer picture of what to test next."
