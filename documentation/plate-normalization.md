# Opening-window plate normalization experiment

This experiment tests a fixed per-robot video transform before another
full media export. It does not replace physical camera calibration.

## Run

From this branch's `code/` directory:

```bash
pixi install --locked
pixi run python -m maus_export.plate_normalization \
  --data-dir /absolute/path/to/repo/data \
  --out /absolute/path/to/repo/data/rectification/morph/plate-statistics
```

The selected output directory must be new, and under gitignored `data/`.
Original exports are unchanged. Do not publish diagnostic images, NPZ files,
per-video reports or matrices. They contain private source-derived data.

## Method

Group videos by robot id and exact pixel dimensions. Sample up to eight
videos distributed across the group's sorted run paths. This is not a
chronological or statistically representative sample of all operations.
Resolution groups are candidate configurations, not verified camera ids.

For each selected video, sample 24 frames across the first 12 seconds.
Require at least 12 decodable samples spanning five seconds. Downsample to
at most 640 pixels wide for bounded memory and compute:

- Per-channel temporal mean and median.
- Grayscale temporal variance and standard deviation.
- A stable-region mask excluding variance above 400 intensity units squared.

A temporal median suppresses transient objects only when they do not occupy
the same pixel for most samples. Stationary tools, glare and food remain.
Low variance alone does not mean a pixel belongs to the hotplate.

Fit ellipse candidates from median-image edge contours. Use explicit
central-position, plausible-size and aspect-ratio priors. Require stable
edge support within three analysis pixels along at least 60% of the rim,
including sufficient support in at least eight of twelve angular sectors.
The highest-supported candidate is saved for **visual review**. This can
still select an inner ring, crêpe boundary, or fixture instead of the rim.
No candidate is physical ground truth.

Convert the ellipse to a geometric centre and covariance. These are
moments of the fitted filled ellipse, not intensity-weighted moments of the
whole camera image. Scale them back to original video coordinates.
Aggregate centres and covariances robustly across independent videos.
Require at least three agreeing videos and a strict majority among fitted
candidates. Centre drift must be at most 2.5% in normalized coordinates,
and relative covariance drift at most 8%. Failed video fits are recorded
but are not evidence of stability; inspect coverage and rejection counts.

The resulting fixed **affine** transform whitens the ellipse covariance,
placing its centre at the middle of a 512-square image and its rim at a
224-pixel radius. It preserves the image-plane orientation rather than
inventing an in-plane physical reference direction. This is an
ellipse-to-circle approximation, not a unique projective reconstruction.

## Artifacts and review

Each group receives a `report.json` with individual failures and candidates,
and a proposed transform only when cross-video checks pass. For each
sampled video, private diagnostics include:

- `*_statistics.npz`: mean, median, variance, sample frames and source size.
- `*_mean.png`, `*_median.png`, `*_stddev.png`: temporal statistics.
- `*_candidate.png`: fitted rim superimposed for human review.

Every proposed transform has `approved_for_export: false`. This command
performs an audit only; it does not launch another full rectification run.
Reject incorrect rim detections even when numeric consensus passes.
Check held-out dates and videos for camera movement before approving reuse.

Do not transfer these transforms to check images based on aspect ratio.
Verify identical camera, crop and optics using independent correspondences,
or estimate and validate a separate image mapping. Physical accuracy still
needs reference geometry and lens calibration. A planar/affine transform
cannot correct parallax of folds and raised tools.

## Tests

Synthetic offline tests check temporal statistics with moving objects,
variance-based rejection, ellipse geometry, cross-video outlier rejection,
insufficient or mixed groups, and covariance normalization to a circle.
