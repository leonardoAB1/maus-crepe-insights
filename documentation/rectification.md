# Experimental camera rectification

This experiment estimates a projective map from a raw check image to its
upstream overlay. It then applies that map to **raw pixels only**, without
copying annotation paint into training images. It is not physical camera
calibration, a damage detector, or evidence of metric accuracy.

## Run locally

From the experimental worktree's `code/` directory, install with
`pixi install --locked`. No additional dependencies are needed.

Set `DATA` to the absolute path of the main checkout's gitignored `data/`
directory. Choose a matching raw/overlay pair locally. Never publish run
paths, source images, transforms, or training labels.

```bash
export DATA=/absolute/path/to/repo/data
pixi run python -m maus_export.rectification --data-dir "$DATA" fit \
  --raw "$DATA/raw/media/<run>/<check>_frame.jpg" \
  --overlay "$DATA/raw/media/<run>/<check>_overlay.jpg" \
  --camera-key '<robot>-<camera-configuration>-900x509' \
  --out "$DATA/rectification/transform.json"

pixi run python -m maus_export.rectification --data-dir "$DATA" apply \
  --transform "$DATA/rectification/transform.json" \
  --input "$DATA/raw/media/<run>/<check>_frame.jpg" \
  --camera-key '<robot>-<camera-configuration>-900x509' \
  --out "$DATA/rectification/frame.png"
```

Image output is lossless PNG, with a sibling `frame_valid.png` mask.
Exclude invalid border pixels from downstream measurements. Existing files
are never overwritten by the CLI, and outputs outside the chosen `data/`
directory or inside its `raw/` subtree are rejected. All artifacts must
remain local. Gitignore is a safeguard, not protection against `git add -f`.

## Estimation and acceptance

- Mask high-saturation/high-value overlay pixels, dilating the mask to
  reduce contamination by painted annotations and JPEG colour fringes.
  This heuristic can also mask saturated real objects and miss paint.
- Match SIFT descriptors with a nearest-neighbour ratio filter and unique
  target features.
- Use a reproducible 70/30 fit/validation split. Fit with RANSAC on the
  training subset only; never refit using validation points.
- Require at least 40 matches, held-out median error at most 1.5 pixels,
  and held-out 95th-percentile error at most 5 pixels. Errors are measured
  in overlay pixels. Require at least 80% of held-out points within 5
  pixels and their convex hull to cover at least 15% of each image.
- Save the matrix, dimensions, camera key and measured validation results.
  A rejected fit writes no transform. Do not loosen thresholds merely to
  obtain an output image.

These are experimental acceptance thresholds, not a physical accuracy
specification. Randomly held-out features from the same pair do not prove
transfer to other runs. Repeated texture, occlusion and moving food can
produce misleading matches. Coverage helps reject tightly clustered fits
but does not ensure uniformly distributed validation.

## Video and camera configuration

Video application uses the same `apply` command with `--video` and an
`.mp4` output path. It streams frames and preserves the reported constant
frame rate, but drops audio and does not preserve variable frame timing or
original container metadata. MP4 target dimensions must be even to avoid
silent codec cropping. A validity mask is currently emitted only by image
application. Decode failure after some frames can resemble normal
end-of-file; verify output duration against the source before use.

Every decoded frame must exactly match the transform's source dimensions.
The tool intentionally does **not** scale an image transform for video:
matching aspect ratios do not establish identical crop, camera, or optics.
A caller-supplied camera key guards accidental reuse but does not identify
a camera automatically. Estimate and independently verify a video-specific
transform using suitable reference correspondences before video use.

## Before model training

1. Obtain original intrinsics, lens-distortion coefficients and plate-plane
   calibration where possible. Lens distortion is **not corrected** here.
2. Validate on independent frames, runs and dates for every camera setup.
   Check a static background or reference target, not only the crêpe.
3. For physical accuracy, measure held-out reference geometry across the
   hotplate. Agreement with an overlay only reproduces upstream processing.
4. Check whether the relevant image is actually after cooking, using phase
   and check timestamps. A check called `spread` alone does not prove it.
5. Use one validated preprocessing contract for training and inference.
   Never feed overlays to the damage model, and split by run before any
   fitting or augmentation. Keep calibration and evaluation groups separate.
6. A planar homography cannot remove parallax from raised folds, utensils,
   or other objects away from the calibration plane.

## Tests

`pixi run test` exercises known homographies, independent point validation,
outlier and low-coverage rejection, annotation-masked synthetic matching,
valid-pixel masks, resolution checks, private output paths and synthetic
video streaming. No dashboard access or real-image fixtures are required.
