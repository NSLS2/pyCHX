# NSLS-II CHX Dataset 1

This fixture contains 50 static-sample frames collected with the Eiger 500k.

- Tiled scan ID: `216964`
- Tiled UID: `6323cb44-3596-48cc-87ef-620ec7eda4c7`
- Original master file: `/nsls2/data/chx/proposals/2026-2/pass-316251/assets/eiger500k-1/2026/08/02/8d9263a6-add2-4a61-baec_83_master.h5`
- Reference notebook environment: prerefactor pyCHX

According to the pyCHX Eiger 500k convention, both `reverse` and `rot90` are
`True`. These transformations must be applied, in that order, before comparing
raw frames with analysis results.

## Layout

- `raw/`: the Eiger master and five externally linked data files.
- `masks/detector.npy`: detector/chip mask passed to `compress_eigerdata()`.
- `masks/polygon.npy`: generated beamstop/polygon mask.
- `masks/roi.npy`: final 12-label SAXS `wide` ROI mask.
- `expected/`: prerefactor numerical results used for characterization.
- `manifest.json`: portable acquisition and test parameters, including the
  SHA-256 of the omitted CMP file.
- `reference/`: the unchanged executed notebook retained for provenance only;
  tests must not execute or import it.

The combined static-analysis mask is `detector.npy & polygon.npy`. There were
no hot pixels in this dataset, so the notebook's hot-pixel mask was all ones.

The 199.6 MB `compressed_data.cmp` is deliberately not versioned. Tests generate
it in a temporary directory, compare its SHA-256 with the manifest, and verify
every decoded frame against the transformed raw detector data. A local source
copy may remain in the ignored `results/` directory.

The original full metadata JSON is also not versioned. Its two large array
fields are already represented by the Eiger master file and masks; all metadata
required by the tests is recorded in `manifest.json`.

## Interpreting the reference results

These results characterize the prerefactor implementation; they are not
guaranteed ground truth. A mismatch is important and must be reviewed, but
does not by itself prove that a newer implementation is wrong. It may instead
expose an existing defect in the reference implementation. Do not update a
reference merely to make a test pass without understanding and documenting the
difference.

The prerefactor parallel compressor had a segment-count bug for this dataset:
the saved average image is exactly half the correct average. Because circular
averaging is linear, the saved I(q) has the same factor-of-two bias. Tests apply
the documented correction without modifying the original reference arrays.
The q positions, frame intensities, g2, and TTCF remain valid references.
