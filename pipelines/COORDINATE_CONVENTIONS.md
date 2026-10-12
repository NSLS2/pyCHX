# Detector coordinates in the XPCS pipelines

Survey of the working tree, 2026-10-11. **For an oriented analysis image,
plot `(x, y)` but index `image[y, x]`.** Beam-center metadata, helper arguments,
and saved masks do not all use the same ordering.

## Screens, files, and analysis images

**Albula `(x, y)` is the source of IOC `beam_center_x/y` and the corresponding
HDF5 fields**, as confirmed by beamline practice. In the acquisition code,
[`update_beam_position_dict`](https://github.com/NSLS2/chx-profile-collection/blob/670a10be52a0db75f090087f6c466eabe48aaf16/startup/30-user.py#L623)
copies these PVs into `RE.md['beam_position_dict']` without swapping or
reversing them. Its comment “from CSS” refers to reading the PV values entered
through the screen. Thus the beam-center entry fields use Albula coordinates;
the CSS/Phoebus **image cursor** may still use a different orientation.

The Python side is explicit. Let `R` be an untransformed detector frame with
`R.shape == (H, W)`, and `A` the image used for analysis. Raw zero-based pixel
coordinates `(x_raw, y_raw)` mean `R[y_raw, x_raw]`.

| Detector selection | Loader flags | Analysis array `A` | Shape | Same pixel at analysis `(x, y)` |
| --- | --- | --- | --- | --- |
| Eiger 4M / `image`, Eiger 1M | `reverse=True, rot90=False` | `R[::-1, :]` | `(H, W)` | `(x_raw, H - 1 - y_raw)` |
| Eiger 500K | `reverse=True, rot90=True` | `np.rot90(R[::-1, :])` | `(W, H)` | `(H - 1 - y_raw, W - 1 - x_raw)` |

These choices appear in [XPCS_auto.ipynb](XPCS_auto.ipynb), “Find beam center,”
and the three [old pipelines](old_pipelines). They are hardcoded by detector
name; no detector mounting angle is consulted. The bundled 500K example has
raw shape `(514, 1030)` and analysis shape `(1030, 514)`.

In [chx_generic_functions.py](../pyCHX/chx_generic_functions.py), `load_data`
reverses rows before rotating, and preserves image metadata without transforming
its coordinates or mask arrays. `load_dask_data` performs the corresponding
operations on stack axes 1 and 2. Its unusual `flip(axis=(1, 1))` applies one
row reversal in the inspected Dask implementation. `rot90_clockwise` actually
calls `np.rot90`: use that expression to identify the transformation, since
its apparent direction depends on the plot origin. Direct HDF5 compression
uses the same reversal-then-rotation in
[chx_compress.py](../pyCHX/chx_compress.py), `_iter_binned_images`.

The notebooks generally set `image.origin='lower'`, and `show_img` defaults to
`origin='lower'`: column/x increases rightward, row/y upward. Ordinary Matplotlib
defaults to `origin='upper'` unless changed. **Changing the plot origin does
not change array indices.** `cmap_albula` changes colors only.

### What this establishes about Albula

Because the input beam-center values come from Albula, the master's metadata
conversion directly gives its **Albula-to-analysis beam-center mapping**:

| Detector | Albula `(ax, ay)` → analysis `(x, y)` as currently coded |
| --- | --- |
| 4M / 1M | `(ax, H - ay)` |
| 500K | `(H - ay, W - ax)` |

This supports Albula x along raw columns and Albula y along raw rows. Relative
to the usual analysis plot, 4M/1M preserve x and reverse y; 500K swap axes and
reverse both. The acquisition helper
[`get_beam_center_update`](https://github.com/NSLS2/chx-profile-collection/blob/670a10be52a0db75f090087f6c466eabe48aaf16/startup/30-user.py#L74)
provides stronger evidence: it writes the raw maximum's zero-based column to
`BeamX` and row to `BeamY` for 4M. For 500K it finds the maximum after `np.rot90`
and undoes that coordinate change with `1030 - 1 - row`, again writing raw
`(column, row)`. The first table therefore gives the expected exact pixel
mapping for Albula readings using this same zero-based convention. The
one-pixel discrepancy with the beam-center conversion is discussed below.

The adjacent
[`calculate_direct_beam_position`](https://github.com/NSLS2/chx-profile-collection/blob/670a10be52a0db75f090087f6c466eabe48aaf16/startup/30-user.py#L644)
also explains the detector mounting difference. For motor displacements
`dx, dy` in mm and `p=0.075` mm/pixel, it predicts (before integer rounding):

| Detector | Change in Albula/IOC `(x, y)` | Change after analysis transformation |
| --- | --- | --- |
| 4M | `(+dx/p, +dy/p)` | `(+dx/p, -dy/p)` |
| 500K | `(+dy/p, -dx/p)` | `(+dx/p, -dy/p)` |

Both analysis images therefore respond to detector motion along the same
axes. The 500K rotation compensates for its different mounting orientation.
This establishes relative axes and signs without needing an Albula screenshot.

## Why beam-center numbers differ

Acquisition has two distinct beam positions: `beam_position_dict` stores the
Albula calibration **reference** and detector motor positions at calibration;
`series(auto_beam_position=True)` updates the current 4M/500K PVs from that
reference and the current motors. It rounds to integer pixels. The nested
reference values can therefore legitimately differ from the exposure's
configuration/HDF5 values after detector motion.

The master notebook's metadata then passes through several stages:

| Stage | Meaning of the values |
| --- | --- |
| `get_meta_data` / `md_blue` | Configuration metadata merged with run-start metadata; start keys override matching configuration keys. `md_blue` is a snapshot of this merge. |
| `md.update(img_md)` | Matching keys are replaced again. Dask obtains beam centers from detector configuration; the fallback uses the image handler's metadata, normally from HDF5. These sources need not agree. |
| “Apply image metadata” | The notebook converts `beam_center_x/y` to analysis `(column, row)` using the Albula mapping above. |
| `inc_x0`, `inc_y0`, `center` | Master `inc_x0/inc_y0` are analysis `(x, y)`. With both arguments supplied, `check_lost_metadata` returns `center=[int(inc_x0), int(inc_y0)]`. |
| `setup_pargs['center']` | Starts as `(x, y)`; is temporarily reversed for template ROI placement on selected detectors; “Setup center args” later sets it to `(y, x)` for SAXS calculations. |
| “Save metadata” | Master SAXS writes `beam_center_x=center[1]`, `beam_center_y=center[0]`: these saved fields are **row, column**. Master WAXS and surface modes save `inc_x0/inc_y0`: **column, row**. |

**Likely one-pixel offset:** the master and old SAXS/GISAXS notebooks use
`size - coordinate` for beam centers, whereas exact zero-based pixel mappings
use `size - 1 - coordinate`. With the PV convention explicitly implemented by
`get_beam_center_update`, the converted center is one pixel too large in
analysis y for 4M/1M, and in both analysis axes for 500K. Manual Albula cursor
numbering is not documented in these sources, so confirm that it matches the
automatic helper before changing historical calibration. A uniform one-based
convention alone would not explain the 4M formula: x would also need `-1`.
Use the exact pixel mappings for zero-based hot-pixel indices.

`check_lost_metadata` adds another trap: without explicit center arguments it
returns `[int(md['beam_center_y']), int(md['beam_center_x'])]`. With both supplied
it returns their original `(x, y)` order. Its internal center changes affect a
copy; it does not rewrite the caller's beam-center fields. Integer conversion
also discards subpixel precision. Master surface q maps use the floating-point
`inc_x0/inc_y0` directly. Manual `surface_beam_center_x/y` fallbacks already mean
analysis coordinates and skip the loader-coordinate conversion.

Compressed headers copy the supplied beam-center fields without a coordinate
conversion. Also, the legacy `Multifile` reader reconstructs images with shape
`(md['ncols'], md['nrows'])`; these historical names are reversed relative to
NumPy terminology. Prefer the actual image shape when identifying axes.

## Helper argument order

These are the implementations in
[chx_generic_functions.py](../pyCHX/chx_generic_functions.py) and
[XPCS_SAXS.py](../pyCHX/XPCS_SAXS.py), regardless of older comments or docstrings.
Reversing a two-element center changes argument order; it does not rotate or
flip an image.

| Operation | Required coordinates |
| --- | --- |
| NumPy indexing; JSON mask nested lists | `[row, column]`, i.e. `[y, x]` / `[y][x]` |
| `plt.plot`; `show_img(center=...)` | `(x, y)`; `show_img` calls `plot1D(y, x)` internally |
| `create_cross_mask`; `create_multi_rotated_rectangle_mask` | `(x, y)`; beam-mask `x_off/y_off` follow these analysis axes |
| `show_ROI_on_image(center=...)` | `(y, x)` for setting the plot limits |
| `create_hot_pixel_mask(center=...)` | `(y, x)`, despite its `(x, y)` docstring |
| `circular_average`; `get_ring_mask`; `get_angular_mask`; `get_qval_qwid_dict` | `(y, x)`, including centers passed through `pargs` |
| `shift_mask`; `get_roi_mask_qval_qwid_by_shift` | Centers in `(y, x)`; the latter takes its effective new center from `setup_pargs`, ignoring its `new_cen` argument |
| New [geometry helpers](../pyCHX/_xpcs_pipeline_geometry.py), including `surface_q_maps` and SAXS ROI validation | `(x, y)` in the oriented image |

For angular SAXS ROIs, zero degrees is toward increasing columns/right;
positive 90 degrees is toward increasing rows/up with `origin='lower'`.
Surface maps likewise use increasing analysis rows for the reflected-beam
displacement and vertical scattering coordinate.

## Masks and adding a hot pixel

The master loads `detector_mask_<detector>.json` arrays **as already oriented**;
it applies no further reversal or rotation. Here `True/1` means keep and
`False/0` means exclude. HDF5 `detectorSpecific/pixel_mask` is different:
zero means good and nonzero values encode defects. The cached
`pixel_mask_compression_*.json` belongs to that detector metadata path;
it is not the editable analysis detector-mask JSON.

[Detector_Masks_Add_Pixels.ipynb](masking_and_rois/Detector_Masks_Add_Pixels.ipynb)
writes `[pixel[0]][pixel[1]]`. Thus `pixels_to_add` must contain **`(row, column)`**,
even though its temporary variables are called `x` and `y`. For example, a pixel
at analysis plot `(x=1119, y=1010)` is:

```python
x, y = 1119, 1010                # analysis-image column, row
mask_data['masks']['pixel_mask'][y][x] = False
# Equivalently, in Detector_Masks_Add_Pixels.ipynb:
pixels_to_add = [(y, x)]

# Verify against an image loaded with the pipeline's detector flags:
plt.imshow(avg_img, origin='lower', interpolation='nearest')
plt.plot(x, y, 'r+')
```

Use the cursor on that oriented image, or convert a verified raw pixel with
the first table. A flat index is `row * image.shape[1] + column`, in the
orientation of that particular array. The add-pixels notebook's `pcolormesh`
uses integer cell edges; its pixel centers are offset by half a pixel from
`imshow`'s integer centers.

New manual mask/ROI inputs must declare `orientation='analysis'` (in the file
or notebook option) and match the oriented image shape. This is a declaration
and validation step, not an automatic conversion. The general SAXS ROI JSON
templates follow a separate path: shift/crop around their `original center`
in `(row, column)` order.

## Legacy differences and remaining ambiguities

Cell numbers below count all notebook cells from one, including Markdown cells.

- [Old SAXS](old_pipelines/XPCS_SAXS_auto.ipynb) follows the master's early
  center conversion. [Old GISAXS](old_pipelines/XPCS_GISAXS_auto.ipynb) swaps
  `inc_x0/inc_y0`, then reverses the returned `center` again.
  [Old WAXS](old_pipelines/XPCS_WAXS_auto.ipynb) retains a different center
  conversion after loading and uses `load_mask(..., reverse=..., rot90=...)`
  for its `.npy` mask. Those files cannot be assumed to have the same saved
  orientation as the newer detector-mask JSONs.
- [Detector_Masks](masking_and_rois/Detector_Masks.ipynb), cell 7, reverses the
  4M metadata pixel mask, only rotates the 500K metadata mask, and leaves the
  1M metadata mask unchanged, while images use the full transformations above.
  Chip masks have their own handling. This is an apparent asymmetry; historical
  external mask files and their provenance are needed to establish correctness.
- [Auto_Mask_SAXS](masking_and_rois/Auto_Mask_SAXS.ipynb), cells 29–31, has an
  inconsistent `hot_beam` path: `np.nonzero` row indices become `x_hots`, and
  reconstruction adds `beam_center_x` to **both** index components. Its plotting
  uses a different ordering. This looks like a defect, not a reliable conversion.
- [Auto_ROI_Mask_Generator](masking_and_rois/Auto_ROI_Mask_Generator.ipynb)
  reverses the center for 4M/500K template placement. The master currently
  checks lowercase `eiger500k` there although loading accepts both spellings;
  1M has no corresponding swap. Some plotting/masking calls also pass `(x, y)`
  to helpers expecting `(y, x)`, including master `show_ROI_on_image` and
  `create_hot_pixel_mask`. Symmetric centers can hide these mismatches.
- [WAXS_Geometry_Calibration](masking_and_rois/WAXS_Geometry_Calibration.ipynb)
  uses a Pilatus image with `np.flipud`, then `(x, y)` beam centers;
  [detector_coverage](masking_and_rois/detector_coverage.ipynb) uses simulated
  geometry. Both pass `poni1=y*pixel1`, `poni2=x*pixel2` to pyFAI. Neither
  establishes the CSS/Albula mapping or a general Pilatus rule for the master.

The Albula/IOC axis mapping and relative detector orientation are supported by
the acquisition and analysis code together. What remains to check is the
Albula cursor's exact pixel origin, the one-pixel center discrepancy, and the
CSS/Phoebus image-cursor mapping. Matching asymmetric pixel landmarks across
those displays and the raw array would settle these points; distinguish the
stored calibration reference from the current exposure's beam-center values.
