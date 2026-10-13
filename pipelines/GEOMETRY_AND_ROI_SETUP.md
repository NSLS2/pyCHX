# Set up ROIs in `XPCS_auto.ipynb`

In the **Run controls** cell, set `scat_geometry` to one of the modes below.
Bluesky run-start `scat_geometry` takes precedence over the notebook value.
The ROI choices are:

| Geometry | Default ROIs | Custom choice |
| --- | --- | --- |
| `saxs` | Ring or q-phi template | Create a named template or set `manual_roi_file`. |
| `gi_saxs` | 8 × 3 q grid | Set `gisaxs_q_parallel_edges`/`gisaxs_q_z_edges` or `surface_roi_config_file`. |
| `waxs` | Intensity bands, indexed 1, 2, ... | Set `manual_roi_file` for custom pixel ROIs. |
| `gi_waxs` | 12 × 4 q grid | Set `giwaxs_q_parallel_edges`/`giwaxs_q_z_edges` or `surface_roi_config_file`. |

GI grid edges and surface JSON box ranges use `(q_parallel, q_z)` in inverse
ångströms. WAXS intensity-band indices are not calibrated q values.

For SAXS templates, add a name and parameters to `mask_param_dict` in
[`Auto_ROI_Mask_Generator.ipynb`](masking_and_rois/Auto_ROI_Mask_Generator.ipynb),
set `mask_selection` to that name, and run it. It writes
`_mask_path_/ROI_masks/general_roi_mask_<name>.json` and will not overwrite an
existing name. Keep `roi_auto=True` to use this template. If run-start
`analysis` names a different template, `analysis_type_auto=True` selects that
instead. To force a notebook choice, set `analysis_type_auto=False`, choose
`qphi_analysis=False` for rings or `True` for q-phi, and set the corresponding
mask name. If you set `roi_auto=False`, you must set `manual_roi_file`.

## Check calibration and masks first

GI-SAXS and GI-WAXS require calibrated q maps. Supply these values in Bluesky
metadata when collecting the run, or use the **Run controls** fallback for an
older run:

| Value | Run metadata | Notebook fallback |
| --- | --- | --- |
| Incident angle (degrees) | `inc_ang`, `incident_angle`, or `incident_angle_deg` | `surface_incident_angle_deg`; required if metadata lacks an angle |
| In-plane angle (degrees) | `inc_phi`, `inplane_angle`, or `inplane_angle_deg` | `surface_inplane_angle_deg` (default 0) |
| Incident beam center | Image `beam_center_x/y` | `surface_beam_center_x/y` if either center coordinate is missing; use oriented image column/row |
| Pixel size | `x_pixel_size` (metres) | `surface_pixel_size_mm` |
| Detector distance | `det_distance` or `detector_distance` (metres) | `surface_detector_distance_mm` |
| Wavelength | `wavelength` or `incident_wavelength` (ångströms) | `surface_wavelength_angstrom` |

The notebook always loads the oriented detector mask from
`_mask_path_/pixel_masks/detector_mask_<detector>.json`. Run
[`Detector_Masks.ipynb`](masking_and_rois/Detector_Masks.ipynb) only if it is
missing or needs rebuilding; use
[`Detector_Masks_Add_Pixels.ipynb`](masking_and_rois/Detector_Masks_Add_Pixels.ipynb)
for known bad pixels (`pixels_to_add` uses **row, column**).

For a beam shadow, put `XPCS_beam_mask_<scat_geometry>.json` in the cycle/user
`masks/` directory, or set `beam_mask_config_file`. SAXS also accepts the old
`XPCS_beam_mask.json`. Alternatively, set `manual_mask_file` to a saved mask
with the oriented image shape and 1=keep/0=exclude. An `.npz` mask must
contain `mask` and `orientation='analysis'`; for legacy `.npy`/`.pkl` masks,
set `manual_mask_orientation='analysis'`. The default GI-SAXS beam mask
excludes a stripe; the default GI-WAXS and WAXS beam masks are empty when
`auto_bst_mask=True`. Check the combined-mask plot before analyzing a new setup.

## Edit or create surface ROIs with JSON

For `gi_saxs` or `gi_waxs`, save a JSON file and set
`surface_roi_config_file` to its path. This example starts from the 2025-3
rheadrick GI-SAXS ROI and q-map files in
`/nsls2/users/jgoodrich/proposals/2025-3/rheadrick/masks/`. Save the JSON
there too so the relative paths work:

```json
{
  "geometry": "gi_saxs",
  "orientation": "analysis",
  "base_roi_file": "uid=0ae234d7_roi_mask.pkl",
  "base_qmap_file": "uid=0ae234d7_qmap.pkl",
  "remove_labels": [1],
  "boxes": [
    {"q_parallel": [0.003, 0.018], "q_z": [0.151, 0.158]}
  ]
}
```

`remove_labels` refers to labels in the saved ROI. Each box uses q values in
inverse ångströms with `[lower, upper)` bounds. Adjust the example ranges for
your run; a box must contain unmasked pixels and cannot overlap a retained
ROI. Set `remove_labels` and `boxes` to `[]` to start with all saved regions.
For a new layout, omit both base-file fields and list your boxes. Change
`geometry` to `gi_waxs` for GI-WAXS.

With `base_qmap_file`, the loader estimates each old ROI's q bounds from its
saved pixels and draws those ranges on the **current** q maps. Inspect the
resulting ROI overlay: this reprojection is an approximation and can produce
empty or overlapping regions after a geometry change. Without
`base_qmap_file`, it uses the saved pixel labels directly and checks their q
values against the current calibration. The other rheadrick starting point is
`uid=151209_roi_mask.pkl` with its matching `uid=151209_qmap.pkl`. The old
notebooks used a hardcoded 0.46° incident angle; verify that angle against the
run metadata before applying it in the unified notebook.

For direct saved ROI use, set `manual_roi_file` instead of
`surface_roi_config_file`. A new `.npz` file needs `roi_mask`, ordered
`q_values` (one row per positive label), and `orientation='analysis'`.
SAXS values are calibrated q; surface values are `(q_parallel, q_z)`; WAXS
values are ROI indices `1, 2, ...`. Labels must be contiguous and inside the
analysis mask. For an old `.pkl` ROI, also set
`manual_roi_orientation='analysis'`. No ROI-generation notebook is needed
for surface JSON recipes or default WAXS bands.

Check the resolved geometry, calibration, combined mask, and ROI overlay for
each new setup. Before committing an edited notebook, run
`pre-commit run nbstripout --files pipelines/XPCS_auto.ipynb`. See
[coordinate conventions](COORDINATE_CONVENTIONS.md) when moving saved pixel
masks between detector orientations.
