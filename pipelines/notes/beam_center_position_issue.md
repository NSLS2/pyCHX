# Beam-center position issue

The [acquisition center finder](https://github.com/NSLS2/chx-profile-collection/blob/670a10be52a0db75f090087f6c466eabe48aaf16/startup/30-user.py#L74) writes zero-based raw pixel coordinates to BeamX/BeamY. `load_data` flips detector rows and, for Eiger 500K, then applies `np.rot90`. For raw `(x, y)` and raw shape `(H, W)`, the matching analysis pixel is:

| Detector | Correct analysis `(x, y)` | Current pipeline | Offset |
| --- | --- | --- | --- |
| Eiger 4M/1M | `(x, H-1-y)` | `(x, H-y)` | `+1` row |
| Eiger 500K | `(H-1-y, W-1-x)` | `(H-y, W-x)` | `+1` on both axes |

The same conversion appears in the SAXS mask and ROI notebooks. It can shift beam masks, ROIs, and calibrated q maps. Separately, the unified pipeline writes SAXS result metadata with `beam_center_x` and `beam_center_y` swapped.

**Check before changing historical results:** compare a direct-beam pixel in a raw frame, its BeamX/BeamY values, and its location in the oriented analysis image for each detector. The automatic finder is zero-based; manually entered centers may have a different convention. See [coordinate conventions](../COORDINATE_CONVENTIONS.md) for the full trace.
