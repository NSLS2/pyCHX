"""Private geometry helpers for the unified XPCS notebook.

Coordinates in this module are in the oriented analysis image: x is a column,
y is a row, distances are millimetres, and q is in inverse Angstroms.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

MODES = ("saxs", "gi_saxs", "waxs", "gi_waxs")


def resolve_geometry(metadata, notebook_value):
    """Start metadata wins; otherwise retain the editable notebook value."""
    mode = metadata.get("scat_geometry", notebook_value)
    if mode not in MODES:
        raise ValueError(f"Unknown scat_geometry {mode!r}; choose one of {', '.join(MODES)}")
    return mode


def _positive_number(value, name):
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return value


def _setting(metadata, explicit, keys, name):
    for key in keys:
        if key in metadata and metadata[key] is not None:
            return metadata[key]
    if explicit is not None:
        return explicit
    raise ValueError(f"Missing {name} in run metadata; set {name} in the notebook")


def surface_q_maps(
    shape, center, dpix, distance, wavelength, metadata, incident_angle_deg=None, inplane_angle_deg=0
):
    """Calculate calibrated q_parallel and q_z for one oriented detector image."""
    angle = _positive_number(
        _setting(
            metadata, incident_angle_deg, ("inc_ang", "incident_angle", "incident_angle_deg"), "incident_angle_deg"
        ),
        "incident_angle_deg",
    )
    azimuth = _setting(
        metadata, inplane_angle_deg, ("inc_phi", "inplane_angle", "inplane_angle_deg"), "inplane_angle_deg"
    )
    azimuth = float(azimuth)
    if not np.isfinite(azimuth):
        raise ValueError("inplane_angle_deg must be finite")
    pixel_mm = _positive_number(dpix, "pixel size")
    distance_mm = _positive_number(distance, "sample-detector distance")
    wavelength_a = _positive_number(wavelength, "wavelength")
    x0, y0 = map(float, center)
    if not np.isfinite([x0, y0]).all():
        raise ValueError("Incident beam center must be finite")
    alpha_i = np.deg2rad(angle)
    phi = np.deg2rad(azimuth)
    reflected_y = y0 + np.tan(2 * alpha_i) * distance_mm / pixel_mm
    reflected_x = x0 - np.tan(phi) * (reflected_y - y0)
    rows, cols = np.indices(shape)
    # This is the same grazing-incidence convention as convert_gisaxs_pixel_to_q.
    alpha_f = np.arctan2((rows - y0) * pixel_mm, distance_mm) - alpha_i
    theta_f = np.arctan2((cols - x0) * pixel_mm, distance_mm) / 2
    qx = np.cos(alpha_f) * np.cos(2 * theta_f) - np.cos(alpha_i)
    qy_lab = np.cos(alpha_f) * np.sin(2 * theta_f)
    qz_lab = np.sin(alpha_f) + np.sin(alpha_i)
    qy = qz_lab * np.sin(phi) + qy_lab * np.cos(phi)
    qz = qz_lab * np.cos(phi) - qy_lab * np.sin(phi)
    factor = 2 * np.pi / wavelength_a
    return {
        "q_parallel": np.hypot(qx, qy) * factor,
        "q_z": qz * factor,
        "incident_angle_deg": angle,
        "inplane_angle_deg": azimuth,
        "incident_center": (x0, y0),
        "reflected_center": (reflected_x, reflected_y),
    }


def _edges(values, valid, requested, count, name):
    if requested is not None:
        edges = np.asarray(requested, dtype=float)
        if edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
            raise ValueError(f"{name} must be strictly increasing finite bin edges")
        return edges
    selected = np.asarray(values)[valid]
    selected = selected[np.isfinite(selected) & (selected > 0)]
    if selected.size < 4:
        raise ValueError(f"Too few valid pixels to derive {name}; supply explicit bin edges")
    low, high = np.quantile(selected, [0.02, 0.98])
    if high <= low:
        raise ValueError(f"No usable {name} range; supply explicit bin edges")
    return np.linspace(low, high, count + 1)


def _rectangular_rois(q_parallel, q_z, valid, parallel_edges, z_edges):
    labels = np.zeros(q_parallel.shape, dtype=np.int32)
    q_values = {}
    for z_lo, z_hi in zip(z_edges[:-1], z_edges[1:]):
        z_band = valid & (q_z >= z_lo) & (q_z < z_hi)
        shared_z = float(np.mean(q_z[z_band])) if z_band.any() else (z_lo + z_hi) / 2
        for p_lo, p_hi in zip(parallel_edges[:-1], parallel_edges[1:]):
            selected = valid & (q_parallel >= p_lo) & (q_parallel < p_hi) & (q_z >= z_lo) & (q_z < z_hi)
            if selected.any():
                number = len(q_values) + 1
                labels[selected] = number
                q_values[number - 1] = np.array([np.mean(q_parallel[selected]), shared_z])
    if not q_values:
        raise ValueError("No populated surface q ROIs; adjust the q bin edges")
    return labels, q_values


def build_surface_rois(q_parallel, q_z, mask, parallel_edges=None, z_edges=None, parallel_count=8, z_count=3):
    valid = np.asarray(mask, dtype=bool) & np.isfinite(q_parallel) & np.isfinite(q_z) & (q_z > 0)
    p_edges = _edges(q_parallel, valid, parallel_edges, parallel_count, "q_parallel edges")
    z_edges = _edges(q_z, valid, z_edges, z_count, "q_z edges")
    roi_mask, qval_dict = _rectangular_rois(q_parallel, q_z, valid, p_edges, z_edges)

    # The legacy GISAXS profile functions accept [start, end, width, number].
    def profile_args(edges):
        widths = np.diff(edges)
        if not np.allclose(widths, widths[0], rtol=1e-10, atol=0):
            return None
        count = len(edges) - 1
        return [float(edges[0]), float(edges[-1]), float(widths[0] * 0.8), count]

    return {
        "roi_mask": roi_mask,
        "qval_dict": qval_dict,
        "parallel_edges": p_edges,
        "z_edges": z_edges,
        "Qr": profile_args(p_edges),
        "Qz": profile_args(z_edges),
    }


def cal_1d_qr_from_roi(data, roi_mask, qval_dict, path=None, uid=None, save=True):
    """Average each surface ROI without integrating unselected q-space gaps."""
    image = np.asarray(data, dtype=float)
    labels = np.asarray(roi_mask)
    if image.shape != labels.shape:
        raise ValueError("Surface profile image and ROI mask must have the same shape")
    if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
        raise ValueError("Surface profile ROI labels must be nonnegative integers")
    count = int(labels.max())
    if count < 1 or not np.array_equal(np.unique(labels[labels > 0]), np.arange(1, count + 1)):
        raise ValueError("Surface profile ROI labels must be contiguous and populated")
    if set(qval_dict) != set(range(count)):
        raise ValueError("Surface profile q values must match the ROI labels")
    centers = np.stack([np.asarray(qval_dict[index], dtype=float)[:2] for index in range(count)])
    if centers.shape != (count, 2) or not np.isfinite(centers).all():
        raise ValueError("Surface profile q values must contain finite q_parallel and q_z centers")

    selected = (labels > 0) & np.isfinite(image)
    totals = np.bincount(labels[selected], weights=image[selected], minlength=count + 1)[1:]
    sizes = np.bincount(labels[selected], minlength=count + 1)[1:]
    means = np.divide(totals, sizes, out=np.full(count, np.nan), where=sizes > 0)
    order = np.argsort(centers[:, 0], kind="stable")
    z_centers = np.round(centers[:, 1], 8)
    columns = {"qr": centers[order, 0]}
    for index, z_center in enumerate(np.unique(z_centers)):
        columns[f"qz{index}={z_center:.4f}"] = np.where(z_centers[order] == z_center, means[order], np.nan)
    profile = pd.DataFrame(columns)
    if save:
        if path is None or uid is None:
            raise ValueError("Surface profile path and uid are required when saving")
        profile.to_csv(Path(path) / f"{uid}_qr_1d.csv")
    return profile


def get_t_qrc_from_roi(FD, frame_edge, roi_mask, qval_dict, get_avg_image, path=None, uid="uid", save=True):
    """Build time-dependent q_parallel profiles from the same selected ROI pixels."""
    frame_edge = list(frame_edge)
    frames = [
        cal_1d_qr_from_roi(
            get_avg_image(FD, beg=start, end=stop, sampling=1, plot_=False), roi_mask, qval_dict, save=False
        )
        for start, stop in frame_edge
    ]
    if not frames:
        raise ValueError("Surface time profiles require at least one frame range")
    columns = {"qr": frames[0]["qr"].to_numpy()}
    for band in frames[0].columns[1:]:
        for (start, stop), frame in zip(frame_edge, frames):
            columns[f"{band}_fra_{start}_to_{stop}"] = frame[band].to_numpy()
    profile = pd.DataFrame(columns)
    if save:
        if path is None:
            raise ValueError("Surface time profile path is required when saving")
        profile.to_csv(Path(path) / f"{uid}_qrt_pds.csv")
    return profile


def build_waxs_rois(image, mask):
    """Preserve the WAXS notebook's smoothed-intensity percentile bands."""
    smoothed = gaussian_filter(np.where(mask, np.asarray(image, dtype=float), 0), 2)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(smoothed)
    if not valid.any():
        raise ValueError("No valid WAXS pixels")
    values = smoothed[valid]
    cutoffs = [100, 99.95, 99.9, 99.8, 99.6, 99.3, 98.5, 97, 95]
    labels = np.zeros(smoothed.shape, dtype=np.int32)
    q_values = {}
    for high, next_high in zip(cutoffs[:-1], cutoffs[1:]):
        low = high - 0.3 * (high - next_high)
        selected = valid & (smoothed <= np.percentile(values, high)) & (smoothed > np.percentile(values, low))
        if selected.any():
            number = len(q_values) + 1
            labels[selected] = number
            q_values[number - 1] = np.array([float(number)])
    if not q_values:
        raise ValueError("No populated WAXS intensity bands")
    return labels, q_values


def _read_saved(path, array_key):
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as loaded:
            return {key: loaded[key].copy() for key in loaded.files}
    if path.suffix == ".npy":
        return {array_key: np.load(path, allow_pickle=False)}
    if path.suffix in (".pkl", ".pickle"):
        with path.open("rb") as stream:
            return pickle.load(stream)
    raise ValueError(f"Unsupported saved input format: {path.suffix}")


def _orientation(saved, explicit):
    orientation = saved.get("orientation", explicit) if isinstance(saved, dict) else explicit
    if isinstance(orientation, np.ndarray):
        orientation = orientation.item()
    if orientation != "analysis":
        raise ValueError("Saved input must declare orientation='analysis' for the oriented detector image")


def load_manual_mask(path, shape, orientation=None, geometry=None):
    saved = _read_saved(path, "mask")
    if isinstance(saved, dict):
        _orientation(saved, orientation)
        if geometry is not None and "geometry" in saved and str(np.asarray(saved["geometry"]).item()) != geometry:
            raise ValueError("Saved mask geometry does not match selected scat_geometry")
        data = saved.get("mask")
    else:
        _orientation(saved, orientation)
        data = saved
    data = np.asarray(data)
    if data.shape != tuple(shape) or not np.isfinite(data).all() or not np.isin(data, [0, 1]).all():
        raise ValueError("Saved mask shape or values do not match the oriented detector image")
    return data.astype(bool)


def combine_detector_masks(saved, shape, detector):
    """Validate and combine the components of an analysis-oriented detector mask."""
    expected = detector.split("_")[0]
    actual = saved.get("detector") if isinstance(saved, dict) else None
    if not isinstance(actual, str) or actual.casefold() != expected.casefold():
        raise ValueError(f"Detector mask identity {actual!r} does not match {expected!r}")
    components = saved.get("masks")
    if not isinstance(components, dict) or not components:
        raise ValueError("Detector mask must contain at least one mask component")
    combined = np.ones(shape, dtype=bool)
    for name, pixels in components.items():
        try:
            part = np.asarray(pixels)
        except ValueError as exc:
            raise ValueError(f"Detector mask component {name!r} has an invalid shape") from exc
        if part.shape != tuple(shape):
            raise ValueError(f"Detector mask component {name!r} does not match the oriented detector shape")
        if not np.isin(part, [0, 1]).all():
            raise ValueError(f"Detector mask component {name!r} must contain only 0/1 values")
        combined &= part.astype(bool)
    return combined


def _radial_q(shape, center, dpix, distance, wavelength):
    rows, cols = np.indices(shape)
    radius_mm = np.hypot(rows - center[1], cols - center[0]) * dpix
    return 4 * np.pi / wavelength * np.sin(np.arctan2(radius_mm, distance) / 2)


def validate_roi(
    roi_mask,
    qval_dict,
    shape,
    geometry,
    mask=None,
    q_maps=None,
    center=None,
    dpix=None,
    distance=None,
    wavelength=None,
):
    roi_mask = np.asarray(roi_mask)
    if roi_mask.shape != tuple(shape) or not np.issubdtype(roi_mask.dtype, np.integer):
        raise ValueError("ROI labels must be an integer image matching the oriented detector shape")
    labels = np.unique(roi_mask)
    positive = labels[labels > 0]
    if np.any(labels < 0) or not np.array_equal(positive, np.arange(1, len(positive) + 1)):
        raise ValueError("ROI labels must be contiguous positive integers starting at 1")
    if not len(positive) or set(qval_dict) != set(range(len(positive))):
        raise ValueError("ROI q-value keys must match every populated ROI label")
    if mask is not None and np.any((roi_mask > 0) & ~np.asarray(mask, dtype=bool)):
        raise ValueError("ROI contains pixels excluded by the analysis mask")
    if geometry == "waxs":
        expected = np.arange(1, len(positive) + 1)
        actual = np.asarray([qval_dict[i][0] for i in range(len(positive))], dtype=float)
        if not np.array_equal(actual, expected):
            raise ValueError("WAXS q values must be ROI indices 1, 2, ...")
        return roi_mask, qval_dict
    if geometry == "saxs":
        if any(value is None for value in (center, dpix, distance, wavelength)):
            raise ValueError("SAXS calibration is required to validate ROI q values")
        maps = (_radial_q(shape, center, dpix, distance, wavelength),)
    else:
        if q_maps is None:
            raise ValueError("Calibrated surface q maps are required to validate ROI q values")
        maps = (q_maps["q_parallel"], q_maps["q_z"])
    for index, label in enumerate(positive):
        values = np.asarray(qval_dict[index], dtype=float).reshape(-1)
        required = len(maps)
        if len(values) < required or not np.isfinite(values).all() or values[0] <= 0:
            raise ValueError(f"Invalid q values for ROI {label}")
        selected = roi_mask == label
        for coordinate, q_map in enumerate(maps):
            observed = np.asarray(q_map)[selected]
            span = max(float(np.ptp(observed)), float(np.ptp(q_map)) * 0.005, 1e-5)
            if not (observed.min() - span <= values[coordinate] <= observed.max() + span):
                raise ValueError(f"ROI {label} q value disagrees with calibrated image orientation")
    return roi_mask, qval_dict


def intersect_roi_with_mask(roi_mask, qval_dict, mask, geometry=None):
    """Apply a later dynamic mask and keep labels aligned with q-value keys."""
    selected = np.where(mask, roi_mask, 0)
    labels = np.unique(selected)
    labels = labels[labels > 0]
    if not len(labels):
        raise ValueError("Dynamic mask removed every ROI")
    renumbered = np.zeros(selected.shape, dtype=np.int32)
    values = {}
    for number, old_label in enumerate(labels, start=1):
        renumbered[selected == old_label] = number
        values[number - 1] = (
            np.array([float(number)]) if geometry == "waxs" else np.asarray(qval_dict[int(old_label) - 1])
        )
    return renumbered, values


def load_manual_roi(path, shape, geometry, orientation=None, **validation):
    saved = _read_saved(path, "roi_mask")
    _orientation(saved, orientation)
    if isinstance(saved, (list, tuple)) and len(saved) == 2:
        roi_mask, qval_dict = saved
    elif isinstance(saved, dict):
        roi_mask = saved.get("roi_mask")
        if "geometry" in saved and str(np.asarray(saved["geometry"]).item()) != geometry:
            raise ValueError("Saved ROI geometry does not match selected scat_geometry")
        if "q_values" in saved:
            qval_dict = {i: np.atleast_1d(row) for i, row in enumerate(np.asarray(saved["q_values"]))}
        else:
            qval_dict = saved.get("qval_dict")
    else:
        raise ValueError("Saved ROI must contain a mask and q values")
    if not isinstance(qval_dict, dict):
        raise ValueError("Saved ROI has no q-value dictionary")
    return validate_roi(roi_mask, qval_dict, shape, geometry, **validation)


def load_surface_roi_config(path, shape, geometry, mask, q_maps):
    """Edit a saved surface ROI or create new q-space boxes from a JSON recipe."""
    if geometry not in ("gi_saxs", "gi_waxs"):
        raise ValueError("Surface ROI configurations require gi_saxs or gi_waxs")
    path = Path(path)
    with path.open() as stream:
        config = json.load(stream)
    if (
        not isinstance(config, dict)
        or config.get("geometry") != geometry
        or config.get("orientation") != "analysis"
    ):
        raise ValueError("Surface ROI configuration needs matching geometry and orientation='analysis'")
    base_file = config.get("base_roi_file")
    old_qmap_file = config.get("base_qmap_file")
    if old_qmap_file is not None and base_file is None:
        raise ValueError("base_qmap_file requires base_roi_file")

    def from_recipe(value):
        source = Path(value)
        return source if source.is_absolute() else path.parent / source

    z_groups = {}
    if base_file is not None:
        old_maps = q_maps
        if old_qmap_file is not None:
            old_data = _read_saved(from_recipe(old_qmap_file), "q_parallel")
            if not isinstance(old_data, (list, tuple)) or len(old_data) < 2:
                raise ValueError("Legacy base q map must contain q_parallel and q_z arrays")
            old_maps = {"q_parallel": np.asarray(old_data[0]), "q_z": np.asarray(old_data[1])}
            if any(values.shape != tuple(shape) for values in old_maps.values()):
                raise ValueError("Legacy base q map shape must match the oriented image")
        roi_mask, base_q_values = load_manual_roi(
            from_recipe(base_file),
            shape,
            geometry,
            orientation="analysis",
            # Removed labels may intersect the current mask; check retained
            # labels against it after applying the recipe.
            mask=None,
            q_maps=old_maps,
        )
        roi_mask = roi_mask.copy()
        # Saved surface ROIs use one q_z value for every label in a z band.
        for label in range(1, int(roi_mask.max()) + 1):
            z_groups[label] = ("base", round(float(base_q_values[label - 1][1]), 8))
    else:
        roi_mask = np.zeros(shape, dtype=np.int32)

    removed = config.get("remove_labels", [])
    if (
        not isinstance(removed, list)
        or any(
            isinstance(label, bool) or not isinstance(label, int) or label < 1 or label > roi_mask.max()
            for label in removed
        )
        or len(set(removed)) != len(removed)
    ):
        raise ValueError("remove_labels must contain distinct labels from the base ROI")
    if removed:
        roi_mask[np.isin(roi_mask, removed)] = 0

    q_parallel = np.asarray(q_maps["q_parallel"])
    q_z = np.asarray(q_maps["q_z"])
    valid = np.asarray(mask, dtype=bool) & np.isfinite(q_parallel) & np.isfinite(q_z) & (q_z > 0)
    if old_qmap_file is not None:
        projected = np.zeros(shape, dtype=np.int32)
        for label in np.unique(roi_mask):
            if label <= 0:
                continue
            old_selected = roi_mask == label
            old_parallel = old_maps["q_parallel"][old_selected]
            old_z = old_maps["q_z"][old_selected]
            p_lo, p_hi = np.nextafter([np.min(old_parallel), np.max(old_parallel)], [-np.inf, np.inf])
            z_lo, z_hi = np.nextafter([np.min(old_z), np.max(old_z)], [-np.inf, np.inf])
            selected = valid & (q_parallel >= p_lo) & (q_parallel <= p_hi) & (q_z >= z_lo) & (q_z <= z_hi)
            if not selected.any():
                raise ValueError(f"Legacy ROI {label} has no pixels in the current q calibration")
            if np.any(projected[selected]):
                raise ValueError(f"Projected legacy ROI {label} overlaps another ROI")
            projected[selected] = label
        roi_mask = projected

    boxes = config.get("boxes", [])
    if not isinstance(boxes, list) or (base_file is None and not boxes):
        raise ValueError("Surface ROI configuration needs a base ROI or at least one box")
    for index, box in enumerate(boxes, start=1):
        if not isinstance(box, dict) or "q_parallel" not in box or "q_z" not in box:
            raise ValueError(f"Box {index} needs q_parallel and q_z ranges")
        bounds = []
        for key in ("q_parallel", "q_z"):
            try:
                pair = np.asarray(box[key], dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Box {index} has invalid {key} bounds") from exc
            if pair.shape != (2,) or not np.isfinite(pair).all() or pair[0] < 0 or pair[1] <= pair[0]:
                raise ValueError(f"Box {index} has invalid {key} bounds")
            bounds.append(pair)
        (p_lo, p_hi), (z_lo, z_hi) = bounds
        selected = valid & (q_parallel >= p_lo) & (q_parallel < p_hi) & (q_z >= z_lo) & (q_z < z_hi)
        if not selected.any():
            raise ValueError(f"Box {index} contains no valid pixels")
        if np.any(roi_mask[selected] != 0):
            raise ValueError(f"Box {index} overlaps an existing ROI; remove its label first")
        label = int(roi_mask.max()) + 1
        roi_mask[selected] = label
        z_groups[label] = ("box", float(z_lo), float(z_hi))

    # A new box can extend a retained saved band. Match only when its z range
    # contains exactly one retained band's pixels in the current calibration.
    base_groups = {}
    for label in np.unique(roi_mask):
        if label > 0 and z_groups[int(label)][0] == "base":
            base_groups.setdefault(z_groups[int(label)], []).append(label)
    base_ranges = {}
    for group, labels in base_groups.items():
        values = q_z[np.isin(roi_mask, labels)]
        base_ranges[group] = (float(np.min(values)), float(np.max(values)))
    for label in np.unique(roi_mask):
        if label <= 0:
            continue
        group = z_groups[int(label)]
        if group[0] != "box":
            continue
        matches = [base for base, (low, high) in base_ranges.items() if group[1] <= low and high < group[2]]
        if len(matches) == 1:
            z_groups[int(label)] = matches[0]

    result = np.zeros(shape, dtype=np.int32)
    q_values = {}
    z_totals = {}
    z_counts = {}
    parallel_centers = {}
    for label in np.unique(roi_mask):
        if label <= 0:
            continue
        selected = roi_mask == label
        group = z_groups[int(label)]
        z_totals[group] = z_totals.get(group, 0.0) + float(np.sum(q_z[selected]))
        z_counts[group] = z_counts.get(group, 0) + int(np.count_nonzero(selected))
        parallel_centers[int(label)] = float(np.mean(q_parallel[selected]))
    for label in np.unique(roi_mask):
        if label <= 0:
            continue
        selected = roi_mask == label
        number = len(q_values) + 1
        result[selected] = number
        group = z_groups[int(label)]
        q_values[number - 1] = np.array([parallel_centers[int(label)], z_totals[group] / z_counts[group]])
    return validate_roi(result, q_values, shape, geometry, mask=mask, q_maps=q_maps)


def plot_surface_roi(image, roi_mask, q_maps, path, uid):
    """Plot calibrated surface coordinates without changing public plot APIs."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(image, origin="lower", norm="log" if np.any(np.asarray(image) > 0) else None)
    ax.contour(roi_mask, levels=np.arange(0.5, int(np.max(roi_mask)) + 1), colors="white", linewidths=0.4)
    for key, color in (("q_parallel", "cyan"), ("q_z", "orange")):
        q_map = np.asarray(q_maps[key])
        levels = np.quantile(q_map[np.isfinite(q_map)], [0.2, 0.4, 0.6, 0.8])
        if np.unique(levels).size > 1:
            ax.contour(q_map, levels=np.unique(levels), colors=color, linewidths=0.5, alpha=0.6)
    ax.set(
        xlabel="detector column",
        ylabel="detector row",
        title=f"{uid} surface q ROIs (cyan: q_parallel; orange: q_z), Å⁻¹",
    )
    fig.savefig(Path(path) / f"{uid}_ROI_on_Image.png", dpi=140)
    plt.close(fig)


def plot_roi_index_rates(rates, path, uid):
    """Show WAXS relaxation rates by ROI index without implying a calibrated q."""
    import matplotlib.pyplot as plt

    values = np.asarray(rates, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(np.arange(1, len(values) + 1), values, "o-")
    ax.set(xlabel="WAXS ROI index", ylabel="relaxation rate (s$^{-1}$)")
    fig.tight_layout()
    fig.savefig(Path(path) / f"{uid}_Q_Rate_fit.png", dpi=140)
    plt.close(fig)


def plot_xpcs_correlation(
    g2_dict,
    taus_dict,
    qval_dict,
    *,
    mode,
    geometry,
    filename,
    path,
    append_name="",
    ylabel="g2",
    g2_labels=None,
    g2_err_dict=None,
    **kwargs,
):
    """Use ROI-index panels for WAXS and the existing plotter elsewhere."""
    if mode != "waxs":
        from pyCHX.chx_generic_functions import plot_g2_general

        return plot_g2_general(
            g2_dict=g2_dict,
            taus_dict=taus_dict,
            qval_dict=qval_dict,
            geometry=geometry,
            filename=filename,
            path=path,
            append_name=append_name,
            ylabel=ylabel,
            g2_labels=g2_labels,
            g2_err_dict=g2_err_dict,
            **kwargs,
        )
    import matplotlib.pyplot as plt

    count = len(qval_dict)
    columns = min(4, count)
    rows = int(np.ceil(count / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(3.4 * columns, 2.8 * rows), squeeze=False)
    for index, ax in enumerate(axes.flat):
        if index >= count:
            ax.axis("off")
            continue
        for series, (key, values) in enumerate(g2_dict.items()):
            tau = np.asarray(taus_dict[key])
            y = np.asarray(values)[:, index]
            label = g2_labels[series] if g2_labels is not None else ("data" if series == 0 else "fit")
            if g2_err_dict is not None and key in g2_err_dict:
                ax.errorbar(
                    tau,
                    y,
                    yerr=np.asarray(g2_err_dict[key])[:, index],
                    marker="o",
                    markersize=2,
                    linewidth=0.8,
                    label=label,
                )
            else:
                ax.plot(tau, y, marker="o" if series == 0 else None, markersize=2, linewidth=0.9, label=label)
        if np.any(tau > 0):
            ax.set_xscale("log")
        ax.set(title=f"ROI {index + 1} (intensity band)", xlabel="delay (s)", ylabel=ylabel)
        if index == 0 and len(g2_dict) > 1:
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(Path(path) / f"{filename}{append_name}.png", dpi=140)
    plt.close(fig)
