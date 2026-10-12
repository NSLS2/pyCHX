"""Synthetic checks for the four modes of the unified XPCS notebook."""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from skbeam.core.correlation import multi_tau_auto_corr

from pyCHX._xpcs_pipeline_geometry import (
    build_surface_rois,
    build_waxs_rois,
    combine_detector_masks,
    intersect_roi_with_mask,
    load_manual_mask,
    load_manual_roi,
    plot_roi_index_rates,
    plot_surface_roi,
    plot_xpcs_correlation,
    resolve_geometry,
    surface_profile_args_from_roi,
    surface_q_maps,
    validate_roi,
)

SHAPE = (64, 80)
CENTER = (40.0, 8.0)
PIXEL_MM = 0.075
DISTANCE_MM = 1500.0
WAVELENGTH_A = 1.0


def _saxs_roi(mask):
    rows, cols = np.indices(SHAPE)
    radius = np.hypot(rows - CENTER[1], cols - CENTER[0])
    labels = np.zeros(SHAPE, dtype=np.int32)
    labels[(radius >= 8) & (radius < 14) & mask] = 1
    labels[(radius >= 18) & (radius < 25) & mask] = 2
    q_map = 4 * np.pi / WAVELENGTH_A * np.sin(np.arctan2(radius * PIXEL_MM, DISTANCE_MM) / 2)
    return labels, {index: np.array([np.mean(q_map[labels == index + 1])]) for index in range(2)}


@pytest.fixture
def synthetic_inputs():
    rng = np.random.default_rng(31415)
    mask = np.ones(SHAPE, dtype=bool)
    mask[:, :2] = False
    mask[-2:, :] = False
    image = rng.gamma(4, 4, SHAPE)
    q_maps = surface_q_maps(SHAPE, CENTER, PIXEL_MM, DISTANCE_MM, WAVELENGTH_A, {"inc_ang": 0.4})
    return mask, image, q_maps


def _mode_roi(mode, mask, image, q_maps):
    if mode == "saxs":
        return _saxs_roi(mask)
    if mode == "waxs":
        return build_waxs_rois(image, mask)
    result = build_surface_rois(
        q_maps["q_parallel"],
        q_maps["q_z"],
        mask,
        parallel_count=4 if mode == "gi_saxs" else 6,
        z_count=2 if mode == "gi_saxs" else 3,
    )
    return result["roi_mask"], result["qval_dict"]


def _validation(mode, mask, q_maps):
    common = {"mask": mask}
    if mode == "saxs":
        common.update(center=CENTER, dpix=PIXEL_MM, distance=DISTANCE_MM, wavelength=WAVELENGTH_A)
    elif mode in ("gi_saxs", "gi_waxs"):
        common["q_maps"] = q_maps
    return common


def test_metadata_geometry_precedence_and_missing_surface_angle():
    assert resolve_geometry({}, "waxs") == "waxs"
    assert resolve_geometry({"scat_geometry": "gi_waxs"}, "saxs") == "gi_waxs"
    with pytest.raises(ValueError, match="Unknown scat_geometry"):
        resolve_geometry({"scat_geometry": "invalid"}, "saxs")
    with pytest.raises(ValueError, match="set incident_angle_deg"):
        surface_q_maps(SHAPE, CENTER, PIXEL_MM, DISTANCE_MM, WAVELENGTH_A, {})


def test_surface_q_calibration_matches_existing_conversion(synthetic_inputs):
    from pyCHX.XPCS_GiSAXS import convert_gisaxs_pixel_to_q

    _, _, maps = synthetic_inputs
    reflected_x, reflected_y = maps["reflected_center"]
    _, _, q_parallel, q_z = convert_gisaxs_pixel_to_q(
        *CENTER,
        reflected_x,
        reflected_y,
        pixelsize=(75, 75),
        Lsd=DISTANCE_MM / 1000,
        dimx=SHAPE[1],
        dimy=SHAPE[0],
        lamda=WAVELENGTH_A,
    )
    np.testing.assert_allclose(maps["q_parallel"], q_parallel, atol=1e-12)
    np.testing.assert_allclose(maps["q_z"], q_z, atol=1e-12)


@pytest.mark.parametrize("mode", ("saxs", "gi_saxs", "waxs", "gi_waxs"))
def test_synthetic_geometry_manual_fallback_correlation_and_hdf5(mode, synthetic_inputs, tmp_path):
    from pyCHX.Create_Report import export_xpcs_results_to_h5

    mask, image, q_maps = synthetic_inputs
    roi_mask, q_values = _mode_roi(mode, mask, image, q_maps)
    validation = _validation(mode, mask, q_maps)
    validate_roi(roi_mask, q_values, SHAPE, mode, **validation)
    assert len(q_values) == roi_mask.max()
    assert not np.any((roi_mask > 0) & ~mask)
    if mode == "waxs":
        np.testing.assert_array_equal([value[0] for value in q_values.values()], np.arange(1, len(q_values) + 1))
    elif mode in ("gi_saxs", "gi_waxs"):
        assert all(len(value) == 2 for value in q_values.values())

    mask_file = tmp_path / "mask.npz"
    roi_file = tmp_path / "roi.npz"
    np.savez(mask_file, mask=mask.astype(np.uint8), orientation="analysis")
    np.savez(
        roi_file,
        roi_mask=roi_mask,
        q_values=np.stack(list(q_values.values())),
        orientation="analysis",
        geometry=mode,
    )
    loaded_mask = load_manual_mask(mask_file, SHAPE)
    loaded_roi, loaded_q = load_manual_roi(roi_file, SHAPE, mode, **validation)
    np.testing.assert_array_equal(loaded_mask, mask)
    np.testing.assert_array_equal(loaded_roi, roi_mask)
    for index in q_values:
        np.testing.assert_allclose(loaded_q[index], q_values[index])

    rng = np.random.default_rng(22)
    images = rng.poisson(image + 2, size=(16, *SHAPE)).astype(float)
    g2, lag_steps = multi_tau_auto_corr(2, 8, loaded_roi, images)
    assert g2.shape == (len(lag_steps), len(q_values))
    assert np.isfinite(g2).all()

    metadata = {
        "uid": "synthetic",
        "scat_geometry": mode,
        "mask_source": str(mask_file),
        "roi_source": str(roi_file),
    }
    export_xpcs_results_to_h5(
        "result.h5",
        str(tmp_path) + "/",
        {"md": metadata, "roi_mask": loaded_roi, "qval_dict": loaded_q, "g2": g2, "taus": lag_steps.astype(float)},
    )
    with h5py.File(tmp_path / "result.h5") as result:
        assert result["md"].attrs["scat_geometry"] == mode
        assert result["md"].attrs["roi_source"] == str(roi_file)
        np.testing.assert_array_equal(result["roi_mask"], loaded_roi)
        np.testing.assert_allclose(result["g2"], g2)
        assert len(result["qval_dict"].attrs) == len(q_values)


def test_saved_inputs_reject_orientation_shape_labels_and_bad_q(synthetic_inputs, tmp_path):
    mask, image, q_maps = synthetic_inputs
    roi_mask, q_values = _mode_roi("gi_waxs", mask, image, q_maps)
    values = np.stack(list(q_values.values()))
    mask_file = tmp_path / "mask.npy"
    np.save(mask_file, mask)
    with pytest.raises(ValueError, match="orientation"):
        load_manual_mask(mask_file, SHAPE)
    assert load_manual_mask(mask_file, SHAPE, orientation="analysis").shape == SHAPE
    np.savez(tmp_path / "wrong_mode_mask.npz", mask=mask, orientation="analysis", geometry="saxs")
    with pytest.raises(ValueError, match="geometry"):
        load_manual_mask(tmp_path / "wrong_mode_mask.npz", SHAPE, geometry="gi_waxs")
    np.savez(tmp_path / "bad_shape.npz", roi_mask=roi_mask.T, q_values=values, orientation="analysis")
    with pytest.raises(ValueError, match="shape"):
        load_manual_roi(tmp_path / "bad_shape.npz", SHAPE, "gi_waxs", **_validation("gi_waxs", mask, q_maps))
    broken = roi_mask.copy()
    broken[broken == 2] = 3
    np.savez(tmp_path / "bad_labels.npz", roi_mask=broken, q_values=values, orientation="analysis")
    with pytest.raises(ValueError, match="contiguous"):
        load_manual_roi(tmp_path / "bad_labels.npz", SHAPE, "gi_waxs", **_validation("gi_waxs", mask, q_maps))
    values[0, 1] += 10
    np.savez(tmp_path / "bad_q.npz", roi_mask=roi_mask, q_values=values, orientation="analysis")
    with pytest.raises(ValueError, match="orientation"):
        load_manual_roi(tmp_path / "bad_q.npz", SHAPE, "gi_waxs", **_validation("gi_waxs", mask, q_maps))


def test_detector_mask_components_reject_broadcasting_bad_values_and_wrong_detector():
    saved = {
        "detector": "eiger500K",
        "masks": {
            "pixel_mask": [[1, 0, 1], [1, 1, 1]],
            "chip_mask": [[1, 1, 1], [0, 1, 1]],
        },
    }
    np.testing.assert_array_equal(
        combine_detector_masks(saved, (2, 3), "eiger500k_single_image"),
        [[True, False, True], [False, True, True]],
    )
    with pytest.raises(ValueError, match="identity"):
        combine_detector_masks(saved, (2, 3), "eiger4m_single_image")
    saved["masks"]["chip_mask"] = [[1, 1, 1]]
    with pytest.raises(ValueError, match="component 'chip_mask'.*shape"):
        combine_detector_masks(saved, (2, 3), "eiger500K_single_image")
    saved["masks"]["chip_mask"] = [[1, 1, 1], [0, 2, 1]]
    with pytest.raises(ValueError, match="component 'chip_mask'.*0/1"):
        combine_detector_masks(saved, (2, 3), "eiger500K_single_image")


def test_dynamic_beam_mask_keeps_roi_labels_and_q_values_aligned(synthetic_inputs):
    mask, image, q_maps = synthetic_inputs
    roi_mask, values = _mode_roi("waxs", mask, image, q_maps)
    later_mask = mask & (roi_mask != 2)
    updated, updated_values = intersect_roi_with_mask(roi_mask, values, later_mask, geometry="waxs")
    assert np.array_equal(np.unique(updated), np.arange(len(updated_values) + 1))
    assert len(updated_values) == len(values) - 1
    assert np.array_equal(updated_values[1], [2])
    validate_roi(updated, updated_values, SHAPE, "waxs", mask=later_mask)


def test_saxs_final_mask_reindexes_q_values_after_hot_pixels():
    import time

    notebook = json.loads((Path(__file__).resolve().parents[2] / "pipelines/XPCS_auto.ipynb").read_text())
    cell = [item for item in notebook["cells"] if item["cell_type"] == "code"][41]
    state = {
        "time": time,
        "np": np,
        "scat_geometry": "saxs",
        "roi_mask2": np.array([[1, 1], [2, 2]], dtype=np.int32),
        "qval_dict": {0: np.array([0.1]), 1: np.array([0.2])},
        "mask_copy": np.ones((2, 2), dtype=bool),
        "avg_img": np.ones((2, 2)),
        "det_mask": np.ones((2, 2), dtype=bool),
        "center": (0, 0),
        "mask_outliers": False,
        "setup_pargs": {},
        "create_hot_pixel_mask": lambda *args, **kwargs: np.array([[0, 0], [1, 1]]),
        "get_circular_average": lambda *args, **kwargs: (np.array([0.1, 0.2]),) * 3,
        "plot_circular_average": lambda *args, **kwargs: None,
        "intersect_roi_with_mask": intersect_roi_with_mask,
        "_record_xpcs_cell_timing": lambda *args: None,
    }
    exec("".join(cell["source"]), state)
    np.testing.assert_array_equal(state["roi_mask"], [[0, 0], [1, 1]])
    np.testing.assert_array_equal(state["qval_dict"][0], [0.2])
    assert len(state["qval_dict"]) == state["roi_mask"].max()


def test_surface_saved_roi_works_when_auto_bins_have_too_few_pixels():
    parallel = np.array([[1.0, 2.0], [3.0, 4.0]])
    q_z = np.array([[1.0, 1.0], [2.0, 2.0]])
    mask = np.array([[True, True], [False, True]])
    with pytest.raises(ValueError, match="Too few valid pixels"):
        build_surface_rois(parallel, q_z, mask)
    roi_mask = np.array([[1, 1], [0, 0]], dtype=np.int32)
    q_values = {0: np.array([1.5, 1.0])}
    maps = {"q_parallel": parallel, "q_z": q_z}
    validate_roi(roi_mask, q_values, roi_mask.shape, "gi_saxs", mask=mask, q_maps=maps)
    profile = surface_profile_args_from_roi(roi_mask, q_values, maps)
    assert profile["Qr"][3] == 1 and profile["Qz"][3] == 1

    root = Path(__file__).resolve().parents[2]
    notebook = json.loads((root / "pipelines/XPCS_auto.ipynb").read_text())
    source = "".join(notebook["cells"][43]["source"])
    assert source.index("if manual_roi_file is not None:", source.index("elif scat_geometry in")) < source.index(
        "build_surface_rois("
    )


def test_surface_plot_uses_report_filename(synthetic_inputs, tmp_path):
    mask, image, q_maps = synthetic_inputs
    roi_mask, _ = _mode_roi("gi_saxs", mask, image, q_maps)
    plot_surface_roi(image, roi_mask, q_maps, tmp_path, "uid=synthetic")
    assert (tmp_path / "uid=synthetic_ROI_on_Image.png").stat().st_size > 0
    plot_roi_index_rates([0.2, 0.4], tmp_path, "uid=synthetic")
    assert (tmp_path / "uid=synthetic_Q_Rate_fit.png").stat().st_size > 0
    plot_xpcs_correlation(
        {1: np.ones((4, 2))},
        {1: np.array([0.1, 0.2, 0.4, 0.8])},
        {0: np.array([1]), 1: np.array([2])},
        mode="waxs",
        geometry="gi_waxs",
        filename="uid=synthetic_g2",
        path=tmp_path,
    )
    assert (tmp_path / "uid=synthetic_g2.png").stat().st_size > 0


def test_beam_mask_configuration_is_geometry_specific(tmp_path):
    import os
    import time
    import warnings

    root = Path(__file__).resolve().parents[2]
    notebook = json.loads((root / "pipelines/XPCS_auto.ipynb").read_text())
    source = "".join(notebook["cells"][29]["source"])
    mask_dir = tmp_path / "cycle" / "user" / "masks"
    mask_dir.mkdir(parents=True)
    generic = mask_dir / "XPCS_beam_mask.json"
    specific = mask_dir / "XPCS_beam_mask_gi_saxs.json"
    generic.write_text('{"masks": {"generic": {}}}')
    specific.write_text('{"masks": {"surface": {}}}')

    def run(mode, manual_mask_file=None):
        state = dict(
            default_dir=str(tmp_path) + "/",
            cycle="cycle",
            username="user",
            scat_geometry=mode,
            beam_mask_config_file=None,
            auto_bst_mask=True,
            manual_mask_file=manual_mask_file,
            os=os,
            json=json,
            time=time,
            warnings=warnings,
            _record_xpcs_cell_timing=lambda *args: None,
        )
        exec(source, state)
        return state

    assert run("saxs")["beam_mask_source"] == str(generic)
    assert run("gi_saxs")["beam_mask_source"] == str(specific)
    with pytest.warns(RuntimeWarning, match="gi_waxs.*empty"):
        assert run("gi_waxs")["beam_mask_source"] == "generated:gi_waxs:beam_mask"
    with pytest.warns(RuntimeWarning, match="waxs.*empty"):
        assert run("waxs")["beam_mask_source"] == "generated:waxs:beam_mask"
    with warnings.catch_warnings(record=True) as caught:
        assert run("waxs", manual_mask_file="manual.npz")["beam_mask_source"] == "generated:waxs:beam_mask"
    assert not caught


@pytest.mark.parametrize("mode", ("saxs", "gi_saxs", "waxs", "gi_waxs"))
def test_report_renders_selected_mode_and_roi_image(mode, synthetic_inputs, tmp_path):
    import matplotlib.pyplot as plt

    from pyCHX.Create_Report import make_pdf_report

    mask, image, q_maps = synthetic_inputs
    roi_mask, _ = _mode_roi(mode, mask, image, q_maps)
    plt.imsave(tmp_path / "uid=synthetic_img_avg.png", image)
    plt.imsave(tmp_path / "uid=synthetic_ROI_on_Image.png", roi_mask)
    plt.imsave(tmp_path / "uid=synthetic_fra_0_16_g2.png", image)
    plt.imsave(tmp_path / "uid=synthetic_fra_0_16_Q_Rate_fit.png", image)
    metadata = {
        "uid": "synthetic",
        "scat_geometry": mode,
        "sample": "synthetic",
        "start_time": "now",
        "stop_time": "now",
        "Measurement": "test",
        "incident_wavelength": WAVELENGTH_A,
        "number of images": 16,
        "count_time": 0.1,
        "frame_time": 0.1,
        "detector_distance": DISTANCE_MM / 1000,
        "feedback_x": "off",
        "feedback_y": "off",
        "shutter mode": "auto",
        "beam_center_x": CENTER[0],
        "beam_center_y": CENTER[1],
        "beam_refl_center_x": CENTER[0],
        "beam_refl_center_y": CENTER[1] + 20,
        "mask_file": "synthetic",
        "roi_mask_file": "synthetic",
        "bad_frame_list": [],
        "transmission": 1,
        "beg": 0,
        "end": 16,
    }
    report_type = "gi_waxs" if mode == "waxs" else mode
    make_pdf_report(
        str(tmp_path) + "/",
        "synthetic",
        str(tmp_path) + "/",
        f"{mode}.pdf",
        "test",
        False,
        True,
        False,
        False,
        False,
        False,
        report_type=report_type,
        md=metadata,
    )
    pdf = (tmp_path / f"{mode}.pdf").read_bytes()
    assert pdf.startswith(b"%PDF") and len(pdf) > 3_000


def test_unified_notebook_preserves_parameters_and_shared_correlation_cells():
    root = Path(__file__).resolve().parents[2]
    original = json.loads((root / "pipelines/old_pipelines/XPCS_SAXS_auto.ipynb").read_text())
    unified = json.loads((root / "pipelines/XPCS_auto.ipynb").read_text())
    assert "# Parameters" in "".join(unified["cells"][7]["source"])
    for index in (7, 12, 22, 52, 89, 103, 104, 112, 113, 148, 152):
        assert unified["cells"][index]["cell_type"] == original["cells"][index]["cell_type"]

    def without_timing_record(cell):
        return "".join(
            line for line in cell["source"] if not line.lstrip().startswith("_record_xpcs_cell_timing(")
        )

    for index in (103, 104, 112, 113, 148):
        assert without_timing_record(unified["cells"][index]) == without_timing_record(original["cells"][index])


def test_all_input_cells_are_numbered_and_papermill_timing_survives_injection():
    import re
    import time

    import nbformat

    pytest.importorskip("papermill")
    from papermill.parameterize import parameterize_notebook

    root = Path(__file__).resolve().parents[2]
    notebook = nbformat.read(root / "pipelines/XPCS_auto.ipynb", as_version=4)
    cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert len(cells) == 104
    for number, cell in enumerate(cells, start=1):
        assert re.search(
            rf'_record_xpcs_cell_timing\("{number:02d} - [^"]+", _xpcs_cell_start\)',
            cell.source,
        )
    assert "parameters" in cells[2].metadata.tags
    assert "injected-parameters" in cells[3].metadata.tags

    injected = parameterize_notebook(notebook, {"uid": "synthetic"})
    injected_cells = [cell for cell in injected.cells if cell.cell_type == "code"]
    assert 'uid = "synthetic"' in injected_cells[3].source
    assert "_record_xpcs_cell_timing" not in injected_cells[3].source

    recorded = []
    state = {
        "time": time,
        "_xpcs_parameters_start": time.time(),
        "_xpcs_parameters_recorded": False,
        "_record_xpcs_cell_timing": lambda label, start: recorded.append(label),
    }
    exec(injected_cells[3].source, state)
    prelude = injected_cells[4].source.split("# Data security modification:", 1)[0]
    exec(prelude, state)
    assert recorded == ["04 - Papermill parameters"]


def test_single_roi_two_time_path_without_one_time_result():
    import itertools
    import time
    from types import SimpleNamespace

    root = Path(__file__).resolve().parents[2]
    notebook = json.loads((root / "pipelines/XPCS_auto.ipynb").read_text())
    calls = {}

    def save_correlation(data, **kwargs):
        calls["csv"] = kwargs
        return kwargs

    state = {
        "np": np,
        "time": time,
        "itertools": itertools,
        "plt": SimpleNamespace(cm=SimpleNamespace(jet=None)),
        "run_two_time": True,
        "g12b": np.ones((16, 16, 1)),
        "lag_steps": np.array([0, 1, 2, 3]),
        "FD": SimpleNamespace(beg=0, end=16),
        "Nimg": 16,
        "timeperframe": 0.1,
        "qval_dict": {0: np.array([0.01, 0.02])},
        "scat_geometry": "gi_waxs",
        "scat_geometry_": "gi_saxs",
        "uid_": "uid=synthetic_fra_0_16",
        "q_mask_name": "gi_waxs",
        "data_dir": "/tmp/",
        "fit_g2_func": "stretched",
        "get_one_time_from_two_time": lambda _: np.ones((16, 1)),
        "save_g2_general": save_correlation,
        "get_g2_fit_general": lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("fit failed")),
        "show_C12": lambda *args, **kwargs: calls.update(q_ind=kwargs["q_ind"]),
        "plot_xpcs_correlation": lambda *args, **kwargs: calls.update(plot_q=kwargs["qval_dict"]),
        "_record_xpcs_cell_timing": lambda *args: None,
    }
    for index in (114, 115, 116, 117):
        exec("".join(notebook["cells"][index]["source"]), state)
    assert calls["q_ind"] == 1
    np.testing.assert_array_equal(calls["csv"]["qz"], [0.02])
    assert state["g2b_fit_paras"].shape == (1, 4)
    assert list(calls["plot_q"]) == [0]
