import ast
import os
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest


@pytest.mark.portable
def test_removed_numpy_and_scipy_apis_are_not_reintroduced():
    package = Path(__file__).parents[1]
    sources = "\n".join(path.read_text() for path in package.glob("*.py"))

    assert "np.in1d" not in sources
    assert "np.lib.pad" not in sources
    assert "scipy.ndimage.measurements" not in sources
    assert "1 * 10 ^ (-5)" not in sources


@pytest.mark.portable
def test_missing_pixel_size_uses_the_requested_fallback():
    from pyCHX.chx_generic_functions import check_lost_metadata

    metadata = {
        "beam_center_x": 10,
        "beam_center_y": 20,
        "incident_wavelength": 1.0,
        "detector_distance": 5.0,
        "count_time": 0.1,
        "frame_time": 0.2,
    }

    dpix, _, _, _, _, _ = check_lost_metadata(metadata, Nimg=3, pixelsize=1e-4)

    assert metadata["x_pixel_size"] == pytest.approx(1e-4)
    assert dpix == pytest.approx(0.1)


@pytest.mark.portable
def test_functions_do_not_use_mutable_literal_defaults():
    package = Path(__file__).parents[1]
    offenders = []
    for path in package.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            defaults = list(node.args.defaults) + [value for value in node.args.kw_defaults if value is not None]
            for default in defaults:
                if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                    offenders.append(f"{path.name}:{node.lineno}:{node.name}")

    assert offenders == []


@pytest.mark.portable
def test_file_consumers_do_not_receive_unmanaged_open_streams():
    package = Path(__file__).parents[1]
    offenders = []
    for path in package.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            consumer = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
            if consumer not in {"Attachment", "dump", "load"}:
                continue
            arguments = [*node.args, *(keyword.value for keyword in node.keywords)]
            has_open_call = any(
                isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == "open"
                for argument in arguments
                for child in ast.walk(argument)
            )
            if has_open_call:
                offenders.append(f"{path.name}:{node.lineno}")

    assert offenders == []


@pytest.mark.portable
def test_fill_pixel_maps_sparse_values_into_the_requested_pixels():
    from pyCHX.chx_correlationc import fill_pixel

    pixel_indices = np.array([1, 3, 5, 8])
    populated_indices = np.array([1, 5])
    values = np.array([10, 50])

    np.testing.assert_array_equal(fill_pixel(populated_indices, values, pixel_indices), [10, 0, 50, 0])


@pytest.mark.portable
def test_roi_intensity_averages_each_labeled_region():
    from pyCHX.chx_generic_functions import get_roi_intensity

    image = np.arange(16, dtype=float).reshape(4, 4)
    roi_mask = np.array(
        [
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [2, 2, 0, 0],
            [2, 2, 0, 0],
        ]
    )

    actual = get_roi_intensity(image, roi_mask)
    expected = [np.mean(image[roi_mask == 1]), np.mean(image[roi_mask == 2])]
    np.testing.assert_allclose(actual, expected)


@pytest.mark.portable
def test_rgb_brightness_scaling_clips_to_integer_range():
    from pyCHX.chx_generic_functions import scale_rgb

    image = np.array([[[10, 100, 200]]], dtype=np.uint8)

    np.testing.assert_array_equal(scale_rgb(image, scale=2), [[[20, 200, 255]]])


@pytest.mark.portable
def test_csv_entry_plot_uses_loaded_csv_data(monkeypatch, tmp_path):
    from pyCHX import chx_xpcs_xsvs_jupyter_V1 as pipeline

    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    (sample_dir / "iq.csv").write_text("q_saxs,iq_saxs\n0.01,2\n0.02,3\n")
    monkeypatch.setattr(pipeline, "get_meta_data", lambda uid: {"uid": f"full-{uid}"})

    figure, axes = pipeline.plot_entries_from_csvlist(
        ["iq.csv"],
        ["sample"],
        os.fspath(tmp_path) + "/",
        key="iq",
        fp_fulluid=False,
        ymulti=2,
    )
    try:
        np.testing.assert_allclose(axes.lines[0].get_xdata(), [0.01, 0.02])
        np.testing.assert_allclose(axes.lines[0].get_ydata(), [4, 6])
    finally:
        plt.close(figure)


@pytest.mark.portable
def test_one_time_correlation_is_mean_of_two_time_diagonals():
    from pyCHX.Two_Time_Correlation_Function import get_one_time_from_two_time

    two_time = np.arange(50, dtype=float).reshape(5, 5, 2)
    expected = np.array([[np.mean(np.diag(two_time[:, :, q], lag)) for q in range(2)] for lag in range(5)])

    np.testing.assert_allclose(get_one_time_from_two_time(two_time), expected)


@pytest.mark.portable
def test_one_time_from_two_time_preserves_nan_and_explicit_normalization_semantics():
    from pyCHX.Two_Time_Correlation_Function import get_one_time_from_two_time

    two_time = np.arange(72, dtype=float).reshape(6, 6, 2)
    two_time[1, 3, 0] = np.nan
    two_time[2, 2, 1] = np.nan
    norms = np.linspace(1.0, 3.0, 12).reshape(6, 2)
    pixel_counts = np.array([3, 7])
    expected = np.empty((6, 2))
    for delay in range(6):
        for roi_index in range(2):
            expected[delay, roi_index] = np.nanmean(np.diag(two_time[:, :, roi_index], delay)) / (
                np.average(norms[delay:, roi_index])
                * np.average(norms[: 6 - delay, roi_index])
                * pixel_counts[roi_index]
            )

    np.testing.assert_allclose(
        get_one_time_from_two_time(two_time, norms=norms, nopr=pixel_counts),
        expected,
        rtol=1e-14,
        atol=0,
    )


@pytest.mark.portable
def test_compiled_two_time_diagonal_reducer_matches_numpy():
    from pyCHX.Two_Time_Correlation_Function import get_one_time_from_two_time

    generator = np.random.default_rng(20260902)
    two_time = generator.random((360, 360, 8))
    two_time[10, 17, 3] = np.nan
    norms = 1.0 + generator.random((360, 8))
    pixel_counts = np.arange(2, 10)
    expected = np.empty((360, 8))
    for delay in range(360):
        diagonal = np.nanmean(two_time.diagonal(delay), axis=1)
        expected[delay] = diagonal / (
            norms[delay:].mean(axis=0) * norms[: 360 - delay].mean(axis=0) * pixel_counts
        )

    np.testing.assert_allclose(
        get_one_time_from_two_time(two_time, norms=norms, nopr=pixel_counts),
        expected,
        rtol=2e-13,
        atol=0,
    )


@pytest.mark.portable
def test_legacy_delay_values_scale_once():
    from skbeam.core.utils import multi_tau_lags

    from pyCHX.Two_Time_Correlation_Function import delays

    unscaled, unscaled_levels = delays(num_lev=4, num_buf=8, time=1)
    scaled, scaled_levels = delays(num_lev=4, num_buf=8, time=0.25)

    np.testing.assert_array_equal(unscaled, multi_tau_lags(4, 8)[1])
    np.testing.assert_allclose(scaled, unscaled * 0.25)
    for level in unscaled_levels:
        np.testing.assert_allclose(scaled_levels[level], unscaled_levels[level] * 0.25)

    from pyCHX.xpcs_timepixel import xpcs

    calculator = xpcs()
    np.testing.assert_allclose(calculator.delays(time=0.25, nolevs=4, nobufs=8, tmaxs=100), scaled)
    for level in scaled_levels:
        np.testing.assert_allclose(calculator.dict_dly[level], scaled_levels[level])


@pytest.mark.portable
def test_legacy_edge_helpers_return_integer_arrays():
    from pyCHX.Two_Time_Correlation_Function import get_qedge, get_qedge2, get_time_edge

    for helper, arguments in (
        (get_qedge, (2, 8, 2, 3)),
        (get_qedge2, (2, 8, 2, 3)),
        (get_time_edge, (2, 8, 2, 3)),
    ):
        edges, centers = helper(*arguments, return_int=True)
        assert isinstance(edges, np.ndarray)
        assert isinstance(centers, np.ndarray)
        assert np.issubdtype(edges.dtype, np.integer)
        assert np.issubdtype(centers.dtype, np.integer)


@pytest.mark.portable
def test_legacy_frame_roi_intensity_uses_passed_data_and_source_indices():
    from pyCHX.Two_Time_Correlation_Function import get_each_frame_ROI_intensity

    data = np.arange(24).reshape(6, 4)
    intensity, bad_frames = get_each_frame_ROI_intensity(data, sampling=2, bad_pixel_threshold=40)

    np.testing.assert_array_equal(intensity, data[::2].sum(axis=1))
    np.testing.assert_array_equal(bad_frames, [4])


@pytest.mark.portable
def test_timepixel_histograms_and_flat_field_correction():
    from pyCHX.xpcs_timepixel import (
        Get_TimePixel_Arrayc,
        get_timepixel_avg_image,
        histogram_pt,
        histogram_xyt,
    )

    x = np.array([0, 0, 0, 1, 1, 1])
    y = np.array([0, 0, 1, 0, 0, 1])
    positions = x * 2 + y
    times = np.array([0, 1, 11, 12, 21, 22])

    expected = np.zeros((2, 2, 3), dtype=int)
    for x_value, y_value, time_value in zip(x, y, times):
        expected[x_value, y_value, time_value // 10] += 1
    np.testing.assert_array_equal(histogram_xyt(x, y, times, binstep=10, detx=2, dety=2), expected)
    np.testing.assert_array_equal(
        histogram_pt(positions, times, binstep=10, detx=2, dety=2), expected.reshape(4, 3)
    )

    raw = Get_TimePixel_Arrayc(positions, times, 10, pixelist=np.array([0, 1]), detx=2, dety=2).get_data()
    corrected = Get_TimePixel_Arrayc(
        positions,
        times,
        10,
        pixelist=np.array([0, 1]),
        flat_correction=np.array([2.0, 4.0]),
        detx=2,
        dety=2,
    ).get_data()
    np.testing.assert_array_equal(raw, [[2, 0], [0, 1]])
    np.testing.assert_allclose(corrected, [[1, 0], [0, 0.25]])

    subset = Get_TimePixel_Arrayc(
        positions,
        times,
        10,
        pixelist=np.array([0, 1]),
        beg=1,
        end=3,
        detx=2,
        dety=2,
    ).get_data()
    np.testing.assert_array_equal(subset, [[0, 1]])

    non_square_average = get_timepixel_avg_image(
        np.array([0, 1]),
        np.array([2, 0]),
        np.array([0, 1]),
        det_shape=(2, 3),
    )
    np.testing.assert_array_equal(non_square_average, [[0, 0, 1], [1, 0, 0]])

    time_window_average = get_timepixel_avg_image(
        np.array([1, 0, 0]),
        np.array([0, 2, 1]),
        np.array([2e12, 0, 1e12]),
        det_shape=(2, 3),
        delta_time=1.5,
    )
    np.testing.assert_array_equal(time_window_average, [[0, 1, 1], [0, 0, 0]])


@pytest.mark.portable
@pytest.mark.parametrize("module_name", ["chx_correlation", "chx_correlationc"])
def test_label_first_two_time_conversion_preserves_each_roi(module_name):
    import importlib

    one_time_from_two_time = importlib.import_module(f"pyCHX.{module_name}").one_time_from_two_time
    two_time = np.stack(
        [
            np.arange(16, dtype=float).reshape(4, 4),
            100 + np.arange(16, dtype=float).reshape(4, 4),
        ]
    )
    expected = np.array([[np.mean(np.diag(correlation, k=lag)) for lag in range(4)] for correlation in two_time])

    np.testing.assert_allclose(one_time_from_two_time(two_time), expected)


@pytest.mark.portable
@pytest.mark.parametrize("normalization", [["regular"], ["symavg"]])
def test_cross_correlator_implementations_agree(normalization):
    from pyCHX.chx_crosscor import CrossCorrelator1, CrossCorrelator2

    first = np.arange(1, 26, dtype=float).reshape(5, 5)
    second = np.flipud(first) + 1
    roi_mask = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 2, 2],
            [0, 1, 1, 2, 2],
            [0, 1, 1, 2, 2],
            [0, 0, 0, 0, 0],
        ]
    )

    reference = CrossCorrelator1(first.shape, mask=roi_mask, normalization=normalization)(first, second)
    current = CrossCorrelator2(
        first.shape,
        mask=roi_mask,
        normalization=normalization,
        progress_bar=False,
    )(first, second)

    assert len(reference) == len(current) == 2
    for reference_roi, current_roi in zip(reference, current):
        np.testing.assert_allclose(current_roi, reference_roi, rtol=1e-13, atol=1e-13)


@pytest.mark.portable
def test_parallel_cross_correlation_handles_default_normalization():
    from pyCHX.chx_crosscor import CrossCorrelator2, run_para_ccorr_sym

    class Frames:
        def __init__(self, frames):
            self.frames = frames
            self.end = len(frames)

        def rdframe(self, index):
            return self.frames[index]

    base = np.arange(1, 26, dtype=float).reshape(5, 5)
    frames = np.stack([base + offset for offset in range(4)])
    roi_mask = np.ones((5, 5), dtype=int)
    correlator = CrossCorrelator2(frames.shape[1:], mask=roi_mask, normalization=["symavg"], progress_bar=False)

    expected = np.mean([correlator(frames[i], frames[i + 1]) for i in range(3)], axis=0)
    actual = run_para_ccorr_sym(correlator, Frames(frames))

    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


@pytest.mark.portable
def test_gpf_reconstruction_supports_sparse_roi_labels():
    from pyCHX.chx_correlationp import get_g2_from_ROI_GPF

    roi_mask = np.array([[0, 2], [5, 5]])
    numerator = np.array([[2.0, 4.0, 6.0], [3.0, 8.0, 10.0]])
    denominator = np.ones_like(numerator)

    g2, error = get_g2_from_ROI_GPF(numerator, denominator, denominator, roi_mask)

    np.testing.assert_allclose(g2, [[2.0, 5.0], [3.0, 9.0]])
    np.testing.assert_allclose(error[:, 0], 0)
    assert np.all(error[:, 1] > 0)


@pytest.mark.portable
def test_array_two_time_helpers_support_sparse_roi_labels():
    from pyCHX.chx_correlationc import auto_two_Arrayc, auto_two_Arrayc_ExplicitNorm, two_time_norm
    from pyCHX.chx_correlationp import auto_two_Arrayp
    from pyCHX.chx_correlationp2 import auto_two_Arrayp as auto_two_Arrayp2
    from pyCHX.Two_Time_Correlation_Function import auto_two_Array, get_mean_intensity

    roi_mask = np.array([[2, 2, 0], [5, 5, 5]])
    data = np.array(
        [
            [1.0, 2.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 4.0, 5.0, 7.0],
            [4.0, 6.0, 5.0, 7.0, 8.0],
        ]
    )

    expected = []
    expected_auto = []
    expected_norm = []
    for selected in (data[:, :2], data[:, 2:]):
        means = selected.mean(axis=1)
        expected.append(np.dot(selected, selected.T) / np.outer(means, means) / selected.shape[1])
        expected_auto.append(
            np.dot(selected, selected.T) / means.reshape(1, -1) / means.reshape(-1, 1) / selected.shape[1]
        )
        expected_norm.append(means.mean())
    expected = np.stack(expected, axis=2)
    expected_auto = np.stack(expected_auto, axis=2)

    np.testing.assert_array_equal(auto_two_Arrayc(data, roi_mask), expected_auto)
    integer_data = data.astype(np.int64)
    integer_expected = []
    for selected in (integer_data[:, :2], integer_data[:, 2:]):
        means = selected.mean(axis=1)
        integer_expected.append(
            np.dot(selected, selected.T) / means.reshape(1, -1) / means.reshape(-1, 1) / selected.shape[1]
        )
    np.testing.assert_array_equal(auto_two_Arrayc(integer_data, roi_mask), np.stack(integer_expected, axis=2))
    np.testing.assert_allclose(auto_two_Arrayc_ExplicitNorm(data, roi_mask, norm=data), expected)
    np.testing.assert_allclose(auto_two_Arrayp(data, roi_mask), expected)
    np.testing.assert_allclose(auto_two_Arrayp2(data, roi_mask), expected)
    np.testing.assert_allclose(auto_two_Array(None, roi_mask, data_pixel=data), expected)
    np.testing.assert_allclose(two_time_norm(data, roi_mask), expected_norm)
    mean_intensity = get_mean_intensity(data, np.array([2, 2, 5, 5, 5]))
    assert set(mean_intensity) == {2, 5}
    np.testing.assert_allclose(mean_intensity[2], data[:, :2].mean(axis=1))
    np.testing.assert_allclose(mean_intensity[5], data[:, 2:].mean(axis=1))
    np.testing.assert_array_equal(auto_two_Arrayc(data, roi_mask, index=5), expected_auto[:, :, 1:])

    with pytest.raises(ValueError, match="ROI labels not present"):
        auto_two_Arrayc(data, roi_mask, index=3)


@pytest.mark.portable
def test_production_two_time_path_prenormalizes_before_symmetric_blas(monkeypatch):
    from pyCHX import chx_correlationc

    generator = np.random.default_rng(9)
    data = 1.0 + generator.random((512, 512))
    means = data.mean(axis=1)
    observed = {}
    original_dsyrk = chx_correlationc.dsyrk

    def recording_dsyrk(alpha, normalized, **kwargs):
        observed["alpha"] = alpha
        observed["means"] = normalized.mean(axis=1)
        return original_dsyrk(alpha, normalized, **kwargs)

    monkeypatch.setattr(chx_correlationc, "dsyrk", recording_dsyrk)
    actual = chx_correlationc._symmetric_two_time_product(data, means, data.shape[1])
    expected = np.dot(data, data.T) / np.outer(means, means) / data.shape[1]

    assert observed["alpha"] == pytest.approx(1 / data.shape[1])
    np.testing.assert_allclose(observed["means"], 1.0, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=1e-14)


@pytest.mark.portable
def test_array_two_time_preserves_zero_intensity_frame_results():
    from pyCHX.chx_correlationc import auto_two_Arrayc

    roi_mask = np.array([[1, 1]])
    data = np.array([[1.0, 2.0], [0.0, 0.0], [2.0, 4.0], [-1.0, 1.0]])
    means = np.average(data, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = np.dot(data, data.T) / means.reshape(1, -1) / means.reshape(-1, 1) / data.shape[1]

    actual = auto_two_Arrayc(data, roi_mask)[:, :, 0]

    np.testing.assert_allclose(actual, expected, equal_nan=True)


@pytest.mark.portable
def test_two_time_uses_all_physical_cores_without_roi_threading(monkeypatch):
    from pyCHX import chx_correlationc

    blas_limits = []
    numba_limits = []

    def capture_limit(*, limits: int, user_api: str):
        assert user_api == "blas"
        blas_limits.append(limits)

        @contextmanager
        def context():
            yield

        return context()

    @contextmanager
    def capture_numba_limit(limit):
        numba_limits.append(limit)
        yield

    def reject_roi_thread_pool(*args, **kwargs):
        raise AssertionError("two-time ROI calculations must not use Python threads around scipy BLAS")

    monkeypatch.setattr(chx_correlationc, "physical_core_count", lambda: 4)
    monkeypatch.setattr(chx_correlationc, "threadpool_limits", capture_limit)
    monkeypatch.setattr(chx_correlationc, "numba_thread_limit", capture_numba_limit)
    monkeypatch.setattr(chx_correlationc, "ThreadPoolExecutor", reject_roi_thread_pool)
    roi_mask = np.array([[1, 1], [2, 2]])
    data = np.arange(16, dtype=float).reshape(4, 4) + 1
    chx_correlationc.auto_two_Arrayc(data, roi_mask)
    chx_correlationc.auto_two_Arrayc_ExplicitNorm(data, roi_mask, norm=data)
    assert blas_limits == [4, 4]
    assert numba_limits == [4, 4]


@pytest.mark.portable
def test_parallel_two_time_matrix_finishing_matches_serial_kernels():
    from pyCHX._performance import (
        mirror_and_normalize_two_time,
        mirror_and_normalize_two_time_parallel,
        mirror_two_time,
        mirror_two_time_parallel,
        store_symmetric_two_time_batch,
    )

    generator = np.random.default_rng(11)
    upper_triangle = np.triu(generator.random((128, 128)))
    expected_mirror = upper_triangle.copy()
    actual_mirror = upper_triangle.copy()
    mirror_two_time(expected_mirror)
    mirror_two_time_parallel(actual_mirror)
    np.testing.assert_array_equal(actual_mirror, expected_mirror)

    norms = generator.random(128)
    expected_normalized = upper_triangle.copy()
    actual_normalized = upper_triangle.copy()
    mirror_and_normalize_two_time(expected_normalized, norms, 17)
    mirror_and_normalize_two_time_parallel(actual_normalized, norms, 17)
    np.testing.assert_array_equal(actual_normalized, expected_normalized)

    upper_batch = np.empty((128, 128, 2), dtype=np.float64, order="F")
    upper_batch[:, :, 0] = upper_triangle
    upper_batch[:, :, 1] = upper_triangle
    row_norms = np.column_stack((np.ones(128), norms))
    output = np.empty((128, 128, 2), dtype=np.float64)
    store_symmetric_two_time_batch(
        upper_batch,
        row_norms,
        np.array([1, 17]),
        np.array([True, False]),
        output,
        0,
        2,
    )
    np.testing.assert_array_equal(output[:, :, 0], expected_mirror)
    np.testing.assert_array_equal(output[:, :, 1], expected_normalized)


@pytest.mark.portable
def test_two_time_batch_size_is_cache_and_memory_bounded(monkeypatch):
    from pyCHX import chx_correlationc

    frame_count = 1_000
    matrix_bytes = frame_count * frame_count * np.dtype(np.float64).itemsize
    monkeypatch.setattr(chx_correlationc, "available_memory_bytes", lambda: matrix_bytes * 40)
    assert chx_correlationc._two_time_batch_size(frame_count, 100) == 2

    monkeypatch.setattr(chx_correlationc, "available_memory_bytes", lambda: matrix_bytes * 1_000)
    assert chx_correlationc._two_time_batch_size(frame_count, 100) == 16
    assert chx_correlationc._two_time_batch_size(frame_count, 7) == 7


@pytest.mark.portable
def test_two_time_supports_hundreds_of_rois_and_returns_c_contiguous_output(monkeypatch):
    from pyCHX import chx_correlationc

    roi_mask = np.arange(1, 201, dtype=np.int64).reshape(10, 20)
    data = 1.0 + np.arange(8 * 200, dtype=np.float64).reshape(8, 200) % 31
    monkeypatch.setattr(chx_correlationc, "physical_core_count", lambda: 8)
    actual = chx_correlationc.auto_two_Arrayc(data, roi_mask)

    assert actual.shape == (8, 8, 200)
    assert actual.flags.c_contiguous
    np.testing.assert_allclose(actual, 1.0)


@pytest.mark.portable
def test_two_time_multiple_output_batches_match_original_operations(monkeypatch):
    from pyCHX import chx_correlationc

    generator = np.random.default_rng(81)
    roi_count = 20
    pixels_per_roi = 3
    frame_count = 12
    roi_mask = np.repeat(np.arange(1, roi_count + 1), pixels_per_roi).reshape(6, 10)
    data = 0.5 + generator.random((frame_count, roi_count * pixels_per_roi))
    explicit_norm = 0.5 + generator.random(data.shape)
    monkeypatch.setattr(chx_correlationc, "_two_time_batch_size", lambda *_: 7)

    expected = []
    expected_explicit = []
    expected_without_norm = []
    for roi_index in range(roi_count):
        start = roi_index * pixels_per_roi
        selected = data[:, start : start + pixels_per_roi]
        means = selected.mean(axis=1)
        expected.append(np.dot(selected, selected.T) / np.outer(means, means) / pixels_per_roi)
        explicit_means = explicit_norm[:, start : start + pixels_per_roi].mean(axis=1)
        expected_explicit.append(
            np.dot(selected, selected.T) / np.outer(explicit_means, explicit_means) / pixels_per_roi
        )
        expected_without_norm.append(np.dot(selected, selected.T) / pixels_per_roi)

    actual = chx_correlationc.auto_two_Arrayc(data, roi_mask)
    actual_explicit = chx_correlationc.auto_two_Arrayc_ExplicitNorm(data, roi_mask, norm=explicit_norm)
    actual_without_norm = chx_correlationc.auto_two_Arrayc_ExplicitNorm(data, roi_mask)
    np.testing.assert_allclose(actual, np.stack(expected, axis=2), rtol=1e-15, atol=1e-15)
    np.testing.assert_allclose(
        actual_explicit,
        np.stack(expected_explicit, axis=2),
        rtol=1e-15,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        actual_without_norm,
        np.stack(expected_without_norm, axis=2),
        rtol=1e-15,
        atol=1e-15,
    )


@pytest.mark.portable
def test_diagonal_numba_limit_uses_physical_core_count(monkeypatch):
    from pyCHX import Two_Time_Correlation_Function as two_time

    observed = []

    @contextmanager
    def recording_limit(limit):
        observed.append(limit)
        yield

    monkeypatch.setattr(two_time, "physical_core_count", lambda: 7)
    monkeypatch.setattr(two_time, "numba_thread_limit", recording_limit)
    data = np.ones((360, 360, 8))
    two_time.get_one_time_from_two_time(data)
    assert observed == [7]


@pytest.mark.portable
def test_get_pixel_array_normalization_modes_are_exact():
    from pyCHX.chx_correlationc import Get_Pixel_Arrayc

    class SparseFrames:
        beg = 1
        end = 4
        md = {"ncols": 2, "nrows": 4}

        def __init__(self, frames):
            self.frames = frames

        def rdrawframe(self, index):
            flattened = self.frames[index].ravel()
            positions = np.flatnonzero(flattened).astype(np.int32)
            return positions, flattened[positions]

    frames = np.arange(1, 33, dtype=np.float64).reshape(4, 2, 4)
    pixel_list = np.array([0, 2, 5, 7])
    qind = np.array([1, 2, 1, 2])
    norm_1d = np.array([1.5, 2.0, 2.5, 4.0])
    norm_2d = np.multiply.outer(np.arange(1.0, 5.0), norm_1d)
    imgsum = np.arange(10.0, 14.0)
    mean_int_sets = np.array([[2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]])
    cases = [
        {},
        {"norm": norm_1d},
        {"norm": norm_2d},
        {"imgsum": imgsum},
        {"mean_int_sets": mean_int_sets, "qind": qind},
        {"norm": norm_2d, "imgsum": imgsum, "mean_int_sets": mean_int_sets, "qind": qind},
    ]

    selected_frames = frames[SparseFrames.beg : SparseFrames.end].reshape(3, -1)[:, pixel_list]
    for kwargs in cases:
        expected = np.zeros_like(selected_frames)
        for output_index, frame_index in enumerate(range(SparseFrames.beg, SparseFrames.end)):
            mean_norm = mean_int_sets[frame_index, qind - 1] if kwargs.get("mean_int_sets") is not None else 1.0
            sum_norm = imgsum[frame_index] if kwargs.get("imgsum") is not None else 1.0
            if kwargs.get("norm") is norm_2d:
                pixel_norm = norm_2d[frame_index]
            elif kwargs.get("norm") is norm_1d:
                pixel_norm = norm_1d
            else:
                pixel_norm = 1.0
            expected[output_index] = selected_frames[output_index] / (mean_norm * sum_norm * pixel_norm)

        actual = Get_Pixel_Arrayc(SparseFrames(frames), pixel_list, **kwargs).get_data()
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.portable
@pytest.mark.parametrize(
    ("cal_error", "use_intensity_cache"),
    [(False, False), (False, True), (True, False)],
)
def test_optimized_one_time_kernel_matches_legacy_operations_exactly(cal_error, use_intensity_cache):
    from pyCHX.chx_correlationc import _one_time_process, _one_time_process_cached, _one_time_process_error

    def legacy_process(
        buf,
        correlation,
        past_norm,
        future_norm,
        labels,
        num_bufs,
        num_pixels,
        images_per_level,
        level,
        buffer_number,
        bad_counts,
        level_lengths,
        error_arrays=None,
    ):
        images_per_level[level] += 1
        minimum = num_bufs // 2 if level else 0
        for delay in range(minimum, min(images_per_level[level], num_bufs)):
            time_index = int(level * num_bufs / 2 + delay)
            past = buf[level, (buffer_number - delay) % num_bufs]
            future = buf[level, buffer_number]
            level_index = int(time_index - level_lengths[:level].sum())
            normalize = images_per_level[level] - delay - bad_counts[level + 1][level_index]
            if np.isnan(past).any() or np.isnan(future).any():
                bad_counts[level + 1][level_index] += 1
            elif error_arrays is None:
                for weights, output in zip(
                    [past * future, past, future],
                    [correlation, past_norm, future_norm],
                ):
                    binned = np.bincount(labels, weights=weights)[1:]
                    output[time_index] += (binned / num_pixels - output[time_index]) / normalize
            else:
                for weights, output in zip([past * future, past, future], error_arrays):
                    output[time_index] += (weights - output[time_index]) / normalize

    buf = np.array(
        [
            [
                [1.25, 2.5, 3.75, 5.0],
                [np.nan, np.nan, np.nan, np.nan],
                [2.0, 4.5, 7.0, 9.5],
                [3.5, 5.25, 8.75, 11.0],
            ]
        ]
    )
    labels = np.array([1, 1, 2, 2])
    num_pixels = np.array([2, 2])
    level_lengths = np.array([4])
    shape = (4, 2)
    legacy_arrays = [np.zeros(shape), np.zeros(shape), np.zeros(shape)]
    optimized_arrays = [array.copy() for array in legacy_arrays]
    legacy_images_per_level = np.array([3])
    optimized_images_per_level = legacy_images_per_level.copy()
    legacy_bad_counts = {1: np.zeros(4, dtype=np.int64)}
    optimized_bad_counts = {1: np.zeros(4, dtype=np.int64)}

    if cal_error:
        legacy_error_arrays = [np.zeros((4, 4)), np.zeros((4, 4)), np.zeros((4, 4))]
        optimized_error_arrays = [array.copy() for array in legacy_error_arrays]
        legacy_process(
            buf,
            *legacy_arrays,
            labels,
            4,
            num_pixels,
            legacy_images_per_level,
            0,
            3,
            legacy_bad_counts,
            level_lengths,
            legacy_error_arrays,
        )
        _one_time_process_error(
            buf,
            *optimized_arrays,
            labels,
            4,
            num_pixels,
            optimized_images_per_level,
            0,
            3,
            optimized_bad_counts,
            level_lengths,
            *optimized_error_arrays,
        )
        for actual, expected in zip(optimized_error_arrays, legacy_error_arrays):
            np.testing.assert_array_equal(actual, expected)
    else:
        legacy_process(
            buf,
            *legacy_arrays,
            labels,
            4,
            num_pixels,
            legacy_images_per_level,
            0,
            3,
            legacy_bad_counts,
            level_lengths,
        )
        optimized_arguments = [
            buf,
            *optimized_arrays,
            labels,
            4,
            num_pixels,
            optimized_images_per_level,
            0,
            3,
            optimized_bad_counts,
            level_lengths,
        ]
        if use_intensity_cache:
            intensity_cache = np.zeros((1, 4, 2), dtype=np.float64)
            for buffer_index, image in enumerate(buf[0]):
                if not np.isnan(image).any():
                    intensity_cache[0, buffer_index] = np.bincount(labels, weights=image)[1:]
            optimized_arguments.append(intensity_cache)
            _one_time_process_cached(*optimized_arguments)
        else:
            _one_time_process(*optimized_arguments)
        for actual, expected in zip(optimized_arrays, legacy_arrays):
            np.testing.assert_array_equal(actual, expected)

    np.testing.assert_array_equal(optimized_images_per_level, legacy_images_per_level)
    np.testing.assert_array_equal(optimized_bad_counts[1], legacy_bad_counts[1])


@pytest.mark.portable
@pytest.mark.parametrize("use_intensity_cache", [False, True])
@pytest.mark.parametrize(("level", "current_time"), [(0, 8), (1, 8.5)])
def test_optimized_two_time_kernel_matches_legacy_operations_exactly(level, current_time, use_intensity_cache):
    from pyCHX.chx_correlationc import _create_intensity_buffer, _two_time_process, _two_time_process_cached

    def legacy_process(
        buf,
        correlation,
        labels,
        num_bufs,
        num_pixels,
        images_per_level,
        lag_steps,
        current_time,
        level,
        buffer_number,
    ):
        images_per_level[level] += 1
        minimum = 0 if level == 0 else num_bufs // 2
        for delay in range(minimum, min(images_per_level[level], num_bufs)):
            time_index = level * num_bufs / 2 + delay
            past = buf[level, (buffer_number - delay) % num_bufs]
            future = buf[level, buffer_number]
            product_sum = np.bincount(labels, weights=past * future)[1:]
            past_sum = np.bincount(labels, weights=past)[1:]
            future_sum = np.bincount(labels, weights=future)[1:]
            first_time = current_time - 1
            second_time = current_time - lag_steps[int(time_index)] - 1
            values = product_sum / (past_sum * future_sum) * num_pixels
            if not isinstance(current_time, int):
                shift = 2 ** (level - 1)
                for offset in range(-shift + 1, shift + 1):
                    correlation[:, int(first_time + offset), int(second_time + offset)] = values
            else:
                correlation[:, int(first_time), int(second_time)] = values

    buf = np.array(
        [
            [
                [1.25, 2.5, 3.75, 5.0],
                [2.0, 4.5, 7.0, 9.5],
                [3.5, 5.25, 8.75, 11.0],
                [4.25, 6.5, 9.25, 12.5],
            ],
            [
                [1.625, 3.5, 5.375, 7.25],
                [2.75, 4.875, 7.875, 10.25],
                [3.875, 5.875, 9.0, 11.75],
                [2.9375, 4.5, 6.5625, 8.875],
            ],
        ]
    )
    labels = np.array([1, 1, 2, 2])
    num_pixels = np.array([2, 2])
    lag_steps = np.array([0, 1, 2, 3, 4, 6])
    legacy_correlation = np.zeros((2, 12, 12))
    optimized_correlation = legacy_correlation.copy()
    legacy_images_per_level = np.array([3, 3])
    optimized_images_per_level = legacy_images_per_level.copy()
    arguments = (
        buf,
        labels,
        4,
        num_pixels,
        lag_steps,
        current_time,
        level,
        3,
    )

    legacy_process(
        arguments[0],
        legacy_correlation,
        arguments[1],
        arguments[2],
        arguments[3],
        legacy_images_per_level,
        *arguments[4:],
    )
    optimized_arguments = [
        arguments[0],
        optimized_correlation,
        arguments[1],
        arguments[2],
        arguments[3],
        optimized_images_per_level,
        *arguments[4:],
    ]
    if use_intensity_cache:
        optimized_arguments.append(_create_intensity_buffer(buf, labels, len(num_pixels)))
        _two_time_process_cached(*optimized_arguments)
    else:
        _two_time_process(*optimized_arguments)

    np.testing.assert_array_equal(optimized_correlation, legacy_correlation)
    np.testing.assert_array_equal(optimized_images_per_level, legacy_images_per_level)


@pytest.mark.portable
def test_two_time_intensity_cache_tracks_frames_before_the_first_correlated_lag():
    from pyCHX.chx_correlationc import _create_intensity_buffer, _two_time_process_cached

    buf = np.zeros((2, 4, 4), dtype=np.float64)
    buf[1, 0] = [1.0, 2.0, 3.0, 4.0]
    labels = np.array([1, 1, 2, 2])
    intensity_buf = _create_intensity_buffer(buf, labels, 2)
    intensity_buf[1, 0] = 0

    _two_time_process_cached(
        buf,
        np.zeros((2, 8, 8)),
        labels,
        4,
        np.array([2, 2]),
        np.zeros(2, dtype=np.int64),
        np.array([0, 1, 2, 3, 4, 6]),
        1.5,
        1,
        0,
        intensity_buf,
    )

    np.testing.assert_array_equal(intensity_buf[1, 0], [3.0, 7.0])


@pytest.mark.portable
def test_bad_frames_mask_rows_and_columns():
    from pyCHX.Two_Time_Correlation_Function import make_g12_mask

    mask = make_g12_mask([1, 3], (5, 5))

    assert mask.mask[:, 1].all()
    assert mask.mask[1, :].all()
    assert mask.mask[:, 3].all()
    assert mask.mask[3, :].all()
    assert not mask.mask[0, 0]


@pytest.mark.portable
def test_diffusion_and_viscosity_conversions_are_inverses():
    from pyCHX.chx_generic_functions import get_diffusion_coefficient, get_viscosity

    viscosity = 8.9e-4
    radius = 125e-9
    diffusion = get_diffusion_coefficient(viscosity, radius)

    assert get_viscosity(diffusion, radius) == pytest.approx(viscosity)


@pytest.mark.portable
def test_mass_center_uses_current_scipy_namespace():
    from pyCHX.chx_generic_functions import get_mass_center_one_roi

    class Frames:
        beg = 0
        end = 2

        def rdframe(self, index):
            frame = np.zeros((4, 4))
            frame[index + 1, index + 2] = 1
            return frame

    roi_mask = np.ones((4, 4), dtype=int)
    x, y = get_mass_center_one_roi(Frames(), roi_mask, 1)

    np.testing.assert_array_equal(x, [1, 2])
    np.testing.assert_array_equal(y, [2, 3])


@pytest.mark.portable
def test_shutter_plot_uses_explicit_uid(monkeypatch):
    from pyCHX.chx_generic_functions import check_shutter_open

    monkeypatch.setattr(plt, "show", lambda: None)
    assert check_shutter_open(np.ones((3, 2, 2)), min_inten=1, plot_=True, uid="sample") == 0
    assert plt.gca().get_title() == "uid=sample--imgsum"
    plt.close("all")


@pytest.mark.portable
def test_modest_image_compatible_imshow_returns_an_artist():
    from pyCHX._optional import imshow

    figure, axes = plt.subplots()
    try:
        artist = imshow(axes, np.arange(9).reshape(3, 3), cmap="gray")
        np.testing.assert_array_equal(artist.get_array(), np.arange(9).reshape(3, 3))
    finally:
        plt.close(figure)


@pytest.mark.portable
def test_image_file_helpers_support_standard_png_images(tmp_path):
    import h5py
    from PIL import Image

    from pyCHX.chx_generic_functions import combine_images, load_pilatus
    from pyCHX.Create_Report import _image_aspect_ratio
    from pyCHX.DataGonio import Mask

    pixels = np.array([[0, 127], [128, 255]], dtype=np.uint8)
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.fromarray(pixels).save(first)
    Image.fromarray(np.flipud(pixels)).save(second)

    np.testing.assert_array_equal(load_pilatus(first), pixels)
    np.testing.assert_array_equal(Mask(first).data, [[0, 0], [1, 1]])
    assert _image_aspect_ratio(first) == pytest.approx(1)

    hdf5_mask = tmp_path / "mask.hdf5"
    with h5py.File(hdf5_mask, "w") as handle:
        handle["mask"] = np.array([[1, 0], [0, 1]])
    np.testing.assert_array_equal(Mask(hdf5_mask).data, [[1, 0], [0, 1]])

    output = tmp_path / "combined.png"
    combine_images([first, second], output, outsize=(8, 8))
    with Image.open(output) as combined:
        assert combined.size == (8, 8)


@pytest.mark.portable
def test_validate_uid_dict_checks_the_passed_mapping(monkeypatch, capsys):
    from pyCHX import chx_generic_functions

    checked = []
    monkeypatch.setattr(
        chx_generic_functions,
        "validate_uid",
        lambda uid: checked.append(uid) or uid != "bad",
    )

    chx_generic_functions.validate_uid_dict({"sample": ["good", "bad"]})

    assert checked == ["good", "bad"]
    assert "1 bad uids:['bad']" in capsys.readouterr().out


@pytest.mark.portable
def test_gisaxs_grid_uses_requested_qr_and_qz_block_sizes():
    from pyCHX.XPCS_GiSAXS import make_gisaxs_grid

    grid = make_gisaxs_grid(qr_w=2, qz_w=3, dim_r=6, dim_z=6)

    np.testing.assert_array_equal(grid[:3], np.repeat([[1, 2, 3]], 3, axis=0).repeat(2, axis=1))
    np.testing.assert_array_equal(grid[3:], np.repeat([[4, 5, 6]], 3, axis=0).repeat(2, axis=1))


@pytest.mark.portable
def test_legacy_qmap_conversion_uses_current_numpy_histograms():
    from pyCHX.DataGonio import convert_Qmap_old

    image = np.array([[1.0, 2.0], [3.0, 4.0]])
    qx = np.array([[0.0, 0.0], [1.0, 1.0]])
    qy = np.array([[0.0, 1.0], [0.0, 1.0]])

    remeshed_2d, _, _ = convert_Qmap_old(
        image,
        qx,
        qy,
        bins=(2, 2),
        rangeq=((-0.5, 1.5), (-0.5, 1.5)),
    )
    remeshed_1d, _, ybins = convert_Qmap_old(
        image,
        np.arange(4, dtype=float).reshape(2, 2),
        bins=4,
        rangeq=(0, 4),
    )

    np.testing.assert_array_equal(remeshed_2d, image)
    np.testing.assert_array_equal(remeshed_1d, image.ravel())
    assert ybins is None


@pytest.mark.portable
def test_waxs_stitching_uses_current_numpy_histograms():
    from pyCHX.Stitching import stitch_WAXS_in_Qspace_CHX

    class Calibration:
        qx_map_lab_data = np.array([[0.0, 0.0], [1.0, 1.0]])
        qy_map_lab_data = np.array([[0.0, 1.0], [0.0, 1.0]])
        qz_map_lab_data = np.array([[0.0, 0.0], [1.0, 1.0]])

        def set_angles(self, **kwargs):
            self.angles = kwargs

        def clear_maps(self):
            pass

        def _generate_qxyz_maps(self):
            pass

    image = np.array([[1.0, 2.0], [3.0, 4.0]])
    xy, zy, zx, qx, qy, qz = stitch_WAXS_in_Qspace_CHX(
        [image],
        {"phi": [0.0]},
        Calibration(),
        qxlim=(0, 2),
        qylim=(0, 2),
        qzlim=(0, 2),
        dq=1,
    )

    np.testing.assert_array_equal(xy, image)
    np.testing.assert_array_equal(zy, image)
    np.testing.assert_array_equal(zx, [[1.5, 0.0], [0.0, 3.5]])
    np.testing.assert_array_equal(qx, [0, 1])
    np.testing.assert_array_equal(qy, [0, 1])
    np.testing.assert_array_equal(qz, [0, 1])


@pytest.mark.portable
def test_gisaxs_rate_fit_honors_fit_range():
    from pyCHX.XPCS_GiSAXS import fit_qr_qz_rate

    qr = np.linspace(0.01, 0.05, 5)
    rates = np.concatenate((2 * qr**2, 3 * qr**2))

    diffusion = fit_qr_qz_rate(qr, [0.1, 0.2], rates, plot_=False, fit_range=(1, 5))

    np.testing.assert_allclose(diffusion, [2, 3], rtol=1e-7)


@pytest.mark.portable
def test_simple_gisaxs_fit_initializes_fixed_alpha(tmp_path):
    from pyCHX.XPCS_GiSAXS import fit_gisaxs_g2, stretched_auto_corr_scat_factor

    taus = np.logspace(-2, 1, 20)
    g2 = stretched_auto_corr_scat_factor(taus, beta=0.2, relaxation_rate=0.7, alpha=1.0, baseline=1.0)
    result = fit_gisaxs_g2(
        g2[:, np.newaxis],
        {
            "taus": taus,
            "qz_center": np.array([0.02]),
            "qr_center": np.array([0.01]),
            "uid": "test",
            "path": os.fspath(tmp_path) + "/",
        },
        function="simple",
        one_plot=True,
    )
    try:
        assert result["alpha"][0] == pytest.approx(1.0)
        assert result["rate"][0] == pytest.approx(0.7, rel=1e-3)
    finally:
        plt.close("all")


@pytest.mark.portable
def test_general_g2_fit_recovers_synthetic_relaxation_rates():
    from pyCHX.chx_generic_functions import get_g2_fit_general

    taus = np.geomspace(1e-3, 10, 40)
    expected_rates = np.array([0.7, 1.4])
    g2 = np.column_stack(
        [
            1 + 0.12 * np.exp(-2 * expected_rates[0] * taus),
            1 + 0.08 * np.exp(-2 * expected_rates[1] * taus),
        ]
    )

    fit_results, fit_taus, fitted_g2 = get_g2_fit_general(
        g2,
        taus,
        function="simple",
        guess_values={"baseline": 1.0, "beta": 0.1, "alpha": 1.0, "relaxation_rate": 1.0},
    )

    actual_rates = [result.best_values["relaxation_rate"] for result in fit_results]
    np.testing.assert_allclose(actual_rates, expected_rates, rtol=1e-6)
    np.testing.assert_array_equal(fit_taus, taus[1:])
    np.testing.assert_allclose(fitted_g2, g2[1:], rtol=1e-6)


@pytest.mark.portable
def test_general_g2_plot_supports_error_bars(tmp_path):
    from pyCHX.chx_generic_functions import plot_g2_general

    taus = np.geomspace(1e-3, 10, 20)
    g2 = np.column_stack([1 + 0.1 * np.exp(-taus), 1 + 0.2 * np.exp(-2 * taus)])
    errors = np.full_like(g2, 0.01)
    figure = plot_g2_general(
        {1: g2},
        {1: taus},
        {0: np.array([0.01]), 1: np.array([0.02])},
        g2_err_dict={1: errors},
        filename="synthetic_g2",
        path=os.fspath(tmp_path) + "/",
        return_fig=True,
    )
    try:
        assert (tmp_path / "synthetic_g2.png").is_file()
        data_axes = [axes for axes in figure.axes if axes.has_data()]
        assert len(data_axes) == 2
        assert all(axes.get_xscale() == "log" for axes in data_axes)
    finally:
        plt.close(figure)


@pytest.mark.portable
def test_general_q_rate_fit_recovers_synthetic_diffusion():
    from pyCHX.chx_generic_functions import get_q_rate_fit_general

    q_values = np.array([0.01, 0.02, 0.03, 0.04])
    qval_dict = {index: np.array([q]) for index, q in enumerate(q_values)}

    diffusion, _ = get_q_rate_fit_general(qval_dict, 2.5 * q_values**2)

    np.testing.assert_allclose(diffusion, [2.5], rtol=1e-8)


@pytest.mark.portable
def test_g2_fitters_reject_unknown_functions(tmp_path):
    from pyCHX.chx_generic_functions import get_g2_fit_general
    from pyCHX.XPCS_SAXS import fit_saxs_rad_ang_g2

    taus = np.array([0.0, 1.0, 2.0])
    g2 = np.ones((3, 1))

    with pytest.raises(ValueError, match="Unsupported correlation function 'unknown'"):
        get_g2_fit_general(g2, taus, function="unknown")

    parameters = {
        "taus": taus,
        "q_ring_center": np.array([0.01]),
        "ang_center": np.array([0.0]),
        "uid": "test",
        "path": os.fspath(tmp_path) + "/",
    }
    with pytest.raises(ValueError, match="Unsupported correlation function 'unknown'"):
        fit_saxs_rad_ang_g2(g2, parameters, function="unknown")


@pytest.mark.portable
def test_plot_gamma_uses_explicit_inputs():
    from pyCHX.XPCS_SAXS import plot_gamma

    figure, axes = plot_gamma("sample", np.array([0.01, 0.02]), {"rate": np.array([2.0, 4.0])})
    try:
        assert axes.get_title() == "Uid= sample--Gamma"
        np.testing.assert_allclose(axes.lines[0].get_ydata(), [0.5, 0.25])
    finally:
        plt.close(figure)


@pytest.mark.portable
def test_plot_t_qrc_can_save_without_notebook_globals(tmp_path):
    from pyCHX.XPCS_GiSAXS import plot_t_qrc

    qr_data = np.array([[0.01, 10.0, 11.0], [0.02, 8.0, 9.0], [0.03, 6.0, 7.0]])
    path = os.fspath(tmp_path) + "/"

    plot_t_qrc(qr_data, [(0, 2), (2, 4)], save=True, pargs={"uid": "test", "path": path})

    assert (tmp_path / "uid=test--Iq-t-.png").is_file()
    assert (tmp_path / "uid=test-q-Iqt").is_file()
    plt.close("all")


@pytest.mark.portable
def test_form_factor_plot_vlim_uses_intensity_data(monkeypatch, tmp_path):
    from pyCHX.SAXS import plot_form_factor_with_fit

    class Result:
        best_values = {"radius": 10.0, "delta_rho": 1.0, "sigma": 0.1}
        best_fit = np.array([2.0, 3.0, 4.0])

    monkeypatch.setattr(plt, "show", lambda: None)
    figure = plot_form_factor_with_fit(
        np.array([0.1, 0.2, 0.3]),
        np.array([1.0, 2.0, 4.0]),
        np.array([0.1, 0.2, 0.3]),
        Result(),
        res_pargs={"uid": "test", "path": os.fspath(tmp_path) + "/"},
        vlim=(0.5, 2),
        return_fig=True,
    )
    try:
        np.testing.assert_allclose(figure.axes[0].get_ylim(), [0.5, 8.0])
    finally:
        plt.close(figure)
