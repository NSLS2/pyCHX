import struct

import numpy as np
import pytest


class _FakePool:
    def __init__(self):
        self.events = []

    def close(self):
        self.events.append("close")

    def terminate(self):
        self.events.append("terminate")

    def join(self):
        self.events.append("join")


class _FakeAsyncResult:
    def __init__(self, value=None, error=None):
        self.value = value
        self.error = error

    def get(self):
        if self.error is not None:
            raise self.error
        return self.value


def _make_compressed_correlation_input(tmp_path):
    from pyCHX.chx_compress import init_compress_eigerdata

    frame_number = np.arange(24, dtype=np.int32)[:, np.newaxis, np.newaxis]
    pixel_number = np.arange(16, dtype=np.int32).reshape(1, 4, 4)
    frames = 1 + (3 * frame_number + 2 * pixel_number) % 17
    ring_mask = np.array(
        [
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 2, 2, 0],
            [0, 2, 2, 0],
        ],
        dtype=np.int64,
    )
    detector_mask = np.ones(ring_mask.shape, dtype=bool)
    filename = tmp_path / "correlation.cmp"

    init_compress_eigerdata(
        frames,
        detector_mask.copy(),
        {"pixel_mask": detector_mask.copy()},
        str(filename),
        nobytes=4,
        with_pickle=False,
    )
    return filename, frames, ring_mask


@pytest.mark.portable
def test_compressed_file_round_trip(tmp_path):
    from pyCHX.chx_compress import Multifile, init_compress_eigerdata

    frames = np.array(
        [
            [[0, 1, 0, 2], [3, 0, 0, 0], [0, 4, 0, 0]],
            [[0, 2, 0, 1], [0, 0, 5, 0], [0, 0, 0, 3]],
            [[1, 0, 0, 0], [0, 6, 0, 0], [2, 0, 0, 0]],
            [[0, 0, 7, 0], [1, 0, 0, 0], [0, 0, 2, 0]],
        ],
        dtype=np.int32,
    )
    mask = np.ones(frames.shape[1:], dtype=bool)
    metadata = {"pixel_mask": mask.copy()}
    filename = tmp_path / "synthetic.cmp"

    actual_mask, average, intensity, bad_frames = init_compress_eigerdata(
        frames,
        mask.copy(),
        metadata,
        str(filename),
        nobytes=4,
        with_pickle=False,
    )

    np.testing.assert_array_equal(actual_mask, mask)
    np.testing.assert_allclose(average, frames.mean(axis=0))
    np.testing.assert_array_equal(intensity, frames.sum(axis=(1, 2)))
    assert bad_frames.size == 0

    with Multifile(str(filename), beg=0, end=len(frames)) as compressed:
        for index, expected in enumerate(frames):
            np.testing.assert_array_equal(compressed.rdframe(index), expected)
    assert compressed.FID.closed


@pytest.mark.portable
@pytest.mark.parametrize(
    ("nobytes", "bins", "value_format"),
    [(2, 1, "h"), (4, 1, "i"), (4, 2, "d")],
)
def test_compressed_payload_keeps_legacy_binary_layout(tmp_path, nobytes, bins, value_format):
    from pyCHX.chx_compress import init_compress_eigerdata

    frames = np.array(
        [
            [[0, 1, 2], [3, 0, 4]],
            [[5, 0, 6], [0, 7, 8]],
            [[9, 10, 0], [11, 12, 0]],
        ],
        dtype=np.int32,
    )
    mask = np.ones(frames.shape[1:], dtype=bool)
    filename = tmp_path / "layout.cmp"

    init_compress_eigerdata(
        frames,
        mask.copy(),
        {"pixel_mask": mask.copy()},
        str(filename),
        nobytes=nobytes,
        bins=bins,
        with_pickle=False,
    )

    expected = bytearray()
    for start in range(0, len(frames), bins):
        image = np.average(frames[start : start + bins], axis=0)
        positions = np.flatnonzero(image.ravel() > 0)
        values = image.ravel()[positions]
        if bins == 1:
            values = values.astype({2: np.int16, 4: np.int32}[nobytes])
        expected.extend(struct.pack("@I", len(positions)))
        expected.extend(struct.pack("@{}i".format(len(positions)), *positions))
        expected.extend(struct.pack("@{}{}".format(len(positions), value_format), *values))

    assert filename.read_bytes()[1024:] == bytes(expected)


@pytest.mark.portable
def test_compress_eigerdata_stages_then_publishes_serial_output(tmp_path):
    from pyCHX.chx_compress import compress_eigerdata

    frames = np.arange(1, 25, dtype=np.int32).reshape(4, 2, 3)
    mask = np.ones(frames.shape[1:], dtype=bool)
    destination = tmp_path / "destination" / "serial.cmp"
    staging = tmp_path / "staging"
    destination.parent.mkdir()
    staging.mkdir()

    compress_eigerdata(
        frames,
        mask.copy(),
        {"pixel_mask": mask.copy()},
        str(destination),
        force_compress=True,
        dtypes="images",
        direct_load_data=False,
        with_pickle=False,
        new_path=str(staging),
    )

    assert destination.is_file()
    assert list(staging.iterdir()) == []


@pytest.mark.portable
def test_parallel_compression_stages_and_publishes_identical_cmp(tmp_path):
    from pyCHX.chx_compress import init_compress_eigerdata, para_compress_eigerdata

    frames = np.arange(1, 31, dtype=np.int32).reshape(5, 2, 3)
    mask = np.ones(frames.shape[1:], dtype=bool)
    metadata = {
        "beam_center_x": 0,
        "beam_center_y": 0,
        "count_time": 0,
        "detector_distance": 0,
        "frame_time": 0,
        "incident_wavelength": 0,
        "pixel_mask": mask.copy(),
        "x_pixel_size": 75,
        "y_pixel_size": 75,
    }
    serial = tmp_path / "serial.cmp"
    parallel = tmp_path / "destination" / "parallel.cmp"
    staging = tmp_path / "staging"
    parallel.parent.mkdir()
    staging.mkdir()

    init_compress_eigerdata(
        frames,
        mask.copy(),
        metadata.copy(),
        str(serial),
        with_pickle=False,
    )
    para_compress_eigerdata(
        frames,
        mask.copy(),
        metadata.copy(),
        str(parallel),
        num_sub=2,
        dtypes="images",
        cpu_core_number=2,
        with_pickle=False,
        copy_rawdata=False,
        new_path=str(staging),
    )

    assert parallel.read_bytes() == serial.read_bytes()
    assert list(staging.iterdir()) == []


@pytest.mark.portable
def test_atomic_publish_preserves_existing_destination_on_copy_failure(monkeypatch, tmp_path):
    from pyCHX import chx_compress

    source = tmp_path / "source.cmp"
    destination = tmp_path / "destination.cmp"
    source.write_bytes(b"new complete data")
    destination.write_bytes(b"existing data")

    def fail_during_copy(_source, temporary):
        with open(temporary, "wb") as stream:
            stream.write(b"partial")
        raise OSError("simulated copy failure")

    monkeypatch.setattr(chx_compress.shutil, "copyfile", fail_during_copy)
    with pytest.raises(OSError, match="simulated copy failure"):
        chx_compress._publish_file(source, destination)

    assert destination.read_bytes() == b"existing data"
    assert list(tmp_path.glob(".destination.cmp.*.tmp")) == []


@pytest.mark.portable
def test_compression_keeps_a_partial_final_frame_bin(tmp_path):
    from pyCHX.chx_compress import (
        Multifile,
        compress_eigerdata,
        get_avg_imgc,
        init_compress_eigerdata,
        segment_compress_eigerdata,
    )

    frames = np.array([1, 3, 5, 7, 9], dtype=np.int32).reshape(-1, 1, 1)
    mask = np.ones((1, 1), dtype=bool)
    metadata = {"pixel_mask": mask.copy()}
    expected = np.array([2.0, 6.0, 9.0])

    serial_filename = tmp_path / "serial-binned.cmp"
    _, serial_average, serial_intensity, serial_bad = init_compress_eigerdata(
        frames,
        mask.copy(),
        metadata,
        str(serial_filename),
        nobytes=4,
        bins=2,
        with_pickle=False,
    )

    np.testing.assert_allclose(serial_intensity, expected)
    np.testing.assert_allclose(serial_average, expected.mean())
    np.testing.assert_array_equal(serial_bad, [])
    with Multifile(str(serial_filename), beg=0, end=len(expected)) as compressed:
        np.testing.assert_allclose([compressed.rdframe(index).item() for index in range(len(expected))], expected)
        np.testing.assert_allclose(get_avg_imgc(compressed, sampling=1, show_progress=False), expected.mean())
        np.testing.assert_allclose(get_avg_imgc(compressed, sampling=2, show_progress=False), expected[::2].mean())
        np.testing.assert_allclose(
            get_avg_imgc(compressed, sampling=1, bad_frame_list=[0], show_progress=False), expected[1:].mean()
        )
        np.testing.assert_allclose(
            get_avg_imgc(compressed, sampling=1, bad_frame_list=0, show_progress=False), expected[1:].mean()
        )
        with pytest.raises(IndexError, match="record out of range"):
            compressed.rdframe(len(expected))

    _, reloaded_average, reloaded_intensity, _ = compress_eigerdata(
        frames,
        mask.copy(),
        metadata,
        str(serial_filename),
        bins=2,
        with_pickle=False,
        dtypes="images",
        direct_load_data=False,
    )
    np.testing.assert_allclose(reloaded_average, expected.mean())
    np.testing.assert_allclose(reloaded_intensity, expected)

    segment_filename = tmp_path / "segment-binned.cmp"
    _, segment_average, segment_intensity, segment_bad = segment_compress_eigerdata(
        frames,
        mask.copy(),
        metadata,
        str(segment_filename),
        nobytes=4,
        bins=2,
        N1=1,
        N2=4,
    )
    np.testing.assert_allclose(segment_intensity, [4, 7])
    np.testing.assert_allclose(segment_average, 5.5)
    np.testing.assert_array_equal(segment_bad, [False, False])


@pytest.mark.portable
def test_parallel_compression_weights_segment_averages_by_valid_frames(monkeypatch, tmp_path):
    from pyCHX import chx_compress

    class Result:
        def __init__(self, value):
            self.value = value

        def get(self):
            return self.value

    mask = np.ones((1, 1), dtype=bool)
    segment_results = {
        0: Result((mask.copy(), np.array([[2.0]]), np.array([1.0, 2.0, 3.0]), np.array([False, True, False]))),
        1: Result((mask.copy(), np.array([[8.0]]), np.array([4.0, 5.0]), np.array([False, False]))),
    }
    monkeypatch.setattr(chx_compress, "para_segment_compress_eigerdata", lambda **kwargs: segment_results)
    monkeypatch.setattr(chx_compress, "create_compress_header", lambda *args, **kwargs: None)
    monkeypatch.setattr(chx_compress, "combine_compressed", lambda *args, **kwargs: None)
    monkeypatch.setattr(chx_compress, "_publish_file", lambda *args, **kwargs: None)

    _, average, intensity, bad_frames = chx_compress.para_compress_eigerdata(
        np.zeros((5, 1, 1)),
        mask,
        {"pixel_mask": mask.copy()},
        str(tmp_path / "parallel.cmp"),
        num_sub=3,
        dtypes="images",
        cpu_core_number=8,
        with_pickle=False,
        copy_rawdata=False,
    )

    np.testing.assert_allclose(average, [[5.0]])
    np.testing.assert_allclose(intensity, [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(bad_frames, [1])


@pytest.mark.portable
def test_compression_handles_an_all_bad_segment_without_dividing_by_zero(tmp_path):
    from pyCHX.chx_compress import init_compress_eigerdata, segment_compress_eigerdata

    frames = np.zeros((3, 2, 2), dtype=np.int32)
    mask = np.ones((2, 2), dtype=bool)
    metadata = {"pixel_mask": mask.copy()}

    _, serial_average, _, serial_bad = init_compress_eigerdata(
        frames,
        mask.copy(),
        metadata,
        str(tmp_path / "serial-all-bad.cmp"),
        with_pickle=False,
    )
    _, segment_average, _, segment_bad = segment_compress_eigerdata(
        frames,
        mask.copy(),
        metadata,
        str(tmp_path / "segment-all-bad.cmp"),
    )

    assert np.isnan(serial_average).all()
    assert np.isnan(segment_average).all()
    np.testing.assert_array_equal(serial_bad, [0, 1, 2])
    np.testing.assert_array_equal(segment_bad, [True, True, True])


@pytest.mark.portable
@pytest.mark.parametrize("bad_frame_list", [[], [3, 7, 18]])
def test_serial_and_parallel_g2_agree_for_compressed_data(tmp_path, bad_frame_list):
    from pyCHX.chx_compress import Multifile
    from pyCHX.chx_correlationc import cal_g2c
    from pyCHX.chx_correlationp import cal_g2p

    filename, frames, ring_mask = _make_compressed_correlation_input(tmp_path)

    serial_file = Multifile(str(filename), beg=0, end=len(frames))
    parallel_file = Multifile(str(filename), beg=0, end=len(frames))
    try:
        serial_g2, serial_lags = cal_g2c(serial_file, ring_mask, bad_frame_list=bad_frame_list)
        parallel_g2, parallel_lags = cal_g2p(parallel_file, ring_mask, bad_frame_list=bad_frame_list)
    finally:
        serial_file.FID.close()
        parallel_file.FID.close()

    np.testing.assert_array_equal(parallel_lags, serial_lags)
    np.testing.assert_allclose(parallel_g2, serial_g2, rtol=1e-13, atol=0)


@pytest.mark.portable
def test_parallel_g2_keeps_first_roi_without_background_pixels(tmp_path):
    from pyCHX.chx_compress import Multifile
    from pyCHX.chx_correlationc import cal_g2c
    from pyCHX.chx_correlationp import cal_g2p

    filename, frames, ring_mask = _make_compressed_correlation_input(tmp_path)
    ring_mask = ring_mask.copy()
    ring_mask[ring_mask == 0] = 1
    serial_file = Multifile(str(filename), beg=0, end=len(frames))
    parallel_file = Multifile(str(filename), beg=0, end=len(frames))
    try:
        serial_g2, serial_lags = cal_g2c(serial_file, ring_mask, bad_frame_list=[])
        parallel_g2, parallel_lags = cal_g2p(parallel_file, ring_mask, bad_frame_list=[])
    finally:
        serial_file.FID.close()
        parallel_file.FID.close()

    assert parallel_g2.shape[1] == 2
    np.testing.assert_array_equal(parallel_lags, serial_lags)
    np.testing.assert_allclose(parallel_g2, serial_g2, rtol=1e-13, atol=0)


@pytest.mark.portable
def test_serial_and_parallel_g2_error_estimates_agree(tmp_path):
    from pyCHX.chx_compress import Multifile
    from pyCHX.chx_correlationc import cal_g2c
    from pyCHX.chx_correlationp import cal_g2p, cal_GPF, get_g2_from_ROI_GPF

    filename, frames, ring_mask = _make_compressed_correlation_input(tmp_path)
    ring_mask = np.select([ring_mask == 1, ring_mask == 2], [2, 5], default=0)
    norm = np.linspace(1.0, 2.0, np.count_nonzero(ring_mask))
    serial_file = Multifile(str(filename), beg=0, end=len(frames))
    parallel_file = Multifile(str(filename), beg=0, end=len(frames))
    gpf_file = Multifile(str(filename), beg=0, end=len(frames))
    try:
        serial_g2, serial_lags, serial_error, _ = cal_g2c(
            serial_file, ring_mask, bad_frame_list=[], norm=norm, cal_error=True
        )
        parallel_g2, parallel_lags, parallel_error = cal_g2p(
            parallel_file, ring_mask, bad_frame_list=[], norm=norm, cal_error=True
        )
        numerator, past, future = cal_GPF(gpf_file, ring_mask, bad_frame_list=[], norm=norm)
    finally:
        serial_file.FID.close()
        parallel_file.FID.close()
        gpf_file.FID.close()

    np.testing.assert_array_equal(parallel_lags, serial_lags)
    np.testing.assert_allclose(parallel_g2, serial_g2, rtol=1e-13, atol=0)
    np.testing.assert_allclose(parallel_error, serial_error, rtol=1e-13, atol=0)
    reconstructed_g2, _ = get_g2_from_ROI_GPF(numerator, past, future, ring_mask)
    np.testing.assert_allclose(reconstructed_g2[: len(serial_g2)], serial_g2, rtol=1e-13, atol=0)


@pytest.mark.portable
@pytest.mark.parametrize(
    ("norm_kind", "cal_error"),
    [(None, False), ("1d", False), ("2d", True)],
)
def test_parallel_g2_worker_grouping_is_exact(tmp_path, monkeypatch, norm_kind, cal_error):
    import skbeam.core.roi as roi

    from pyCHX import chx_correlationp
    from pyCHX.chx_compress import Multifile

    filename, frames, ring_mask = _make_compressed_correlation_input(tmp_path)
    ring_mask = np.select([ring_mask == 1, ring_mask == 2], [2, 5], default=0)
    _, pixel_list = roi.extract_label_indices(ring_mask)
    base_norm = np.linspace(1.0, 2.0, len(pixel_list))
    if norm_kind == "1d":
        norm = base_norm
    elif norm_kind == "2d":
        norm = np.multiply.outer(np.linspace(1.0, 1.5, len(frames)), base_norm)
    else:
        norm = None

    def calculate(worker_count):
        monkeypatch.setattr(chx_correlationp, "_available_cpu_count", lambda: worker_count)
        with Multifile(str(filename), beg=0, end=len(frames)) as compressed:
            return chx_correlationp.cal_g2p(
                compressed,
                ring_mask,
                bad_frame_list=[3, 7],
                imgsum=np.linspace(10.0, 20.0, len(frames)),
                norm=norm,
                cal_error=cal_error,
            )

    grouped = calculate(1)
    one_roi_per_worker = calculate(8)
    for grouped_value, ungrouped_value in zip(grouped, one_roi_per_worker):
        np.testing.assert_array_equal(grouped_value, ungrouped_value)


@pytest.mark.portable
def test_serial_and_parallel_two_time_agree_for_compressed_data(tmp_path):
    from pyCHX.chx_compress import Multifile
    from pyCHX.chx_correlationc import cal_c12c
    from pyCHX.chx_correlationp import cal_c12p

    filename, frames, ring_mask = _make_compressed_correlation_input(tmp_path)
    serial_file = Multifile(str(filename), beg=0, end=len(frames))
    parallel_file = Multifile(str(filename), beg=0, end=len(frames))
    try:
        serial_c12, serial_lags = cal_c12c(serial_file, ring_mask, bad_frame_list=[])
        parallel_c12, parallel_lags = cal_c12p(parallel_file, ring_mask, bad_frame_list=[])
    finally:
        serial_file.FID.close()
        parallel_file.FID.close()

    np.testing.assert_array_equal(parallel_lags, serial_lags)
    np.testing.assert_allclose(parallel_c12, serial_c12, rtol=1e-13, atol=0)


@pytest.mark.portable
@pytest.mark.parametrize("has_background", [True, False])
def test_serial_and_parallel_xsvs_agree_for_compressed_data(tmp_path, has_background):
    from pyCHX.chx_compress import Multifile
    from pyCHX.chx_specklecp import xsvsc, xsvsp

    filename, frames, ring_mask = _make_compressed_correlation_input(tmp_path)
    if not has_background:
        ring_mask = ring_mask.copy()
        ring_mask[ring_mask == 0] = 1
    serial_file = Multifile(str(filename), beg=0, end=len(frames))
    parallel_file = Multifile(str(filename), beg=0, end=len(frames))
    try:
        serial = xsvsc(serial_file, ring_mask, only_two_levels=True, max_cts=20)
        parallel = xsvsp(parallel_file, ring_mask, only_two_levels=True, max_cts=20)
    finally:
        serial_file.FID.close()
        parallel_file.FID.close()

    assert parallel[1].shape == (2, 2)
    for serial_edges, parallel_edges in zip(serial[0], parallel[0]):
        np.testing.assert_array_equal(parallel_edges, serial_edges)
    for serial_values, parallel_values in zip(serial[1:3], parallel[1:3]):
        for index in np.ndindex(serial_values.shape):
            np.testing.assert_allclose(parallel_values[index], serial_values[index], rtol=1e-13, atol=0)
    np.testing.assert_allclose(parallel[3], serial[3], rtol=1e-13, atol=0)


@pytest.mark.portable
def test_mean_intensity_supports_sparse_labels_and_partial_sampling(tmp_path):
    from pyCHX.chx_compress import Multifile, mean_intensityc

    filename, frames, _ = _make_compressed_correlation_input(tmp_path)
    roi_mask = np.array(
        [
            [2, 2, 5, 5],
            [2, 2, 5, 5],
            [2, 2, 5, 5],
            [2, 2, 5, 5],
        ]
    )
    sampled_frames = frames[::5]
    expected = np.column_stack([sampled_frames[:, roi_mask == label].mean(axis=1) for label in (2, 5)])

    serial_file = Multifile(str(filename), beg=0, end=len(frames))
    parallel_file = Multifile(str(filename), beg=0, end=len(frames))
    try:
        serial, serial_labels = mean_intensityc(serial_file, roi_mask, sampling=5, multi_cor=False)
        parallel, parallel_labels = mean_intensityc(parallel_file, roi_mask, sampling=5, multi_cor=True)
    finally:
        serial_file.FID.close()
        parallel_file.FID.close()

    np.testing.assert_array_equal(serial_labels, [2, 5])
    np.testing.assert_array_equal(parallel_labels, serial_labels)
    np.testing.assert_allclose(serial, expected)
    np.testing.assert_allclose(parallel, expected)

    subset_file = Multifile(str(filename), beg=0, end=len(frames))
    try:
        subset, subset_labels = mean_intensityc(subset_file, roi_mask, sampling=5, index=5, multi_cor=True)
    finally:
        subset_file.FID.close()

    np.testing.assert_array_equal(subset_labels, [5])
    np.testing.assert_allclose(subset[:, 0], expected[:, 1])


@pytest.mark.portable
def test_frame_intensity_sampling_reports_source_frame_indices(tmp_path):
    from pyCHX.chx_compress import Multifile, get_each_frame_intensityc

    filename, frames, _ = _make_compressed_correlation_input(tmp_path)
    sample_indices = np.arange(0, len(frames), 5)
    expected_intensity = frames[sample_indices].sum(axis=(1, 2))
    threshold = np.median(expected_intensity)

    compressed = Multifile(str(filename), beg=0, end=len(frames))
    try:
        intensity, bad_frames = get_each_frame_intensityc(
            compressed,
            sampling=5,
            bad_pixel_threshold=threshold,
            bad_pixel_low_threshold=-1,
        )
    finally:
        compressed.FID.close()

    np.testing.assert_array_equal(intensity, expected_intensity)
    np.testing.assert_array_equal(bad_frames, sample_indices[expected_intensity > threshold])


@pytest.mark.portable
def test_collect_pool_results_always_reaps_workers():
    from pyCHX.chx_compress import _collect_pool_results

    successful_pool = _FakePool()
    results = {2: _FakeAsyncResult("second"), 1: _FakeAsyncResult("first")}
    assert _collect_pool_results(successful_pool, results) == ["first", "second"]
    assert successful_pool.events == ["close", "join"]

    failed_pool = _FakePool()
    results = {1: _FakeAsyncResult(error=RuntimeError("worker failed"))}
    with pytest.raises(RuntimeError, match="worker failed"):
        _collect_pool_results(failed_pool, results)
    assert failed_pool.events == ["close", "terminate", "join"]


@pytest.mark.portable
def test_pool_size_is_limited_by_available_cpus(monkeypatch):
    from pyCHX import chx_compress

    created_with = []
    sentinel = object()
    monkeypatch.setattr(chx_compress, "cpu_count", lambda: 4)
    monkeypatch.setattr(chx_compress, "Pool", lambda processes: created_with.append(processes) or sentinel)

    assert chx_compress._make_pool(10) is sentinel
    assert created_with == [4]
    with pytest.raises(ValueError, match="at least one"):
        chx_compress._make_pool(0)


@pytest.mark.portable
def test_pool_size_respects_cpu_affinity(monkeypatch):
    from pyCHX import chx_compress

    created_with = []
    sentinel = object()
    monkeypatch.setattr(chx_compress, "cpu_count", lambda: 32)
    monkeypatch.setattr(chx_compress.os, "sched_getaffinity", lambda _pid: set(range(3)))
    monkeypatch.setattr(chx_compress, "Pool", lambda processes: created_with.append(processes) or sentinel)

    assert chx_compress._make_pool(10) is sentinel
    assert created_with == [3]
