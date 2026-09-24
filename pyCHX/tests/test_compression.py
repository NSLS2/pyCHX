import pickle
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
def test_parallel_compression_keeps_partial_final_bin_byte_identical(tmp_path):
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
        "x_pixel_size": 0,
        "y_pixel_size": 0,
    }
    serial = tmp_path / "serial-binned.cmp"
    parallel = tmp_path / "parallel-binned.cmp"

    expected = init_compress_eigerdata(
        frames,
        mask.copy(),
        metadata.copy(),
        str(serial),
        bins=2,
        with_pickle=False,
    )
    actual = para_compress_eigerdata(
        frames,
        mask.copy(),
        metadata.copy(),
        str(parallel),
        num_sub=2,
        bins=2,
        dtypes="images",
        cpu_core_number=2,
        with_pickle=False,
        copy_rawdata=False,
        new_path=str(tmp_path),
    )

    assert parallel.read_bytes() == serial.read_bytes()
    for actual_value, expected_value in zip(actual, expected):
        np.testing.assert_allclose(actual_value, expected_value)


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
def test_unbinned_eiger_blocks_preserve_integer_frames_without_cast_warnings(tmp_path):
    from pyCHX.chx_compress import _compress_segment

    invalid = np.iinfo(np.uint32).max
    frames = np.array([[[1, invalid, 2]], [[3, invalid, 4]]], dtype=np.uint32)

    class DirectEigerFrames:
        images_per_file = len(frames)
        valid_keys = ["data_000001"]
        _entry = {"data_000001": frames}

    detector_mask = np.array([[True, False, True]])
    with np.errstate(all="raise"):
        final_mask, average, intensity, bad_frames = _compress_segment(
            DirectEigerFrames(),
            detector_mask.copy(),
            str(tmp_path / "segment.cmp"),
            bad_pixel_threshold=1e15,
            hot_pixel_threshold=2**30,
            bad_pixel_low_threshold=0,
            nobytes=4,
            bins=1,
            start=0,
            stop=len(frames),
        )

    np.testing.assert_array_equal(final_mask, detector_mask)
    np.testing.assert_array_equal(average, [[2, 0, 3]])
    np.testing.assert_array_equal(intensity, [3, 7])
    np.testing.assert_array_equal(bad_frames, [False, False])


@pytest.mark.portable
def test_parallel_compression_weights_segment_averages_by_valid_frames(monkeypatch, tmp_path):
    from pyCHX import chx_compress

    mask = np.ones((1, 1), dtype=bool)
    segment_results = [
        (0, (mask.copy(), np.array([[2.0]]), np.array([1.0, 2.0, 3.0]), np.array([False, True, False]))),
        (1, (mask.copy(), np.array([[8.0]]), np.array([4.0, 5.0]), np.array([False, False]))),
    ]
    monkeypatch.setattr(chx_compress, "_iter_parallel_segment_results", lambda **kwargs: segment_results)
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
def test_parallel_hot_pixel_masking_preserves_segment_boundaries(tmp_path):
    from pyCHX.chx_compress import Multifile, para_compress_eigerdata

    frames = np.array([[[200, 1]], [[2, 2]], [[5, 3]], [[6, 4]]], dtype=np.int32)
    mask = np.ones((1, 2), dtype=bool)
    metadata = {
        "beam_center_x": 0,
        "beam_center_y": 0,
        "count_time": 0,
        "detector_distance": 0,
        "frame_time": 0,
        "incident_wavelength": 0,
        "pixel_mask": mask.copy(),
        "x_pixel_size": 0,
        "y_pixel_size": 0,
    }
    filename = tmp_path / "hot-pixel.cmp"
    final_mask, _, _, _ = para_compress_eigerdata(
        frames,
        mask.copy(),
        metadata,
        str(filename),
        num_sub=2,
        hot_pixel_threshold=100,
        dtypes="images",
        cpu_core_number=2,
        with_pickle=False,
        copy_rawdata=False,
        new_path=str(tmp_path),
    )

    assert not final_mask[0, 0]
    with Multifile(str(filename), 0, len(frames)) as compressed:
        assert compressed.rdframe(1)[0, 0] == 0
        assert compressed.rdframe(2)[0, 0] == 5


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

    reordered_file = Multifile(str(filename), beg=0, end=len(frames))
    try:
        reordered, reordered_labels = mean_intensityc(reordered_file, roi_mask, sampling=5, index=[5, 2])
    finally:
        reordered_file.close()

    np.testing.assert_array_equal(reordered_labels, [5, 2])
    np.testing.assert_allclose(reordered, expected[:, ::-1])

    duplicate_file = Multifile(str(filename), beg=0, end=len(frames))
    try:
        with np.errstate(invalid="ignore"):
            duplicated, duplicate_labels = mean_intensityc(duplicate_file, roi_mask, sampling=5, index=[2, 2])
    finally:
        duplicate_file.close()

    np.testing.assert_array_equal(duplicate_labels, [2, 2])
    assert np.isnan(duplicated[:, 0]).all()
    np.testing.assert_allclose(duplicated[:, 1], expected[:, 0])


@pytest.mark.portable
def test_mean_intensity_uses_buffered_sequential_reads(tmp_path, monkeypatch):
    from pyCHX.chx_compress import Multifile, mean_intensityc

    filename, frames, ring_mask = _make_compressed_correlation_input(tmp_path)
    expected = np.column_stack([frames[:, ring_mask == label].mean(axis=1) for label in (1, 2)])

    with Multifile(str(filename), beg=0, end=len(frames)) as compressed:
        calls = []
        compressed._buffered_read_stats = []
        original = compressed._iter_raw_frames_buffered

        def counted(start, end, block_size=8 * 1024**2):
            for frame_index, frame in original(start, end, block_size):
                calls.append(frame_index)
                yield frame_index, frame

        monkeypatch.setattr(
            compressed,
            "_iter_raw_frames_buffered",
            counted,
        )
        monkeypatch.setattr(compressed, "rdrawframe", lambda _frame_index: pytest.fail("used per-frame reads"))
        actual, labels = mean_intensityc(compressed, ring_mask)
        traversed = compressed._bytes_traversed
        read_stats = compressed._buffered_read_stats

    assert calls == list(range(len(frames)))
    assert traversed == filename.stat().st_size - 1024
    assert sum(read_size for _, read_size, _ in read_stats) == traversed
    assert all(elapsed >= 0 for _, _, elapsed in read_stats)
    np.testing.assert_array_equal(labels, [1, 2])
    np.testing.assert_allclose(actual, expected)


@pytest.mark.portable
def test_mean_intensity_bounded_prefetch_matches_serial_reader(tmp_path):
    from pyCHX.chx_compress import Multifile, mean_intensityc

    filename, _, ring_mask = _make_compressed_correlation_input(tmp_path)
    with (
        Multifile(str(filename), beg=0, end=20) as serial_file,
        Multifile(str(filename), beg=0, end=20) as prefetched_file,
    ):
        prefetched_file._roi_intensity_prefetch = True
        serial, serial_labels = mean_intensityc(serial_file, ring_mask)
        prefetched, prefetched_labels = mean_intensityc(prefetched_file, ring_mask)

    np.testing.assert_array_equal(prefetched_labels, serial_labels)
    np.testing.assert_array_equal(prefetched, serial)


@pytest.mark.portable
def test_bounded_prefetch_propagates_errors_and_closes_early():
    import threading

    from pyCHX.chx_compress import _iter_prefetched_frames

    def broken_frames():
        yield 1
        raise RuntimeError("read failed")

    prefetched = _iter_prefetched_frames(broken_frames())
    assert next(prefetched) == 1
    with pytest.raises(RuntimeError, match="read failed"):
        next(prefetched)

    prefetched = _iter_prefetched_frames(iter(range(100)))
    assert next(prefetched) == 0
    prefetched.close()
    assert not any(thread.name == "pychx-cmp-prefetch" for thread in threading.enumerate())


@pytest.mark.portable
def test_mean_intensity_honors_private_buffer_size(tmp_path, monkeypatch):
    from pyCHX.chx_compress import Multifile, mean_intensityc

    filename, frames, ring_mask = _make_compressed_correlation_input(tmp_path)
    with Multifile(str(filename), beg=0, end=len(frames)) as compressed:
        block_sizes = []
        compressed._roi_intensity_stats = {}
        original = compressed._iter_raw_frames_buffered

        def counted(start, end, block_size=8 * 1024**2):
            block_sizes.append(block_size)
            yield from original(start, end, block_size)

        monkeypatch.setattr(compressed, "_iter_raw_frames_buffered", counted)
        compressed._buffered_read_block_size = 64
        mean_intensityc(compressed, ring_mask)

    assert block_sizes == [64]
    assert set(compressed._roi_intensity_stats) == {
        "roi_setup_seconds",
        "read_and_reduce_seconds",
        "sparse_reduction_seconds",
        "reader_and_iteration_seconds",
        "division_seconds",
    }


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
def test_read_compressed_reconstructs_average_and_bad_frames_in_one_pass(tmp_path, monkeypatch):
    from pyCHX import chx_compress

    filename, frames, _ = _make_compressed_correlation_input(tmp_path)
    threshold = float(frames[7].sum() - 1)
    calls = []
    original = chx_compress.Multifile._raw_frame_view

    def counted(self, frame_index):
        calls.append(frame_index)
        return original(self, frame_index)

    monkeypatch.setattr(chx_compress.Multifile, "_raw_frame_view", counted)
    _, average, intensity, bad_frames = chx_compress.read_compressed_eigerdata(
        np.ones(frames.shape[1:], dtype=bool),
        str(filename),
        0,
        len(frames),
        bad_pixel_threshold=threshold,
        bad_pixel_low_threshold=-1,
        bad_frame_list=[2],
        with_pickle=False,
    )
    expected_bad = np.unique(np.concatenate(([2], np.flatnonzero(frames.sum(axis=(1, 2)) > threshold))))

    assert calls == list(range(len(frames)))
    np.testing.assert_array_equal(intensity, frames.sum(axis=(1, 2)))
    np.testing.assert_array_equal(bad_frames, expected_bad)
    np.testing.assert_allclose(average, np.delete(frames, expected_bad, axis=0).mean(axis=0))


@pytest.mark.portable
def test_waterfall_sparse_extraction_matches_dense_frames(tmp_path):
    from pyCHX.chx_compress import Multifile
    from pyCHX.chx_compress_analysis import cal_waterfallc

    filename, frames, ring_mask = _make_compressed_correlation_input(tmp_path)
    with Multifile(str(filename), beg=0, end=len(frames)) as compressed:
        actual = cal_waterfallc(compressed, ring_mask, qindex=2)

    np.testing.assert_array_equal(actual, frames[:, ring_mask == 2])


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
    monkeypatch.setattr(chx_compress, "_available_cpu_count", lambda: 4)
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
    monkeypatch.setattr(chx_compress, "physical_core_count", lambda cpu_ids: len(cpu_ids))
    monkeypatch.setattr(chx_compress, "Pool", lambda processes: created_with.append(processes) or sentinel)

    assert chx_compress._make_pool(10) is sentinel
    assert created_with == [3]


@pytest.mark.portable
def test_pool_size_prefers_physical_cores_within_affinity(monkeypatch):
    from pyCHX import chx_compress

    monkeypatch.setattr(chx_compress, "cpu_count", lambda: 256)
    monkeypatch.setattr(chx_compress.os, "sched_getaffinity", lambda _pid: set(range(56)))
    monkeypatch.setattr(chx_compress, "physical_core_count", lambda cpu_ids: 28)
    assert chx_compress._available_cpu_count() == 28


@pytest.mark.portable
@pytest.mark.parametrize(("nobytes", "dtype"), [(2, np.uint16), (4, np.uint32), (8, np.float64)])
def test_multifile_indexed_views_support_legacy_value_widths_and_random_access(tmp_path, nobytes, dtype):
    from pyCHX.chx_compress import Multifile, create_compress_header

    filename = tmp_path / f"legacy-{nobytes}.cmp"
    metadata = {"img_shape": (2, 3)}
    create_compress_header(metadata, str(filename), nobytes=nobytes)
    values = [np.asarray([1, 3], dtype=dtype), np.asarray([], dtype=dtype), np.asarray([7], dtype=dtype)]
    positions = [
        np.asarray([0, 5], dtype=np.int32),
        np.asarray([], dtype=np.int32),
        np.asarray([2], dtype=np.int32),
    ]
    with filename.open("ab") as stream:
        for frame_positions, frame_values in zip(positions, values):
            stream.write(np.asarray(len(frame_positions), dtype=np.uint32).tobytes())
            stream.write(frame_positions.tobytes())
            stream.write(frame_values.tobytes())

    with Multifile(str(filename), beg=1, end=3) as compressed:
        buffered_frames = list(compressed._iter_raw_frames_buffered(1, 3, block_size=7))
        assert [frame_index for frame_index, _ in buffered_frames] == [1, 2]
        for frame_index, (actual_positions, actual_values) in buffered_frames:
            np.testing.assert_array_equal(actual_positions, positions[frame_index])
            np.testing.assert_array_equal(actual_values, values[frame_index])
            assert not actual_positions.flags.writeable
            assert not actual_values.flags.writeable

        for frame_index in (2, 1, 2):
            actual_positions, actual_values = compressed._raw_frame_view(frame_index)
            np.testing.assert_array_equal(actual_positions, positions[frame_index])
            np.testing.assert_array_equal(actual_values, values[frame_index])
            assert not actual_positions.flags.writeable
            assert not actual_values.flags.writeable

        restored = pickle.loads(pickle.dumps(compressed))
        try:
            public_positions, public_values = restored.rdrawframe(2)
            assert public_positions.flags.writeable
            assert public_values.flags.writeable
            public_values[:] = 0
            np.testing.assert_array_equal(restored.rdrawframe(2)[1], values[2])
        finally:
            restored.close()

    compressed.reopen()
    try:
        np.testing.assert_array_equal(compressed.rdrawframe(1)[0], positions[1])
    finally:
        compressed.close()
    closed_copy = pickle.loads(pickle.dumps(compressed))
    assert closed_copy.FID.closed
    closed_copy.reopen()
    try:
        np.testing.assert_array_equal(closed_copy.rdrawframe(2)[1], values[2])
    finally:
        closed_copy.close()


@pytest.mark.portable
@pytest.mark.parametrize("corruption", ["header", "frame_header", "payload", "negative_length"])
def test_multifile_rejects_malformed_or_truncated_input(tmp_path, corruption):
    from pyCHX.chx_compress import Multifile, create_compress_header

    filename = tmp_path / "broken.cmp"
    if corruption == "header":
        filename.write_bytes(b"Version-COMP0001")
        with pytest.raises(ValueError, match="header"):
            Multifile(str(filename), 0, 1)
        return

    create_compress_header({"img_shape": (2, 2)}, str(filename), nobytes=4)
    if corruption == "frame_header":
        with pytest.raises(ValueError, match="first frame"):
            Multifile(str(filename), 0, 1)
        return
    with filename.open("ab") as stream:
        stream.write(struct.pack("@i", -1 if corruption == "negative_length" else 2))
        if corruption == "payload":
            stream.write(np.asarray([0], dtype=np.int32).tobytes())
    if corruption == "negative_length":
        with pytest.raises(ValueError, match="negative"):
            Multifile(str(filename), 0, 1)
    else:
        with Multifile(str(filename), 0, 1) as compressed:
            with pytest.raises(ValueError, match="truncated"):
                compressed._raw_frame_view(0)
            with pytest.raises(ValueError, match="truncated"):
                list(compressed._iter_raw_frames_buffered(0, 1, block_size=7))


@pytest.mark.portable
def test_cal_g2p_traverses_each_cmp_frame_once(tmp_path, monkeypatch):
    from pyCHX.chx_compress import Multifile
    from pyCHX.chx_correlationp import cal_g2p

    filename, frames, ring_mask = _make_compressed_correlation_input(tmp_path)
    with Multifile(str(filename), 0, len(frames)) as compressed:
        calls = []
        original = compressed._iter_raw_frames_buffered

        def counted(start, end, block_size=8 * 1024**2):
            for frame_index, frame in original(start, end, block_size):
                calls.append(frame_index)
                yield frame_index, frame

        monkeypatch.setattr(compressed, "_iter_raw_frames_buffered", counted)
        monkeypatch.setattr(compressed, "_raw_frame_view", lambda _frame_index: pytest.fail("used indexed reads"))
        cal_g2p(compressed, ring_mask, bad_frame_list=[])
        traversed = compressed._bytes_traversed

    assert calls == list(range(len(frames)))
    assert traversed == filename.stat().st_size - 1024


@pytest.mark.portable
def test_cal_g2p_buffered_reader_matches_indexed_reader(tmp_path):
    from pyCHX.chx_compress import Multifile
    from pyCHX.chx_correlationp import cal_g2p

    filename, frames, ring_mask = _make_compressed_correlation_input(tmp_path)
    with (
        Multifile(str(filename), 0, len(frames)) as buffered_file,
        Multifile(str(filename), 0, len(frames)) as indexed_file,
        Multifile(str(filename), 0, len(frames)) as previously_scanned_file,
    ):
        indexed_file._one_time_use_buffered_reader = False
        buffered_g2, buffered_lags = cal_g2p(buffered_file, ring_mask, bad_frame_list=[3])
        indexed_g2, indexed_lags = cal_g2p(indexed_file, ring_mask, bad_frame_list=[3])
        list(previously_scanned_file._iter_raw_frames_buffered(0, len(frames)))
        previously_scanned_file._one_time_stats = {}
        scanned_g2, scanned_lags = cal_g2p(previously_scanned_file, ring_mask, bad_frame_list=[3])

    np.testing.assert_array_equal(buffered_lags, indexed_lags)
    np.testing.assert_array_equal(buffered_g2, indexed_g2)
    assert buffered_file._last_buffered_scan == (0, len(frames))
    np.testing.assert_array_equal(scanned_lags, indexed_lags)
    np.testing.assert_array_equal(scanned_g2, indexed_g2)
    assert previously_scanned_file._one_time_stats["reader"] == "buffered"


@pytest.mark.portable
def test_cal_g2p_double_buffer_matches_single_buffer(tmp_path):
    from pyCHX.chx_compress import Multifile
    from pyCHX.chx_correlationp import cal_g2p

    filename, frames, ring_mask = _make_compressed_correlation_input(tmp_path)
    with (
        Multifile(str(filename), 0, len(frames)) as single_file,
        Multifile(str(filename), 0, len(frames)) as double_file,
    ):
        single_file._one_time_stats = {}
        double_file._one_time_stats = {}
        single_file._one_time_double_buffer = False
        double_file._one_time_double_buffer = True
        single_g2, single_lags = cal_g2p(single_file, ring_mask, bad_frame_list=[3])
        double_g2, double_lags = cal_g2p(double_file, ring_mask, bad_frame_list=[3])

    np.testing.assert_array_equal(double_lags, single_lags)
    np.testing.assert_array_equal(double_g2, single_g2)
    assert single_file._one_time_stats["double_buffer"] is False
    assert double_file._one_time_stats["double_buffer"] is True


@pytest.mark.portable
def test_cal_g2p_supports_hundreds_of_sparse_rois(tmp_path, monkeypatch):
    from pyCHX import chx_correlationp
    from pyCHX.chx_compress import Multifile, init_compress_eigerdata
    from pyCHX.chx_correlationc import cal_g2c

    roi_mask = np.arange(1, 201, dtype=np.int64).reshape(10, 20)
    frame_number = np.arange(16, dtype=np.int32)[:, None, None]
    frames = 1 + (frame_number + roi_mask[None, :, :]) % 23
    detector_mask = np.ones(roi_mask.shape, dtype=bool)
    filename = tmp_path / "many-rois.cmp"
    init_compress_eigerdata(
        frames,
        detector_mask.copy(),
        {"pixel_mask": detector_mask.copy()},
        str(filename),
        with_pickle=False,
    )
    monkeypatch.setattr(chx_correlationp, "_available_cpu_count", lambda: 8)
    monkeypatch.setattr(chx_correlationp, "available_memory_bytes", lambda: 1)
    with (
        Multifile(str(filename), 0, len(frames)) as serial_file,
        Multifile(str(filename), 0, len(frames)) as parallel_file,
    ):
        expected, expected_lags = cal_g2c(serial_file, roi_mask, bad_frame_list=[5])
        actual, actual_lags = chx_correlationp.cal_g2p(parallel_file, roi_mask, bad_frame_list=[5])

    np.testing.assert_array_equal(actual_lags, expected_lags)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=0)
