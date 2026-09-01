"""Offline end-to-end characterization tests for the bundled ds1 Eiger data.

IMPORTANT: The numerical files under ``data/ds1/expected`` characterize the
prerefactor pyCHX implementation; they are not guaranteed ground truth. A
mismatch is important and must be reviewed, but does not by itself prove that
a newer implementation is wrong. It may expose an existing defect in the
reference implementation. Never update a reference merely to make this suite
pass without understanding and documenting the difference.

The prerefactor parallel compressor had a known segment-count bug for this
50-frame/100-frame-segment dataset. It divided the average image by two. Since
circular averaging is linear, the saved I(q) has the same factor-of-two bias.
The tests preserve those files unchanged and compare current results with the
explicitly corrected values.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

# Importing hdf5plugin registers the LZ4 filter used by the detector files.
pytest.importorskip("eiger_io", reason="ds1 requires the optional Eiger file reader")
pytest.importorskip("hdf5plugin", reason="ds1 requires the HDF5 LZ4 filter")
h5py = pytest.importorskip("h5py")


pytestmark = pytest.mark.data_regression

DATA_DIR = Path(__file__).parent / "data" / "ds1"
RAW_DIR = DATA_DIR / "raw"
MASK_DIR = DATA_DIR / "masks"
EXPECTED_DIR = DATA_DIR / "expected"
MASTER_FILE = RAW_DIR / "8d9263a6-add2-4a61-baec_83_master.h5"
MANIFEST = json.loads((DATA_DIR / "manifest.json").read_text())
UID = MANIFEST["uid"]
FRAME_COUNT = MANIFEST["frame_count"]
IMAGES_PER_FILE = MANIFEST["images_per_file"]
RAW_FRAME_SHAPE = tuple(MANIFEST["raw_frame_shape"])
ANALYSIS_FRAME_SHAPE = tuple(MANIFEST["analysis_frame_shape"])
ANALYSIS_CENTER = tuple(MANIFEST["analysis"]["center"])

LEGACY_REVIEW_NOTE = (
    "ds1 references characterize prerefactor behavior and are not guaranteed ground truth; "
    "investigate this difference before changing either the implementation or reference"
)


def _master_metadata():
    """Return the minimal metadata needed to reproduce the CMP header."""
    with h5py.File(MASTER_FILE, "r") as master:
        pixel_mask = master["entry/instrument/detector/detectorSpecific/pixel_mask"][:]

    metadata = dict(MANIFEST["compression_header"])
    metadata.update({"pixel_mask": pixel_mask, "uid": UID})
    return metadata


def _analysis_parameters():
    analysis = MANIFEST["analysis"]
    return {
        "uid": f"uid={UID}",
        "dpix": analysis["dpix"],
        "Ldet": analysis["detector_distance"],
        "lambda_": analysis["wavelength"],
        "exposuretime": analysis["exposure_time"],
        "timeperframe": analysis["time_per_frame"],
        "center": list(ANALYSIS_CENTER),
        "path": "",
    }


def _orient_eiger500k_frame(frame):
    """Apply the historical Eiger 500k ``reverse=True, rot90=True`` convention."""
    return np.rot90(np.asarray(frame)[::-1, :])


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.fixture(scope="module")
def detector_mask():
    return np.load(MASK_DIR / "detector.npy", allow_pickle=False)


@pytest.fixture(scope="module")
def polygon_mask():
    return np.load(MASK_DIR / "polygon.npy", allow_pickle=False)


@pytest.fixture(scope="module")
def roi_mask():
    return np.load(MASK_DIR / "roi.npy", allow_pickle=False)


@pytest.fixture(scope="module")
def compressed_ds1(tmp_path_factory, detector_mask):
    from pyCHX.chx_compress import compress_eigerdata

    filename = tmp_path_factory.mktemp("ds1-compressed") / "generated.cmp"
    mask, average, intensity, bad_frames = compress_eigerdata(
        np.ones(FRAME_COUNT, dtype=np.uint8),
        detector_mask.copy(),
        _master_metadata(),
        str(filename),
        force_compress=True,
        bad_pixel_threshold=MANIFEST["compression"]["bad_pixel_threshold"],
        nobytes=MANIFEST["compression"]["bytes_per_pixel"],
        bins=MANIFEST["compression"]["bins"],
        para_compress=MANIFEST["compression"]["parallel"],
        num_sub=MANIFEST["compression"]["num_sub"],
        num_max_para_process=MANIFEST["compression"]["num_max_parallel_processes"],
        hot_pixel_threshold=MANIFEST["compression"]["hot_pixel_threshold"],
        with_pickle=False,
        direct_load_data=True,
        data_path=str(MASTER_FILE),
        copy_rawdata=False,
        reverse=MANIFEST["transform"]["reverse"],
        rot90=MANIFEST["transform"]["rot90"],
    )
    return {
        "average": average,
        "bad_frames": bad_frames,
        "filename": filename,
        "intensity": intensity,
        "mask": mask,
    }


@pytest.fixture(scope="module")
def circular_average_result(compressed_ds1, detector_mask, polygon_mask):
    from pyCHX.XPCS_SAXS import get_circular_average

    analysis_mask = detector_mask & polygon_mask
    qp, intensity, q = get_circular_average(
        compressed_ds1["average"] * detector_mask,
        analysis_mask,
        pargs=_analysis_parameters(),
        save=False,
    )
    return {"iq": intensity, "q": q, "qp": qp}


def test_ds1_hdf5_fixture_is_self_contained():
    with h5py.File(MASTER_FILE, "r") as master:
        data_group = master["entry/data"]
        assert sorted(data_group) == [f"data_{index:06d}" for index in range(1, 6)]

        for index in range(1, 6):
            name = f"data_{index:06d}"
            link = data_group.get(name, getlink=True)
            assert isinstance(link, h5py.ExternalLink)
            assert link.filename == f"8d9263a6-add2-4a61-baec_83_data_{index:06d}.h5"
            assert link.path == "/entry/data/data"
            assert data_group[name].shape == (IMAGES_PER_FILE, *RAW_FRAME_SHAPE)
            assert data_group[name].dtype == np.dtype("uint32")


def test_ds1_masks_have_expected_geometry(detector_mask, polygon_mask, roi_mask):
    assert detector_mask.shape == ANALYSIS_FRAME_SHAPE
    assert detector_mask.dtype == np.dtype("bool")
    assert polygon_mask.shape == ANALYSIS_FRAME_SHAPE
    assert polygon_mask.dtype == np.dtype("bool")
    assert roi_mask.shape == ANALYSIS_FRAME_SHAPE
    np.testing.assert_array_equal(np.unique(roi_mask), np.arange(13))
    assert not np.any((roi_mask > 0) & ~(detector_mask & polygon_mask))


def test_ds1_parallel_compression_matches_frame_references(compressed_ds1, detector_mask):
    expected_intensity = np.load(EXPECTED_DIR / "intensity_vs_frame.npy", allow_pickle=False)
    legacy_average = np.load(EXPECTED_DIR / "legacy_prerefactor_average.npy", allow_pickle=False)

    np.testing.assert_array_equal(compressed_ds1["mask"], detector_mask)
    np.testing.assert_array_equal(compressed_ds1["bad_frames"], [])
    np.testing.assert_array_equal(compressed_ds1["intensity"], expected_intensity)
    np.testing.assert_array_equal(
        compressed_ds1["average"],
        MANIFEST["legacy_reference_corrections"]["average_image_scale"] * legacy_average,
        err_msg=LEGACY_REVIEW_NOTE,
    )

    assert _sha256(compressed_ds1["filename"]) == MANIFEST["compressed_sha256"], LEGACY_REVIEW_NOTE


def test_ds1_generated_cmp_round_trips_all_oriented_frames(compressed_ds1, detector_mask):
    from eiger_io.fs_handler import EigerImages

    from pyCHX.chx_compress import Multifile

    raw_images = EigerImages(str(MASTER_FILE), IMAGES_PER_FILE, md=_master_metadata())
    try:
        assert len(raw_images) == FRAME_COUNT
        with Multifile(str(compressed_ds1["filename"]), 0, FRAME_COUNT) as compressed:
            assert compressed.md["bytes"] == 4
            assert (compressed.md["ncols"], compressed.md["nrows"]) == ANALYSIS_FRAME_SHAPE
            for index in range(FRAME_COUNT):
                oriented = _orient_eiger500k_frame(raw_images[index]).astype(np.int32)
                expected = np.where((oriented > 0) & detector_mask, oriented, 0)
                np.testing.assert_array_equal(compressed.rdframe(index), expected)
    finally:
        raw_images.close()


def test_ds1_circular_average_matches_corrected_legacy_reference(circular_average_result):
    with np.load(EXPECTED_DIR / "legacy_prerefactor_iq.npz", allow_pickle=False) as reference:
        np.testing.assert_array_equal(
            circular_average_result["qp"],
            reference["qp_saxs"],
            err_msg=LEGACY_REVIEW_NOTE,
        )
        np.testing.assert_allclose(
            circular_average_result["iq"],
            MANIFEST["legacy_reference_corrections"]["iq_scale"] * reference["iq_saxs"],
            # Weighted histogram reductions vary slightly across CPU implementations.
            rtol=2e-11,
            atol=0,
            err_msg=LEGACY_REVIEW_NOTE,
        )


def test_ds1_g2_matches_all_legacy_rois(compressed_ds1, circular_average_result, roi_mask):
    import skbeam.core.roi as roi

    from pyCHX.chx_compress import Multifile
    from pyCHX.chx_correlationc import get_pixelist_interp_iq
    from pyCHX.chx_correlationp import cal_g2p

    _, pixel_list = roi.extract_label_indices(roi_mask)
    norm = get_pixelist_interp_iq(
        circular_average_result["qp"],
        circular_average_result["iq"],
        roi_mask,
        ANALYSIS_CENTER,
    )
    expected_intensity = np.load(EXPECTED_DIR / "intensity_vs_frame.npy", allow_pickle=False)

    with Multifile(str(compressed_ds1["filename"]), 0, FRAME_COUNT) as compressed:
        actual_g2, actual_lags = cal_g2p(
            compressed,
            roi_mask,
            bad_frame_list=[],
            good_start=0,
            num_buf=MANIFEST["analysis"]["g2_buffers"],
            num_lev=None,
            imgsum=expected_intensity,
            norm=norm,
            cal_error=False,
        )

    with np.load(EXPECTED_DIR / "g2.npz", allow_pickle=False) as reference:
        assert actual_g2.shape[1] == MANIFEST["analysis"]["roi_count"]
        assert pixel_list.size == np.count_nonzero(roi_mask)
        np.testing.assert_array_equal(actual_lags, reference["lag_steps"], err_msg=LEGACY_REVIEW_NOTE)
        np.testing.assert_allclose(
            actual_g2,
            reference["g2"],
            rtol=1e-13,
            atol=1e-14,
            err_msg=LEGACY_REVIEW_NOTE,
        )


def test_ds1_ttcf_matches_all_legacy_rois(compressed_ds1, circular_average_result, roi_mask):
    import skbeam.core.roi as roi

    from pyCHX.chx_compress import Multifile
    from pyCHX.chx_correlationc import Get_Pixel_Arrayc, auto_two_Arrayc, get_pixelist_interp_iq

    _, pixel_list = roi.extract_label_indices(roi_mask)
    norm = get_pixelist_interp_iq(
        circular_average_result["qp"],
        circular_average_result["iq"],
        roi_mask,
        ANALYSIS_CENTER,
    )

    with Multifile(str(compressed_ds1["filename"]), 0, FRAME_COUNT) as compressed:
        data = Get_Pixel_Arrayc(compressed, pixel_list, norm=norm).get_data()
    actual = auto_two_Arrayc(data, roi_mask, index=None)
    expected = np.load(EXPECTED_DIR / "ttcf.npy", allow_pickle=False)

    assert actual.shape == (FRAME_COUNT, FRAME_COUNT, MANIFEST["analysis"]["roi_count"])
    np.testing.assert_allclose(
        actual,
        expected,
        # The dot-product reduction is sensitive to the host BLAS/CPU implementation.
        rtol=5e-13,
        atol=1e-14,
        err_msg=LEGACY_REVIEW_NOTE,
    )
