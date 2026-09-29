import numpy as np
import pandas as pd
import pytest


@pytest.mark.portable
def test_hdf5_dictionary_round_trip(tmp_path):
    from pyCHX.Create_Report import load_dict_from_hdf5, save_dict_to_hdf5

    filename = tmp_path / "results.h5"
    original = {
        "title": "synthetic XPCS result",
        "count": 3,
        "scale": 1.25,
        "curve": np.array([1.0, 0.75, 0.5]),
        "nested": {"lags": np.array([0, 1, 2], dtype=np.int64)},
    }

    save_dict_to_hdf5(original, filename)
    loaded = load_dict_from_hdf5(filename)

    assert loaded["title"] == original["title"]
    assert loaded["count"] == original["count"]
    assert loaded["scale"] == pytest.approx(original["scale"])
    np.testing.assert_array_equal(loaded["curve"], original["curve"])
    np.testing.assert_array_equal(loaded["nested"]["lags"], original["nested"]["lags"])


@pytest.mark.portable
def test_g2_csv_round_trip_preserves_lags_and_curves(tmp_path):
    from pyCHX.chx_generic_functions import save_g2_general

    taus = np.array([0.0, 0.1, 1.0])
    g2 = np.array([[1.2, 1.3], [1.1, 1.15], [1.01, 1.02]])

    saved = save_g2_general(
        g2,
        taus,
        qr=np.array([0.01, 0.02]),
        uid="synthetic_g2.csv",
        path=str(tmp_path),
        return_res=True,
    )

    assert (tmp_path / "synthetic_g2.csv").is_file()
    np.testing.assert_allclose(saved["tau"], taus)
    np.testing.assert_allclose(saved[["0.01", "0.02"]], g2)


@pytest.mark.portable
def test_eiger_images_per_file_reads_first_dataset(tmp_path):
    import h5py

    from pyCHX.chx_generic_functions import get_eigerImage_per_file

    master = tmp_path / "master.h5"
    with h5py.File(master, "w") as handle:
        data = handle.create_group("entry/data")
        data.create_dataset("data_000002", shape=(7, 2, 3), dtype="u4")
        data.create_dataset("data_000001", shape=(5, 2, 3), dtype="u4")

    assert get_eigerImage_per_file(master) == 5


@pytest.mark.portable
def test_xpcs_result_export_preserves_array_dataframe_and_metadata_layout(tmp_path, monkeypatch):
    import h5py

    from pyCHX.Create_Report import export_xpcs_results_to_h5, extract_xpcs_results_from_h5

    filename = "result.h5"
    g12b = np.arange(36, dtype=np.float64).reshape(3, 3, 4)
    g12b[0, 1, 2] = np.nan
    fit = pd.DataFrame(
        {"baseline": [1.0, 1.1], "beta": [0.2, 0.3]},
        index=pd.Index([3, 8], name="roi"),
    )
    exported = {
        "md": {"uid": "synthetic", "frame_count": np.int64(3)},
        "qval_dict": {0: np.asarray([0.01]), 1: np.asarray([0.02])},
        "g2": np.asarray([[1.2, 1.3], [1.1, 1.15]], dtype=np.float64),
        "g12b": g12b,
        "g2_fit_paras": fit,
    }
    dataframe_writes = []
    original_to_hdf = pd.DataFrame.to_hdf

    def counted_to_hdf(frame, path_or_buf, *args, **kwargs):
        dataframe_writes.append(kwargs["key"])
        return original_to_hdf(frame, path_or_buf, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_hdf", counted_to_hdf)

    export_xpcs_results_to_h5(filename, f"{tmp_path}/", exported)

    assert dataframe_writes == ["g2_fit_paras"]
    with h5py.File(tmp_path / filename, "r") as handle:
        assert isinstance(handle["g2"], h5py.Dataset)
        assert handle["g2"].chunks is None
        assert handle["g2"].compression is None
        assert handle["g12b"].chunks is None
        assert handle["g12b"].compression is None
        assert handle["g12b"].fillvalue == 0.0
        np.testing.assert_array_equal(handle["g12b"][:], g12b)
        assert handle["md"].shape == (1,)
        assert handle["md"].dtype == np.dtype("i")
        assert handle["md"].attrs["uid"] == "synthetic"
        np.testing.assert_array_equal(handle["qval_dict"].attrs["0"], [0.01])

    pd.testing.assert_frame_equal(pd.read_hdf(tmp_path / filename, key="g2_fit_paras"), fit)
    extracted = extract_xpcs_results_from_h5(filename, f"{tmp_path}/")
    np.testing.assert_array_equal(extracted["g2"], exported["g2"])
    np.testing.assert_array_equal(extracted["g12b"], g12b)
    pd.testing.assert_frame_equal(extracted["g2_fit_paras"], fit)
