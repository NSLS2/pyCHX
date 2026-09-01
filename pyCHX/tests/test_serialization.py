import numpy as np
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
