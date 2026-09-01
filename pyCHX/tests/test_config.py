import os

import pytest


@pytest.mark.portable
def test_facility_configuration_uses_environment_overrides(monkeypatch, tmp_path):
    from pyCHX.config import get_analysis_root, get_compressed_data_dir, get_olog_url

    analysis_root = tmp_path / "analysis"
    compressed_data = tmp_path / "compressed"
    monkeypatch.setenv("PYCHX_ANALYSIS_ROOT", os.fspath(analysis_root))
    monkeypatch.setenv("PYCHX_COMPRESSED_DATA_DIR", os.fspath(compressed_data))
    monkeypatch.setenv("PYCHX_OLOG_URL", "https://olog.example.invalid/Olog")

    assert get_analysis_root() == os.fspath(analysis_root)
    assert get_compressed_data_dir() == os.fspath(compressed_data)
    assert get_compressed_data_dir(local=True) == os.fspath(compressed_data)
    assert get_olog_url() == "https://olog.example.invalid/Olog"


@pytest.mark.portable
def test_create_user_folder_uses_configured_analysis_root(monkeypatch, tmp_path):
    from pyCHX.chx_generic_functions import create_user_folder

    monkeypatch.setenv("PYCHX_ANALYSIS_ROOT", os.fspath(tmp_path))

    result = create_user_folder("2026_3", username="tester")

    assert result == os.fspath(tmp_path / "2026_3" / "tester" / "Results") + "/"
    assert os.path.isdir(result)


@pytest.mark.portable
def test_pipeline_modules_do_not_mutate_proxy_environment():
    from pathlib import Path

    package = Path(__file__).parents[1]
    sources = "\n".join(
        (package / filename).read_text()
        for filename in ("chx_xpcs_xsvs_jupyter_V1.py", "XPCS_XSVS_SAXS_Multi_2017_V4.py")
    )

    assert 'os.environ["HTTPS_PROXY"]' not in sources
    assert 'os.environ["no_proxy"]' not in sources
