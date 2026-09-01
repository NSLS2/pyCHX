import pytest


@pytest.mark.portable
def test_package_imports():
    """The package metadata import must not require facility services."""
    import pyCHX

    assert pyCHX.__name__ == "pyCHX"


@pytest.mark.live_facility
def test_public_api_imports_with_live_facility():
    """Opt-in smoke test for the configured CHX catalog and handlers."""
    import pyCHX.chx_packages

    assert pyCHX.chx_packages.__name__ == "pyCHX.chx_packages"
