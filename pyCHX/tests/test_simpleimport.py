import importlib


def test_public_api_imports():
    """Check that the aggregate pyCHX function imports load without errors."""
    public_api = importlib.import_module("pyCHX.chx_packages")

    assert public_api.__name__ == "pyCHX.chx_packages"
