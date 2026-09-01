def test_public_api_imports():
    """Check that the aggregate pyCHX function imports load without errors."""
    import pyCHX.chx_packages

    assert pyCHX.chx_packages.__name__ == "pyCHX.chx_packages"
