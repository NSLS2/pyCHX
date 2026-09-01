import os

import pytest


@pytest.mark.portable
def test_package_imports():
    """The package metadata import must not require facility services."""
    import pyCHX

    assert pyCHX.__name__ == "pyCHX"


@pytest.mark.live_facility
def test_public_api_imports_with_live_facility():
    """Opt-in smoke test for the configured CHX catalog and handlers.

    Set ``PYCHX_LIVE_UID`` to additionally verify a read-only catalog lookup.
    """
    from pyCHX.chx_handlers import get_catalog, initialize_facility

    catalog = get_catalog()
    assert initialize_facility(catalog=catalog) is catalog

    uid = os.environ.get("PYCHX_LIVE_UID")
    if uid:
        assert catalog[uid] is not None
