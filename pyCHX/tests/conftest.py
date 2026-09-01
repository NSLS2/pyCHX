import os

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--live-facility",
        action="store_true",
        default=False,
        help="run tests that connect to live CHX facility services",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--live-facility") or os.environ.get("PYCHX_LIVE_FACILITY") == "1":
        return

    skip_live = pytest.mark.skip(reason="requires --live-facility or PYCHX_LIVE_FACILITY=1")
    for item in items:
        if "live_facility" in item.keywords:
            item.add_marker(skip_live)
