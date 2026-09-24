import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.portable
def test_legacy_star_import_namespace_and_effective_bindings():
    import pyCHX.chx_compress as compress
    import pyCHX.chx_generic_functions as generic
    import pyCHX.chx_packages as packages
    import pyCHX.chx_xpcs_xsvs_jupyter_V1 as pipeline

    added = {"colors", "get_catalog", "initialize_facility", "markers", "register_eiger_handlers"}
    legacy_exports = sorted(set(packages.__all__) - added)

    # Snapshot of every public name exported before the import cleanup.
    assert len(legacy_exports) == 265
    assert hashlib.sha256("\n".join(legacy_exports).encode()).hexdigest() == (
        "50f566671736a2f7bf3c850576c72801c83101bca15ec97a08351d033d27f526"
    )
    assert len(packages.__all__) == len(set(packages.__all__))
    assert all(hasattr(packages, name) for name in packages.__all__)

    namespace = {}
    exec("from pyCHX.chx_packages import *", namespace)
    assert set(namespace) - {"__builtins__"} == set(packages.__all__)

    # Later legacy imports intentionally replaced these earlier bindings.
    assert packages.compress_eigerdata is compress.compress_eigerdata
    assert packages.get_eigerImage_per_file is generic.get_eigerImage_per_file
    assert packages.get_fra_num_by_dose is pipeline.get_fra_num_by_dose
    assert packages.cal_g2 is generic.cal_g2
    assert packages.save_lists is generic.save_lists


@pytest.mark.portable
def test_production_wildcard_import_order_keeps_optimized_bindings():
    from pyCHX.chx_compress import compress_eigerdata
    from pyCHX.chx_correlationc import Get_Pixel_Arrayc, auto_two_Arrayc
    from pyCHX.chx_correlationp import cal_g2p
    from pyCHX.Two_Time_Correlation_Function import get_one_time_from_two_time

    namespace = {}
    exec(
        "from pyCHX.chx_packages import *\nfrom pyCHX.chx_xpcs_xsvs_jupyter_V1 import *",
        namespace,
    )
    assert namespace["compress_eigerdata"] is compress_eigerdata
    assert namespace["cal_g2p"] is cal_g2p
    assert namespace["Get_Pixel_Arrayc"] is Get_Pixel_Arrayc
    assert namespace["auto_two_Arrayc"] is auto_two_Arrayc
    assert namespace["get_one_time_from_two_time"] is get_one_time_from_two_time


@pytest.mark.portable
def test_parallel_correlation_wildcard_does_not_export_instrumentation_modules():
    namespace = {}
    exec("from pyCHX.chx_correlationp import *", namespace)
    assert "time" not in namespace


@pytest.mark.portable
def test_final_marker_and_color_values_are_preserved():
    import pyCHX.chx_packages as packages

    markers = packages.markers.tolist()
    colors = packages.colors.tolist()
    assert len(markers) == 2300
    assert len(colors) == 14500
    assert hashlib.sha256(json.dumps(markers, separators=(",", ":")).encode()).hexdigest() == (
        "32d7bd384f8f93a5d4b34c9a21d0093712146dff0448ceaa24476ac57aa3caf6"
    )
    assert hashlib.sha256(json.dumps(colors, separators=(",", ":")).encode()).hexdigest() == (
        "a5c9f36e801a44eced36547274f0f807ac66ad2148441d7f00eb333ecbf8068d"
    )


@pytest.mark.portable
def test_all_modules_import_without_network_access():
    code = """
import pkgutil
import socket

class NetworkBlockedSocket(socket.socket):
    def connect(self, *args, **kwargs):
        raise AssertionError("network access attempted during import")

    def connect_ex(self, *args, **kwargs):
        raise AssertionError("network access attempted during import")

socket.socket = NetworkBlockedSocket
socket.create_connection = lambda *args, **kwargs: (_ for _ in ()).throw(
    AssertionError("network access attempted during import")
)

import pyCHX
for module in pkgutil.walk_packages(pyCHX.__path__, pyCHX.__name__ + "."):
    if ".tests" not in module.name:
        __import__(module.name)
"""
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(Path(os.environ.get("TMPDIR", "/tmp")) / "pychx-matplotlib")
    subprocess.run([sys.executable, "-c", code], check=True, env=env, timeout=60)


@pytest.mark.portable
def test_facility_classes_have_helpful_stubs_when_dependencies_are_missing():
    code = """
import importlib.abc
import sys

class BlockFacilityImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'databroker', 'eiger_io', 'modest_image', 'pyOlog'}:
            raise ModuleNotFoundError(fullname)
        return None

sys.meta_path.insert(0, BlockFacilityImports())
import pyCHX.chx_packages as packages

for cls, dependency in ((packages.EigerImages, 'eiger-io'), (packages.Attachment, 'pyOlog')):
    try:
        cls()
    except ImportError as exc:
        assert dependency in str(exc)
    else:
        raise AssertionError(f'{cls.__name__} did not report its missing dependency')

assert "not initialized" in repr(packages.db)
"""
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(Path(os.environ.get("TMPDIR", "/tmp")) / "pychx-matplotlib")
    subprocess.run([sys.executable, "-c", code], check=True, env=env, timeout=60)
