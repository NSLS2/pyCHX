"""Runtime configuration for CHX facility paths and services.

The defaults preserve the historical CHX deployment. Environment variables
allow tests and other installations to choose local resources without editing
library code.
"""

from __future__ import annotations

import os

DEFAULT_ANALYSIS_ROOT = "/XF11ID/analysis"
DEFAULT_COMPRESSED_DATA_DIR = "/XF11ID/analysis/Compressed_Data"
DEFAULT_LOCAL_COMPRESSED_DATA_DIR = "/tmp_data/compressed"
DEFAULT_OLOG_URL = "https://epics-services-chx.nsls2.bnl.local:38981/Olog"


def _environment_value(name: str, default: str) -> str:
    """Return a non-empty environment override or its historical default."""
    return os.environ.get(name) or default


def get_analysis_root() -> str:
    """Return the root directory used for CHX analysis results."""
    return _environment_value("PYCHX_ANALYSIS_ROOT", DEFAULT_ANALYSIS_ROOT)


def get_compressed_data_dir(*, local: bool = False) -> str:
    """Return the configured directory used for compressed detector data."""
    default = DEFAULT_LOCAL_COMPRESSED_DATA_DIR if local else DEFAULT_COMPRESSED_DATA_DIR
    return _environment_value("PYCHX_COMPRESSED_DATA_DIR", default)


def get_olog_url() -> str:
    """Return the configured CHX Olog endpoint."""
    return _environment_value("PYCHX_OLOG_URL", DEFAULT_OLOG_URL)


__all__ = [
    "DEFAULT_ANALYSIS_ROOT",
    "DEFAULT_COMPRESSED_DATA_DIR",
    "DEFAULT_LOCAL_COMPRESSED_DATA_DIR",
    "DEFAULT_OLOG_URL",
    "get_analysis_root",
    "get_compressed_data_dir",
    "get_olog_url",
]
