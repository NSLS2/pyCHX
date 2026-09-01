"""Lazy CHX catalog access and Eiger handler registration.

Importing this module is deliberately side-effect free. The historical
``db`` object opens the CHX catalog and registers PIMS handlers only when it
is first used.
"""

from __future__ import annotations

from threading import RLock
from typing import Any


def _missing_dependency_stub(symbol: str, distribution: str):
    message = (
        f"{symbol} requires the optional {distribution!r} dependency. "
        "Install pyCHX with the 'facility' extra (or the facility-source "
        "dependency group when working from a source checkout)."
    )

    class MissingOptionalDependency:
        def __init__(self, *args, **kwargs):
            raise ImportError(message)

    MissingOptionalDependency.__name__ = symbol
    MissingOptionalDependency.__qualname__ = symbol
    MissingOptionalDependency.__doc__ = message
    return MissingOptionalDependency


try:
    from eiger_io.fs_handler import EigerHandler as EigerHandlerPIMS
    from eiger_io.fs_handler import EigerImages as EigerImagesPIMS
except ImportError:
    EigerHandlerPIMS = _missing_dependency_stub("EigerHandler", "eiger-io")
    EigerImagesPIMS = _missing_dependency_stub("EigerImages", "eiger-io")

try:
    from eiger_io.fs_handler_dask import EigerHandlerDask, EigerImagesDask
except ImportError:
    EigerHandlerDask = _missing_dependency_stub("EigerHandlerDask", "eiger-io")
    EigerImagesDask = _missing_dependency_stub("EigerImagesDask", "eiger-io")


# Historical defaults retained for callers importing these names directly.
EigerHandler = EigerHandlerPIMS
EigerImages = EigerImagesPIMS


def get_catalog(name: str = "chx"):
    """Open a databroker catalog by name without registering handlers."""
    try:
        from databroker import Broker
    except ImportError as exc:
        raise ImportError(
            "CHX catalog access requires the optional 'databroker' dependency. "
            "Install pyCHX with the 'facility' extra."
        ) from exc
    return Broker.named(name)


def _eiger_backend(backend: str):
    if backend == "pims":
        try:
            from eiger_io.fs_handler import EigerHandler, EigerImages
        except ImportError as exc:
            raise ImportError(
                "PIMS Eiger support requires the optional 'eiger-io' dependency. "
                "Install the facility-source dependency group."
            ) from exc
        return EigerHandler, EigerImages
    if backend == "dask":
        try:
            from eiger_io.fs_handler_dask import EigerHandlerDask, EigerImagesDask
        except ImportError as exc:
            raise ImportError(
                "Dask Eiger support requires the optional 'eiger-io' dependency. "
                "Install the facility-source dependency group."
            ) from exc
        return EigerHandlerDask, EigerImagesDask
    raise ValueError(f"unknown Eiger backend {backend!r}; expected 'pims' or 'dask'")


def register_eiger_handlers(catalog, backend: str = "pims"):
    """Register the three historical Eiger specs on *catalog*."""
    handler, _ = _eiger_backend(backend)
    for spec in ("AD_EIGER2", "AD_EIGER", "AD_EIGER_SLICE"):
        catalog.reg.register_handler(spec, handler, overwrite=True)
    return catalog


def initialize_facility(name: str = "chx", backend: str = "pims", catalog=None):
    """Open (or accept) a catalog and register its Eiger handlers."""
    if catalog is None:
        catalog = get_catalog(name)
    return register_eiger_handlers(catalog, backend=backend)


def use_pims(catalog):
    """Compatibility wrapper selecting and registering the PIMS handler."""
    global EigerHandler, EigerImages
    EigerHandler, EigerImages = _eiger_backend("pims")
    return register_eiger_handlers(catalog, backend="pims")


def use_dask(catalog):
    """Compatibility wrapper selecting and registering the Dask handler."""
    global EigerHandler, EigerImages
    EigerHandler, EigerImages = _eiger_backend("dask")
    return register_eiger_handlers(catalog, backend="dask")


class LazyCatalog:
    """Transparent, thread-safe proxy for the default CHX catalog."""

    def __init__(self, name: str = "chx", backend: str = "pims"):
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_backend", backend)
        object.__setattr__(self, "_catalog", None)
        object.__setattr__(self, "_lock", RLock())

    def _resolve(self):
        catalog = object.__getattribute__(self, "_catalog")
        if catalog is None:
            with object.__getattribute__(self, "_lock"):
                catalog = object.__getattribute__(self, "_catalog")
                if catalog is None:
                    catalog = initialize_facility(
                        name=object.__getattribute__(self, "_name"),
                        backend=object.__getattribute__(self, "_backend"),
                    )
                    object.__setattr__(self, "_catalog", catalog)
        return catalog

    def _reset(self):
        """Clear the cached catalog (primarily useful for tests)."""
        with object.__getattribute__(self, "_lock"):
            object.__setattr__(self, "_catalog", None)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._resolve(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._resolve(), name, value)

    def __getitem__(self, key):
        return self._resolve()[key]

    def __call__(self, *args, **kwargs):
        return self._resolve()(*args, **kwargs)

    def __iter__(self):
        return iter(self._resolve())

    def __contains__(self, key):
        return key in self._resolve()

    def __len__(self):
        return len(self._resolve())

    def __repr__(self):
        catalog = object.__getattribute__(self, "_catalog")
        if catalog is None:
            name = object.__getattribute__(self, "_name")
            return f"<LazyCatalog {name!r} (not initialized)>"
        return repr(catalog)


db = LazyCatalog()


__all__ = [
    "EigerHandler",
    "EigerHandlerDask",
    "EigerHandlerPIMS",
    "EigerImages",
    "EigerImagesDask",
    "EigerImagesPIMS",
    "LazyCatalog",
    "db",
    "get_catalog",
    "initialize_facility",
    "register_eiger_handlers",
    "use_dask",
    "use_pims",
]
