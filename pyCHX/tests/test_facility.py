import pytest


class FakeRegistry:
    def __init__(self):
        self.calls = []

    def register_handler(self, spec, handler, overwrite=False):
        self.calls.append((spec, handler, overwrite))


class FakeCatalog:
    def __init__(self):
        self.reg = FakeRegistry()
        self.items = {"uid": "run"}

    def __getitem__(self, key):
        return self.items[key]


@pytest.mark.mocked_facility
def test_register_eiger_handlers(monkeypatch):
    from pyCHX import chx_handlers

    class Handler:
        pass

    catalog = FakeCatalog()
    monkeypatch.setattr(chx_handlers, "_eiger_backend", lambda backend: (Handler, object))

    assert chx_handlers.register_eiger_handlers(catalog) is catalog
    assert catalog.reg.calls == [
        ("AD_EIGER2", Handler, True),
        ("AD_EIGER", Handler, True),
        ("AD_EIGER_SLICE", Handler, True),
    ]


@pytest.mark.mocked_facility
def test_lazy_catalog_initializes_once_on_first_real_use(monkeypatch):
    from pyCHX import chx_handlers

    catalog = FakeCatalog()
    calls = []

    def initialize(name="chx", backend="pims", catalog=None):
        calls.append((name, backend))
        return catalog_instance

    catalog_instance = catalog
    monkeypatch.setattr(chx_handlers, "initialize_facility", initialize)
    proxy = chx_handlers.LazyCatalog()

    assert "not initialized" in repr(proxy)
    assert calls == []
    assert proxy["uid"] == "run"
    assert proxy.reg is catalog.reg
    assert calls == [("chx", "pims")]


@pytest.mark.mocked_facility
def test_initialize_facility_accepts_an_existing_catalog(monkeypatch):
    from pyCHX import chx_handlers

    catalog = FakeCatalog()
    registered = []
    monkeypatch.setattr(
        chx_handlers,
        "register_eiger_handlers",
        lambda value, backend="pims": registered.append((value, backend)) or value,
    )

    assert chx_handlers.initialize_facility(catalog=catalog, backend="dask") is catalog
    assert registered == [(catalog, "dask")]


@pytest.mark.mocked_facility
def test_olog_client_is_constructed_only_on_use(monkeypatch):
    from pyCHX import chx_olog

    calls = []

    class FakeClient:
        def __init__(self, url):
            calls.append(url)

        def log(self, text, logbooks):
            return text, logbooks

    chx_olog.get_olog_client.cache_clear()
    monkeypatch.setattr(chx_olog, "SimpleOlogClient", FakeClient)

    assert repr(chx_olog.olog_client) == "<OlogClient (lazy)>"
    assert calls == []
    assert chx_olog.create_olog_entry("ready") == ("ready", "Data Acquisition")
    assert calls == [chx_olog.OLOG_URL]
