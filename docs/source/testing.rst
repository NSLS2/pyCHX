Testing
-------

Install the test dependencies and run the offline suite::

    $ python -m pip install -e ".[test]"
    $ PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest

Tests marked ``portable`` use no facility services. Tests marked
``mocked_facility`` exercise catalog and Olog integration with local fakes.
The bundled Eiger regression data uses the source-only Eiger reader and can be
tested without connecting to facility services::

    $ python -m pip install --group facility-source
    $ pytest -m data_regression

The data-regression suite generates an approximately 200 MB temporary CMP file.
Its prerefactor numerical outputs are characterization references, not guaranteed
ground truth. A mismatch must be investigated and may indicate either a new
regression or a defect in the historical implementation; references must not be
updated without documenting that decision.

The ``live_facility`` tests are opt-in and may be run only in an appropriately
configured CHX environment::

    $ pytest -m live_facility --live-facility

Set ``PYCHX_LIVE_UID`` to a stable, readable run UID to include a catalog
lookup in that smoke test. Live tests must remain read-only.

Deprecation, pending-deprecation, and future warnings emitted by pyCHX are
test failures. A scheduled pre-release dependency job provides additional
early warning for upcoming NumPy and other dependency changes. The lint job
also runs Ruff's NumPy compatibility rules over every module, including code
paths that are not yet exercised by tests.

The representative ds1 fixture is self-contained and documented in
``pyCHX/tests/data/ds1/README.md``. It is excluded from built distributions but
retained in the source repository so fresh clones can run the regression suite.
