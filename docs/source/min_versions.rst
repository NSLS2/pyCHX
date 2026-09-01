===================================
Minimum Version of Python and NumPy
===================================


- This project supports at least the minor versions of Python
  initially released 42 months prior to a planned project release
  date.
- The project will always support at least the 2 latest minor
  versions of Python.
- The project will support minor versions of ``numpy`` initially
  released in the 24 months prior to a planned project release date or
  the oldest version that supports the minimum Python version
  (whichever is higher).
- The project will always support at least the 3 latest minor
  versions of NumPy.

The minimum supported version of Python is set by ``requires-python`` in
``pyproject.toml``. All supported minor versions of Python are included in the
continuous-integration test matrix. pyCHX currently supports Python 3.11
through 3.14 and NumPy 1.23 through the latest 2.x release.

The project should adjust upward the minimum Python and NumPy
version support on every minor and major release, but never on a
patch release.

This is consistent with NumPy `NEP 29
<https://numpy.org/neps/nep-0029-deprecation_policy.html>`__.
