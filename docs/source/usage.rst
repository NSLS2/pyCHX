=====
Usage
=====

Start by importing pyCHX.

.. code-block:: python

    import pyCHX

Existing CHX notebooks may continue to use the historical compatibility
namespace::

    from pyCHX.chx_packages import *

Importing that namespace does not connect to facility services. The default
``db`` object opens the CHX catalog on its first real use. New code may make
that initialization explicit::

    from pyCHX.chx_handlers import initialize_facility

    catalog = initialize_facility()

Facility access requires the optional dependencies described in
:doc:`installation`.

Facility configuration
======================

The historical CHX locations remain the defaults. They can be overridden at
runtime without changing pyCHX or a notebook:

.. code-block:: bash

    export PYCHX_ANALYSIS_ROOT=/path/to/analysis
    export PYCHX_COMPRESSED_DATA_DIR=/path/to/compressed-data
    export PYCHX_OLOG_URL=https://olog.example/Olog

pyCHX does not set proxy environment variables. If an Olog deployment needs a
proxy, configure it in the launching shell or Jupyter environment.
