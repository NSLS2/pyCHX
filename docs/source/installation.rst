============
Installation
============

Install the published package in a Python 3.11 or newer environment::

    $ python -m pip install pyCHX

For CHX catalog and Olog access, install the facility extra::

    $ python -m pip install "pyCHX[facility]"

Developers working from a source checkout can install the pinned Eiger and
ModestImage implementations without publishing their Git URLs as package
metadata::

    $ python -m pip install --group facility-source
    $ python -m pip install -e ".[test,docs,facility]"
