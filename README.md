# pyCHX

Python tools for X-ray photon correlation spectroscopy (XPCS) data collection
and analysis at the NSLS-II Coherent Hard X-ray Scattering (CHX) beamline.

## Installation

Install the published package in a Python 3.11 or newer environment:

```bash
python -m pip install pyCHX
```

CHX catalog and Olog access can be added with
`python -m pip install 'pyCHX[facility]'`. For a source checkout, install the
pinned Eiger and ModestImage implementations with
`python -m pip install --group facility-source`.

See [CONTRIBUTING.rst](CONTRIBUTING.rst) for development setup and checks.
