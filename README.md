pyCHX -- CHX XPCS Data Analysis Packages
========

Repository for data collection and analysis scripts that are useful at the
CHX beamline at NSLS-II (11-ID) developed by Dr. Yugang Zhang (yuzhang@bnl.gov).

Installation instructions on Linux:

Install miniconda from https://conda.io/miniconda.html. Then create the environment for pyCHX:
```
conda create --name pyCHX python=3.6 numpy scipy matplotlib
source activate pyCHX
pip install -r https://raw.githubusercontent.com/NSLS-II-CHX/pyCHX/master/requirements.txt
pip install git+https://github.com/NSLS-II-CHX/pyCHX


Pixi Install:
- pixi install: installs only main dependencies
- pixi install --with dev: installs main and dev dependencies
- pixi run --with dev pytest: runs pytest with dev tols available
```
