__author__ = "Yugang Zhang"

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pyCHX")
except PackageNotFoundError:
    __version__ = "0+unknown"
