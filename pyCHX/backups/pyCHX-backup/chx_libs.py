"""
Dec 10, 2015 Developed by Y.G.@CHX
yuzhang@bnl.gov
This module is for the necessary packages for the XPCS analysis
"""

## Import all the required packages for  Data Analysis
from databroker import Broker
from databroker.assets.path_only_handlers import RawHandler

# edit handlers here to switch to PIMS or dask
# this does the databroker import
# from chxtools.handlers import EigerHandler
from eiger_io.fs_handler import EigerHandler
from IPython.core.magics.display import Javascript
from modest_image import imshow
from skbeam.core.utils import multi_tau_lags
from skimage.draw import disk, ellipse, line, line_aa, polygon

db = Broker.named("chx")
import collections
import copy
import getpass
import itertools
import os
import pickle
import random
import sys
import time
import warnings
from datetime import datetime

import h5py
import matplotlib as mpl
import matplotlib.cm as mcm
import matplotlib.pyplot as plt
import numpy as np
import pims
import skbeam.core.correlation as corr
import skbeam.core.roi as roi
import skbeam.core.utils as utils

# * scikit-beam - data analysis tools for X-ray science
#    - https://github.com/scikit-beam/scikit-beam
# * xray-vision - plotting helper functions for X-ray science
#    - https://github.com/Nikea/xray-vision
import xray_vision
import xray_vision.mpl_plotting as mpl_plot
from lmfit import Model, Parameter, Parameters, minimize, report_fit
from matplotlib import gridspec
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pandas import DataFrame
from PIL import Image
from tqdm import tqdm
from xray_vision.mask.manual_mask import ManualMask
from xray_vision.mpl_plotting import speckle

mcolors = itertools.cycle(["b", "g", "r", "c", "m", "y", "k", "darkgoldenrod", "oldlace", "brown", "dodgerblue"])
markers = itertools.cycle(list(plt.Line2D.filled_markers))
lstyles = itertools.cycle(["-", "--", "-.", ".", ":"])
colors = itertools.cycle(
    [
        "blue",
        "darkolivegreen",
        "brown",
        "m",
        "orange",
        "hotpink",
        "darkcyan",
        "red",
        "gray",
        "green",
        "black",
        "cyan",
        "purple",
        "navy",
    ]
)
colors_copy = itertools.cycle(
    [
        "blue",
        "darkolivegreen",
        "brown",
        "m",
        "orange",
        "hotpink",
        "darkcyan",
        "red",
        "gray",
        "green",
        "black",
        "cyan",
        "purple",
        "navy",
    ]
)
markers = itertools.cycle(
    [
        "o",
        "2",
        "p",
        "1",
        "s",
        "*",
        "4",
        "+",
        "8",
        "v",
        "3",
        "D",
        "H",
        "^",
    ]
)
markers_copy = itertools.cycle(
    [
        "o",
        "2",
        "p",
        "1",
        "s",
        "*",
        "4",
        "+",
        "8",
        "v",
        "3",
        "D",
        "H",
        "^",
    ]
)
RUN_GUI = False  # if True for gui setup; else for notebook; the main code difference is the Figure() or plt.figure(figsize=(8, 6))
markers = [
    "o",
    "D",
    "v",
    "^",
    "<",
    ">",
    "p",
    "s",
    "H",
    "h",
    "*",
    "d",
    "$I$",
    "$L$",
    "$O$",
    "$V$",
    "$E$",
    "$c$",
    "$h$",
    "$x$",
    "$b$",
    "$e$",
    "$a$",
    "$m$",
    "$l$",
    "$i$",
    "$n$",
    "$e$",
    "8",
    "1",
    "3",
    "2",
    "4",
    "+",
    "x",
    "_",
    "|",
    ",",
    "1",
]
markers = np.array(markers * 100)
markers = [
    "o",
    "D",
    "v",
    "^",
    "<",
    ">",
    "p",
    "s",
    "H",
    "h",
    "*",
    "d",
    "8",
    "1",
    "3",
    "2",
    "4",
    "+",
    "x",
    "_",
    "|",
    ",",
    "1",
]
markers = np.array(markers * 100)
colors = np.array(
    [
        "darkorange",
        "mediumturquoise",
        "seashell",
        "mediumaquamarine",
        "darkblue",
        "yellowgreen",
        "mintcream",
        "royalblue",
        "springgreen",
        "slategray",
        "yellow",
        "slateblue",
        "darkslateblue",
        "papayawhip",
        "bisque",
        "firebrick",
        "burlywood",
        "dodgerblue",
        "dimgrey",
        "chartreuse",
        "deepskyblue",
        "honeydew",
        "orchid",
        "teal",
        "steelblue",
        "limegreen",
        "antiquewhite",
        "linen",
        "saddlebrown",
        "grey",
        "khaki",
        "hotpink",
        "darkslategray",
        "forestgreen",
        "lightsalmon",
        "turquoise",
        "navajowhite",
        "darkgrey",
        "darkkhaki",
        "slategrey",
        "indigo",
        "darkolivegreen",
        "aquamarine",
        "moccasin",
        "beige",
        "ivory",
        "olivedrab",
        "whitesmoke",
        "paleturquoise",
        "blueviolet",
        "tomato",
        "aqua",
        "palegoldenrod",
        "cornsilk",
        "navy",
        "mediumvioletred",
        "palevioletred",
        "aliceblue",
        "azure",
        "orangered",
        "lightgrey",
        "lightpink",
        "orange",
        "wheat",
        "darkorchid",
        "mediumslateblue",
        "lightslategray",
        "green",
        "lawngreen",
        "mediumseagreen",
        "darksalmon",
        "pink",
        "oldlace",
        "sienna",
        "dimgray",
        "fuchsia",
        "lemonchiffon",
        "maroon",
        "salmon",
        "gainsboro",
        "indianred",
        "crimson",
        "mistyrose",
        "lightblue",
        "darkgreen",
        "lightgreen",
        "deeppink",
        "palegreen",
        "thistle",
        "lightcoral",
        "lightgray",
        "lightskyblue",
        "mediumspringgreen",
        "mediumblue",
        "peru",
        "lightgoldenrodyellow",
        "darkseagreen",
        "mediumorchid",
        "coral",
        "lightyellow",
        "chocolate",
        "lavenderblush",
        "darkred",
        "lightseagreen",
        "darkviolet",
        "lightcyan",
        "cadetblue",
        "blanchedalmond",
        "midnightblue",
        "lightsteelblue",
        "darkcyan",
        "floralwhite",
        "darkgray",
        "lavender",
        "sandybrown",
        "cornflowerblue",
        "gray",
        "mediumpurple",
        "lightslategrey",
        "seagreen",
        "silver",
        "darkmagenta",
        "darkslategrey",
        "darkgoldenrod",
        "rosybrown",
        "goldenrod",
        "darkturquoise",
        "plum",
        "purple",
        "olive",
        "gold",
        "powderblue",
        "peachpuff",
        "violet",
        "lime",
        "greenyellow",
        "tan",
        "skyblue",
        "magenta",
        "black",
        "brown",
        "green",
        "cyan",
        "red",
        "blue",
    ]
    * 100
)

colors = colors[::-1]
colors_ = itertools.cycle(colors)
# colors_ = itertools.cycle(sorted_colors_ )
markers_ = itertools.cycle(markers)
# Custom colormaps
################################################################################
# ROYGBVR but with Cyan-Blue instead of Blue
color_list_cyclic_spectrum = [
    [1.0, 0.0, 0.0],
    [1.0, 165.0 / 255.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.2, 1.0],
    [148.0 / 255.0, 0.0, 211.0 / 255.0],
    [1.0, 0.0, 0.0],
]
cmap_cyclic_spectrum = mpl.colors.LinearSegmentedColormap.from_list(
    "cmap_cyclic_spectrum", color_list_cyclic_spectrum
)

# classic jet, slightly tweaked
# (bears some similarity to mpl.cm.nipy_spectral)
color_list_jet_extended = [
    [0, 0, 0],
    [0.18, 0, 0.18],
    [0, 0, 0.5],
    [0, 0, 1],
    [0.0, 0.38888889, 1.0],
    [0.0, 0.83333333, 1.0],
    [0.3046595, 1.0, 0.66308244],
    [0.66308244, 1.0, 0.3046595],
    [1.0, 0.90123457, 0.0],
    [1.0, 0.48971193, 0.0],
    [1.0, 0.0781893, 0.0],
    [1, 0, 0],
    [0.5, 0.0, 0.0],
]
cmap_jet_extended = mpl.colors.LinearSegmentedColormap.from_list("cmap_jet_extended", color_list_jet_extended)

# Tweaked version of "view.gtk" default color scale
color_list_vge = [
    [0.0 / 255.0, 0.0 / 255.0, 0.0 / 255.0],
    [0.0 / 255.0, 0.0 / 255.0, 254.0 / 255.0],
    [188.0 / 255.0, 2.0 / 255.0, 107.0 / 255.0],
    [254.0 / 255.0, 55.0 / 255.0, 0.0 / 255.0],
    [254.0 / 255.0, 254.0 / 255.0, 0.0 / 255.0],
    [254.0 / 255.0, 254.0 / 255.0, 254.0 / 255.0],
]
cmap_vge = mpl.colors.LinearSegmentedColormap.from_list("cmap_vge", color_list_vge)

# High-dynamic-range (HDR) version of VGE
color_list_vge_hdr = [
    [255.0 / 255.0, 255.0 / 255.0, 255.0 / 255.0],
    [0.0 / 255.0, 0.0 / 255.0, 0.0 / 255.0],
    [0.0 / 255.0, 0.0 / 255.0, 255.0 / 255.0],
    [188.0 / 255.0, 0.0 / 255.0, 107.0 / 255.0],
    [254.0 / 255.0, 55.0 / 255.0, 0.0 / 255.0],
    [254.0 / 255.0, 254.0 / 255.0, 0.0 / 255.0],
    [254.0 / 255.0, 254.0 / 255.0, 254.0 / 255.0],
]
cmap_vge_hdr = mpl.colors.LinearSegmentedColormap.from_list("cmap_vge_hdr", color_list_vge_hdr)

# Simliar to Dectris ALBULA default color-scale
color_list_hdr_albula = [
    [255.0 / 255.0, 255.0 / 255.0, 255.0 / 255.0],
    [0.0 / 255.0, 0.0 / 255.0, 0.0 / 255.0],
    [255.0 / 255.0, 0.0 / 255.0, 0.0 / 255.0],
    [255.0 / 255.0, 255.0 / 255.0, 0.0 / 255.0],
    # [ 255.0/255.0, 255.0/255.0, 255.0/255.0],
]
cmap_hdr_albula = mpl.colors.LinearSegmentedColormap.from_list("cmap_hdr_albula", color_list_hdr_albula)
cmap_albula = cmap_hdr_albula
cmap_albula_r = mpl.colors.LinearSegmentedColormap.from_list("cmap_hdr_r", color_list_hdr_albula[::-1])

# Ugly color-scale, but good for highlighting many features in HDR data
color_list_cur_hdr_goldish = [
    [255.0 / 255.0, 255.0 / 255.0, 255.0 / 255.0],  # white
    [0.0 / 255.0, 0.0 / 255.0, 0.0 / 255.0],  # black
    [100.0 / 255.0, 127.0 / 255.0, 255.0 / 255.0],  # light blue
    [0.0 / 255.0, 0.0 / 255.0, 127.0 / 255.0],  # dark blue
    # [ 0.0/255.0, 127.0/255.0, 0.0/255.0], # dark green
    [127.0 / 255.0, 60.0 / 255.0, 0.0 / 255.0],  # orange
    [255.0 / 255.0, 255.0 / 255.0, 0.0 / 255.0],  # yellow
    [200.0 / 255.0, 0.0 / 255.0, 0.0 / 255.0],  # red
    [255.0 / 255.0, 255.0 / 255.0, 255.0 / 255.0],  # white
]
cmap_hdr_goldish = mpl.colors.LinearSegmentedColormap.from_list("cmap_hdr_goldish", color_list_cur_hdr_goldish)
