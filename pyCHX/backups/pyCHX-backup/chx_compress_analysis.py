from __future__ import absolute_import, division, print_function

import logging
import os
import struct
from collections import namedtuple

import matplotlib.pyplot as plt
from skbeam.core.roi import extract_label_indices
from skbeam.core.utils import multi_tau_lags
from tqdm import tqdm

from pyCHX.chx_generic_functions import save_arrays

# from pyCHX.chx_generic_functions import (get_circular_average)
# from pyCHX.XPCS_SAXS import (get_circular_average)
from pyCHX.chx_libs import (
    RUN_GUI,
    Figure,
    LogNorm,
    colors,
    colors_,
    datetime,
    db,
    getpass,
    markers,
    markers_,
    np,
    os,
    roi,
    time,
)

logger = logging.getLogger(__name__)

from modest_image import imshow

from pyCHX.chx_compress import (
    Multifile,
    compress_eigerdata,
    get_avg_imgc,
    get_each_frame_intensityc,
    init_compress_eigerdata,
    mean_intensityc,
    pass_FD,
    read_compressed_eigerdata,
)
from pyCHX.chx_generic_functions import find_bad_pixels_FD

# from pyCHX.chx_compress import *


def get_time_edge_avg_img(FD, frame_edge, show_progress=True, apply_threshold=False, threshold=15):
    """YG Dev Nov 14, 2017@CHX
    Update@2019/6/12 with option of apply a threshold for each frame
    Get averaged img by giving FD and frame edges
    Parameters
    ----------
    FD: Multifile class
        compressed file
    frame_edge: np.array, can be created by create_time_slice( Nimg, slice_num= 3,
        slice_width= 1, edges = None )
        e.g., np.array([[   5,    6],
           [2502, 2503],
           [4999, 5000]])
    apply_threshold: if True, will mask out all the pixels with intensity above the threshold
    threshold: 15 (for Eiger500K burst mode)
    Return:
        array: (N of frame_edge,  averaged image) , i.e., d[0] gives the first averaged image
    """

    Nt = len(frame_edge)
    d = np.zeros(Nt, dtype=object)
    if apply_threshold:
        avg_imgi = FD.rdframe(FD.beg)
    for i in tqdm(range(Nt)):
        t1, t2 = frame_edge[i]
        if not apply_threshold:
            d[i] = get_avg_imgc(FD, beg=t1, end=t2, sampling=1, plot_=False, show_progress=show_progress)
        else:
            dti = np.zeros([t2 - t1, avg_imgi.shape[0], avg_imgi.shape[1]])
            j = 0
            for ti in range(t1, t2):
                # print( j, ti )
                badpi = find_bad_pixels_FD(
                    np.arange(ti, ti + 1), FD, img_shape=avg_imgi.shape, threshold=threshold, show_progress=False
                )
                badpi = np.array(badpi, dtype=float)
                badpi[badpi == 0] = np.nan
                dti[j] = FD.rdframe(ti) * badpi
                j += 1
                # print(dti.shape)
            d[i] = np.nanmean(dti, axis=0)  # average_array_withNan( dti, axis=0 )
    return d


def plot_imgs(imgs, image_name=None, *argv, **kwargs):
    # NOT WORKing NOW....
    N = len(imgs)
    sx = np.ceil(np.sqrt(N))
    pass


def cal_waterfallc(
    FD, labeled_array, qindex=1, bin_waterfall=False, waterfall_roi_size=None, save=False, *argv, **kwargs
):
    """Compute the mean intensity for each ROI in the compressed file (FD)

    Parameters
    ----------
    FD: Multifile class
        compressed file
    labeled_array : array
        labeled array; 0 is background.
        Each ROI is represented by a nonzero integer. It is not required that
        the ROI labels are contiguous
    qindex : int, qindex=1, give the first ring in SAXS geometry. NOTE: qindex=0 is non-photon pixels.
        The ROI's to use.

    bin_waterfall: if True, will bin the waterfall along y-axis
    waterfall_roi_size: the size of waterfall roi,  (x-size, y-size), if bin, will bin along y
    save: save the waterfall

    Returns
    -------
    waterfall : array
        The mean intensity of each ROI for all `images`
        Dimensions:
            len(mean_intensity) == len(index)
            len(mean_intensity[0]) == len(images)
    index : list
        The labels for each element of the `mean_intensity` list
    """
    sampling = 1

    labeled_array_ = np.array(labeled_array == qindex, dtype=np.int64)

    qind, pixelist = roi.extract_label_indices(labeled_array_)

    if labeled_array_.shape != (FD.md["ncols"], FD.md["nrows"]):
        raise ValueError(
            " `image` shape (%d, %d) in FD is not equal to the labeled_array shape (%d, %d)"
            % (FD.md["ncols"], FD.md["nrows"], labeled_array_.shape[0], labeled_array_.shape[1])
        )

    # pre-allocate an array for performance
    # might be able to use list comprehension to make this faster

    watf = np.zeros([int((FD.end - FD.beg) / sampling), len(qind)])

    # fra_pix = np.zeros_like( pixelist, dtype=np.float64)

    timg = np.zeros(FD.md["ncols"] * FD.md["nrows"], dtype=np.int32)
    timg[pixelist] = np.arange(1, len(pixelist) + 1)

    # maxqind = max(qind)
    norm = np.bincount(qind)[1:]
    n = 0
    # for  i in tqdm(range( FD.beg , FD.end )):
    for i in tqdm(range(FD.beg, FD.end, sampling), desc="Get waterfall for q index=%s" % qindex):
        (p, v) = FD.rdrawframe(i)
        w = np.where(timg[p])[0]
        pxlist = timg[p[w]] - 1

        watf[n][pxlist] = v[w]
        n += 1

    if bin_waterfall:
        watf_ = watf.copy()
        watf = np.zeros([watf_.shape[0], waterfall_roi_size[0]])
        for i in range(waterfall_roi_size[1]):
            watf += watf_[:, waterfall_roi_size[0] * i : waterfall_roi_size[0] * (i + 1)]
        watf /= waterfall_roi_size[0]

    if save:
        path = kwargs["path"]
        uid = kwargs["uid"]
        np.save(path + "%s_waterfall" % uid, watf)

    return watf


def plot_waterfallc(
    wat,
    qindex=1,
    aspect=None,
    vmax=None,
    vmin=None,
    interpolation="none",
    save=False,
    return_fig=False,
    cmap="viridis",
    *argv,
    **kwargs
):
    """plot waterfall for a giving compressed file

    FD: class object, the compressed file handler
    labeled_array: np.array, a ROI mask
    qindex: the index number of q, will calculate where( labeled_array == qindex)
    aspect: the aspect ratio of the plot

    Return waterfall
    Plot the waterfall

    """
    # wat = cal_waterfallc( FD, labeled_array, qindex=qindex)
    if RUN_GUI:
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
    else:
        fig, ax = plt.subplots(figsize=(8, 6))

    if "uid" in kwargs:
        uid = kwargs["uid"]
    else:
        uid = "uid"
    # fig, ax = plt.subplots(figsize=(8,6))
    ax.set_ylabel("Pixel")
    ax.set_xlabel("Frame")
    ax.set_title("%s_Waterfall_Plot_@qind=%s" % (uid, qindex))
    if "beg" in kwargs:
        beg = kwargs["beg"]
    else:
        beg = 0
    extent = [beg, len(wat) + beg, 0, len(wat.T)]
    if vmax is None:
        vmax = wat.max()
    if vmin is None:
        vmin = wat.min()
    if aspect is None:
        aspect = wat.shape[0] / wat.shape[1]
    im = imshow(ax, wat.T, cmap=cmap, vmax=vmax, extent=extent, interpolation=interpolation)
    # im = ax.imshow(wat.T, cmap='viridis', vmax=vmax,extent= extent,interpolation = interpolation )
    fig.colorbar(im)
    ax.set_aspect(aspect)

    if save:
        # dt =datetime.now()
        # CurTime = '%s%02d%02d-%02d%02d-' % (dt.year, dt.month, dt.day,dt.hour,dt.minute)
        path = kwargs["path"]

        # fp = path + "uid= %s--Waterfall-"%uid + CurTime + '.png'
        fp = path + "%s_waterfall" % uid + ".png"
        plt.savefig(fp, dpi=fig.dpi)

    # plt.show()
    if return_fig:
        return fig, ax, im


def get_waterfallc(FD, labeled_array, qindex=1, aspect=1.0, vmax=None, save=False, *argv, **kwargs):
    """plot waterfall for a giving compressed file

    FD: class object, the compressed file handler
    labeled_array: np.array, a ROI mask
    qindex: the index number of q, will calculate where( labeled_array == qindex)
    aspect: the aspect ratio of the plot

    Return waterfall
    Plot the waterfall
    """

    wat = cal_waterfallc(FD, labeled_array, qindex=qindex)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_ylabel("Pixel")
    ax.set_xlabel("Frame")
    ax.set_title("Waterfall_Plot_@qind=%s" % qindex)

    im = ax.imshow(wat.T, cmap="viridis", vmax=vmax)
    fig.colorbar(im)
    ax.set_aspect(aspect)

    if save:
        # dt =datetime.now()
        # CurTime = '%s%02d%02d-%02d%02d-' % (dt.year, dt.month, dt.day,dt.hour,dt.minute)
        path = kwargs["path"]
        if "uid" in kwargs:
            uid = kwargs["uid"]
        else:
            uid = "uid"
        # fp = path + "uid= %s--Waterfall-"%uid + CurTime + '.png'
        fp = path + "uid=%s--Waterfall-" % uid + ".png"
        fig.savefig(fp, dpi=fig.dpi)

    # plt.show()
    return wat


def cal_each_ring_mean_intensityc(FD, ring_mask, sampling=1, timeperframe=None, multi_cor=False, *argv, **kwargs):
    """
    get time dependent mean intensity of each ring
    """

    mean_int_sets, index_list = mean_intensityc(FD, ring_mask, sampling, index=None, multi_cor=multi_cor)
    if timeperframe is None:
        times = np.arange(FD.end - FD.beg) + FD.beg  # get the time for each frame
    else:
        times = (FD.beg + np.arange(FD.end - FD.beg)) * timeperframe
    num_rings = len(np.unique(ring_mask)[1:])
    return times, mean_int_sets


def plot_each_ring_mean_intensityc(times, mean_int_sets, xlabel="Frame", save=False, *argv, **kwargs):
    """
    Plot time dependent mean intensity of each ring
    """
    num_rings = mean_int_sets.shape[1]

    fig, ax = plt.subplots(figsize=(8, 8))
    uid = "uid"
    if "uid" in kwargs.keys():
        uid = kwargs["uid"]
    ax.set_title("%s--Mean intensity of each ROI" % uid)
    for i in range(num_rings):
        # print(  markers[i],  colors[i] )
        ax.plot(times, mean_int_sets[:, i], label="ROI " + str(i + 1), marker=markers[i], color=colors[i], ls="-")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Mean Intensity")
    ax.legend(loc="best", fontsize="x-small", fancybox=True, framealpha=0.5)

    if save:
        path = kwargs["path"]
        fp = path + "%s_t_ROIs" % uid + ".png"
        fig.savefig(fp, dpi=fig.dpi)
        save_arrays(
            np.hstack([times.reshape(len(times), 1), mean_int_sets]),
            label=["frame"] + ["ROI_%d" % i for i in range(num_rings)],
            filename="%s_t_ROIs" % uid,
            path=path,
        )
    # plt.show()


def get_each_ring_mean_intensityc(
    FD, ring_mask, sampling=1, timeperframe=None, plot_=False, save=False, *argv, **kwargs
):
    """
    get time dependent mean intensity of each ring
    """

    mean_int_sets, index_list = mean_intensityc(FD, ring_mask, sampling, index=None)
    if timeperframe is None:
        times = np.arange(FD.end - FD.beg) + FD.beg  # get the time for each frame
    else:
        times = (FD.beg + np.arange(FD.end - FD.beg)) * timeperframe
    num_rings = len(np.unique(ring_mask)[1:])

    if plot_:
        fig, ax = plt.subplots(figsize=(8, 8))
        uid = "uid"
        if "uid" in kwargs.keys():
            uid = kwargs["uid"]

        ax.set_title("%s--Mean intensity of each ROI" % uid)
        for i in range(num_rings):
            ax.plot(times, mean_int_sets[:, i], label="ROI " + str(i + 1), marker="o", ls="-")
            if timeperframe is not None:
                ax.set_xlabel("Time, sec")
            else:
                ax.set_xlabel("Frame")
            ax.set_ylabel("Mean Intensity")
        ax.legend(loc="best", fontsize="x-small")

        if save:
            # dt =datetime.now()
            # CurTime = '%s%02d%02d-%02d%02d-' % (dt.year, dt.month, dt.day,dt.hour,dt.minute)
            path = kwargs["path"]
            # fp = path + "uid= %s--Mean intensity of each ring-"%uid + CurTime + '.png'
            fp = path + "%s_Mean_intensity_of_each_ROI" % uid + ".png"
            fig.savefig(fp, dpi=fig.dpi)

            save_arrays(
                np.hstack([times.reshape(len(times), 1), mean_int_sets]),
                label=["frame"] + ["ROI_%d" % i for i in range(num_rings)],
                filename="%s_t_ROIs" % uid,
                path=path,
            )

        # plt.show()

    return times, mean_int_sets
