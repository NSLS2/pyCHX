"""
X-ray speckle visibility spectroscopy(XSVS) - Dynamic information of
the speckle patterns are obtained by analyzing the speckle statistics
and calculating the speckle contrast in single scattering patterns.
This module will provide XSVS analysis tools
"""

from __future__ import absolute_import, division, print_function

import logging
import time

import six
from skbeam.core import roi
from skbeam.core.utils import bin_edges_to_centers, geometric_series

logger = logging.getLogger(__name__)

import itertools
import os
import sys
from datetime import datetime
from multiprocessing import Pool

import dill
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.stats as st
from matplotlib.colors import LogNorm
from scipy.optimize import leastsq, minimize
from tqdm import tqdm

from pyCHX.chx_compress import apply_async, go_through_FD, map_async, pass_FD, run_dill_encoded
from pyCHX.chx_generic_functions import trans_data_to_pd


def xsvsp(
    FD,
    label_array,
    only_two_levels=True,
    only_first_level=False,
    timebin_num=2,
    time_bin=None,
    max_cts=None,
    bad_images=None,
    threshold=1e8,
    imgsum=None,
    norm=None,
):
    """
    FD: a list of FD or a single FD
    See other parameters in xsvsc funcs
    """

    if not isinstance(FD, list):
        bin_edges, prob_k, prob_k_std_dev, his_sum = xsvsp_single(
            FD,
            label_array,
            only_two_levels,
            only_first_level,
            timebin_num,
            time_bin,
            max_cts,
            bad_images,
            threshold,
            imgsum,
            norm,
        )
    else:
        bin_edges, prob_k, prob_k_std_dev, his_sum = xsvsp_multi(
            FD,
            label_array,
            only_two_levels,
            only_first_level,
            timebin_num,
            time_bin,
            max_cts,
            bad_images,
            threshold,
            imgsum,
            norm,
        )

    return bin_edges, prob_k, prob_k_std_dev, his_sum


def xsvsp_multi(
    FD_set,
    label_array,
    only_two_levels=True,
    only_first_level=False,
    timebin_num=2,
    time_bin=None,
    max_cts=None,
    bad_images=None,
    threshold=1e8,
    imgsum=None,
    norm=None,
):
    """
    FD_set: a list of FD
    See other parameters in xsvsc funcs
    """
    N = len(FD_set)
    for n in range(N):
        bin_edges, prob_k, prob_k_std_dev, his_sum = xsvsp_single(
            FD_set[n],
            label_array,
            only_two_levels,
            only_first_level,
            timebin_num,
            time_bin,
            max_cts,
            bad_images,
            threshold,
            imgsum,
            norm,
        )
        if n == 0:
            prob_k_all = prob_k
            his_sum_all = his_sum
            prob_k_std_dev_all = prob_k_std_dev
        else:
            prob_k_all += (prob_k - prob_k_all) / (n + 1)
            his_sum_all += (his_sum - his_sum_all) / (n + 1)
            prob_k_std_dev_all += (prob_k_std_dev - prob_k_std_dev_all) / (n + 1)

    return bin_edges, prob_k_all, prob_k_std_dev_all, his_sum_all


def xsvsp_single(
    FD,
    label_array,
    only_two_levels=True,
    only_first_level=False,
    timebin_num=2,
    time_bin=None,
    max_cts=None,
    bad_images=None,
    threshold=1e8,
    imgsum=None,
    norm=None,
):
    """
    Calculate probability density of detecting photons using parallel algorithm
    """

    noframes = FD.end - FD.beg + 1  # number of frames, not "no frames"
    number_of_img = noframes
    for i in range(FD.beg, FD.end):
        pass_FD(FD, i)
    label_arrays = [np.array(label_array == i, dtype=np.int64) for i in np.unique(label_array)[1:]]
    qind, pixelist = roi.extract_label_indices(label_array)
    if norm is not None:
        norms = [
            norm[np.in1d(pixelist, extract_label_indices(np.array(label_array == i, dtype=np.int64))[1])]
            for i in np.unique(label_array)[1:]
        ]

    inputs = range(len(label_arrays))

    pool = Pool(processes=len(inputs))
    print("Starting assign the tasks...")
    results = {}
    progress_bar = False
    if norm is not None:
        for i in tqdm(inputs):
            results[i] = apply_async(
                pool,
                xsvsc_single,
                (
                    FD,
                    label_arrays[i],
                    only_two_levels,
                    only_first_level,
                    timebin_num,
                    time_bin,
                    max_cts,
                    bad_images,
                    threshold,
                    imgsum,
                    norms[i],
                    progress_bar,
                ),
            )
    else:
        for i in tqdm(inputs):
            results[i] = apply_async(
                pool,
                xsvsc_single,
                (
                    FD,
                    label_arrays[i],
                    only_two_levels,
                    only_first_level,
                    timebin_num,
                    time_bin,
                    max_cts,
                    bad_images,
                    threshold,
                    imgsum,
                    None,
                    progress_bar,
                ),
            )
    pool.close()

    print("Starting running the tasks...")
    res = [results[k].get() for k in tqdm(list(sorted(results.keys())))]

    u_labels = list(np.unique(qind))
    num_roi = len(u_labels)
    if time_bin is None:
        time_bin = geometric_series(timebin_num, number_of_img)
    if only_first_level:
        time_bin = [1]
    elif only_two_levels:
        time_bin = [1, 2]
    # print(time_bin)
    # number of times in the time bin
    num_times = len(time_bin)
    prob_k = np.zeros([num_times, num_roi], dtype=np.object)
    prob_k_std_dev = np.zeros_like(prob_k)
    his_sum = np.zeros([num_times, num_roi])
    # print( len(res) )
    # print( prob_k.shape )

    for i in inputs:
        # print( i)
        bin_edges, prob_k[:, i], prob_k_std_dev[:, i], his_sum[:, i] = (
            res[i][0],
            res[i][1][:, 0],
            res[i][2][:, 0],
            res[i][3][:, 0],
        )

    print("Histogram calculation DONE!")
    del results
    del res
    return bin_edges, prob_k, prob_k_std_dev, his_sum


def xsvsc(
    FD,
    label_array,
    only_two_levels=True,
    only_first_level=False,
    timebin_num=2,
    time_bin=None,
    max_cts=None,
    bad_images=None,
    threshold=1e8,
    imgsum=None,
    norm=None,
):
    """
    FD: a list of FD or a single FD
    See other parameters in xsvsc funcs
    """

    if not isinstance(FD, list):
        bin_edges, prob_k, prob_k_std_dev, his_sum = xsvsc_single(
            FD,
            label_array,
            only_two_levels,
            only_first_level,
            timebin_num,
            time_bin,
            max_cts,
            bad_images,
            threshold,
            imgsum,
            norm,
        )
    else:
        bin_edges, prob_k, prob_k_std_dev, his_sum = xsvsc_multi(
            FD,
            label_array,
            only_two_levels,
            only_first_level,
            timebin_num,
            time_bin,
            max_cts,
            bad_images,
            threshold,
            imgsum,
            norm,
        )

    return bin_edges, prob_k, prob_k_std_dev, his_sum


def xsvsc_multi(
    FD_set,
    label_array,
    only_two_levels=True,
    only_first_level=False,
    timebin_num=2,
    time_bin=None,
    max_cts=None,
    bad_images=None,
    threshold=1e8,
    imgsum=None,
    norm=None,
):
    """
    FD_set: a list of FD
    See other parameters in xsvsc funcs
    """
    N = len(FD_set)
    for n in range(N):
        bin_edges, prob_k, prob_k_std_dev, his_sum = xsvsc_single(
            FD_set[n],
            label_array,
            only_two_levels,
            only_first_level,
            timebin_num,
            time_bin,
            max_cts,
            bad_images,
            threshold,
            imgsum,
            norm,
        )
        if n == 0:
            prob_k_all = prob_k
            prob_k_std_dev_all = prob_k_std_dev
            his_sum_all = his_sum
        else:
            prob_k_all += (prob_k - prob_k_all) / (n + 1)
            his_sum_all += (his_sum - his_sum_all) / (n + 1)
            prob_k_std_dev_all += (prob_k_std_dev - prob_k_std_dev_all) / (n + 1)
    return bin_edges, prob_k_all, prob_k_std_dev_all, his_sum_all


def xsvsc_single(
    FD,
    label_array,
    only_two_levels=True,
    only_first_level=False,
    timebin_num=2,
    time_bin=None,
    max_cts=None,
    bad_images=None,
    threshold=1e8,
    imgsum=None,
    norm=None,
    progress_bar=True,
):
    """YG MOD@Octo 12, 2017, Change photon statistic error bar from sampling statistic bar to error bar with phisical meaning,
    photon_number@one_particular_count = photon_tolal_number * photon_distribution@one_particular_count +/-
                                                  sqrt( photon_number@one_particular_count )


    This function will provide the probability density of detecting photons
    for different integration times.
    The experimental probability density P(K) of detecting photons K is
    obtained by histogramming the speckle counts over an ensemble of
    equivalent pixels and over a number of speckle patterns recorded
    with the same integration time T under the same condition.
    Parameters
    ----------
    image_iterable : FD, a compressed eiger file by Multifile class

    label_array : array
        labeled array; 0 is background.
        Each ROI is represented by a distinct label (i.e., integer).
    number_of_img : int
        number of images (how far to go with integration times when finding
        the time_bin, using skxray.utils.geometric function)
    timebin_num : int, optional
        integration time; default is 2
    max_cts : int, optional
       the brightest pixel in any ROI in any image in the image set.
       defaults to using skxray.core.roi.roi_max_counts to determine
       the brightest pixel in any of the ROIs


    bad_images: array, optional
        the bad images number list, the XSVS will not analyze the binning image groups which involve any bad images
    threshold: float, optional
        If one image involves a pixel with intensity above threshold, such image will be considered as a bad image.


    Returns
    -------
    prob_k_all : array
        probability density of detecting photons
    prob_k_std_dev : array
        standard deviation of probability density of detecting photons
    Notes
    -----
    These implementation is based on following references
    References: text [1]_, text [2]_
    .. [1] L. Li, P. Kwasniewski, D. Oris, L Wiegart, L. Cristofolini,
       C. Carona and A. Fluerasu , "Photon statistics and speckle visibility
       spectroscopy with partially coherent x-rays" J. Synchrotron Rad.,
       vol 21, p 1288-1295, 2014.
    .. [2] R. Bandyopadhyay, A. S. Gittings, S. S. Suh, P.K. Dixon and
       D.J. Durian "Speckle-visibilty Spectroscopy: A tool to study
       time-varying dynamics" Rev. Sci. Instrum. vol 76, p  093110, 2005.
    There is an example in https://github.com/scikit-xray/scikit-xray-examples
    It will demonstrate the use of these functions in this module for
    experimental data.
    """

    label_array = np.int_(label_array)
    if max_cts is None:
        max_cts = roi.roi_max_counts(FD, label_array)
    # find the label's and pixel indices for ROI's
    labels, indices = roi.extract_label_indices(label_array)
    nopixels = len(indices)
    # number of ROI's
    u_labels = list(np.unique(labels))
    num_roi = len(u_labels)
    # create integration times
    noframes = FD.end - FD.beg + 1
    number_of_img = noframes
    if time_bin is None:
        time_bin = geometric_series(timebin_num, noframes)
    if only_first_level:
        time_bin = [1]
    elif only_two_levels:
        time_bin = [1, 2]
    # print(time_bin)
    # number of times in the time bin
    num_times = len(time_bin)
    # number of pixels per ROI
    num_pixels = np.bincount(labels, minlength=(num_roi + 1))[1:]
    # probability density of detecting photons
    prob_k = np.zeros([num_times, num_roi], dtype=np.object)
    his_sum = np.zeros([num_times, num_roi])
    # square of probability density of detecting photons
    prob_k_pow = np.zeros_like(prob_k)
    # standard deviation of probability density of detecting photons
    prob_k_std_dev = np.zeros_like(prob_k)
    # get the bin edges for each time bin for each ROI
    bin_edges = np.zeros(prob_k.shape[0], dtype=prob_k.dtype)
    for i in range(num_times):
        bin_edges[i] = np.arange(max_cts * timebin_num**i)
    # start_time = time.time()  # used to log the computation time (optionally)
    bad_frame_list = bad_images
    if bad_frame_list is None:
        bad_frame_list = []
    pixelist = indices
    fra_pix = np.zeros_like(pixelist, dtype=np.float64)

    timg = np.zeros(FD.md["ncols"] * FD.md["nrows"], dtype=np.int32)
    timg[pixelist] = np.arange(1, len(pixelist) + 1)
    # print( n, FD, num_times, timebin_num, nopixels )
    buf = np.zeros([num_times, timebin_num, nopixels], dtype=np.float64)
    # to track processing each time level
    track_level = np.zeros(num_times)  # ,  dtype=bool)
    track_bad_level = np.zeros(num_times)  # , dtype=np.int32   )
    # to increment buffer
    cur = np.int_(np.full(num_times, timebin_num, dtype=np.int64))
    # cur =  np.full(num_times, timebin_num)
    # to track how many images processed in each level
    img_per_level = np.zeros(num_times, dtype=np.int64)

    if progress_bar:
        xr = tqdm(range(FD.beg, FD.end))
    else:
        xr = range(FD.beg, FD.end)

    for i in xr:
        # Ring buffer, a buffer with periodic boundary conditions.
        # Images must be keep for up to maximum delay in buf.
        # buf = np.zeros([num_times, timebin_num], dtype=np.object)  # matrix of buffers
        if i in bad_frame_list:
            fra_pix[:] = np.nan
            # print( 'here is a bad frmae--%i'%i )
        else:
            fra_pix[:] = 0
            (p, v) = FD.rdrawframe(i)
            w = np.where(timg[p])[0]
            pxlist = timg[p[w]] - 1
            if imgsum is None:
                if norm is None:
                    fra_pix[pxlist] = v[w]
                else:
                    fra_pix[pxlist] = v[w] / norm[pxlist]  # -1.0
            else:
                if norm is None:
                    fra_pix[pxlist] = v[w] / imgsum[i]
                else:
                    fra_pix[pxlist] = v[w] / imgsum[i] / norm[pxlist]
        # level =0
        cur[0] = 1 + cur[0] % timebin_num
        # read each frame
        # Put the image into the ring buffer.
        img_ = fra_pix
        # Put the ROI pixels into the ring buffer.
        # fra_pix[:]=0
        if threshold is not None:
            if img_.max() >= threshold:
                print("bad image: %s here!" % n)
                img_[:] = np.nan
        buf[0, cur[0] - 1] = img_
        _process(
            num_roi,
            0,
            cur[0] - 1,
            buf,
            img_per_level,
            labels,
            max_cts,
            bin_edges[0],
            prob_k,
            prob_k_pow,
            track_bad_level,
        )
        # check whether the number of levels is one, otherwise
        # continue processing the next level
        level = 1
        if number_of_img > 1:
            processing = 1
        else:
            processing = 0
        if only_first_level:
            processing = 0

        while processing:
            # print( 'here')
            if track_level[level]:
                prev = 1 + (cur[level - 1] - 2) % timebin_num
                cur[level] = 1 + cur[level] % timebin_num
                bufa = buf[level - 1, prev - 1]
                bufb = buf[level - 1, cur[level - 1] - 1]
                buf[level, cur[level] - 1] = bufa + bufb
                # print(  buf[level, cur[level]-1]  )
                track_level[level] = 0
                _process(
                    num_roi,
                    level,
                    cur[level] - 1,
                    buf,
                    img_per_level,
                    labels,
                    max_cts,
                    bin_edges[level],
                    prob_k,
                    prob_k_pow,
                    track_bad_level,
                )
                level += 1
                if level < num_times:
                    processing = 1
                else:
                    processing = 0
            else:
                track_level[level] = 1
                processing = 0
            # print( level )
        # prob_k_std_dev = np.power((prob_k_pow -
        #                       np.power(prob_k, 2)), .5)

    for i in range(num_times):
        for j in range(num_roi):
            # print( prob_k[i,j] )
            his_sum[i, j] = np.sum(np.array(prob_k[i, j])) / 1.0
            # print(his_sum[i,j])
            prob_k_std_dev[i, j] = np.sqrt(prob_k[i, j]) / his_sum[i, j]
            prob_k[i, j] = prob_k[i, j] / his_sum[i, j]

    # for i in range(num_times):
    #    if  isinstance(prob_k[i,0], float ) or isinstance(prob_k[i,0], int ):
    #        pass

    return bin_edges, prob_k, prob_k_std_dev, his_sum


def _process(
    num_roi, level, buf_no, buf, img_per_level, labels, max_cts, bin_edges, prob_k, prob_k_pow, track_bad_level
):
    """
    Internal helper function. This modifies inputs in place.
    This helper function calculate probability of detecting photons for
    each integration time.
    .. warning :: This function mutates the input values.
    Parameters
    ----------
    num_roi : int
        number of ROI's
    level : int
        current time level(integration time)
    buf_no : int
        current buffer number
    buf : array
        image data array to use for XSVS
    img_per_level : int
        to track how many images processed in each level
    labels : array
        labels of the required region of interests(ROI's)
    max_cts: int
        maximum pixel count
    bin_edges : array
        bin edges for each integration times and each ROI
    prob_k : array
        probability density of detecting photons
    prob_k_pow : array
        squares of probability density of detecting photons
    """
    img_per_level[level] += 1
    data = buf[level, buf_no]
    if np.isnan(data).any():
        track_bad_level[level] += 1
    # print (img_per_level,track_bad_level)
    u_labels = list(np.unique(labels))
    ##############
    ##To Do list here, change histogram to bincount
    ##Change error bar calculation
    if not (np.isnan(data).any()):
        for j, label in enumerate(u_labels):
            roi_data = data[labels == label]
            # print(  np.max( bin_edges), prob_k[level, j]  )
            # print( level, j, bin_edges )
            spe_hist = np.bincount(np.int_(roi_data), minlength=np.max(bin_edges))
            # spe_hist, bin_edges = np.histogram(roi_data, bins=bin_edges,  density=True)
            spe_hist = np.nan_to_num(spe_hist)
            # print( spe_hist.shape )
            # prob_k[level, j] += (spe_hist -
            #                 prob_k[level, j])/( img_per_level[level] - track_bad_level[level] )
            # print(  prob_k[level, j] )
            prob_k[level, j] += spe_hist
            # print( spe_hist.shape, prob_k[level, j] )
            # prob_k_pow[level, j] += (np.power(spe_hist, 2) -
            #                     prob_k_pow[level, j])/(img_per_level[level] - track_bad_level[level])


def normalize_bin_edges(num_times, num_rois, mean_roi, max_cts):
    """
    This will provide the normalized bin edges and bin centers for each
    integration time.
    Parameters
    ----------
    num_times : int
        number of integration times for XSVS
    num_rois : int
        number of ROI's
    mean_roi : array
        mean intensity of each ROI
        shape (number of ROI's)
    max_cts : int
        maximum pixel counts
    Returns
    -------
    norm_bin_edges : array
        normalized speckle count bin edges
         shape (num_times, num_rois)
    norm_bin_centers :array
        normalized speckle count bin centers
        shape (num_times, num_rois)
    """
    norm_bin_edges = np.zeros((num_times, num_rois), dtype=object)
    norm_bin_centers = np.zeros_like(norm_bin_edges)
    for i in range(num_times):
        for j in range(num_rois):
            norm_bin_edges[i, j] = np.arange(max_cts * 2**i) / (mean_roi[j] * 2**i)
            norm_bin_centers[i, j] = bin_edges_to_centers(norm_bin_edges[i, j])

    return norm_bin_edges, norm_bin_centers


def get_his_std_qi(data_pixel_qi, max_cts=None):
    """
    YG. Dev 16, 2016
    Calculate the photon histogram for one q by giving
    Parameters:
        data_pixel_qi: one-D array, for the photon counts
        max_cts: for bin max, bin will be [0,1,2,..., max_cts]
    Return:
        bins
        his
        std
    """
    if max_cts is None:
        max_cts = np.max(data_pixel_qi) + 1
    bins = np.arange(max_cts)
    dqn, dqm = data_pixel_qi.shape
    # get histogram here
    H = np.apply_along_axis(np.bincount, 1, np.int_(data_pixel_qi), minlength=max_cts) / dqm
    # do average for different frame
    his = np.average(H, axis=0)
    std = np.std(H, axis=0)
    # cal average photon counts
    kmean = np.average(data_pixel_qi)
    return bins, his, std, kmean


def get_his_std(data_pixel, rois, max_cts=None):
    """
    YG. Dev 16, 2016
    Calculate the photon histogram for multi-q by giving
    Parameters:
        data_pixel: multi-D array, for the photon counts
        max_cts: for bin max, bin will be [0,1,2,..., max_cts]
    Return:
        bins
        his
        std
    """
    if max_cts is None:
        max_cts = np.max(data_pixel) + 1
    qind, pixelist = roi.extract_label_indices(rois)
    noqs = len(np.unique(qind))
    his = np.zeros([noqs], dtype=np.object)
    std = np.zeros_like(his, dtype=np.object)
    kmean = np.zeros_like(his, dtype=np.object)
    for qi in range(noqs):
        pixelist_qi = np.where(qind == qi + 1)[0]
        # print(qi, max_cts)
        bins, his[qi], std[qi], kmean[qi] = get_his_std_qi(data_pixel[:, pixelist_qi], max_cts)
    return bins, his, std, kmean


def save_bin_his_std(spec_bins, spec_his, spec_std, filename, path):
    """YG. Dec 23, 2016
    save spec_bin, spec_his, spec_std to a csv file
    """
    ql = spec_his.shape[1]
    tl = spec_bins[-1].shape[0] - 1
    # print( tl, ql)
    mhis, nhis = spec_his.shape
    mstd, nstd = spec_std.shape
    spec_data = np.zeros([tl, 1 + ql * (mhis + mstd)])
    spec_data[:, :] = np.nan
    spec_data[:, 0] = spec_bins[-1][:-1]
    for i in range(mhis):
        max_m = spec_his[i, 0].shape[0]
        for j in range(nhis):
            spec_data[:max_m, 1 + i * ql + j] = spec_his[i, j]
    for i in range(mstd):
        max_m = spec_std[i, 0].shape[0]
        for j in range(nstd):
            spec_data[:max_m, 1 + mhis * ql + i * ql + j] = spec_std[i, j]
    label = ["count"]
    for l in range(mhis):
        for q in range(nhis):
            label += ["his_level_%s_q_%s" % (l, q)]
    for l in range(mstd):
        for q in range(nstd):
            label += ["std_level_%s_q_%s" % (l, q)]
    spec_pds = trans_data_to_pd(spec_data, label, "array")
    filename_ = os.path.join(path, filename)
    spec_pds.to_csv(filename_)
    print("The file: %s is saved in %s" % (filename, path))
    return spec_pds


def reshape_array(array, new_len):
    """
    array: shape= [M,N]
    new_len: the length of the new array, the reshaped shape will be [M//new_len, new_len, N]

    """
    M, N = array.shape
    m = M // new_len
    return array[: m * new_len, :].reshape([m, new_len, N])


def get_binned_his_std_qi(data_pixel_qi, lag_steps, max_cts=None):
    """
    YG. Dev 16, 2016
    Calculate the binned photon histogram for one q by giving
    Parameters:
        data_pixel_qi: one-D array, for the photon counts
        lag_steps: the binned number
        max_cts: for bin max, bin will be [0,1,2,..., max_cts]
    Return:
        bins
        his
        std
    """
    if max_cts is None:
        max_cts = np.max(data_pixel_qi) + 1
    lag_steps = np.array(lag_steps)
    lag_steps = lag_steps[np.nonzero(lag_steps)]
    nologs = len(lag_steps)
    his = np.zeros([nologs], dtype=np.object)
    bins = np.zeros_like(his, dtype=np.object)
    std = np.zeros_like(his, dtype=np.object)
    kmean = np.zeros_like(his, dtype=np.object)
    i = 0
    for lag in lag_steps:
        data_pixel_qi_ = np.sum(reshape_array(data_pixel_qi, lag), axis=1)
        bins[i], his[i], std[i], kmean[i] = get_his_std_qi(data_pixel_qi_, max_cts * lag)
        i += 1
    return bins, his, std, kmean


def get_binned_his_std(data_pixel, rois, lag_steps, max_cts=None):
    """
    YG. Dev 16, 2016
    Calculate the binned photon histogram qs by giving
    Parameters:
        data_pixel: one-D array, for the photon counts
        lag_steps: the binned number
        max_cts: for bin max, bin will be [0,1,2,..., max_cts]
    Return:
        bins
        his
        std
    """
    if max_cts is None:
        max_cts = np.max(data_pixel) + 1
    qind, pixelist = roi.extract_label_indices(rois)
    noqs = len(np.unique(qind))

    lag_steps = np.array(lag_steps)
    lag_steps = lag_steps[np.nonzero(lag_steps)]

    nologs = len(lag_steps)
    his = np.zeros([nologs, noqs], dtype=np.object)
    bins = np.zeros([nologs], dtype=np.object)
    std = np.zeros_like(his, dtype=np.object)
    kmean = np.zeros_like(his, dtype=np.object)
    i = 0
    for lag in tqdm(lag_steps):
        data_pixel_ = np.sum(reshape_array(data_pixel, lag), axis=1)
        # print( data_pixel_.shape)
        for qi in range(noqs):
            pixelist_qi = np.where(qind == qi + 1)[0]
            bins[i], his[i, qi], std[i, qi], kmean[i, qi] = get_his_std_qi(
                data_pixel_[:, pixelist_qi], max_cts * lag
            )
        i += 1

    return bins, his, std, kmean


def get_bin_edges(num_times, num_rois, mean_roi, max_cts):
    """
    This will provide the normalized bin edges and bin centers for each
    integration time.
    Parameters
    ----------
    num_times : int
        number of integration times for XSVS
    num_rois : int
        number of ROI's
    mean_roi : array
        mean intensity of each ROI
        shape (number of ROI's)
    max_cts : int
        maximum pixel counts
    Returns
    -------
    norm_bin_edges : array
        normalized speckle count bin edges
         shape (num_times, num_rois)
    norm_bin_centers :array
        normalized speckle count bin centers
        shape (num_times, num_rois)
    """
    norm_bin_edges = np.zeros((num_times, num_rois), dtype=object)
    norm_bin_centers = np.zeros_like(norm_bin_edges)

    bin_edges = np.zeros((num_times, num_rois), dtype=object)
    bin_centers = np.zeros_like(bin_edges)

    for i in range(num_times):
        for j in range(num_rois):
            bin_edges[i, j] = np.arange(max_cts * 2**i)
            bin_centers[i, j] = bin_edges_to_centers(bin_edges[i, j])
            norm_bin_edges[i, j] = bin_edges[i, j] / (mean_roi[j] * 2**i)
            norm_bin_centers[i, j] = bin_edges_to_centers(norm_bin_edges[i, j])

    return bin_edges, bin_centers, norm_bin_edges, norm_bin_centers


#################
##for fit
###################

from scipy import stats
from scipy.special import gamma, gammaln

###########################3
##Dev at Nov 18, 2016
#


def nbinomres_old(p, hist, x, hist_err=None, N=1):
    """residuals to leastsq() to fit normal chi-square"""
    mu, M = p
    Np = N * st.nbinom.pmf(x, M, 1.0 / (1.0 + mu / M))
    err = (hist - Np) / np.sqrt(Np)
    return err


def nbinomlog_old(p, hist, x, hist_err=None, N=1):
    """Residuals for maximum likelihood fit to nbinom distribution.
    Vary M (shape param) and mu (count rate) vary (using leastsq)"""
    mu, M = p
    mu = abs(mu)
    M = abs(M)
    w = np.where(hist > 0.0)
    Np = N * st.nbinom.pmf(x, M, 1.0 / (1.0 + mu / M))
    err = 2 * (Np - hist)
    err = err[w] - 2 * hist[w] * np.log(Np[w] / hist[w])  # note: sum(Np-hist)==0
    return np.sqrt(np.abs(err))
    # return err


def nbinomlog1_old(p, hist, x, hist_err=None, N=1, mu=1):
    """Residuals for maximum likelihood fit to nbinom distribution.
    Vary M (shape param) but mu (count rate) fixed (using leastsq)"""
    M = abs(p[0])
    w = np.where(hist > 0.0)
    Np = N * st.nbinom.pmf(x, M, 1.0 / (1.0 + mu / M))
    err = 2 * (Np - hist)
    err[w] = err[w] - 2 * hist[w] * np.log(Np[w] / hist[w])  # note: sum(Np-hist)==0
    return np.sqrt(np.abs(err))


def nbinomres(p, hist, x, hist_err=None, N=1):
    """residuals to leastsq() to fit normal chi-square"""
    mu, M = p
    Np = N * st.nbinom.pmf(x, M, 1.0 / (1.0 + mu / M))
    w = np.where(hist > 0.0)
    if hist_err is None:
        hist_err = np.ones_like(Np)
    scale = np.sqrt(Np[w] / hist_err[w])
    # scale = 1
    err = (hist[w] - Np[w]) / scale
    return err


###########
##Dev at Octo 12, 2017


def nbinom(p, x, mu):
    """Residuals for maximum likelihood fit to nbinom distribution.
    Vary M (shape param) but mu (count rate) fixed (using leastsq)"""
    M = abs(p[0])
    Np = st.nbinom.pmf(x, M, 1.0 / (1.0 + mu / M))
    return Np


def nbinomlog1(p, hist, x, N, mu):
    """Residuals for maximum likelihood fit to nbinom distribution.
    Vary M (shape param) but mu (count rate) fixed (using leastsq)

    p: fitting parameter, in this case is M, coherent mode number
    hist: histogram of photon count for each bin (is a number not probablity)
    x: photon count
    N: total photons count in the statistics, ( probablity = hist / N )
    mu: average photon count for each bin

    """
    M = abs(p[0])
    w = np.where(hist > 0.0)
    Np = N * st.nbinom.pmf(x, M, 1.0 / (1.0 + mu / M))
    err = 2 * (Np[w] - hist[w])
    err = err - 2 * hist[w] * np.log(Np[w] / hist[w])  # note: sum(Np-hist)==0
    return np.sqrt(np.abs(err))


def nbinomlog(p, hist, x, N):
    """Residuals for maximum likelihood fit to nbinom distribution.
    Vary M (shape param) and mu (count rate) vary (using leastsq)"""
    mu, M = p
    mu = abs(mu)
    M = abs(M)
    w = np.where(hist > 0.0)
    Np = N * st.nbinom.pmf(x, M, 1.0 / (1.0 + mu / M))
    err = 2 * (Np[w] - hist[w])
    err = err - 2 * hist[w] * np.log(Np[w] / hist[w])  # note: sum(Np-hist)==0
    return np.sqrt(np.abs(err))


def get_roi(data, threshold=1e-3):
    ind = np.where(data > threshold)[0]
    if len(ind) > len(data) - 3:
        ind = (np.array(ind[:-3]),)
    elif len(ind) < 3:
        ind = np.where(data >= 0)[0]
    return ind


def get_xsvs_fit(
    spe_cts_all,
    spec_sum,
    K_mean,
    spec_std=None,
    spec_bins=None,
    lag_steps=None,
    varyK=True,
    qth=None,
    max_bins=None,
    min_cutoff_count=None,
    g2=None,
    times=None,
    taus=None,
    fit_range=None,
):
    """
    Fit the xsvs by Negative Binomial Function using max-likelihood chi-squares
    """

    max_cts = spe_cts_all[0][0].shape[0] - 1
    num_times, num_rings = spe_cts_all.shape
    if max_bins is not None:
        num_times = min(num_times, max_bins)
    if spec_bins is None:
        # print(   num_times, num_rings, K_mean[0], int(max_cts+2)  )
        bin_edges, bin_centers, Knorm_bin_edges, Knorm_bin_centers = get_bin_edges(
            num_times, num_rings, K_mean[0], int(max_cts + 2)
        )

    else:
        bin_edges = spec_bins
        if lag_steps is None:
            print("Please give lag_steps")
            lag_steps = [1, 2]
            print("The lag_steps is changed to %s" % lag_steps)
        lag_steps = np.array(lag_steps)
        lag_steps = lag_steps[np.nonzero(lag_steps)]

    if g2 is not None:
        g2c = g2.copy()
        g2c[0] = g2[1]
    ML_val = {}
    KL_val = {}
    # K_ = np.zeros_like(  K_mean  )
    K_ = []
    if qth is not None:
        range_ = range(qth, qth + 1)
    else:
        range_ = range(num_rings)
    for i in range_:
        ML_val[i] = []
        KL_val[i] = []
        if g2 is not None:
            mi_g2 = 1 / (g2c[:, i] - 1)
            m_ = np.interp(times, taus, mi_g2)
        for j in range(num_times):
            kmean_guess = K_mean[j, i]
            N = spec_sum[j, i]
            if spec_bins is None:
                x_, x, y = bin_edges[j, i][:-1], Knorm_bin_edges[j, i][:-1], spe_cts_all[j, i]
            else:
                x_, x, y = bin_edges[j], bin_edges[j] / kmean_guess, spe_cts_all[j, i]

            if spec_std is not None:
                yerr = spec_std[j, i]
            else:
                yerr = None
            if g2 is not None:
                m0 = m_[j]
            else:
                m0 = 10
            if min_cutoff_count is not None:
                ind = get_roi(y, threshold=min_cutoff_count)
                y = y[ind]
                x_ = x_[ind]
                yerr = yerr[ind]
                # print(y)
            if fit_range is not None:
                f1, f2 = fit_range
                y, x_, yerr = y[f1:f2], x_[f1:f2], yerr[f1:f2]

            if not varyK:
                fit_func = nbinomlog1
                # print(i,j,m0,)
                # print(y,N, x_, kmean_guess)
                resultL = leastsq(
                    fit_func,
                    [m0],
                    args=(y * N, x_, N, kmean_guess),
                    ftol=1.49012e-38,
                    xtol=1.49012e-38,
                    factor=100,
                    full_output=1,
                )
                ML_val[i].append(abs(resultL[0][0]))
                KL_val[i].append(kmean_guess)  #   resultL[0][0] )
            else:
                # vary M and K
                fit_func = nbinomlog
                resultL = leastsq(
                    fit_func,
                    [kmean_guess, m0],
                    args=(y * N, x_, N),
                    ftol=1.49012e-38,
                    xtol=1.49012e-38,
                    factor=100,
                    full_output=1,
                )

                ML_val[i].append(abs(resultL[0][1]))
                KL_val[i].append(abs(resultL[0][0]))  #   resultL[0][0] )
                # print( j, m0, resultL[0][1], resultL[0][0], K_mean[i] * 2**j    )
            if j == 0:
                K_.append(KL_val[i][0])
    return ML_val, KL_val, np.array(K_)


def plot_xsvs_fit(
    spe_cts_all,
    ML_val,
    KL_val,
    K_mean,
    spec_std=None,
    xlim=[0, 15],
    vlim=[0.9, 1],
    q_ring_center=None,
    max_bins=None,
    uid="uid",
    qth=None,
    times=None,
    fontsize=3,
    path=None,
    logy=True,
    spec_bins=None,
    lag_steps=None,
    figsize=[10, 10],
):
    """
    Plot visibility with fit
    if qth is not None, should be an integer, starting from 1
    """

    # if qth is None:
    #    fig = plt.figure(figsize=(10,12))
    # else:
    #    fig = plt.figure(figsize=(8,8))

    max_cts = spe_cts_all[0][0].shape[0] - 1
    num_times, num_rings = spe_cts_all.shape

    if spec_bins is None:
        bin_edges, bin_centers, Knorm_bin_edges, Knorm_bin_centers = get_bin_edges(
            num_times, num_rings, K_mean[0], int(max_cts + 2)
        )

    else:
        bin_edges = spec_bins
        if lag_steps is None:
            # lag_steps = times[:num_times]
            print("Please give lag_steps")

        lag_steps = np.array(lag_steps)
        lag_steps = lag_steps[np.nonzero(lag_steps)]

    if qth is not None:
        range_ = range(qth - 1, qth)
        num_times = len(ML_val[qth - 1])
    else:
        range_ = range(num_rings)
        num_times = len(ML_val[0])
    # for i in range(num_rings):

    if max_bins is not None:
        num_times = min(num_times, max_bins)

    sx = int(round(np.sqrt(len(range_))))

    if len(range_) % sx == 0:
        sy = int(len(range_) / sx)
    else:
        sy = int(len(range_) / sx + 1)
    n = 1
    if qth is not None:
        fontsize = 14

    if qth is None:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure()
    title = "%s" % uid + "-NB-Fit"
    plt.title(title, fontsize=16, y=1.08)
    # plt.axes(frameon=False)
    flag = True
    if (qth is None) and (num_rings != 1):
        plt.axis("off")
        plt.xticks([])
        plt.yticks([])
        flag = False

    for i in range_:
        axes = fig.add_subplot(sx, sy, n)
        axes.set_xlabel("K/<K>")
        axes.set_ylabel("P(K)")
        n += 1
        for j in range(num_times):
            kmean_guess = K_mean[j, i]
            L = len(spe_cts_all[j, i])
            if spec_bins is None:
                max_cts_ = max_cts * 2**j
                x_, x, y = bin_edges[j, i][:L], Knorm_bin_edges[j, i][:L], spe_cts_all[j, i]
                xscale = (x_ / x)[1]  # bin_edges[j, i][:-1][1]/ Knorm_bin_edges[j, i][:-1][1]
                # print( xscale )
            else:
                max_cts_ = max_cts * lag_steps[j]
                x_, x, y = bin_edges[j][:L], bin_edges[j][:L] / kmean_guess, spe_cts_all[j, i]
                xscale = kmean_guess
            # Using the best K and M values interpolate and get more values for fitting curve

            fitx = np.linspace(0, max_cts_, 5000)
            fitx_ = fitx / xscale

            # print (j,i,kmean_guess,  xscale,   fitx.shape,KL_val[i][j], ML_val[i][j] )
            # fity = nbinom_dist( fitx, K_val[i][j], M_val[i][j] )
            fitL = nbinom_dist(fitx, KL_val[i][j], ML_val[i][j])

            if qth is None:
                ith = 0
            else:
                ith = qth

            # print( i, ith, qth )
            if i == ith:
                if times is not None:
                    label = "Data--" + str(round(times[j] * 1000, 3)) + " ms"
                else:
                    label = "Bin_%s" % (2**j)
            else:
                if qth is not None:
                    if times is not None:
                        label = "Data--" + str(round(times[j] * 1000, 3)) + " ms"
                    else:
                        label = "Bin_%s" % (2**j)
                else:
                    label = ""

            if spec_std is None:
                (art,) = axes.plot(x, y, "o", label=label)
            else:
                yerr = spec_std[j][i]
                # print(x.shape, y.shape, yerr.shape)
                axes.errorbar(x, y, yerr, marker="o", label=label)

            # if j == 0:
            if j < 2:
                label = "nbinom_L"
                txts = r"$M=%s$" % round(ML_val[i][j], 2) + "," + r"$K=%s$" % round(KL_val[i][j], 2)
                # print( ML_val[i] )
                x = 0.05
                y0 = 0.2 - j * 0.1
                if qth is None:
                    fontsize_ = fontsize * 2
                else:
                    fontsize_ = 18
                axes.text(x=x, y=y0, s=txts, fontsize=fontsize_, transform=axes.transAxes)
            else:
                label = ""
            (art,) = axes.plot(fitx_, fitL, "-r", label=label)
            if logy:
                axes.set_yscale("log")
            ny = y[np.nonzero(y)]
            axes.set_ylim(vlim[0] * ny.min(), vlim[1] * ny.max())

        axes.set_xlim(xlim)
        til = "Q=" + "%.4f  " % (q_ring_center[i]) + r"$\AA^{-1}$"
        if flag:
            til = title + "--" + til
        axes.set_title(til, fontsize=12, y=1.0)
        axes.legend(loc="best", fontsize=fontsize, fancybox=True, framealpha=0.5)

    if qth is None:
        file_name = "%s_xsvs_fit" % (uid)
    else:
        file_name = "%s_xsvs_fit_q=%s" % (uid, qth)
    fp = path + file_name + ".png"
    fig.tight_layout()
    plt.savefig(fp, dpi=fig.dpi)

    # plt.show()


def get_contrast(ML_val):
    nq, nt = len(ML_val.keys()), len(ML_val[list(ML_val.keys())[0]])
    contrast_factorL = np.zeros([nq, nt])
    for i in range(nq):
        for j in range(nt):
            contrast_factorL[i, j] = 1 / ML_val[i][j]
    return contrast_factorL


def get_K(KL_val):
    nq, nt = len(KL_val.keys()), len(KL_val[list(KL_val.keys())[0]])
    K_ = np.zeros([nq, nt])
    for i in range(nq):
        for j in range(nt):
            K_[i, j] = KL_val[i][j]
    return K_


def save_KM(K_mean, KL_val, ML_val, qs=None, level_time=None, uid=None, path=None):
    """save Kmean, K_val, M_val as dataframe"""

    import os

    from pandas import DataFrame

    kl = get_K(KL_val)
    ml = 1 / get_contrast(ML_val)
    L, n = kl.shape
    m2, m1 = K_mean.shape
    # print(L,n,m2,m1)
    if level_time is None:
        l = (
            ["K_mean_%d" % i for i in range(m2)]
            + ["K_fit_Bin_%i" % s for s in range(1, n + 1)]
            + ["M_Fit_Bin_%i" % s for s in range(1, n + 1)]
            + ["Contrast_Fit_Bin_%i" % s for s in range(1, n + 1)]
        )
    else:
        l = (
            ["K_mean_%s" % i for i in level_time]
            + ["K_fit_%s" % s for s in level_time]
            + ["M_Fit_%s" % s for s in level_time]
            + ["Contrast_Fit_%s" % s for s in level_time]
        )
    data = np.hstack([(K_mean).T, kl.reshape(L, n), ml.reshape(L, n), (1 / ml).reshape(L, n)])
    if qs is not None:
        qs = np.array(qs)
        l = ["q"] + l
        # print(   (K_mean).T,  (K_mean).T.shape )
        # print(  qs )
        data = np.hstack(
            [qs.reshape(L, 1), (K_mean).T, kl.reshape(L, n), ml.reshape(L, n), (1 / ml).reshape(L, n)]
        )

    df = DataFrame(data)
    df.columns = (x for x in l)
    filename = "%s_xsvs_fitted_KM.csv" % (uid)
    filename1 = os.path.join(path, filename)
    print("The K-M values are saved as %s in %s." % (filename, path))
    df.to_csv(filename1)
    return df


def get_his_std_from_pds(spec_pds, his_shapes=None):
    """Y.G.Dec 22, 2016
    get spec_his, spec_std from a pandas.dataframe file
    Parameters:
        spec_pds: pandas.dataframe, contains columns as 'count',
                        spec_his (as 'his_level_0_q_0'), spec_std (as 'std_level_0_q_0')
        his_shapes: the shape of the returned spec_his, if None, shapes = (2, (len(spec_pds.keys)-1)/4) )
    Return:
        spec_his: array, shape as his_shapes
        spec_std, array, shape as his_shapes
    """
    spkeys = list(spec_pds.keys())
    if his_shapes is None:
        M, N = 2, int((len(spkeys) - 1) / 4)
    # print(M,N)
    spec_his = np.zeros([M, N], dtype=np.object)
    spec_std = np.zeros([M, N], dtype=np.object)
    for i in range(M):
        for j in range(N):
            spec_his[i, j] = np.array(spec_pds[spkeys[1 + i * N + j]][~np.isnan(spec_pds[spkeys[1 + i * N + j]])])
            spec_std[i, j] = np.array(
                spec_pds[spkeys[1 + 2 * N + i * N + j]][~np.isnan(spec_pds[spkeys[1 + 2 * N + i * N + j]])]
            )
    return spec_his, spec_std


def plot_g2_contrast(
    contrast_factorL,
    g2,
    times,
    taus,
    q_ring_center=None,
    uid=None,
    vlim=[0.8, 1.2],
    qth=None,
    path=None,
    legend_size=16,
    figsize=[10, 10],
):
    nq, nt = contrast_factorL.shape

    if qth is not None:
        range_ = range(qth, qth + 1)
    else:
        range_ = range(nq)
    num_times = nt
    nr = len(range_)
    sx = int(round(np.sqrt(nr)))
    if nr % sx == 0:
        sy = int(nr / sx)
    else:
        sy = int(nr / sx + 1)
    # fig = plt.figure(figsize=(14, 10))
    if qth is not None:
        fig = plt.figure()
    else:
        if len(contrast_factorL) == 1:
            fig = plt.figure()
        else:
            fig = plt.figure(figsize=figsize)
            # print('here')
    title = "%s_" % uid + "Contrast"
    plt.title(title, fontsize=14, y=1.08)
    if qth is None:
        if len(contrast_factorL) != 1:
            plt.axis("off")

    n = 1
    for sn in range_:
        # print( sn )
        ax = fig.add_subplot(sx, sy, n)
        n += 1
        yL = contrast_factorL[sn, :]
        ax.semilogx(times[:nt], yL, "-bs", label="xsvs")
        ylim = [yL.min() * vlim[0], yL.max() * vlim[1]]
        if (g2 != []) and (g2 is not None):
            g = g2[1:, sn] - 1
            ax.semilogx(taus[1:], g, "-rx", label="xpcs")
            ylim = [g.min() * vlim[0], g.max() * vlim[1]]
        # ax.semilogx([times[:nt][-1], taus[1:][0]], [yL[-1],g[0]], "--bs",  label='')
        til = " Q=" + "%.5f  " % (q_ring_center[sn]) + r"$\AA^{-1}$"
        if qth is not None:
            til = title + til
        ax.set_title(til)
        # ym = np.mean( g )
        ax.set_ylim(ylim)
        # if qth is not None:legend_size=12
        if n == 2:
            ax.legend(loc="best", fontsize=legend_size)

    if qth is None:
        file_name = "%s_contrast" % (uid)
    else:
        file_name = "%s_contrast_q=%s" % (uid, qth)

    fig.tight_layout()
    fp = path + file_name + ".png"
    plt.savefig(fp, dpi=fig.dpi)

    # plt.show()


def get_xsvs_fit_old(spe_cts_all, K_mean, varyK=True, qth=None, max_bins=2, g2=None, times=None, taus=None):
    """
    Fit the xsvs by Negative Binomial Function using max-likelihood chi-squares
    """

    max_cts = spe_cts_all[0][0].shape[0] - 1
    num_times, num_rings = spe_cts_all.shape
    if max_bins is not None:
        num_times = min(num_times, max_bins)

    bin_edges, bin_centers, Knorm_bin_edges, Knorm_bin_centers = get_bin_edges(
        num_times, num_rings, K_mean, int(max_cts + 2)
    )

    if g2 is not None:
        g2c = g2.copy()
        g2c[0] = g2[1]
    ML_val = {}
    KL_val = {}
    K_ = []
    if qth is not None:
        range_ = range(qth, qth + 1)
    else:
        range_ = range(num_rings)
    for i in range_:
        N = 1
        ML_val[i] = []
        KL_val[i] = []

        if g2 is not None:
            mi_g2 = 1 / (g2c[:, i] - 1)
            m_ = np.interp(times, taus, mi_g2)
        for j in range(num_times):
            x_, x, y = bin_edges[j, i][:-1], Knorm_bin_edges[j, i][:-1], spe_cts_all[j, i]
            if g2 is not None:
                m0 = m_[j]
            else:
                m0 = 10
            # resultL = minimize(nbinom_lnlike,  [K_mean[i] * 2**j, m0], args=(x_, y) )
            # the normal leastsq
            # result_n = leastsq(nbinomres, [K_mean[i] * 2**j, m0], args=(y,x_,N),full_output=1)
            # not vary K
            if not varyK:
                resultL = leastsq(
                    nbinomlog1,
                    [m0],
                    args=(y, x_, N, K_mean[i] * 2**j),
                    ftol=1.49012e-38,
                    xtol=1.49012e-38,
                    factor=100,
                    full_output=1,
                )
                ML_val[i].append(abs(resultL[0][0]))
                KL_val[i].append(K_mean[i] * 2**j)  #   resultL[0][0] )

            else:
                # vary M and K
                resultL = leastsq(
                    nbinomlog,
                    [K_mean[i] * 2**j, m0],
                    args=(y, x_, N),
                    ftol=1.49012e-38,
                    xtol=1.49012e-38,
                    factor=100,
                    full_output=1,
                )

                ML_val[i].append(abs(resultL[0][1]))
                KL_val[i].append(abs(resultL[0][0]))  #   resultL[0][0] )
                # print( j, m0, resultL[0][1], resultL[0][0], K_mean[i] * 2**j    )
            if j == 0:
                K_.append(KL_val[i][0])
    # if max_bins==2:
    #    ML_val = np.array( [ML_val[k][0] for k in sorted(list(ML_val.keys()))] )
    #    KL_val = np.array( [KL_val[k][0] for k in sorted(list(KL_val.keys()))] )

    return ML_val, KL_val, np.array(K_)


def gammaDist(x, params):
    """Gamma distribution function
    M,K = params, where K is  average photon counts <x>,
    M is the number of coherent modes,
    In case of high intensity, the beam behavors like wave and
    the probability density of photon, P(x), satify this gamma function.
    """

    K, M = params
    K = float(K)
    M = float(M)
    coeff = np.exp(M * np.log(M) + (M - 1) * np.log(x) - gammaln(M) - M * np.log(K))
    Gd = coeff * np.exp(-M * x / K)
    return Gd


def gamma_dist(bin_values, K, M):
    """
    Gamma distribution function
    Parameters
    ----------
    bin_values : array
        scattering intensities
    K : int
        average number of photons
    M : int
        number of coherent modes
    Returns
    -------
    gamma_dist : array
        Gamma distribution
    Notes
    -----
    These implementations are based on the references under
    nbinom_distribution() function Notes

    : math ::
        P(K) =(\frac{M}{<K>})^M \frac{K^(M-1)}{\Gamma(M)}\exp(-M\frac{K}{<K>})
    """

    # gamma_dist = (stats.gamma(M, 0., K/M)).pdf(bin_values)
    x = bin_values
    coeff = np.exp(M * np.log(M) + (M - 1) * np.log(x) - gammaln(M) - M * np.log(K))
    gamma_dist = coeff * np.exp(-M * x / K)
    return gamma_dist


def nbinom_dist(bin_values, K, M):
    """
    Negative Binomial (Poisson-Gamma) distribution function
    Parameters
    ----------
    bin_values : array
        scattering bin values
    K : int
        number of photons
    M : int
        number of coherent modes
    Returns
    -------
    nbinom : array
        Negative Binomial (Poisson-Gamma) distribution function
    Notes
    -----
    The negative-binomial distribution function
    :math ::
        P(K) = \frac{\\Gamma(K + M)} {\\Gamma(K + 1) ||Gamma(M)}(\frac {M} {M + <K>})^M (\frac {<K>}{M + <K>})^K

    These implementation is based on following references

    References: text [1]_
    .. [1] L. Li, P. Kwasniewski, D. Oris, L Wiegart, L. Cristofolini,
       C. Carona and A. Fluerasu , "Photon statistics and speckle visibility
       spectroscopy with partially coherent x-rays" J. Synchrotron Rad.,
       vol 21, p 1288-1295, 2014.

    """
    co_eff = np.exp(gammaln(bin_values + M) - gammaln(bin_values + 1) - gammaln(M))

    nbinom = co_eff * np.power(M / (K + M), M) * np.power(K / (M + K), bin_values)

    return nbinom


#########poisson
def poisson(x, K):
    """Poisson distribution function.
    K is  average photon counts
    In case of low intensity, the beam behavors like particle and
    the probability density of photon, P(x), satify this poisson function.
    """
    K = float(K)
    Pk = np.exp(-K) * power(K, x) / gamma(x + 1)
    return Pk


def poisson_dist(bin_values, K):
    """
    Poisson Distribution
    Parameters
    ---------
    K : int
        average counts of photons
    bin_values : array
        scattering bin values
    Returns
    -------
    poisson_dist : array
       Poisson Distribution
    Notes
    -----
    These implementations are based on the references under
    nbinom_distribution() function Notes
    :math ::
        P(K) = \frac{<K>^K}{K!}\exp(-<K>)
    """
    # poisson_dist = stats.poisson.pmf(K, bin_values)
    K = float(K)
    poisson_dist = np.exp(-K) * np.power(K, bin_values) / gamma(bin_values + 1)
    return poisson_dist


def diff_mot_con_factor(times, relaxation_rate, contrast_factor, cf_baseline=0):
    """
    This will provide the speckle contrast factor of samples undergoing
    a diffusive motion.

    Parameters
    ----------
    times : array
        integration times

    relaxation_rate : float
        relaxation rate

    contrast_factor : float
        contrast factor

    cf_baseline : float, optional
        the baseline for the contrast factor

    Return
    ------
    diff_contrast_factor : array
        speckle contrast factor for samples undergoing a diffusive motion

    Notes
    -----
    integration times more information - geometric_series function in
    skxray.core.utils module

    These implementations are based on the references under
    negative_binom_distribution() function Notes

    """
    co_eff = (np.exp(-2 * relaxation_rate * times) - 1 + 2 * relaxation_rate * times) / (
        2 * (relaxation_rate * times) ** 2
    )

    return contrast_factor * co_eff + cf_baseline


def plot_sxvs(Knorm_bin_edges, spe_cts_all, uid=None, q_ring_center=None, xlim=[0, 3.5], time_steps=None):
    """a convinent function to plot sxvs results"""
    num_rings = spe_cts_all.shape[1]
    num_times = Knorm_bin_edges.shape[0]
    sx = int(round(np.sqrt(num_rings)))
    if num_rings % sx == 0:
        sy = int(num_rings / sx)
    else:
        sy = int(num_rings / sx + 1)
    fig = plt.figure(figsize=(10, 6))
    plt.title("uid= %s" % uid, fontsize=20, y=1.02)
    plt.axes(frameon=False)
    plt.xticks([])
    plt.yticks([])
    if time_steps is None:
        time_steps = [2**i for i in range(num_times)]
    for i in range(num_rings):
        for j in range(num_times):
            axes = fig.add_subplot(sx, sy, i + 1)
            axes.set_xlabel("K/<K>")
            axes.set_ylabel("P(K)")
            (art,) = axes.plot(
                Knorm_bin_edges[j, i][:-1], spe_cts_all[j, i], "-o", label=str(time_steps[j]) + " ms"
            )
            axes.set_xlim(xlim)
            axes.set_title("Q " + "%.4f  " % (q_ring_center[i]) + r"$\AA^{-1}$")
            axes.legend(loc="best", fontsize=6)
    # plt.show()
    fig.tight_layout()


def fit_xsvs1(
    Knorm_bin_edges,
    bin_edges,
    spe_cts_all,
    K_mean=None,
    func="bn",
    threshold=1e-7,
    uid=None,
    q_ring_center=None,
    xlim=[0, 3.5],
    ylim=None,
    time_steps=None,
):
    """a convinent function to plot sxvs results
    supporting fit function include:
    'bn': Negative Binomaial Distribution
    'gm': Gamma Distribution
    'ps': Poission Distribution

    """
    from lmfit import Model
    from scipy.interpolate import UnivariateSpline

    if func == "bn":
        mod = Model(nbinom_dist)
    elif func == "gm":
        mod = Model(gamma_dist, indepdent_vars=["K"])
    elif func == "ps":
        mod = Model(poisson_dist)
    else:
        print("the current supporting function include 'bn', 'gm','ps'")

    # g_mod = Model(gamma_dist, indepdent_vars=['K'])
    # g_mod = Model( gamma_dist )
    # n_mod = Model(nbinom_dist)
    # p_mod = Model(poisson_dist)
    # dc_mod = Model(diff_mot_con_factor)

    num_rings = spe_cts_all.shape[1]
    num_times = Knorm_bin_edges.shape[0]

    M_val = {}
    K_val = {}
    sx = int(round(np.sqrt(num_rings)))
    if num_rings % sx == 0:
        sy = int(num_rings / sx)
    else:
        sy = int(num_rings / sx + 1)
    fig = plt.figure(figsize=(10, 6))
    plt.title("uid= %s" % uid + " Fitting with Negative Binomial Function", fontsize=20, y=1.02)
    plt.axes(frameon=False)
    plt.xticks([])
    plt.yticks([])
    if time_steps is None:
        time_steps = [2**i for i in range(num_times)]

    for i in range(num_rings):
        M_val[i] = []
        K_val[i] = []
        for j in range(num_times):
            # find the best values for K and M from fitting
            if threshold is not None:
                rois = get_roi(data=spe_cts_all[j, i], threshold=threshold)
            else:
                rois = range(len(spe_cts_all[j, i]))

            # print ( rois )
            if func == "bn":
                result = mod.fit(spe_cts_all[j, i][rois], bin_values=bin_edges[j, i][:-1][rois], K=5 * 2**j, M=12)
            elif func == "gm":
                result = mod.fit(
                    spe_cts_all[j, i][rois], bin_values=bin_edges[j, i][:-1][rois], K=K_mean[i] * 2**j, M=20
                )
            elif func == "ps":
                result = mod.fit(
                    spe_cts_all[j, i][rois], bin_values=bin_edges[j, i][:-1][rois], K=K_mean[i] * 2**j
                )
            else:
                pass

            if func == "bn":
                K_val[i].append(result.best_values["K"])
                M_val[i].append(result.best_values["M"])
            elif func == "gm":
                M_val[i].append(result.best_values["M"])
            elif func == "ps":
                K_val[i].append(result.best_values["K"])
            else:
                pass

            axes = fig.add_subplot(sx, sy, i + 1)
            axes.set_xlabel("K/<K>")
            axes.set_ylabel("P(K)")

            #  Using the best K and M values interpolate and get more values for fitting curve
            fitx_ = np.linspace(0, max(Knorm_bin_edges[j, i][:-1]), 1000)
            fitx = np.linspace(0, max(bin_edges[j, i][:-1]), 1000)
            if func == "bn":
                fity = nbinom_dist(fitx, K_val[i][j], M_val[i][j])  # M and K are fitted best values
                label = "nbinom"
                txt = "K=" + "%.3f" % (K_val[i][0]) + "," + "M=" + "%.3f" % (M_val[i][0])
            elif func == "gm":
                fity = gamma_dist(fitx, K_mean[i] * 2**j, M_val[i][j])
                label = "gamma"
                txt = "M=" + "%.3f" % (M_val[i][0])
            elif func == "ps":
                fity = poisson_dist(fitx, K_val[i][j])
                label = "poisson"
                txt = "K=" + "%.3f" % (K_val[i][0])
            else:
                pass

            if j == 0:
                (art,) = axes.plot(fitx_, fity, "-b", label=label)
            else:
                (art,) = axes.plot(fitx_, fity, "-b")

            if i == 0:
                (art,) = axes.plot(
                    Knorm_bin_edges[j, i][:-1], spe_cts_all[j, i], "o", label=str(time_steps[j]) + " ms"
                )
            else:
                (art,) = axes.plot(
                    Knorm_bin_edges[j, i][:-1],
                    spe_cts_all[j, i],
                    "o",
                )

            axes.set_xlim(0, 3.5)
            if ylim is not None:
                axes.set_ylim(ylim)
            # Annotate the best K and M values on the plot

            axes.annotate(
                r"%s" % txt,
                xy=(1, 0.25),
                xycoords="axes fraction",
                fontsize=10,
                horizontalalignment="right",
                verticalalignment="bottom",
            )
            axes.set_title("Q " + "%.4f  " % (q_ring_center[i]) + r"$\AA^{-1}$")
            axes.legend(loc="best", fontsize=6)
    # plt.show()
    fig.tight_layout()

    return M_val, K_val


def plot_xsvs_g2(g2, taus, res_pargs=None, *argv, **kwargs):
    """plot g2 results,
    g2: one-time correlation function
    taus: the time delays
    res_pargs, a dict, can contains
        uid/path/qr_center/qz_center/
    kwargs: can contains
        vlim: [vmin,vmax]: for the plot limit of y, the y-limit will be [vmin * min(y), vmx*max(y)]
        ylim/xlim: the limit of y and x

    e.g.
    plot_gisaxs_g2( g2b, taus= np.arange( g2b.shape[0]) *timeperframe, q_ring_center = q_ring_center, vlim=[.99, 1.01] )

    """

    if res_pargs is not None:
        uid = res_pargs["uid"]
        path = res_pargs["path"]
        q_ring_center = res_pargs["q_ring_center"]

    else:
        if "uid" in kwargs.keys():
            uid = kwargs["uid"]
        else:
            uid = "uid"

        if "q_ring_center" in kwargs.keys():
            q_ring_center = kwargs["q_ring_center"]
        else:
            q_ring_center = np.arange(g2.shape[1])

        if "path" in kwargs.keys():
            path = kwargs["path"]
        else:
            path = ""

    num_rings = g2.shape[1]
    sx = int(round(np.sqrt(num_rings)))
    if num_rings % sx == 0:
        sy = int(num_rings / sx)
    else:
        sy = int(num_rings / sx + 1)

    # print (num_rings)
    if num_rings != 1:
        # fig = plt.figure(figsize=(14, 10))
        fig = plt.figure(figsize=(12, 10))
        plt.axis("off")
        # plt.axes(frameon=False)
        # print ('here')
        plt.xticks([])
        plt.yticks([])

    else:
        fig = plt.figure(figsize=(8, 8))

    plt.title("uid= %s" % uid, fontsize=20, y=1.06)
    for i in range(num_rings):
        ax = fig.add_subplot(sx, sy, i + 1)
        ax.set_ylabel("beta")
        ax.set_title(" Q= " + "%.5f  " % (q_ring_center[i]) + r"$\AA^{-1}$")
        y = g2[:, i]
        # print (y)
        ax.semilogx(taus, y, "-o", markersize=6)
        # ax.set_ylim([min(y)*.95, max(y[1:])*1.05 ])
        if "ylim" in kwargs:
            ax.set_ylim(kwargs["ylim"])
        elif "vlim" in kwargs:
            vmin, vmax = kwargs["vlim"]
            ax.set_ylim([min(y) * vmin, max(y[1:]) * vmax])
        else:
            pass
        if "xlim" in kwargs:
            ax.set_xlim(kwargs["xlim"])

    dt = datetime.now()
    CurTime = "%s%02d%02d-%02d%02d-" % (dt.year, dt.month, dt.day, dt.hour, dt.minute)

    fp = path + "g2--uid=%s" % (uid) + CurTime + ".png"
    fig.savefig(fp, dpi=fig.dpi)
    fig.tight_layout()
    # plt.show()


def get_xsvs_fit_old1(
    spe_cts_all, K_mean, spec_std=None, varyK=True, qth=None, max_bins=None, g2=None, times=None, taus=None
):
    """
    Fit the xsvs by Negative Binomial Function using max-likelihood chi-squares
    """

    max_cts = spe_cts_all[0][0].shape[0] - 1
    num_times, num_rings = spe_cts_all.shape
    if max_bins is not None:
        num_times = min(num_times, max_bins)

    bin_edges, bin_centers, Knorm_bin_edges, Knorm_bin_centers = get_bin_edges(
        num_times, num_rings, K_mean, int(max_cts + 2)
    )

    if g2 is not None:
        g2c = g2.copy()
        g2c[0] = g2[1]
    ML_val = {}
    KL_val = {}
    K_ = []
    if qth is not None:
        range_ = range(qth, qth + 1)
    else:
        range_ = range(num_rings)
    for i in range_:
        N = 1
        ML_val[i] = []
        KL_val[i] = []

        if g2 is not None:
            mi_g2 = 1 / (g2c[:, i] - 1)
            m_ = np.interp(times, taus, mi_g2)
        for j in range(num_times):
            x_, x, y = bin_edges[j, i][:-1], Knorm_bin_edges[j, i][:-1], spe_cts_all[j, i]
            if spec_std is not None:
                yerr = spec_std[j, i]
            else:
                yerr = None
            if g2 is not None:
                m0 = m_[j]
            else:
                m0 = 10
            # resultL = minimize(nbinom_lnlike,  [K_mean[i] * 2**j, m0], args=(x_, y) )
            # the normal leastsq
            # result_n = leastsq(nbinomres, [K_mean[i] * 2**j, m0], args=(y,x_,N),full_output=1)
            # not vary K

            if not varyK:
                if yerr is None:
                    fit_func = nbinomlog1  # _old
                else:
                    fit_func = nbinomlog1
                # print(j,i,m0, y.shape, x_.shape, yerr.shape, N,  K_mean[i] * 2**j)
                resultL = leastsq(
                    fit_func,
                    [m0],
                    args=(y, x_, yerr, N, K_mean[i] * 2**j),
                    ftol=1.49012e-38,
                    xtol=1.49012e-38,
                    factor=100,
                    full_output=1,
                )

                ML_val[i].append(abs(resultL[0][0]))
                KL_val[i].append(K_mean[i] * 2**j)  #   resultL[0][0] )

            else:
                # vary M and K
                if yerr is None:
                    fit_func = nbinomlog  # _old
                else:
                    fit_func = nbinomlog
                resultL = leastsq(
                    fit_func,
                    [K_mean[i] * 2**j, m0],
                    args=(y, x_, yerr, N),
                    ftol=1.49012e-38,
                    xtol=1.49012e-38,
                    factor=100,
                    full_output=1,
                )

                ML_val[i].append(abs(resultL[0][1]))
                KL_val[i].append(abs(resultL[0][0]))  #   resultL[0][0] )
                # print( j, m0, resultL[0][1], resultL[0][0], K_mean[i] * 2**j    )
            if j == 0:
                K_.append(KL_val[i][0])

    return ML_val, KL_val, np.array(K_)
