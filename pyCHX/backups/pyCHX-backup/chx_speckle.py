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

import sys
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.stats as st
from matplotlib.colors import LogNorm
from scipy.optimize import leastsq, minimize


def xsvs(
    image_sets,
    label_array,
    number_of_img,
    timebin_num=2,
    time_bin=None,
    only_first_level=False,
    max_cts=None,
    bad_images=None,
    threshold=None,
):
    """
    This function will provide the probability density of detecting photons
    for different integration times.
    The experimental probability density P(K) of detecting photons K is
    obtained by histogramming the speckle counts over an ensemble of
    equivalent pixels and over a number of speckle patterns recorded
    with the same integration time T under the same condition.
    Parameters
    ----------
    image_sets : array
        sets of images
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
    if max_cts is None:
        max_cts = roi.roi_max_counts(image_sets, label_array)

    # find the label's and pixel indices for ROI's
    labels, indices = roi.extract_label_indices(label_array)
    nopixels = len(indices)
    # number of ROI's
    u_labels = list(np.unique(labels))
    num_roi = len(u_labels)

    # create integration times
    if time_bin is None:
        time_bin = geometric_series(timebin_num, number_of_img)
    if only_first_level:
        time_bin = [1]
    # number of times in the time bin
    num_times = len(time_bin)

    # number of pixels per ROI
    num_pixels = np.bincount(labels, minlength=(num_roi + 1))[1:]

    # probability density of detecting photons
    prob_k_all = np.zeros([num_times, num_roi], dtype=np.object)

    # square of probability density of detecting photons
    prob_k_pow_all = np.zeros_like(prob_k_all)

    # standard deviation of probability density of detecting photons
    prob_k_std_dev = np.zeros_like(prob_k_all)

    # get the bin edges for each time bin for each ROI
    bin_edges = np.zeros(prob_k_all.shape[0], dtype=prob_k_all.dtype)
    for i in range(num_times):
        bin_edges[i] = np.arange(max_cts * 2**i)

    start_time = time.time()  # used to log the computation time (optionally)

    for i, images in enumerate(image_sets):
        # print( i, images )
        # Ring buffer, a buffer with periodic boundary conditions.
        # Images must be keep for up to maximum delay in buf.
        # buf = np.zeros([num_times, timebin_num], dtype=np.object)  # matrix of buffers

        buf = np.ma.zeros([num_times, timebin_num, nopixels])
        buf.mask = True

        # to track processing each time level
        track_level = np.zeros(num_times)
        track_bad_level = np.zeros(num_times)
        # to increment buffer
        cur = np.int_(np.full(num_times, timebin_num))

        # to track how many images processed in each level
        img_per_level = np.zeros(num_times, dtype=np.int64)

        prob_k = np.zeros_like(prob_k_all)
        prob_k_pow = np.zeros_like(prob_k_all)

        try:
            noframes = len(images)
        except:
            noframes = images.length

        # Num= { key: [0]* len(  dict_dly[key] ) for key in list(dict_dly.keys())  }

        for n, img in enumerate(images):
            cur[0] = 1 + cur[0] % timebin_num
            # read each frame
            # Put the image into the ring buffer.

            img_ = (np.ravel(img))[indices]

            if threshold is not None:
                if img_.max() >= threshold:
                    print("bad image: %s here!" % n)
                    img_ = np.ma.zeros(len(img_))
                    img_.mask = True

            if bad_images is not None:
                if n in bad_images:
                    print("bad image: %s here!" % n)
                    img_ = np.ma.zeros(len(img_))
                    img_.mask = True

            buf[0, cur[0] - 1] = img_

            # print( n, np.sum(buf[0, cur[0] - 1] ), np.sum( img ) )

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

            # print (0, img_per_level)

            # check whether the number of levels is one, otherwise
            # continue processing the next level
            level = 1
            if number_of_img > 1:
                processing = 1
            else:
                processing = 0
            # print ('track_level: %s'%track_level)
            # while level < num_times:
            # if not track_level[level]:
            # track_level[level] = 1
            if only_first_level:
                processing = 0
            while processing:
                if track_level[level]:
                    prev = 1 + (cur[level - 1] - 2) % timebin_num
                    cur[level] = 1 + cur[level] % timebin_num

                    bufa = buf[level - 1, prev - 1]
                    bufb = buf[level - 1, cur[level - 1] - 1]

                    if (bufa.data == 0).all():
                        buf[level, cur[level] - 1] = bufa
                    elif (bufb.data == 0).all():
                        buf[level, cur[level] - 1] = bufb
                    else:
                        buf[level, cur[level] - 1] = bufa + bufb

                    # print (level, cur[level]-1)

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
                # print ('track_level: %s'%track_level)

            if noframes >= 10 and n % (int(noframes / 10)) == 0:
                sys.stdout.write("#")
                sys.stdout.flush()

            prob_k_all += (prob_k - prob_k_all) / (i + 1)
            prob_k_pow_all += (prob_k_pow - prob_k_pow_all) / (i + 1)

    prob_k_std_dev = np.power((prob_k_pow_all - np.power(prob_k_all, 2)), 0.5)

    for i in range(num_times):
        if isinstance(prob_k_all[i, 0], float):
            for j in range(len(u_labels)):
                prob_k_all[i, j] = np.array([0] * (len(bin_edges[i]) - 1))
                prob_k_std_dev[i, j] = np.array([0] * (len(bin_edges[i]) - 1))

    logger.info("Processing time for XSVS took %s seconds." "", (time.time() - start_time))
    elapsed_time = time.time() - start_time
    # print (Num)
    print("Total time: %.2f min" % (elapsed_time / 60.0))

    # print (img_per_level - track_bad_level)
    # print (buf)

    return bin_edges, prob_k_all, prob_k_std_dev


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
    if (data.data == 0).all():
        track_bad_level[level] += 1

    # print (img_per_level,track_bad_level)

    u_labels = list(np.unique(labels))

    if not (data.data == 0).all():
        for j, label in enumerate(u_labels):
            roi_data = data[labels == label]
            spe_hist, bin_edges = np.histogram(roi_data, bins=bin_edges, density=True)
            spe_hist = np.nan_to_num(spe_hist)
            prob_k[level, j] += (spe_hist - prob_k[level, j]) / (img_per_level[level] - track_bad_level[level])

            prob_k_pow[level, j] += (np.power(spe_hist, 2) - prob_k_pow[level, j]) / (
                img_per_level[level] - track_bad_level[level]
            )


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


def get_roi(data, threshold=1e-3):
    roi = np.where(data > threshold)
    if len(roi[0]) > len(data) - 3:
        roi = (np.array(roi[0][:-3]),)
    elif len(roi[0]) < 3:
        roi = np.where(data >= 0)
    return roi[0]


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


###########################3

#


def nbinomlog(p, hist, x, N):
    """Residuals for maximum likelihood fit to nbinom distribution.
    Vary M (shape param) and mu (count rate) vary (using leastsq)"""
    mu, M = p
    mu = abs(mu)
    M = abs(M)
    w = np.where(hist > 0.0)
    Np = N * st.nbinom.pmf(x, M, 1.0 / (1.0 + mu / M))
    err = 2 * (Np - hist)
    err[w] = err[w] - 2 * hist[w] * np.log(Np[w] / hist[w])  # note: sum(Np-hist)==0
    return np.sqrt(np.abs(err))
    # return err


def nbinomlog1(p, hist, x, N, mu):
    """Residuals for maximum likelihood fit to nbinom distribution.
    Vary M (shape param) but mu (count rate) fixed (using leastsq)"""
    M = abs(p[0])
    w = np.where(hist > 0.0)
    Np = N * st.nbinom.pmf(x, M, 1.0 / (1.0 + mu / M))
    err = 2 * (Np - hist)
    err[w] = err[w] - 2 * hist[w] * np.log(Np[w] / hist[w])  # note: sum(Np-hist)==0
    return np.sqrt(np.abs(err))


def nbinomlog1_notworknow(p, hist, x, N, mu):
    """Residuals for maximum likelihood fit to nbinom distribution.
    Vary M (shape param) but mu (count rate) fixed (using leastsq)"""
    M = abs(p[0])
    w = np.where(hist > 0.0)
    Np = N * st.nbinom.pmf(x, M, 1.0 / (1.0 + mu / M))
    err = 2 * (Np - hist)
    err[w] = err[w] - 2 * hist[w] * np.log(Np[w] / hist[w])  # note: sum(Np-hist)==0
    # return np.sqrt(err)
    return err


def nbinomres(p, hist, x, N):
    """residuals to leastsq() to fit normal chi-square"""
    mu, M = p
    Np = N * st.nbinom.pmf(x, M, 1.0 / (1.0 + mu / M))
    err = (hist - Np) / np.sqrt(Np)
    return err


def get_xsvs_fit(spe_cts_all, K_mean, varyK=True, max_bins=None, qth=None, g2=None, times=None, taus=None):
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
    return ML_val, KL_val, np.array(K_)


def plot_xsvs_fit(
    spe_cts_all,
    ML_val,
    KL_val,
    K_mean,
    xlim=[0, 15],
    ylim=[1e-8, 1],
    q_ring_center=None,
    uid="uid",
    qth=None,
    times=None,
    fontsize=3,
):
    fig = plt.figure(figsize=(9, 6))
    plt.title("uid= %s" % uid + " Fitting with Negative Binomial Function", fontsize=20, y=1.02)
    plt.axes(frameon=False)
    plt.xticks([])
    plt.yticks([])

    max_cts = spe_cts_all[0][0].shape[0] - 1
    num_times, num_rings = spe_cts_all.shape

    bin_edges, bin_centers, Knorm_bin_edges, Knorm_bin_centers = get_bin_edges(
        num_times, num_rings, K_mean, int(max_cts + 2)
    )

    if qth is not None:
        range_ = range(qth, qth + 1)
        num_times = len(ML_val[qth])
    else:
        range_ = range(num_rings)
        num_times = len(ML_val[0])
    # for i in range(num_rings):

    sx = int(round(np.sqrt(len(range_))))

    if len(range_) % sx == 0:
        sy = int(len(range_) / sx)
    else:
        sy = int(len(range_) / sx + 1)
    n = 1
    for i in range_:
        axes = fig.add_subplot(sx, sy, n)
        axes.set_xlabel("K/<K>")
        axes.set_ylabel("P(K)")
        n += 1
        for j in range(num_times):
            # print( i, j )
            x_, x, y = bin_edges[j, i][:-1], Knorm_bin_edges[j, i][:-1], spe_cts_all[j, i]
            # Using the best K and M values interpolate and get more values for fitting curve

            xscale = bin_edges[j, i][:-1][1] / Knorm_bin_edges[j, i][:-1][1]
            fitx = np.linspace(0, max_cts * 2**j, 5000)
            fitx_ = fitx / xscale

            # fity = nbinom_dist( fitx, K_val[i][j], M_val[i][j] )
            fitL = nbinom_dist(fitx, KL_val[i][j], ML_val[i][j])

            if j == 0:
                (art,) = axes.semilogy(fitx_, fitL, "-r", label="nbinom_L")
                # art, = axes.semilogy( fitx_,fity, '--b',  label="nbinom")
            else:
                (art,) = axes.plot(fitx_, fitL, "-r")
                # art, = axes.plot( fitx_,fity, '--b')
            if i == 0:
                if times is not None:
                    label = str(times[j] * 1000) + " ms"
                else:
                    label = "Bin_%s" % (2**j)

                (art,) = axes.plot(x, y, "o", label=label)
            else:
                (art,) = axes.plot(
                    x,
                    y,
                    "o",
                )

            axes.set_xlim(xlim)
            axes.set_ylim(ylim)

            axes.set_title("Q=" + "%.4f  " % (q_ring_center[i]) + r"$\AA^{-1}$")
            axes.legend(loc="best", fontsize=fontsize)
    # plt.show()
    fig.tight_layout()


def get_max_countc(FD, labeled_array):
    """Compute the max intensity of ROIs in the compressed file (FD)

    Parameters
    ----------
    FD: Multifile class
        compressed file
    labeled_array : array
        labeled array; 0 is background.
        Each ROI is represented by a nonzero integer. It is not required that
        the ROI labels are contiguous
    index : int, list, optional
        The ROI's to use. If None, this function will extract averages for all
        ROIs

    Returns
    -------
    max_intensity : a float
    index : list
        The labels for each element of the `mean_intensity` list
    """

    qind, pixelist = roi.extract_label_indices(labeled_array)
    timg = np.zeros(FD.md["ncols"] * FD.md["nrows"], dtype=np.int32)
    timg[pixelist] = np.arange(1, len(pixelist) + 1)

    if labeled_array.shape != (FD.md["ncols"], FD.md["nrows"]):
        raise ValueError(
            " `image` shape (%d, %d) in FD is not equal to the labeled_array shape (%d, %d)"
            % (FD.md["ncols"], FD.md["nrows"], labeled_array.shape[0], labeled_array.shape[1])
        )

    max_inten = 0
    for i in tqdm(range(FD.beg, FD.end, 1), desc="Get max intensity of ROIs in all frames"):
        (p, v) = FD.rdrawframe(i)
        w = np.where(timg[p])[0]

        max_inten = max(max_inten, np.max(v[w]))
    return max_inten


def get_contrast(ML_val):
    nq, nt = len(ML_val.keys()), len(ML_val[list(ML_val.keys())[0]])
    contrast_factorL = np.zeros([nq, nt])
    for i in range(nq):
        for j in range(nt):
            contrast_factorL[i, j] = 1 / ML_val[i][j]
    return contrast_factorL


def plot_g2_contrast(contrast_factorL, g2, times, taus, q_ring_center=None, uid=None, vlim=[0.8, 1.2], qth=None):
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

    fig = plt.figure()
    plt.title("uid= %s_" % uid + "Contrast Factor for Each Q Rings", fontsize=14, y=1.08)
    if qth is None:
        plt.axis("off")
    n = 1
    for sn in range_:
        # print( sn )
        ax = fig.add_subplot(sx, sy, n)
        n += 1
        yL = contrast_factorL[sn, :]
        g = g2[1:, sn] - 1
        ax.semilogx(times[:nt], yL, "-bs", label="vis")
        ax.semilogx(taus[1:], g, "-rx", label="xpcs")
        ax.set_title(" Q=" + "%.5f  " % (q_ring_center[sn]) + r"$\AA^{-1}$")

        # ym = np.mean( g )
        ax.set_ylim([g.min() * vlim[0], g.max() * vlim[1]])

    fig.tight_layout()
