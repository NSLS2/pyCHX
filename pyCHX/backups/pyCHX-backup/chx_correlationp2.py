"""
Aug 10, Developed by Y.G.@CHX
yuzhang@bnl.gov
This module is for parallel computation of time correlation
Feb 20, 2018
The chx_correlationp2 is for dedug g2
"""

from __future__ import absolute_import, division, print_function

import logging
import sys
from collections import namedtuple
from multiprocessing import Pool

import dill
import numpy as np
import skbeam.core.roi as roi
from skbeam.core.roi import extract_label_indices
from skbeam.core.utils import multi_tau_lags

from pyCHX.chx_compress import apply_async, go_through_FD, map_async, pass_FD, run_dill_encoded
from pyCHX.chx_correlationc import _one_time_process as _one_time_processp
from pyCHX.chx_correlationc import _one_time_process_error as _one_time_process_errorp
from pyCHX.chx_correlationc import _two_time_process as _two_time_processp
from pyCHX.chx_correlationc import _validate_and_transform_inputs, get_pixelist_interp_iq
from pyCHX.chx_libs import tqdm

logger = logging.getLogger(__name__)


class _init_state_two_timep:
    def __init__(self, num_levels, num_bufs, labels, num_frames):
        (
            label_array,
            pixel_list,
            num_rois,
            num_pixels,
            lag_steps,
            buf,
            img_per_level,
            track_level,
            cur,
            norm,
            lev_len,
        ) = _validate_and_transform_inputs(num_bufs, num_levels, labels)
        count_level = np.zeros(num_levels, dtype=np.int64)
        # current image time
        current_img_time = 0
        # generate a time frame for each level
        time_ind = {key: [] for key in range(num_levels)}
        # two time correlation results (array)
        g2 = np.zeros((num_rois, num_frames, num_frames), dtype=np.float64)

        (
            self.buf,
            self.img_per_level,
            self.label_array,
            self.track_level,
            self.cur,
            self.pixel_list,
            self.num_pixels,
            self.lag_steps,
            self.g2,
            self.count_level,
            self.current_img_time,
            self.time_ind,
            self.norm,
            self.lev_len,
        ) = (
            buf,
            img_per_level,
            label_array,
            track_level,
            cur,
            pixel_list,
            num_pixels,
            lag_steps,
            g2,
            count_level,
            current_img_time,
            time_ind,
            norm,
            lev_len,
        )

    def __getstate__(self):
        """This is called before pickling."""
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        """This is called while unpickling."""
        self.__dict__.update(state)


def lazy_two_timep(
    FD, num_levels, num_bufs, labels, internal_state=None, bad_frame_list=None, imgsum=None, norm=None
):
    """Generator implementation of two-time correlation
    If you do not want multi-tau correlation, set num_levels to 1 and
    num_bufs to the number of images you wish to correlate
    Multi-tau correlation uses a scheme to achieve long-time correlations
    inexpensively by downsampling the data, iteratively combining successive
    frames.
    The longest lag time computed is num_levels * num_bufs.
    ** see comments on multi_tau_auto_corr
    Parameters
    ----------
    FD: the handler of compressed data
    num_levels : int, optional
        how many generations of downsampling to perform, i.e.,
        the depth of the binomial tree of averaged frames
        default is one
    num_bufs : int, must be even
        maximum lag step to compute in each generation of
        downsampling
    labels : array
        labeled array of the same shape as the image stack;
        each ROI is represented by a distinct label (i.e., integer)
    two_time_internal_state: None


    Yields
    ------
    namedtuple
        A ``results`` object is yielded after every image has been processed.
        This `reults` object contains, in this order:
        - ``g2``: the normalized correlation
          shape is (num_rois, len(lag_steps), len(lag_steps))
        - ``lag_steps``: the times at which the correlation was computed
        - ``_internal_state``: all of the internal state. Can be passed back in
          to ``lazy_one_time`` as the ``internal_state`` parameter
    Notes
    -----
    The two-time correlation function is defined as
    .. math::
        C(q,t_1,t_2) = \\frac{<I(q,t_1)I(q,t_2)>}{<I(q, t_1)><I(q,t_2)>}
    Here, the ensemble averages are performed over many pixels of detector,
    all having the same ``q`` value. The average time or age is equal to
    ``(t1+t2)/2``, measured by the distance along the ``t1 = t2`` diagonal.
    The time difference ``t = |t1 - t2|``, with is distance from the
    ``t1 = t2`` diagonal in the perpendicular direction.
    In the equilibrium system, the two-time correlation functions depend only
    on the time difference ``t``, and hence the two-time correlation contour
    lines are parallel.
    References
    ----------
    .. [1]
        A. Fluerasu, A. Moussaid, A. Mandsen and A. Schofield, "Slow dynamics
        and aging in collodial gels studied by x-ray photon correlation
        spectroscopy," Phys. Rev. E., vol 76, p 010401(1-4), 2007.
    """
    num_frames = FD.end - FD.beg
    if internal_state is None:
        internal_state = _init_state_two_timep(num_levels, num_bufs, labels, num_frames)
    # create a shorthand reference to the results and state named tuple
    s = internal_state

    qind, pixelist = roi.extract_label_indices(labels)
    # iterate over the images to compute multi-tau correlation
    fra_pix = np.zeros_like(pixelist, dtype=np.float64)
    timg = np.zeros(FD.md["ncols"] * FD.md["nrows"], dtype=np.int32)
    timg[pixelist] = np.arange(1, len(pixelist) + 1)
    if bad_frame_list is None:
        bad_frame_list = []

    for i in range(FD.beg, FD.end):
        if i in bad_frame_list:
            fra_pix[:] = np.nan
        else:
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

        level = 0
        # increment buffer
        s.cur[0] = (1 + s.cur[0]) % num_bufs
        s.count_level[0] = 1 + s.count_level[0]
        # get the current image time

        # s = s._replace(current_img_time=(s.current_img_time + 1))
        s.current_img_time += 1

        # Put the ROI pixels into the ring buffer.
        s.buf[0, s.cur[0] - 1] = fra_pix
        fra_pix[:] = 0
        _two_time_processp(
            s.buf,
            s.g2,
            s.label_array,
            num_bufs,
            s.num_pixels,
            s.img_per_level,
            s.lag_steps,
            s.current_img_time,
            level=0,
            buf_no=s.cur[0] - 1,
        )
        # time frame for each level
        s.time_ind[0].append(s.current_img_time)
        # check whether the number of levels is one, otherwise
        # continue processing the next level
        processing = num_levels > 1
        # Compute the correlations for all higher levels.
        level = 1
        while processing:
            if not s.track_level[level]:
                s.track_level[level] = 1
                processing = False
            else:
                prev = 1 + (s.cur[level - 1] - 2) % num_bufs
                s.cur[level] = 1 + s.cur[level] % num_bufs
                s.count_level[level] = 1 + s.count_level[level]
                s.buf[level, s.cur[level] - 1] = (
                    s.buf[level - 1, prev - 1] + s.buf[level - 1, s.cur[level - 1] - 1]
                ) / 2
                t1_idx = (s.count_level[level] - 1) * 2
                current_img_time = ((s.time_ind[level - 1])[t1_idx] + (s.time_ind[level - 1])[t1_idx + 1]) / 2.0
                # time frame for each level
                s.time_ind[level].append(current_img_time)
                # make the track_level zero once that level is processed
                s.track_level[level] = 0
                # call the _two_time_process function for each multi-tau level
                # for multi-tau levels greater than one
                # Again, this is modifying things in place. See comment
                # on previous call above.
                _two_time_processp(
                    s.buf,
                    s.g2,
                    s.label_array,
                    num_bufs,
                    s.num_pixels,
                    s.img_per_level,
                    s.lag_steps,
                    current_img_time,
                    level=level,
                    buf_no=s.cur[level] - 1,
                )
                level += 1

                # Checking whether there is next level for processing
                processing = level < num_levels
        # print (s.g2[1,:,1] )
        # yield s
    for q in range(np.max(s.label_array)):
        x0 = (s.g2)[q, :, :]
        (s.g2)[q, :, :] = np.tril(x0) + np.tril(x0).T - np.diag(np.diag(x0))
    return s.g2, s.lag_steps


def cal_c12p(FD, ring_mask, bad_frame_list=None, good_start=0, num_buf=8, num_lev=None, imgsum=None, norm=None):
    """calculation g2 by using a multi-tau algorithm
    for a compressed file with parallel calculation
    """
    FD.beg = max(FD.beg, good_start)
    noframes = FD.end - FD.beg  # +1   # number of frames, not "no frames"
    for i in range(FD.beg, FD.end):
        pass_FD(FD, i)
    if num_lev is None:
        num_lev = int(np.log(noframes / (num_buf - 1)) / np.log(2) + 1) + 1
    print("In this g2 calculation, the buf and lev number are: %s--%s--" % (num_buf, num_lev))
    if bad_frame_list is not None:
        if len(bad_frame_list) != 0:
            print("Bad frame involved and will be precessed!")
            noframes -= len(np.where(np.in1d(bad_frame_list, range(good_start, FD.end)))[0])
    print("%s frames will be processed..." % (noframes))
    ring_masks = [np.array(ring_mask == i, dtype=np.int64) for i in np.unique(ring_mask)[1:]]
    qind, pixelist = roi.extract_label_indices(ring_mask)
    if norm is not None:
        norms = [
            norm[np.in1d(pixelist, extract_label_indices(np.array(ring_mask == i, dtype=np.int64))[1])]
            for i in np.unique(ring_mask)[1:]
        ]
    inputs = range(len(ring_masks))
    pool = Pool(processes=len(inputs))
    internal_state = None
    print("Starting assign the tasks...")
    results = {}
    if norm is not None:
        for i in tqdm(inputs):
            # for i in  inputs:
            results[i] = apply_async(
                pool,
                lazy_two_timep,
                (
                    FD,
                    num_lev,
                    num_buf,
                    ring_masks[i],
                    internal_state,
                    bad_frame_list,
                    imgsum,
                    norms[i],
                ),
            )
    else:
        # print ('for norm is None')
        for i in tqdm(inputs):
            # for i in  inputs:
            results[i] = apply_async(
                pool,
                lazy_two_timep,
                (
                    FD,
                    num_lev,
                    num_buf,
                    ring_masks[i],
                    internal_state,
                    bad_frame_list,
                    imgsum,
                    None,
                ),
            )
    pool.close()
    print("Starting running the tasks...")
    res = [results[k].get() for k in tqdm(list(sorted(results.keys())))]

    c12 = np.zeros([noframes, noframes, len(ring_masks)])
    for i in inputs:
        # print( res[i][0][:,0].shape, g2.shape )
        c12[:, :, i] = res[i][0][0]  # [:len_lag, :len_lag]
        if i == 0:
            lag_steps = res[0][1]

    print("G2 calculation DONE!")
    del results
    del res
    return c12, lag_steps[lag_steps < noframes]


class _internal_statep:
    def __init__(self, num_levels, num_bufs, labels, cal_error=False):
        """YG. DEV Nov, 2016, Initialize class for the generator-based multi-tau
        for one time correlation

             Jan 1, 2018, Add cal_error option to calculate signal to noise to one time correaltion

        """
        (
            label_array,
            pixel_list,
            num_rois,
            num_pixels,
            lag_steps,
            buf,
            img_per_level,
            track_level,
            cur,
            norm,
            lev_len,
        ) = _validate_and_transform_inputs(num_bufs, num_levels, labels)

        G = np.zeros((int((num_levels + 1) * num_bufs / 2), num_rois), dtype=np.float64)
        # matrix for normalizing G into g2
        past_intensity = np.zeros_like(G)
        # matrix for normalizing G into g2
        future_intensity = np.zeros_like(G)
        (
            self.buf,
            self.G,
            self.past_intensity,
            self.future_intensity,
            self.img_per_level,
            self.label_array,
            self.track_level,
            self.cur,
            self.pixel_list,
            self.num_pixels,
            self.lag_steps,
            self.norm,
            self.lev_len,
        ) = (
            buf,
            G,
            past_intensity,
            future_intensity,
            img_per_level,
            label_array,
            track_level,
            cur,
            pixel_list,
            num_pixels,
            lag_steps,
            norm,
            lev_len,
        )
        if cal_error:
            self.G_all = np.zeros((int((num_levels + 1) * num_bufs / 2), len(pixel_list)), dtype=np.float64)
            # matrix for normalizing G into g2
            self.past_intensity_all = np.zeros_like(self.G_all)
            # matrix for normalizing G into g2
            self.future_intensity_all = np.zeros_like(self.G_all)

    def __getstate__(self):
        """This is called before pickling."""
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        """This is called while unpickling."""
        self.__dict__.update(state)


def lazy_one_timep(
    FD,
    num_levels,
    num_bufs,
    labels,
    internal_state=None,
    bad_frame_list=None,
    imgsum=None,
    norm=None,
    cal_error=False,
):
    if internal_state is None:
        internal_state = _internal_statep(num_levels, num_bufs, labels, cal_error)
    # create a shorthand reference to the results and state named tuple
    s = internal_state
    qind, pixelist = roi.extract_label_indices(labels)
    # iterate over the images to compute multi-tau correlation
    fra_pix = np.zeros_like(pixelist, dtype=np.float64)
    timg = np.zeros(FD.md["ncols"] * FD.md["nrows"], dtype=np.int32)
    timg[pixelist] = np.arange(1, len(pixelist) + 1)
    if bad_frame_list is None:
        bad_frame_list = []
    # for  i in tqdm(range( FD.beg , FD.end )):
    for i in range(FD.beg, FD.end):
        if i in bad_frame_list:
            fra_pix[:] = np.nan
        else:
            (p, v) = FD.rdrawframe(i)
            w = np.where(timg[p])[0]
            pxlist = timg[p[w]] - 1
            if imgsum is None:
                if norm is None:
                    # print ('here')
                    fra_pix[pxlist] = v[w]
                else:
                    fra_pix[pxlist] = v[w] / norm[pxlist]  # -1.0
            else:
                if norm is None:
                    fra_pix[pxlist] = v[w] / imgsum[i]
                else:
                    fra_pix[pxlist] = v[w] / imgsum[i] / norm[pxlist]

        level = 0
        # increment buffer
        s.cur[0] = (1 + s.cur[0]) % num_bufs
        # Put the ROI pixels into the ring buffer.
        s.buf[0, s.cur[0] - 1] = fra_pix
        fra_pix[:] = 0
        # print( i, len(p), len(w), len( pixelist))

        # print ('i= %s init fra_pix'%i )
        buf_no = s.cur[0] - 1
        # Compute the correlations between the first level
        # (undownsampled) frames. This modifies G,
        # past_intensity, future_intensity,
        # and img_per_level in place!
        # print (s.G)
        if cal_error:
            _one_time_process_errorp(
                s.buf,
                s.G,
                s.past_intensity,
                s.future_intensity,
                s.label_array,
                num_bufs,
                s.num_pixels,
                s.img_per_level,
                level,
                buf_no,
                s.norm,
                s.lev_len,
                s.G_all,
                s.past_intensity_all,
                s.future_intensity_all,
            )
        else:
            _one_time_processp(
                s.buf,
                s.G,
                s.past_intensity,
                s.future_intensity,
                s.label_array,
                num_bufs,
                s.num_pixels,
                s.img_per_level,
                level,
                buf_no,
                s.norm,
                s.lev_len,
            )

        # print (s.G)
        # check whether the number of levels is one, otherwise
        # continue processing the next level
        processing = num_levels > 1
        level = 1
        while processing:
            if not s.track_level[level]:
                s.track_level[level] = True
                processing = False
            else:
                prev = 1 + (s.cur[level - 1] - 2) % num_bufs
                s.cur[level] = 1 + s.cur[level] % num_bufs

                s.buf[level, s.cur[level] - 1] = (
                    s.buf[level - 1, prev - 1] + s.buf[level - 1, s.cur[level - 1] - 1]
                ) / 2

                # make the track_level zero once that level is processed
                s.track_level[level] = False

                # call processing_func for each multi-tau level greater
                # than one. This is modifying things in place. See comment
                # on previous call above.
                buf_no = s.cur[level] - 1
                if cal_error:
                    _one_time_process_errorp(
                        s.buf,
                        s.G,
                        s.past_intensity,
                        s.future_intensity,
                        s.label_array,
                        num_bufs,
                        s.num_pixels,
                        s.img_per_level,
                        level,
                        buf_no,
                        s.norm,
                        s.lev_len,
                        s.G_all,
                        s.past_intensity_all,
                        s.future_intensity_all,
                    )
                else:
                    _one_time_processp(
                        s.buf,
                        s.G,
                        s.past_intensity,
                        s.future_intensity,
                        s.label_array,
                        num_bufs,
                        s.num_pixels,
                        s.img_per_level,
                        level,
                        buf_no,
                        s.norm,
                        s.lev_len,
                    )

                level += 1
                # Checking whether there is next level for processing
                processing = level < num_levels

    # If any past intensities are zero, then g2 cannot be normalized at
    # those levels. This if/else code block is basically preventing
    # divide-by-zero errors.
    if not cal_error:
        if len(np.where(s.past_intensity == 0)[0]) != 0:
            g_max1 = np.where(s.past_intensity == 0)[0][0]
        else:
            g_max1 = s.past_intensity.shape[0]
        if len(np.where(s.future_intensity == 0)[0]) != 0:
            g_max2 = np.where(s.future_intensity == 0)[0][0]
        else:
            g_max2 = s.future_intensity.shape[0]
        g_max = min(g_max1, g_max2)
        g2 = s.G[:g_max] / (s.past_intensity[:g_max] * s.future_intensity[:g_max])
    # sys.stdout.write('#')
    # del FD
    # sys.stdout.flush()
    # print (g2)
    # return results(g2, s.lag_steps[:g_max], s)
    if cal_error:
        # return g2, s.lag_steps[:g_max], s.G[:g_max],s.past_intensity[:g_max], s.future_intensity[:g_max] #, s
        return (None, s.lag_steps, s.G_all, s.past_intensity_all, s.future_intensity_all)  # , s )
    else:
        return g2, s.lag_steps[:g_max]  # , s


def cal_g2p(
    FD,
    ring_mask,
    bad_frame_list=None,
    good_start=0,
    num_buf=8,
    num_lev=None,
    imgsum=None,
    norm=None,
    cal_error=False,
):
    """calculation g2 by using a multi-tau algorithm
    for a compressed file with parallel calculation
    if return_g2_details: return g2 with g2_denomitor, g2_past, g2_future
    """
    FD.beg = max(FD.beg, good_start)
    noframes = FD.end - FD.beg + 1  # number of frames, not "no frames"
    for i in range(FD.beg, FD.end):
        pass_FD(FD, i)
    if num_lev is None:
        num_lev = int(np.log(noframes / (num_buf - 1)) / np.log(2) + 1) + 1
    print("In this g2 calculation, the buf and lev number are: %s--%s--" % (num_buf, num_lev))
    if bad_frame_list is not None:
        if len(bad_frame_list) != 0:
            print("%s Bad frames involved and will be discarded!" % len(bad_frame_list))
            noframes -= len(np.where(np.in1d(bad_frame_list, range(good_start, FD.end)))[0])
    print("%s frames will be processed..." % (noframes - 1))
    ring_masks = [np.array(ring_mask == i, dtype=np.int64) for i in np.unique(ring_mask)[1:]]
    qind, pixelist = roi.extract_label_indices(ring_mask)
    noqs = len(np.unique(qind))
    nopr = np.bincount(qind, minlength=(noqs + 1))[1:]
    if norm is not None:
        norms = [
            norm[np.in1d(pixelist, extract_label_indices(np.array(ring_mask == i, dtype=np.int64))[1])]
            for i in np.unique(ring_mask)[1:]
        ]

    inputs = range(len(ring_masks))

    pool = Pool(processes=len(inputs))
    internal_state = None
    print("Starting assign the tasks...")
    results = {}
    if norm is not None:
        for i in tqdm(inputs):
            results[i] = apply_async(
                pool,
                lazy_one_timep,
                (FD, num_lev, num_buf, ring_masks[i], internal_state, bad_frame_list, imgsum, norms[i], cal_error),
            )
    else:
        # print ('for norm is None')
        for i in tqdm(inputs):
            results[i] = apply_async(
                pool,
                lazy_one_timep,
                (FD, num_lev, num_buf, ring_masks[i], internal_state, bad_frame_list, imgsum, None, cal_error),
            )
    pool.close()
    print("Starting running the tasks...")
    res = [results[k].get() for k in tqdm(list(sorted(results.keys())))]
    len_lag = 10**10
    for i in inputs:  # to get the smallest length of lag_step,
        ##*****************************
        ##Here could result in problem for significantly cut useful data if some Q have very short tau list
        ##****************************
        if len_lag > len(res[i][1]):
            lag_steps = res[i][1]
            len_lag = len(lag_steps)

    # lag_steps  = res[0][1]
    if not cal_error:
        g2 = np.zeros([len(lag_steps), len(ring_masks)])
    else:
        g2 = np.zeros([int((num_lev + 1) * num_buf / 2), len(ring_masks)])
        g2_err = np.zeros_like(g2)

        g2_G = np.zeros((int((num_lev + 1) * num_buf / 2), len(pixelist)))
        g2_P = np.zeros_like(g2_G)
        g2_F = np.zeros_like(g2_G)

    Gmax = 0
    lag_steps_err = res[0][1]
    nopr_ = np.lib.pad(np.cumsum(nopr), [1], mode="constant", constant_values=(0))[:-1]
    for i in inputs:
        # print( res[i][0][:,0].shape, g2.shape )
        if not cal_error:
            g2[:, i] = res[i][0][:, 0][:len_lag]
        else:
            s_Gall_qi = res[i][2]  # [:len_lag]
            s_Pall_qi = res[i][3]  # [:len_lag]
            s_Fall_qi = res[i][4]  # [:len_lag]
            # print( s_Gall_qi.shape,s_Pall_qi.shape,s_Fall_qi.shape )
            avgGi = np.average(s_Gall_qi, axis=1)
            devGi = np.std(s_Gall_qi, axis=1)
            avgPi = np.average(s_Pall_qi, axis=1)
            devPi = np.std(s_Pall_qi, axis=1)
            avgFi = np.average(s_Fall_qi, axis=1)
            devFi = np.std(s_Fall_qi, axis=1)

            if len(np.where(avgPi == 0)[0]) != 0:
                g_max1 = np.where(avgPi == 0)[0][0]
            else:
                g_max1 = avgPi.shape[0]
            if len(np.where(avgFi == 0)[0]) != 0:
                g_max2 = np.where(avgFi == 0)[0][0]
            else:
                g_max2 = avgFi.shape[0]
            g_max = min(g_max1, g_max2)
            g2[:g_max, i] = avgGi[:g_max] / (avgPi[:g_max] * avgFi[:g_max])
            g2_err[:g_max, i] = np.sqrt(
                (1 / (avgFi[:g_max] * avgPi[:g_max])) ** 2 * devGi[:g_max] ** 2
                + (avgGi[:g_max] / (avgFi[:g_max] ** 2 * avgPi[:g_max])) ** 2 * devFi[:g_max] ** 2
                + (avgGi[:g_max] / (avgFi[:g_max] * avgPi[:g_max] ** 2)) ** 2 * devPi[:g_max] ** 2
            )
            Gmax = max(g_max, Gmax)
            lag_stepsi = res[i][1]
            if len(lag_steps_err) < len(lag_stepsi):
                lag_steps_err = lag_stepsi

            g2_G[:, nopr_[i] : nopr_[i + 1]] = s_Gall_qi
            g2_P[:, nopr_[i] : nopr_[i + 1]] = s_Pall_qi
            g2_F[:, nopr_[i] : nopr_[i + 1]] = s_Fall_qi

    del results
    del res
    if cal_error:
        print("G2 with error bar calculation DONE!")
        return g2[:Gmax, :], lag_steps_err[:Gmax], g2_err[:Gmax, :] / np.sqrt(nopr), g2_G, g2_P, g2_F
    else:
        print("G2 calculation DONE!")
        return g2, lag_steps


def auto_two_Arrayp(data_pixel, rois, index=None):
    """
    TODO list
    will try to use dask

    Dec 16, 2015, Y.G.@CHX
    a numpy operation method to get two-time correlation function using parallel computation

    Parameters:
        data:  images sequence, shape as [img[0], img[1], imgs_length]
        rois: 2-D array, the interested roi, has the same shape as image, can be rings for saxs, boxes for gisaxs

    Options:

        data_pixel: if not None,
                    2-D array, shape as (len(images), len(qind)),
                    use function Get_Pixel_Array( ).get_data(  ) to get


    Return:
        g12: a 3-D array, shape as ( imgs_length, imgs_length, q)

    One example:
        g12 = auto_two_Array( imgsr, ring_mask, data_pixel = data_pixel )
    """
    qind, pixelist = roi.extract_label_indices(rois)
    noqs = len(np.unique(qind))
    nopr = np.bincount(qind, minlength=(noqs + 1))[1:]
    noframes = data_pixel.shape[0]
    g12b = np.zeros([noframes, noframes, noqs])

    if index is None:
        index = np.arange(1, noqs + 1)
    else:
        try:
            len(index)
            index = np.array(index)
        except TypeError:
            index = np.array([index])
    qlist = np.arange(1, noqs + 1)[index - 1]

    inputs = range(len(qlist))

    data_pixel_qis = [0] * len(qlist)
    for i in inputs:
        pixelist_qi = np.where(qind == qlist[i])[0]
        data_pixel_qis[i] = data_pixel[:, pixelist_qi]

    # pool =  Pool(processes= len(inputs) )
    # results = [ apply_async( pool, _get_two_time_for_one_q, ( qlist[i],
    #                                    data_pixel_qis[i], nopr, noframes ) ) for i in tqdm( inputs )  ]
    # res = [r.get() for r in results]

    pool = Pool(processes=len(inputs))
    results = {}
    for i in inputs:
        results[i] = pool.apply_async(_get_two_time_for_one_q, [qlist[i], data_pixel_qis[i], nopr, noframes])
    pool.close()
    pool.join()
    res = np.array([results[k].get() for k in list(sorted(results.keys()))])

    # print('here')

    for i in inputs:
        qi = qlist[i]
        g12b[:, :, qi - 1] = res[i]
    print("G12 calculation DONE!")
    return g12b  # g12b


def _get_two_time_for_one_q(qi, data_pixel_qi, nopr, noframes):
    # print( data_pixel_qi.shape)

    sum1 = (np.average(data_pixel_qi, axis=1)).reshape(1, noframes)
    sum2 = sum1.T
    two_time_qi = np.dot(data_pixel_qi, data_pixel_qi.T) / sum1 / sum2 / nopr[qi - 1]
    return two_time_qi
