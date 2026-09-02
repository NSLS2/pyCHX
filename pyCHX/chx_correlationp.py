"""
Aug 10, Developed by Y.G.@CHX
yuzhang@bnl.gov
This module is for parallel computation of time correlation
"""

from __future__ import absolute_import, division, print_function

import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import skbeam.core.roi as roi
from skbeam.core.utils import multi_tau_lags
from tqdm import tqdm

from pyCHX._performance import available_memory_bytes, process_one_time_block, sparse_scatter_normalized
from pyCHX.chx_compress import _available_cpu_count, _collect_pool_results, _make_pool, apply_async, pass_FD
from pyCHX.chx_correlationc import _create_intensity_buffer
from pyCHX.chx_correlationc import _one_time_process_cached as _one_time_processp_cached
from pyCHX.chx_correlationc import _one_time_process_error as _one_time_process_errorp
from pyCHX.chx_correlationc import _select_two_time_rois
from pyCHX.chx_correlationc import _two_time_process_cached as _two_time_processp_cached
from pyCHX.chx_correlationc import _validate_and_transform_inputs

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

    pixelist = s.pixel_list
    # iterate over the images to compute multi-tau correlation
    fra_pix = np.zeros_like(pixelist, dtype=np.float64)
    timg = np.zeros(FD.md["ncols"] * FD.md["nrows"], dtype=np.int32)
    timg[pixelist] = np.arange(1, len(pixelist) + 1)
    if bad_frame_list is None:
        bad_frame_list = []
    bad_frames = set(bad_frame_list)
    has_imgsum_norm = imgsum is not None
    has_pixel_norm = norm is not None
    pixel_norm_is_2d = has_pixel_norm and len(norm.shape) > 1
    intensity_buf = _create_intensity_buffer(s.buf, s.label_array, len(s.num_pixels))

    for i in range(FD.beg, FD.end):
        if i in bad_frames:
            fra_pix[:] = np.nan
        else:
            p, v = FD.rdrawframe(i)
            mapped_pixels = timg[p]
            selected = mapped_pixels != 0
            pxlist = mapped_pixels[selected] - 1
            values = v[selected]
            if not has_imgsum_norm:
                if not has_pixel_norm:
                    fra_pix[pxlist] = values
                else:
                    if pixel_norm_is_2d:
                        fra_pix[pxlist] = values / norm[i, pxlist]  # -1.0
                    else:
                        fra_pix[pxlist] = values / norm[pxlist]  # -1.0
            else:
                if not has_pixel_norm:
                    fra_pix[pxlist] = values / imgsum[i]
                else:
                    if pixel_norm_is_2d:
                        fra_pix[pxlist] = values / imgsum[i] / norm[i, pxlist]
                    else:
                        fra_pix[pxlist] = values / imgsum[i] / norm[pxlist]
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
        _two_time_processp_cached(
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
            intensity_buf=intensity_buf,
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
                level_buffer = s.buf[level, s.cur[level] - 1]
                np.add(
                    s.buf[level - 1, prev - 1],
                    s.buf[level - 1, s.cur[level - 1] - 1],
                    out=level_buffer,
                )
                level_buffer /= 2
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
                _two_time_processp_cached(
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
                    intensity_buf=intensity_buf,
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
            noframes -= len(np.where(np.isin(bad_frame_list, range(good_start, FD.end)))[0])
    print("%s frames will be processed..." % (noframes))
    roi_labels = np.unique(ring_mask)
    roi_labels = roi_labels[roi_labels > 0]
    ring_masks = [np.array(ring_mask == label, dtype=np.int64) for label in roi_labels]
    qind, _ = roi.extract_label_indices(ring_mask)
    if norm is not None:
        if len(norm.shape) > 1:
            norms = [norm[:, qind == label] for label in roi_labels]
        else:
            norms = [norm[qind == label] for label in roi_labels]
    inputs = range(len(ring_masks))
    pool = _make_pool(len(inputs))
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
    print("Starting running the tasks...")
    res = _collect_pool_results(pool, results, show_progress=True)

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
        else:
            self.intensity_buf = np.zeros((num_levels, num_bufs, num_rois), dtype=np.float64)

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
    pixelist = s.pixel_list
    # iterate over the images to compute multi-tau correlation
    fra_pix = np.zeros_like(pixelist, dtype=np.float64)
    timg = np.zeros(FD.md["ncols"] * FD.md["nrows"], dtype=np.int32)
    timg[pixelist] = np.arange(1, len(pixelist) + 1)
    if bad_frame_list is None:
        bad_frame_list = []
    intensity_buf = getattr(s, "intensity_buf", None)
    bad_frames = set(bad_frame_list)
    has_imgsum_norm = imgsum is not None
    has_pixel_norm = norm is not None
    pixel_norm_is_2d = has_pixel_norm and len(norm.shape) > 1
    # for  i in tqdm(range( FD.beg , FD.end )):
    for i in range(FD.beg, FD.end):
        if i in bad_frames:
            fra_pix[:] = np.nan
        else:
            p, v = FD.rdrawframe(i)
            mapped_pixels = timg[p]
            selected = mapped_pixels != 0
            pxlist = mapped_pixels[selected] - 1
            values = v[selected]
            if not has_imgsum_norm:
                if not has_pixel_norm:
                    # print ('here')
                    fra_pix[pxlist] = values
                else:
                    if pixel_norm_is_2d:
                        fra_pix[pxlist] = values / norm[i, pxlist]  # -1.0
                    else:
                        fra_pix[pxlist] = values / norm[pxlist]  # -1.0
            else:
                if not has_pixel_norm:
                    fra_pix[pxlist] = values / imgsum[i]
                else:
                    if pixel_norm_is_2d:
                        fra_pix[pxlist] = values / imgsum[i] / norm[i, pxlist]
                    else:
                        fra_pix[pxlist] = values / imgsum[i] / norm[pxlist]

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
            _one_time_processp_cached(
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
                intensity_buf,
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

                level_buffer = s.buf[level, s.cur[level] - 1]
                np.add(
                    s.buf[level - 1, prev - 1],
                    s.buf[level - 1, s.cur[level - 1] - 1],
                    out=level_buffer,
                )
                level_buffer /= 2

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
                    _one_time_processp_cached(
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
                        intensity_buf,
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


def _balance_roi_jobs(pixel_counts, worker_count):
    """Assign ROI indices to workers using largest-pixel-count-first packing."""
    roi_count = len(pixel_counts)
    if roi_count <= worker_count:
        return [[index] for index in range(roi_count)]

    groups = [[] for _ in range(worker_count)]
    group_pixels = np.zeros(worker_count, dtype=np.int64)
    for index in sorted(range(roi_count), key=lambda item: (-pixel_counts[item], item)):
        group = int(np.argmin(group_pixels))
        groups[group].append(index)
        group_pixels[group] += pixel_counts[index]
    return groups


def _run_one_time_group(
    FD,
    num_levels,
    num_bufs,
    ring_mask,
    jobs,
    bad_frame_list,
    imgsum,
    cal_error,
):
    """Calculate one or more ROIs sequentially inside one worker."""
    group_results = []
    for index, label, norm in jobs:
        label_mask = np.asarray(ring_mask == label, dtype=np.int64)
        result = lazy_one_timep(
            FD,
            num_levels,
            num_bufs,
            label_mask,
            None,
            bad_frame_list,
            imgsum,
            norm,
            cal_error,
        )
        group_results.append((index, result))
    return group_results


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
    noframes = FD.end - FD.beg + 1  # preserve the historical level-selection convention
    if num_lev is None:
        num_lev = int(np.log(noframes / (num_buf - 1)) / np.log(2) + 1) + 1
    print("In this g2 calculation, the buf and lev number are: %s--%s--" % (num_buf, num_lev))
    bad_frames = set() if bad_frame_list is None else set(np.atleast_1d(bad_frame_list).tolist())
    if bad_frames:
        print("%s Bad frames involved and will be discarded!" % len(bad_frames))
        noframes -= len(np.where(np.isin(list(bad_frames), range(good_start, FD.end)))[0])
    print("%s frames will be processed..." % (noframes - 1))

    qind, pixel_list = roi.extract_label_indices(ring_mask)
    roi_labels = np.unique(qind)
    if roi_labels.size == 0:
        raise ValueError("ring_mask contains no positive ROI labels")
    original_columns = [np.flatnonzero(qind == label) for label in roi_labels]
    permutation = np.concatenate(original_columns)
    grouped_pixels = np.asarray(pixel_list[permutation], dtype=np.int64)
    pixel_counts = np.asarray([len(columns) for columns in original_columns], dtype=np.int64)
    roi_starts = np.concatenate(([0], np.cumsum(pixel_counts))).astype(np.int64)

    lookup = np.full(FD.md["ncols"] * FD.md["nrows"], -1, dtype=np.int64)
    lookup[grouped_pixels] = np.arange(grouped_pixels.size, dtype=np.int64)
    norm_is_2d = norm is not None and np.ndim(norm) > 1
    norm_1d = np.asarray(norm if norm is not None and not norm_is_2d else np.ones(1), dtype=np.float64)
    norm_2d = np.asarray(norm if norm_is_2d else np.ones((1, 1)), dtype=np.float64)
    norm_columns = np.asarray(permutation if norm is not None else np.zeros(grouped_pixels.size), dtype=np.int64)
    image_sums = np.asarray(imgsum if imgsum is not None else np.ones(1), dtype=np.float64)
    normalization_flags = np.asarray(
        [norm is not None and not norm_is_2d, norm_is_2d, imgsum is not None, False], dtype=np.bool_
    )
    dummy_means = np.ones((1, 1), dtype=np.float64)
    dummy_qind = np.zeros(grouped_pixels.size, dtype=np.int64)

    _, lag_steps, dict_lags = multi_tau_lags(num_lev, num_buf)
    level_lengths = np.asarray([len(dict_lags[key]) for key in dict_lags], dtype=np.int64)
    level_offsets = np.zeros(num_lev, dtype=np.int64)
    if num_lev > 1:
        level_offsets[1:] = np.cumsum(level_lengths[:-1])
    lag_capacity = int((num_lev + 1) * num_buf / 2)

    states = []
    for pixel_count in pixel_counts:
        error_shape = (lag_capacity, int(pixel_count)) if cal_error else (1, 1)
        states.append(
            {
                "buf": np.zeros((num_lev, num_buf, int(pixel_count)), dtype=np.float64),
                "G": np.zeros(lag_capacity, dtype=np.float64),
                "past": np.zeros(lag_capacity, dtype=np.float64),
                "future": np.zeros(lag_capacity, dtype=np.float64),
                "images_per_level": np.zeros(num_lev, dtype=np.int64),
                "track_level": np.zeros(num_lev, dtype=np.bool_),
                "current": np.ones(num_lev, dtype=np.int64),
                "bad_counts": np.zeros((num_lev, num_buf), dtype=np.int64),
                "intensity_buf": np.zeros((num_lev, num_buf), dtype=np.float64),
                "G_all": np.zeros(error_shape, dtype=np.float64),
                "past_all": np.zeros(error_shape, dtype=np.float64),
                "future_all": np.zeros(error_shape, dtype=np.float64),
            }
        )

    worker_count = min(len(roi_labels), _available_cpu_count())
    groups = _balance_roi_jobs(pixel_counts, worker_count)
    target_bytes = min(256 * 1024**2, max(8 * grouped_pixels.size, int(available_memory_bytes() * 0.10)))
    block_frames = max(1, min(1024, target_bytes // max(8, 8 * grouped_pixels.size)))

    def process_group(group, block):
        for roi_index in group:
            start, stop = roi_starts[roi_index : roi_index + 2]
            state = states[roi_index]
            process_one_time_block(
                block[:, start:stop],
                state["buf"],
                state["G"],
                state["past"],
                state["future"],
                state["images_per_level"],
                state["track_level"],
                state["current"],
                state["bad_counts"],
                level_offsets,
                state["intensity_buf"],
                state["G_all"],
                state["past_all"],
                state["future_all"],
                cal_error,
            )

    if hasattr(FD, "_ensure_index"):
        FD._ensure_index()
    executor = ThreadPoolExecutor(max_workers=worker_count) if worker_count > 1 else None
    try:
        starts = range(FD.beg, FD.end, block_frames)
        for block_start in tqdm(starts, desc="Correlating frame blocks"):
            block_stop = min(FD.end, block_start + block_frames)
            block = np.zeros((block_stop - block_start, grouped_pixels.size), dtype=np.float64)
            for output_row, frame_index in enumerate(range(block_start, block_stop)):
                if frame_index in bad_frames:
                    block[output_row].fill(np.nan)
                    continue
                if hasattr(FD, "_raw_frame_view"):
                    positions, values = FD._raw_frame_view(frame_index)
                else:
                    positions, values = FD.rdrawframe(frame_index)
                sparse_scatter_normalized(
                    positions,
                    values,
                    lookup,
                    block,
                    output_row,
                    frame_index,
                    norm_1d,
                    norm_2d,
                    norm_columns,
                    image_sums,
                    dummy_means,
                    dummy_qind,
                    normalization_flags,
                )
            if executor is None:
                process_group(groups[0], block)
            else:
                futures = [executor.submit(process_group, group, block) for group in groups]
                for future in futures:
                    future.result()
    finally:
        if executor is not None:
            executor.shutdown()

    if not cal_error:
        roi_results = []
        valid_lengths = []
        for state in states:
            zero_past = np.flatnonzero(state["past"] == 0)
            zero_future = np.flatnonzero(state["future"] == 0)
            g_max1 = int(zero_past[0]) if zero_past.size else state["past"].size
            g_max2 = int(zero_future[0]) if zero_future.size else state["future"].size
            valid_length = min(g_max1, g_max2)
            valid_lengths.append(valid_length)
            roi_results.append(
                state["G"][:valid_length] / (state["past"][:valid_length] * state["future"][:valid_length])
            )
        common_length = min(valid_lengths)
        g2 = np.column_stack([result[:common_length] for result in roi_results])
        print("G2 calculation DONE!")
        return g2, lag_steps[:common_length]

    g2 = np.zeros((lag_capacity, len(roi_labels)), dtype=np.float64)
    g2_err = np.zeros_like(g2)
    maximum_length = 0
    for roi_index, state in enumerate(states):
        avg_g = np.average(state["G_all"], axis=1)
        dev_g = np.std(state["G_all"], axis=1)
        avg_past = np.average(state["past_all"], axis=1)
        dev_past = np.std(state["past_all"], axis=1)
        avg_future = np.average(state["future_all"], axis=1)
        dev_future = np.std(state["future_all"], axis=1)
        zero_past = np.flatnonzero(avg_past == 0)
        zero_future = np.flatnonzero(avg_future == 0)
        g_max1 = int(zero_past[0]) if zero_past.size else avg_past.size
        g_max2 = int(zero_future[0]) if zero_future.size else avg_future.size
        valid_length = min(g_max1, g_max2)
        g2[:valid_length, roi_index] = avg_g[:valid_length] / (avg_past[:valid_length] * avg_future[:valid_length])
        g2_err[:valid_length, roi_index] = np.sqrt(
            (1 / (avg_future[:valid_length] * avg_past[:valid_length])) ** 2 * dev_g[:valid_length] ** 2
            + (avg_g[:valid_length] / (avg_future[:valid_length] ** 2 * avg_past[:valid_length])) ** 2
            * dev_future[:valid_length] ** 2
            + (avg_g[:valid_length] / (avg_future[:valid_length] * avg_past[:valid_length] ** 2)) ** 2
            * dev_past[:valid_length] ** 2
        )
        maximum_length = max(maximum_length, valid_length)
    print("G2 with error bar calculation DONE!")
    return (
        g2[:maximum_length],
        lag_steps[:maximum_length],
        g2_err[:maximum_length] / np.sqrt(pixel_counts),
    )


def cal_GPF(
    FD,
    ring_mask,
    bad_frame_list=None,
    good_start=0,
    num_buf=8,
    num_lev=None,
    imgsum=None,
    norm=None,
    cal_error=True,
):
    """calculation G,P,D by using a multi-tau algorithm
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
            noframes -= len(np.where(np.isin(bad_frame_list, range(good_start, FD.end)))[0])
    print("%s frames will be processed..." % (noframes - 1))
    roi_labels = np.unique(ring_mask)
    roi_labels = roi_labels[roi_labels > 0]
    ring_masks = [np.array(ring_mask == label, dtype=np.int64) for label in roi_labels]
    qind, pixelist = roi.extract_label_indices(ring_mask)
    if norm is not None:
        norms = [norm[qind == label] for label in roi_labels]

    inputs = range(len(ring_masks))
    pool = _make_pool(len(inputs))
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
    print("Starting running the tasks...")
    res = _collect_pool_results(pool, results, show_progress=True)

    # lag_steps  = res[0][1]
    g2_G = np.zeros((int((num_lev + 1) * num_buf / 2), len(pixelist)))
    g2_P = np.zeros_like(g2_G)
    g2_F = np.zeros_like(g2_G)
    _ = 0
    _ = res[0][1]
    # print('Here')
    for i in inputs:
        selected = qind == roi_labels[i]
        g2_G[:, selected] = res[i][2]  # [:len_lag]
        g2_P[:, selected] = res[i][3]  # [:len_lag]
        g2_F[:, selected] = res[i][4]  # [:len_lag]
    del results
    del res
    return g2_G, g2_P, g2_F


def get_g2_from_ROI_GPF(G, P, F, roi_mask):
    """YG. 2018.10.26. Get g2 from G, P, F by giving bins (roi_mask)
    Input:
        G: <I(t) * I(t+tau)>t
        P: <  I(t)  >t
        F: <  I(t+tau) >t
        roi_mask: the roi mask

    Output:
       g2 and g2_err

    """

    qind, pixelist = roi.extract_label_indices(roi_mask)
    roi_labels = np.unique(qind)
    noqs = len(roi_labels)
    g2 = np.zeros([G.shape[0], noqs])
    g2_err = np.zeros([G.shape[0], noqs])
    for column, label in enumerate(roi_labels):
        # G[0].shape is the same as roi_mask shape
        if len(G.shape) > 2:
            s_Gall_qi = G[:, roi_mask == label]
            s_Pall_qi = P[:, roi_mask == label]
            s_Fall_qi = F[:, roi_mask == label]
        # G[0].shape is the same length as pixelist
        else:
            s_Gall_qi = G[:, qind == label]
            s_Pall_qi = P[:, qind == label]
            s_Fall_qi = F[:, qind == label]

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
        # print()
        g2[:g_max, column] = avgGi[:g_max] / (avgPi[:g_max] * avgFi[:g_max])
        g2_err[:g_max, column] = np.sqrt(
            (1 / (avgFi[:g_max] * avgPi[:g_max])) ** 2 * devGi[:g_max] ** 2
            + (avgGi[:g_max] / (avgFi[:g_max] ** 2 * avgPi[:g_max])) ** 2 * devFi[:g_max] ** 2
            + (avgGi[:g_max] / (avgFi[:g_max] * avgPi[:g_max] ** 2)) ** 2 * devPi[:g_max] ** 2
        )

    return g2, g2_err


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
    qind, qlist, nopr = _select_two_time_rois(rois, index)
    noframes = data_pixel.shape[0]
    g12b = np.zeros([noframes, noframes, len(qlist)])

    inputs = range(len(qlist))

    data_pixel_qis = [0] * len(qlist)
    for i in inputs:
        pixelist_qi = np.where(qind == qlist[i])[0]
        data_pixel_qis[i] = data_pixel[:, pixelist_qi]

    # pool =  Pool(processes= len(inputs) )
    # results = [ apply_async( pool, _get_two_time_for_one_q, ( qlist[i],
    #                                    data_pixel_qis[i], nopr, noframes ) ) for i in tqdm( inputs )  ]
    # res = [r.get() for r in results]

    pool = _make_pool(len(inputs))
    results = {}
    for i in inputs:
        results[i] = pool.apply_async(_get_two_time_for_one_q, [data_pixel_qis[i], nopr[i], noframes])
    res = np.array(_collect_pool_results(pool, results))

    # print('here')

    for i in inputs:
        g12b[:, :, i] = res[i]
    print("G12 calculation DONE!")
    return g12b  # g12b


def _get_two_time_for_one_q(data_pixel_qi, pixel_count, noframes):
    # print( data_pixel_qi.shape)

    sum1 = (np.average(data_pixel_qi, axis=1)).reshape(1, noframes)
    sum2 = sum1.T
    two_time_qi = np.dot(data_pixel_qi, data_pixel_qi.T) / sum1 / sum2 / pixel_count
    return two_time_qi
