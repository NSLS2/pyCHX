import os
import pickle as pkl
import struct
import sys

# from Init_for_Timepix import * # the setup file
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pds
from numpy import (
    apply_over_axes,
    arange,
    arctan,
    around,
    array,
    digitize,
    dot,
    exp,
    histogram,
    histogramdd,
    hstack,
    hypot,
    indices,
    int_,
    intersect1d,
    linspace,
    load,
    log,
    log10,
    ma,
    mean,
    mgrid,
    ones,
    pi,
    poly1d,
    polyfit,
    power,
    ravel,
    reshape,
    round,
    save,
    shape,
    sin,
    sqrt,
    std,
    sum,
    unique,
    vstack,
    where,
    zeros,
    zeros_like,
)
from numpy.linalg import lstsq
from tqdm import tqdm

from pyCHX.chx_compress import Multifile, go_through_FD, pass_FD
from pyCHX.chx_libs import multi_tau_lags


def get_timepixel_data(data_dir, filename, time_unit=1):
    """give a csv file of a timepixel data, return x,y,t
    x, pos_x in pixel
    y, pos_y in pixel
    t, arrival time
    time_unit, t*time_unit will convert to second,  in reality, this value is 6.1e-12
    return x,y,t (in second, starting from zero)

    """
    data = pds.read_csv(data_dir + filename)
    #'#Col', ' #Row', ' #ToA',
    # return np.array( data['Col'] ), np.array(data['Row']), np.array(data['GlobalTimeFine']) #*6.1  #in ps
    if time_unit != 1:
        try:
            x, y, t = np.array(data["#Col"]), np.array(data["#Row"]), np.array(data["#ToA"]) * time_unit
        except:
            x, y, t = np.array(data["#Col"]), np.array(data[" #Row"]), np.array(data[" #ToA"]) * time_unit
    else:
        try:
            x, y, t = np.array(data["#Col"]), np.array(data["#Row"]), np.array(data["#ToA"])
        except:
            x, y, t = np.array(data["#Col"]), np.array(data[" #Row"]), np.array(data[" #ToA"])
    return x, y, t - t.min()  # * 25/4096.  #in ns


def get_pvlist_from_post(p, t, binstep=100, detx=256, dety=256):
    """YG.DEV@CHX Nov, 2017 to get a pos, val list of phonton hitting detector by giving
    p (photon hit pos_x * detx +  y (photon hit pos_y), t  (photon hit time), and the time bin
    The most important function for timepix
    Input:
        p: array, int64, coordinate-x * det_x +  coordinate-y
        t: list, int64, photon hit time
        binstep: int,  binstep (in t unit) period
        detx,dety: int/int, the detector size in x and y
    Output:
        positions: int array, (x*detx +y)
        vals: int array, counts of that positions
        counts: int array, counts of that positions in each binstep
    """
    v = (t - t[0]) // binstep
    L = np.max(v) + 1
    arr = np.ravel_multi_index([p, v], [detx * dety, L])
    uval, ind, count = np.unique(arr, return_counts=True, return_index=True)
    ind2 = np.lexsort((p[ind], v[ind]))
    ps = (p[ind])[ind2]
    vs = count[ind2]
    cs = np.bincount(v[ind])
    return ps, vs, cs


def histogram_pt(p, t, binstep=100, detx=256, dety=256):
    """YG.DEV@CHX Nov, 2017 to get a histogram of phonton counts by giving
    p (photon hit pos_x * detx +  y (photon hit pos_y), t  (photon hit time), and the time bin
    The most important function for timepix
    Input:
        p: coordinate-x * det_x +  coordinate-y
        t: photon hit time
        bin t in binstep (in t unit) period
        detx,dety: the detector size in x and y
    Output:
        the hitorgram of photons with bins as binstep (in time unit)
    """
    L = np.max((t - t[0]) // binstep) + 1
    # print(L,x,y, (t-t[0])//binstep)
    arr = np.ravel_multi_index([p, (t - t[0]) // binstep], [detx * dety, L])
    M, N = arr.max(), arr.min()
    da = np.zeros([detx * dety, L])
    da.flat[np.arange(N, M)] = np.bincount(arr - N)
    return da


def histogram_xyt(x, y, t, binstep=100, detx=256, dety=256):
    """YG.DEV@CHX Mar, 2017 to get a histogram of phonton counts by giving
    x (photon hit pos_x), y (photon hit pos_y), t  (photon hit time), and the time bin
    The most important function for timepix
    Input:
        x: coordinate-x
        y: coordinate-y
        t: photon hit time
        bin t in binstep (in t unit) period
        detx,dety: the detector size in x and y
    Output:
        the hitorgram of photons with bins as binstep (in time unit)


    """
    L = np.max((t - t[0]) // binstep) + 1
    # print(L,x,y, (t-t[0])//binstep)
    arr = np.ravel_multi_index([x, y, (t - t[0]) // binstep], [detx, dety, L])
    M, N = arr.max(), arr.min()
    da = np.zeros([detx, dety, L])
    da.flat[np.arange(N, M)] = np.bincount(arr - N)
    return da


def get_FD_end_num(FD, maxend=1e10):
    N = maxend
    for i in range(0, int(maxend)):
        try:
            FD.seekimg(i)
        except:
            N = i
            break
    FD.seekimg(0)
    return N


def compress_timepix_data(
    pos, t, tbins, filename=None, md=None, force_compress=False, nobytes=2, with_pickle=True
):
    """YG.Dev@CHX Nov 20, 2017
    Compress the timepixeldata, in a format of x, y, t
    x: pos_x in pixel
    y: pos_y in pixel
    timepix3 det size 256, 256
    TODOLIST: mask is not working now
    Input:
      pos: 256 * y + x
      t: arrival time in sec
      filename: the output filename
      md: a dict to describle the data info
      force_compress: if False,
                          if already compressed, just it
                          else: compress
                      if True, compress and, if exist, overwrite the already-coompress data
    Return:
      avg_img, imgsum, N (frame number)

    """
    if filename is None:
        filename = "/XF11ID/analysis/Compressed_Data" + "/timpix_uid_%s.cmp" % md["uid"]

    if force_compress:
        print("Create a new compress file with filename as :%s." % filename)
        return init_compress_timepix_data(
            pos, t, tbins, filename=filename, md=md, nobytes=nobytes, with_pickle=with_pickle
        )
    else:
        if not os.path.exists(filename):
            print("Create a new compress file with filename as :%s." % filename)
            return init_compress_timepix_data(
                pos, t, tbins, filename=filename, md=md, nobytes=nobytes, with_pickle=with_pickle
            )
        else:
            print("Using already created compressed file with filename as :%s." % filename)
            return pkl.load(open(filename + ".pkl", "rb"))

            # FD = Multifile(filename, 0, int(1e25)  )
            # return    get_FD_end_num(FD)


def create_timepix_compress_header(md, filename, nobytes=2, bins=1):
    """
    Create the head for a compressed eiger data, this function is for parallel compress
    """
    fp = open(filename, "wb")
    # Make Header 1024 bytes
    # md = images.md
    if bins != 1:
        nobytes = 8
    Header = struct.pack(
        "@16s8d7I916x",
        b"Version-COMPtpx1",
        md["beam_center_x"],
        md["beam_center_y"],
        md["count_time"],
        md["detector_distance"],
        md["frame_time"],
        md["incident_wavelength"],
        md["x_pixel_size"],
        md["y_pixel_size"],
        nobytes,
        md["sy"],
        md["sx"],
        0,
        256,
        0,
        256,
    )
    fp.write(Header)
    fp.close()


def init_compress_timepix_data(pos, t, binstep, filename, mask=None, md=None, nobytes=2, with_pickle=True):
    """YG.Dev@CHX Nov 19, 2017 with optimal algorithm by using complex index techniques

    Compress the timepixeldata, in a format of x, y, t
    x: pos_x in pixel
    y: pos_y in pixel
    timepix3 det size 256, 256
    TODOLIST: mask is not working now
    Input:
      pos: 256 * x + y   #can't be 256*x + y
      t: arrival time in sec
      binstep: int,  binstep (in t unit) period
      filename: the output filename
      md: a dict to describle the data info
    Return:
      N (frame number)

    """
    fp = open(filename, "wb")
    if md is None:
        md = {}
        md["beam_center_x"] = 0
        md["beam_center_y"] = 0
        md["count_time"] = 0
        md["detector_distance"] = 0
        md["frame_time"] = 0
        md["incident_wavelength"] = 0
        md["x_pixel_size"] = 45
        md["y_pixel_size"] = 45
        # nobytes = 2
        md["sx"] = 256
        md["sy"] = 256

    # TODList: for different detector using different md structure, March 2, 2017,

    # 8d include,
    #'bytes', 'nrows', 'ncols', (detsize)
    #'rows_begin', 'rows_end', 'cols_begin', 'cols_end'  (roi)
    Header = struct.pack(
        "@16s8d7I916x",
        b"Version-COMPtpx1",
        md["beam_center_x"],
        md["beam_center_y"],
        md["count_time"],
        md["detector_distance"],
        md["frame_time"],
        md["incident_wavelength"],
        md["x_pixel_size"],
        md["y_pixel_size"],
        nobytes,
        md["sy"],
        md["sx"],
        0,
        256,
        0,
        256,
    )
    fp.write(Header)

    N_ = np.int(np.ceil((t.max() - t.min()) / binstep))
    print("There are %s frames to be compressed..." % (N_ - 1))

    ps, vs, cs = get_pvlist_from_post(pos, t, binstep, detx=md["sx"], dety=md["sy"])
    N = len(cs) - 1  # the last one might don't have full number for bings, so kick off
    css = np.cumsum(cs)
    imgsum = np.zeros(N)
    good_count = 0
    avg_img = np.zeros(
        [md["sy"], md["sx"]], dtype=np.float64
    )  # changed depreciated np.float to np.float64   LW @06/11/2023

    for i in tqdm(range(0, N)):
        if i == 0:
            ind1 = 0
            ind2 = css[i]
        else:
            ind1 = css[i - 1]
            ind2 = css[i]
        # print( ind1, ind2 )
        good_count += 1
        psi = ps[ind1:ind2]
        vsi = vs[ind1:ind2]
        dlen = cs[i]
        imgsum[i] = vsi.sum()
        np.ravel(avg_img)[psi] += vsi
        # print(vs.sum())
        fp.write(struct.pack("@I", dlen))
        fp.write(struct.pack("@{}i".format(dlen), *psi))
        fp.write(struct.pack("@{}{}".format(dlen, "ih"[nobytes == 2]), *vsi))
    fp.close()
    avg_img /= good_count
    # return N -1
    if with_pickle:
        pkl.dump([avg_img, imgsum, N], open(filename + ".pkl", "wb"))
    return avg_img, imgsum, N


def init_compress_timepix_data_light_duty(
    pos, t, binstep, filename, mask=None, md=None, nobytes=2, with_pickle=True
):
    """YG.Dev@CHX Nov 19, 2017
    Compress the timepixeldata, in a format of x, y, t
    x: pos_x in pixel
    y: pos_y in pixel
    timepix3 det size 256, 256
    TODOLIST: mask is not working now
    Input:
      pos: 256 * x + y   #can't be 256*x + y
      t: arrival time in sec
      filename: the output filename
      md: a dict to describle the data info
    Return:
      N (frame number)

    """
    fp = open(filename, "wb")
    if md is None:
        md = {}
        md["beam_center_x"] = 0
        md["beam_center_y"] = 0
        md["count_time"] = 0
        md["detector_distance"] = 0
        md["frame_time"] = 0
        md["incident_wavelength"] = 0
        md["x_pixel_size"] = 45
        md["y_pixel_size"] = 45
        # nobytes = 2
        md["sx"] = 256
        md["sy"] = 256

    # TODList: for different detector using different md structure, March 2, 2017,

    # 8d include,
    #'bytes', 'nrows', 'ncols', (detsize)
    #'rows_begin', 'rows_end', 'cols_begin', 'cols_end'  (roi)
    Header = struct.pack(
        "@16s8d7I916x",
        b"Version-COMPtpx1",
        md["beam_center_x"],
        md["beam_center_y"],
        md["count_time"],
        md["detector_distance"],
        md["frame_time"],
        md["incident_wavelength"],
        md["x_pixel_size"],
        md["y_pixel_size"],
        nobytes,
        md["sy"],
        md["sx"],
        0,
        256,
        0,
        256,
    )
    fp.write(Header)

    tx = np.arange(t.min(), t.max(), binstep)
    N = len(tx)
    imgsum = np.zeros(N - 1)
    print("There are %s frames to be compressed..." % (N - 1))
    good_count = 0
    avg_img = np.zeros(
        [md["sy"], md["sx"]], dtype=np.float64
    )  # changed depreciated np.float to np.float64   LW @06/11/2023
    for i in tqdm(range(N - 1)):
        ind1 = np.argmin(np.abs(tx[i] - t))
        ind2 = np.argmin(np.abs(tx[i + 1] - t))
        # print( 'N=%d:'%i, ind1, ind2 )
        p_i = pos[ind1:ind2]
        ps, vs = np.unique(p_i, return_counts=True)
        np.ravel(avg_img)[ps] += vs
        good_count += 1
        dlen = len(ps)
        imgsum[i] = vs.sum()
        # print(vs.sum())
        fp.write(struct.pack("@I", dlen))
        fp.write(struct.pack("@{}i".format(dlen), *ps))
        fp.write(struct.pack("@{}{}".format(dlen, "ih"[nobytes == 2]), *vs))
    fp.close()
    avg_img /= good_count
    # return N -1
    if with_pickle:
        pkl.dump([avg_img, imgsum, N - 1], open(filename + ".pkl", "wb"))
    return avg_img, imgsum, N - 1


def compress_timepix_data_old(data_pixel, filename, rois=None, md=None, nobytes=2):
    """
    Compress the timepixeldata
    md: a dict to describle the data info
    rois: [y1,y2, x1, x2]

    """
    fp = open(filename, "wb")
    if md is None:
        md = {}
        md["beam_center_x"] = 0
        md["beam_center_y"] = 0
        md["count_time"] = 0
        md["detector_distance"] = 0
        md["frame_time"] = 0
        md["incident_wavelength"] = 0
        md["x_pixel_size"] = 25
        md["y_pixel_size"] = 25
        # nobytes = 2
        md["sx"] = 256
        md["sy"] = 256
        md["roi_rb"] = 0
        md["roi_re"] = md["sy"]
        md["roi_cb"] = 0
        md["roi_ce"] = md["sx"]
    if rois is not None:
        md["roi_rb"] = rois[2]
        md["roi_re"] = rois[3]
        md["roi_cb"] = rois[1]
        md["roi_ce"] = rois[0]

        md["sy"] = md["roi_cb"] - md["roi_ce"]
        md["sx"] = md["roi_re"] - md["roi_rb"]

    # TODList: for different detector using different md structure, March 2, 2017,

    # 8d include,
    #'bytes', 'nrows', 'ncols', (detsize)
    #'rows_begin', 'rows_end', 'cols_begin', 'cols_end'  (roi)
    Header = struct.pack(
        "@16s8d7I916x",
        b"Version-COMPtpx1",
        md["beam_center_x"],
        md["beam_center_y"],
        md["count_time"],
        md["detector_distance"],
        md["frame_time"],
        md["incident_wavelength"],
        md["x_pixel_size"],
        md["y_pixel_size"],
        nobytes,
        md["sy"],
        md["sx"],
        md["roi_rb"],
        md["roi_re"],
        md["roi_cb"],
        md["roi_ce"],
    )

    fp.write(Header)
    fp.write(data_pixel)


class Get_TimePixel_Arrayc(object):
    """
    a class to get intested pixels from a images sequence,
    load ROI of all images into memory
    get_data: to get a 2-D array, shape as (len(images), len(pixellist))

    One example:
        data_pixel =   Get_Pixel_Array( imgsr, pixelist).get_data()
    """

    def __init__(
        self, pos, hitime, tbins, pixelist, beg=None, end=None, norm=None, flat_correction=None, detx=256, dety=256
    ):
        """
        indexable: a images sequences
        pixelist:  1-D array, interest pixel list
        #flat_correction, normalized by flatfield
        #norm, normalized by total intensity, like a incident beam intensity
        """
        self.hitime = hitime
        self.tbins = tbins
        self.tx = np.arange(self.hitime.min(), self.hitime.max(), self.tbins)
        N = len(self.tx)
        if beg is None:
            beg = 0
        if end is None:
            end = N

        self.beg = beg
        self.end = end
        self.length = self.end - self.beg
        self.pos = pos
        self.pixelist = pixelist
        self.norm = norm
        self.flat_correction = flat_correction
        self.detx = detx
        self.dety = dety

    def get_data(self):
        """
        To get intested pixels array
        Return: 2-D array, shape as (len(images), len(pixellist))
        """
        norm = self.norm
        data_array = np.zeros([self.length - 1, len(self.pixelist)])
        print(data_array.shape)

        # fra_pix = np.zeros_like( pixelist, dtype=np.float64)
        timg = np.zeros(self.detx * self.dety, dtype=np.int32)
        timg[self.pixelist] = np.arange(1, len(self.pixelist) + 1)
        n = 0
        tx = self.tx
        N = len(self.tx)
        print("The Produced Array Length is  %d." % (N - 1))
        flat_correction = self.flat_correction
        # imgsum  =  np.zeros(    N   )
        for i in tqdm(range(N - 1)):
            ind1 = np.argmin(np.abs(tx[i] - self.hitime))
            ind2 = np.argmin(np.abs(tx[i + 1] - self.hitime))
            # print( 'N=%d:'%i, ind1, ind2 )
            p_i = self.pos[ind1:ind2]
            pos, val = np.unique(p_i, return_counts=True)
            # print( val.sum() )
            w = np.where(timg[pos])[0]
            pxlist = timg[pos[w]] - 1
            # print( val[w].sum() )
            # fra_pix[ pxlist] = v[w]
            if flat_correction is not None:
                # normalized by flatfield
                data_array[n][pxlist] = val[w]
            else:
                data_array[n][pxlist] = val[w] / flat_correction[pxlist]  # -1.0
            if norm is not None:
                # normalized by total intensity, like a incident beam intensity
                data_array[n][pxlist] /= norm[i]
            n += 1
        return data_array


def apply_timepix_mask(x, y, t, roi):
    y1, y2, x1, x2 = roi
    w = (x < x2) & (x >= x1) & (y < y2) & (y >= y1)
    return x[w], y[w], t[w]


def get_timepixel_data_from_series(data_dir, filename_prefix, total_filenum=72, colms=int(1e5)):
    x = np.zeros(total_filenum * colms)
    y = np.zeros(total_filenum * colms)
    t = zeros(total_filenum * colms)
    for n in range(total_filenum):
        filename = filename_prefix + "_%s.csv" % n
        data = get_timepixel_data(data_dir, filename)
        if n != total_filenum - 1:
            (x[n * colms : (n + 1) * colms], y[n * colms : (n + 1) * colms], t[n * colms : (n + 1) * colms]) = (
                data[0],
                data[1],
                data[2],
            )
        else:
            # print(  filename_prefix + '_%s.csv'%n )
            ln = len(data[0])
            # print( ln )
            (x[n * colms : n * colms + ln], y[n * colms : n * colms + ln], t[n * colms : n * colms + ln]) = (
                data[0],
                data[1],
                data[2],
            )

    return x[: n * colms + ln], y[: n * colms + ln], t[: n * colms + ln]


def get_timepixel_avg_image(x, y, t, det_shape=[256, 256], delta_time=None):
    """YG.Dev@CHX, 2016
    give x,y, t data to get image in a period of delta_time (in second)
    x, pos_x in pixel
    y, pos_y in pixel
    t, arrival time


    """
    t0 = t.min()
    tm = t.max()

    if delta_time is not None:
        delta_time *= 1e12
        if delta_time > tm:
            delta_time = tm
    else:
        delta_time = t.max()
    # print( delta_time)
    t_ = t[t < delta_time]
    x_ = x[: len(t_)]
    y_ = y[: len(t_)]

    img = np.zeros(det_shape, dtype=np.int32)
    pixlist = x_ * det_shape[0] + y_
    his = np.histogram(pixlist, bins=np.arange(det_shape[0] * det_shape[1] + 1))[0]
    np.ravel(img)[:] = his
    print("The max photon count is %d." % img.max())
    return img


def get_his_taus(t, bin_step):
    """Get taus and  histrogram of photons
    Parameters:
         t: the time stamp of photon hitting the detector
         bin_step: bin time step, in unit of ms
    Return:
         taus, in ms
         histogram of photons

    """

    bins = np.arange(t.min(), t.max(), bin_step)  # 1e6 for us
    # print( bins )
    td = np.histogram(t, bins=bins)[0]  # do histogram
    taus = bins - bins[0]
    return taus[1:], td


def get_multi_tau_lags(oned_count, num_bufs=8):
    n = len(oned_count)
    num_levels = int(np.log(n / (num_bufs - 1)) / np.log(2) + 1) + 1
    tot_channels, lag_steps, dict_lag = multi_tau_lags(num_levels, num_bufs)
    return lag_steps[lag_steps < n]


def get_timepixel_multi_tau_g2(oned_count, num_bufs=8):
    n = len(oned_count)
    lag_steps = get_multi_tau_lags(oned_count, num_bufs)
    g2 = np.zeros(len(lag_steps))

    for tau_ind, tau in enumerate(lag_steps):
        IP = oned_count[: n - tau]
        IF = oned_count[tau:n]
        # print( IP.shape,IF.shape, tau, n )
        g2[tau_ind] = np.dot(IP, IF) / (IP.mean() * IF.mean() * float(n - tau))
    return g2


def get_timepixel_c12(oned_count):
    noframes = len(oned_count)
    oned_count = oned_count.reshape(noframes, 1)
    return np.dot(oned_count, oned_count.T) / oned_count / oned_count.T / noframes


def get_timepixel_g2(oned_count):
    n = len(oned_count)
    norm = (
        np.arange(n, 0, -1)
        * np.array([np.average(oned_count[i:]) for i in range(n)])
        * np.array([np.average(oned_count[0 : n - i]) for i in range(n)])
    )
    return np.correlate(oned_count, oned_count, mode="full")[-n:] / norm


#########################################
T = True
F = False


def read_xyt_frame(n=1):
    """Load the xyt txt files:
    x,y is the detector (x,y) coordinates
    t is the time-encoder (when hitting the detector at that (x,y))
    DATA_DIR is the data filefold path
    DataPref is the data prefix
    n is file number
    the data name will be like: DATA_DIR/DataPref_0001.txt
    return the histogram of the hitting event
    """
    import numpy as np

    ni = "%04d" % n
    fp = DATA_DIR + DataPref + "%s.txt" % ni
    data = np.genfromtxt(fp, skiprows=0)[:, 2]  # take the time encoder
    td = np.histogram(data, bins=np.arange(11810))[0]  # do histogram
    return td


def readframe_series(n=1):
    """Using this universe name for all the loading fucntions"""
    return read_xyt_frame(n)


class xpcs(object):
    def __init__(self):
        """DOCUMENT __init__(   )
        the initilization of the XPCS class
        """
        self.version = "version_0"
        self.create_time = "July_14_2015"
        self.author = "Yugang_Zhang@chx11id_nsls2_BNL"

    def delays(self, time=1, nolevs=None, nobufs=None, tmaxs=None):
        """DOCUMENT delays(time=)
        Using the lev,buf concept, to generate array of time delays
        return array of delays.
        KEYWORD:  time: scale delays by time ( should be time between frames)
                  nolevs: lev (a integer number)
                  nobufs: buf (a integer number)
                  tmax:   the max time in the calculation, usually, the noframes

        """

        if nolevs is None:
            nolevs = nolev  # defined by the set-up file
        if nobufs is None:
            nobufs = nobuf  # defined by the set-up file
        if tmaxs is None:
            tmaxs = tmax  # defined by the set-up file
        if nobufs % 2 != 0:
            print("nobuf must be even!!!")
        dly = zeros((nolevs + 1) * nobufs / 2 + 1)
        dict_dly = {}
        for i in range(1, nolevs + 1):
            if i == 1:
                imin = 1
            else:
                imin = nobufs / 2 + 1
            ptr = (i - 1) * nobufs / 2 + arange(imin, nobufs + 1)
            dly[ptr] = arange(imin, nobufs + 1) * 2 ** (i - 1)
            dict_dly[i] = dly[ptr - 1]
        dly *= time
        dly = dly[:-1]
        dly_ = dly[: where(dly < tmaxs)[0][-1] + 1]
        self.dly = dly
        self.dly_ = dly_
        self.dict_dly = dict_dly
        return dly

    def make_qlist(self):
        """DOCUMENT make_qlist( )
        Giving the noqs, qstart,qend,qwidth, defined by the set-up file
        return qradi: a list of q values, [qstart, ...,qend] with length as noqs
               qlist: a list of q centered at qradi with qwidth.
        KEYWORD:  noqs, qstart,qend,qwidth::defined by the set-up file
        """
        qradi = linspace(qstart, qend, noqs)
        qlist = zeros(2 * noqs)
        qlist[::2] = round(qradi - qwidth / 2)  # render  even value
        qlist[1::2] = round(qradi + (1 + qwidth) / 2)  # render odd value
        qlist[::2] = int_(qradi - qwidth / 2)  # render  even value
        qlist[1::2] = int_(qradi + (1 + qwidth) / 2)  # render odd value
        if qlist_ != None:
            qlist = qlist_
        return qlist, qradi

    def calqlist(self, qmask=None, shape="circle"):
        """DOCUMENT calqlist( qmask=,shape=, )
        calculate the equvilent pixel with a shape,
        return
            qind: the index of q
            pixellist: the list of pixle
            nopr: pixel number in each q
            nopixels: total pixel number
        KEYWORD:
            qmask, a mask file;
            qlist,qradi is calculated by make_qlist()
            shape='circle', give a circle shaped qlist
            shape='column', give a column shaped qlist
            shape='row', give a row shaped qlist
        """

        qlist, qradi = self.make_qlist()
        y, x = indices([dimy, dimx])
        if shape == "circle":
            y_ = y - ceny + 1
            x_ = x - cenx + 1
            r = int_(hypot(x_, y_) + 0.5)
        elif shape == "column":
            r = x
        elif shape == "row":
            r = y
        else:
            pass
        r = r.flatten()
        noqrs = len(qlist)
        qind = digitize(r, qlist)
        if qmask is None:
            w_ = where((qind) % 2)  # qind should be odd;print 'Yes'
            w = w_[0]
        else:
            a = where((qind) % 2)[0]
            b = where(mask.flatten() == False)[0]
            w = intersect1d(a, b)
        nopixels = len(w)
        qind = qind[w] / 2
        pixellist = (y * dimx + x).flatten()[w]
        nopr, bins = histogram(qind, bins=range(len(qradi) + 1))
        return qind, pixellist, nopr, nopixels

    ###########################################################################
    ########for one_time correlation function for xyt frames
    ##################################################################

    def autocor_xytframe(self, n):
        """Do correlation for one xyt frame--with data name as n"""
        dly_ = xp.dly_
        # cal=0
        gg2 = zeros(len(dly_))
        data = read_xyt_frame(n)  # load data
        datm = len(data)
        for tau_ind, tau in enumerate(dly_):
            IP = data[: datm - tau]
            IF = data[tau:datm]
            gg2[tau_ind] = dot(IP, IF) / (IP.mean() * IF.mean() * float(datm - tau))

        return gg2

    def autocor(self, noframes=10):
        """Do correlation for xyt file,
        noframes is the frame number to be correlated
        """
        start_time = time.time()
        for n in range(1, noframes + 1):  # the main loop for correlator
            gg2 = self.autocor_xytframe(n)
            if n == 1:
                g2 = zeros_like(gg2)
            g2 += (gg2 - g2) / float(n)  # average  g2
            # print n
            if noframes > 10:  # print progress...
                if n % (noframes / 10) == 0:
                    sys.stdout.write("#")
                    sys.stdout.flush()
        elapsed_time = time.time() - start_time
        print("Total time: %.2f min" % (elapsed_time / 60.0))
        return g2

    def plot(self, y, x=None):
        """a simple plot"""
        if x is None:
            x = arange(len(y))
        plt.plot(x, y, "ro", ls="-")
        plt.show()

    def g2_to_pds(self, dly, g2, tscale=None):
        """convert g2 to a pandas frame"""
        if len(g2.shape) == 1:
            g2 = g2.reshape([len(g2), 1])
        tn, qn = g2.shape
        tindex = xrange(tn)
        qcolumns = ["t"] + ["g2"]
        if tscale is None:
            tscale = 1.0
        g2t = hstack([dly[:tn].reshape(tn, 1) * tscale, g2])
        g2p = pd.DataFrame(data=g2t, index=tindex, columns=qcolumns)
        return g2p

    def show(self, g2p, title):
        t = g2p.t
        N = len(g2p)
        ylim = [g2p.g2.min(), g2p[1:N].g2.max()]
        g2p.plot(x=t, y="g2", marker="o", ls="--", logx=T, ylim=ylim)
        plt.xlabel("time delay, ns", fontsize=12)
        plt.title(title)
        plt.savefig(RES_DIR + title + ".png")
        plt.show()


######################################################

if False:
    xp = xpcs()
    # use the xpcs class
    dly = xp.delays()
    if T:
        fnum = 100
        g2 = xp.autocor(fnum)
        filename = "g2_-%s-" % (fnum)
        save(RES_DIR + FOUT + filename, g2)
        ##g2= load(RES_DIR + FOUT + filename +'.npy')
        g2p = xp.g2_to_pds(dly, g2, tscale=20)
        xp.show(g2p, "g2_run_%s" % fnum)
