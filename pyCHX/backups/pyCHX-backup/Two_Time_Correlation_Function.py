######################################################################################
########Dec 16, 2015, Yugang Zhang, yuzhang@bnl.gov, CHX, NSLS-II, BNL################
########Time correlation function, include one-time, two-time, four-time##############
########Muli-tau method, array-operation method#######################################
######################################################################################


import itertools
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import skbeam.core.roi as roi
from matplotlib import gridspec
from matplotlib.colors import LogNorm
from modest_image import ModestImage, imshow
from tqdm import tqdm

# from pyCHX.chx_libs import  colors_ as mcolors,  markers_ as markers
from pyCHX.chx_libs import RUN_GUI, Figure
from pyCHX.chx_libs import colors
from pyCHX.chx_libs import colors as colors_array
from pyCHX.chx_libs import lstyles
from pyCHX.chx_libs import markers
from pyCHX.chx_libs import markers as markers_array
from pyCHX.chx_libs import markers_copy, mcolors, multi_tau_lags


def delays(num_lev=3, num_buf=4, time=1):
    """DOCUMENT delays(time=)
    return array of delays.
    KEYWORD:  time: scale delays by time ( should be time between frames)
    """
    if num_buf % 2 != 0:
        print("nobuf must be even!!!")
    dly = np.zeros((num_lev + 1) * int(num_buf / 2) + 1)
    dict_dly = {}
    for i in range(1, num_lev + 1):
        if i == 1:
            imin = 1
        else:
            imin = int(num_buf / 2) + 1
        ptr = (i - 1) * int(num_buf / 2) + np.arange(imin, num_buf + 1)
        dly[ptr] = np.arange(imin, num_buf + 1) * 2 ** (i - 1)
        dict_dly[i] = dly[ptr - 1]
        dly *= time
        # print (i, ptr, imin)
    return dly, dict_dly


class Get_Pixel_Array(object):
    """
    Dec 16, 2015, Y.G.@CHX
    a class to get intested pixels from a images sequence,
    load ROI of all images into memory
    get_data: to get a 2-D array, shape as (len(images), len(pixellist))

    One example:
        data_pixel =   Get_Pixel_Array( imgsr, pixelist).get_data()
    """

    def __init__(self, indexable, pixelist):
        """
        indexable: a images sequences
        pixelist:  1-D array, interest pixel list
        """
        self.indexable = indexable
        self.pixelist = pixelist
        # self.shape = indexable.shape
        try:
            self.length = len(indexable)
        except:
            self.length = indexable.length

    def get_data(self):
        """
        To get intested pixels array
        Return: 2-D array, shape as (len(images), len(pixellist))
        """

        # print (self.length)
        data_array = np.zeros([self.length, len(self.pixelist)])
        for key in tqdm(range(self.length)):
            data_array[key] = np.ravel(self.indexable[key])[self.pixelist]
        return data_array


class Reverse_Coordinate(object):
    """obsolete codes"""

    def __init__(self, indexable, mask):
        self.indexable = indexable
        self.mask = mask
        try:
            self.shape = indexable.shape
        except:
            # if
            self.shape = [len(indexable), indexable[0].shape[0], indexable[0].shape[1]]
        # self.shape = indexable.shape
        self.length = len(indexable)

    def __getitem__(self, key):
        if self.mask is not None:
            img = self.indexable[key] * self.mask
        else:
            img = self.indexable[key]

        if len(img.shape) == 3:
            img_ = img[:, ::-1, :]
        if len(img.shape) == 2:
            img_ = img[::-1, :]
        return img_


def get_mean_intensity(data_pixel, qind):
    """
    Dec 16, 2015, Y.G.@CHX
    a function to get mean intensity as a function of time (image number)

    Parameters:
        data_pixel: 2-D array, shape as (len(images), len(qind)),
                    use function Get_Pixel_Array( ).get_data(  ) to get
        qind: 1-D int array, a index list of interest pixel, values change from 1 to int number

    Return:
        mean_inten: a dict, with keys as the unique values of qind,
                    each dict[key]: 1-D array, with shape as data_pixel.shape[0],namely, len(images)

    One example:
        mean_inten = get_mean_intensity( data_pixel, qind)
    """

    noqs = len(np.unique(qind))
    mean_inten = {}

    for qi in range(1, noqs + 1):
        pixelist_qi = np.where(qind == qi)[0]
        # print (pixelist_qi.shape,  data_pixel[qi].shape)
        data_pixel_qi = data_pixel[:, pixelist_qi]
        mean_inten[qi] = data_pixel_qi.mean(axis=1)
    return mean_inten


def run_time(t0):
    """Calculate running time of a program
    Dec 16, 2015, Y.G.@CHX
    Parameters
    ----------
    t0: time_string, t0=time.time()
        The start time
    Returns
    -------
    Print the running time

    One usage
    ---------
    t0=time.time()
    .....(the running code)
    run_time(t0)
    """

    elapsed_time = time.time() - t0
    print("Total time: %.2f min" % (elapsed_time / 60.0))


def get_each_frame_ROI_intensity(data_pixel, bad_pixel_threshold=1e10, plot_=False, *argv, **kwargs):
    """
    Dec 16, 2015, Y.G.@CHX
    Get the ROI intensity of each frame
       Also get bad_frame_list by check whether above  bad_pixel_threshold

       Usuage:
       imgsum, bad_frame_list = get_each_frame_intensity( data_pixel,
                                bad_pixel_threshold=1e10,  plot_ = True)
    """

    # print ( argv, kwargs )
    imgsum = np.array([np.sum(img) for img in tqdm(data_series[::sampling], leave=True)])
    if plot_:
        uid = "uid"
        if "uid" in kwargs.keys():
            uid = kwargs["uid"]
        fig, ax = plt.subplots()
        ax.plot(imgsum, "bo")
        ax.set_title("uid= %s--imgsum" % uid)
        ax.set_xlabel("Frame_bin_%s" % sampling)
        ax.set_ylabel("Total_Intensity")

        if save:
            # dt =datetime.now()
            # CurTime = '%s%02d%02d-%02d%02d-' % (dt.year, dt.month, dt.day,dt.hour,dt.minute)
            path = kwargs["path"]
            # fp = path + "uid= %s--Waterfall-"%uid + CurTime + '.png'
            fp = path + "uid=%s--imgsum-" % uid + ".png"
            fig.savefig(fp, dpi=fig.dpi)
        # plt.show()
    bad_frame_list = np.where(np.array(imgsum) > bad_pixel_threshold)[0]
    if len(bad_frame_list):
        print("Bad frame list are: %s" % bad_frame_list)
    else:
        print("No bad frames are involved.")
    return imgsum, bad_frame_list


def auto_two_Array(data, rois, data_pixel=None):
    """
    Dec 16, 2015, Y.G.@CHX
    a numpy operation method to get two-time correlation function

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

    start_time = time.time()

    qind, pixelist = roi.extract_label_indices(rois)
    noqs = len(np.unique(qind))
    nopr = np.bincount(qind, minlength=(noqs + 1))[1:]

    if data_pixel is None:
        data_pixel = Get_Pixel_Array(data, pixelist).get_data()
        # print (data_pixel.shape)

    noframes = data_pixel.shape[0]
    g12b = np.zeros([noframes, noframes, noqs])
    Unitq = noqs / 10
    proi = 0

    for qi in tqdm(range(1, noqs + 1)):
        pixelist_qi = np.where(qind == qi)[0]
        # print (pixelist_qi.shape,  data_pixel[qi].shape)
        data_pixel_qi = data_pixel[:, pixelist_qi]

        sum1 = (np.average(data_pixel_qi, axis=1)).reshape(1, noframes)
        sum2 = sum1.T

        g12b[:, :, qi - 1] = np.dot(data_pixel_qi, data_pixel_qi.T) / sum1 / sum2 / nopr[qi - 1]
        # print ( proi, int( qi //( Unitq) ) )
    #        if  int( qi //( Unitq) ) == proi:
    #            sys.stdout.write("#")
    #            sys.stdout.flush()
    #            proi += 1

    elapsed_time = time.time() - start_time
    print("Total time: %.2f min" % (elapsed_time / 60.0))

    return g12b


####################################
##Derivation of Two time correlation
#####################################


#####################################
# get one-time @different age
#####################################


def get_qedge2(qstart, qend, qwidth, noqs, return_int=False):
    """DOCUMENT make_qlist( )
    give qstart,qend,qwidth,noqs
    return a qedge by giving the noqs, qstart,qend,qwidth.
           a qcenter, which is center of each qedge
    KEYWORD:  None"""
    import numpy as np

    qcenter = np.linspace(qstart, qend, noqs)
    # print ('the qcenter is:  %s'%qcenter )
    qedge = np.zeros(2 * noqs)
    qedge[::2] = qcenter - (qwidth / 2)  # +1  #render  even value
    qedge[1::2] = qcenter + qwidth / 2  # render odd value
    if not return_int:
        return qedge, qcenter
    else:
        return np.int(qedge), np.int(qcenter)


def get_qedge(qstart, qend, qwidth, noqs, return_int=False):
    """DOCUMENT make_qlist( )
    give qstart,qend,qwidth,noqs
    return a qedge by giving the noqs, qstart,qend,qwidth.
           a qcenter, which is center of each qedge
    KEYWORD:  None"""
    import numpy as np

    qcenter = np.linspace(qstart, qend, noqs)
    # print ('the qcenter is:  %s'%qcenter )
    qedge = np.zeros([noqs, 2])
    qedge[:, 0] = qcenter - (qwidth / 2)  # +1  #render  even value
    qedge[:, 1] = qcenter + qwidth / 2  # render odd value
    if not return_int:
        return qedge, qcenter
    else:
        return np.int(qedge), np.int(qcenter)


def get_time_edge(tstart, tend, twidth, nots, return_int=False):
    """Get time edge and time center by giving tstart, tend, twidth,nots
    Return:
        tedge: array, [ [ tedge1_start, tedge1_end], [ tedge2_start, tedge2_end], ...  ]
        tcenter: array, [tcenter1, tcenter2, ...]
        if return_int = True, return tedge, tcenter in integer
    """
    import numpy as np

    tcenter = np.linspace(tstart, tend, nots)
    # print ('the qcenter is:  %s'%qcenter )
    tedge = np.zeros([nots, 2])
    tedge[:, 0] = tcenter - (twidth / 2)  # +1  #render  even value
    tedge[:, 1] = tcenter + twidth / 2  # render odd value
    if not return_int:
        return tedge, tcenter
    else:
        return np.int(tedge), np.int(tcenter)


def rotate_g12q_to_rectangle(g12q):
    """
    Dec 16, 2015, Y.G.@CHX
    Rotate anti clockwise 45 of a one-q two correlation function along diagonal to a masked array
    the shape ( imgs_length, imgs_length )  of g12q will change to ( imgs_length, 2*imgs_length -1)


    Parameters:
        g12q: a 2-D array, one-q two correlation function, shape as ( imgs_length, imgs_length )


    Return:
        g12qr: a masked 2-D array, shape as ( imgs_length, 2*imgs_length -1 )
           x-axis: taus, from 0 to imgs_length
           y-axis: ages, from 0 to imgs_length( the middle of y) to 2imgs_length-1 (top)
    One example:
        g12qr = rotate_g12q_to_rectangle(g12bm[:,:,0] )
    """
    M, N = g12q.shape
    g12qr = np.ma.empty((2 * N - 1, N))
    g12qr.mask = True
    for i in range(N):
        g12qr[i : (2 * N - 1 - i) : 2, i] = g12q.diagonal(i)
    return g12qr


def get_aged_g2_from_g12(g12, age_edge, age_center):
    """
    Dec 16, 2015, Y.G.@CHX
    Get one-time correlation function of different age from two correlation function
    namely, calculate the different aged mean of each diag line of g12 to get one-time correlation fucntion

    Parameters:
        g12: a 3-D array, a  two correlation function, shape as ( imgs_length, imgs_length, noqs )

    Options:
        slice_num: int, the slice number of the diagonal of g12
        slice_width: int,  each slice width in unit of pixel
        slice start: int, can start from 0
        slice end: int, can end at 2*imgs_length -1


    Return:
        g2_aged: a dict, one time correlation function at different age
                 the keys of dict is ages in unit of pixel
                 dict[key]:
                           a two-D array, shape as ( imgs_length ),
                           a multi-q one-time correlation function
    One example:
        g2_aged = get_aged_g2_from_g12( g12, slice_num =3, slice_width= 500,
                slice_start=4000, slice_end= 20000-4000  )
    """

    m, n, noqs = g12.shape
    g2_aged = {}
    for q in range(noqs):
        g12q = g12[:, :, q]
        g2q_aged = get_aged_g2_from_g12q(g12q, age_edge, age_center)
        if q == 0:
            keys = list(g2q_aged.keys())
        for key in keys:
            if q == 0:
                g2_aged[key] = np.zeros([len(g2q_aged[key]), noqs])
            g2_aged[key][:, q] = g2q_aged[key]
            # print( q, key )

    return g2_aged


def get_aged_g2_from_g12q(g12q, age_edge, age_center=None, timeperframe=1, time_sampling="log", num_bufs=8):
    """


    Revised at Octo 20, 2017, correct age, should be (t1+t2)/2, namely, age_edge will *2, age_center will keep same

    Revised at Sep 28, 2017 add time_sampling='log', num_bufs=8 options
    Dec 16, 2015, Y.G.@CHX
    Revised at April 19, 2017
    Get one-time correlation function of different age from 1q-two correlation function
    namely, calculate the different aged mean of each diag line of g12 to get one-time correlation fucntion

    Parameters:
        g12q: a 2-D array, one-q two correlation function, shape as ( imgs_length, imgs_length )
        age_edge, list, e.g.,  [[0, 500], [2249, 2749], [4500, 5000]]
                       can be obtained by function:
                    age_edge = create_time_slice( len(imgsa), slice_num= 3, slice_width= 500, edges = None )
    Options:
       timeperframe: in unit of sec
       age_center: None, will use the center of age_edge
       time_sampling: 'log', log sampling of taus; Or 'linear', namely, delta tau = 1
       num_bufs: 8, buf number for log sampling tau


    Return:

        g2_aged: a dict, one time correlation function at different age
                 the keys of dict is ages in unit of pixel
                 dict[key]:
                           a 1-D array, shape as ( imgs_length ),
                           a one-q one-time correlation function
    One example:
        g2_aged = get_aged_g2_from_g12q( g12q, age_edge  )
    """

    arr = rotate_g12q_to_rectangle(g12q)
    m, n = arr.shape  # m should be 2*n-1
    # age_edge, age_center = get_qedge( qstart=slice_start,qend= slice_end,
    #                 qwidth = slice_width, noqs =slice_num  )
    # print(arr.shape)
    age_edge = np.int_(age_edge)
    if age_center is None:
        age_center = (age_edge[:, 0] + age_edge[:, 1]) // 2

    age_edge_ = age_edge * 2
    age_center_ = age_center * timeperframe
    g2_aged = {}
    lag_dict = {}
    # print( age_edge, age_center)
    for i, age in enumerate(age_center_):
        age_edges_0, age_edges_1 = age_edge_[i][0], age_edge_[i][1]
        # print(i, age, age_edges_0, age_edges_1)
        g2i = arr[age_edges_0:age_edges_1].mean(axis=0)
        # print('here')
        g2i_ = np.array(g2i)
        g2_aged[age] = g2i_[np.nonzero(g2i_)[0]]
        N = len(g2_aged[age])
        lag_dict[age] = np.arange(N) * 1.0
        if time_sampling == "log":
            num_levels = int(np.log(N / (num_bufs - 1)) / np.log(2) + 1) + 1
            tot_channels, lag_steps, dict_lag = multi_tau_lags(num_levels, num_bufs)
            # max_taus= lag_steps[age].max()
            lag_steps_ = lag_steps[lag_steps < N]
            # print(i, age, lag_steps, N, lag_steps_, len(g2_aged[age]))
            g2_aged[age] = g2_aged[age][lag_steps_]
            lag_dict[age] = lag_steps_ * 1.0
            # print( lag_dict[age] )
        lag_dict[age] *= timeperframe

    return lag_dict, g2_aged


def get_aged_g2_from_g12q2(g12q, slice_num=6, slice_width=5, slice_start=0, slice_end=1):
    """
    Dec 16, 2015, Y.G.@CHX
    Get one-time correlation function of different age from two correlation function
    namely, calculate the different aged mean of each diag line of g12 to get one-time correlation fucntion

    Parameters:
        g12q: a 2-D array, one-q two correlation function, shape as ( imgs_length, imgs_length )

    Options:
        slice_num: int, the slice number of the diagonal of g12
        slice_width: int,  each slice width in unit of pixel
        slice start: int, can start from 0
        slice end: int, can end at 2*imgs_length -1


    Return:
        g2_aged: a dict, one time correlation function at different age
                 the keys of dict is ages in unit of pixel
                 dict[key]:
                           a 1-D array, shape as ( imgs_length ),
                           a one-q one-time correlation function
    One example:
        g2_aged = get_aged_g2_from_g12q( g12q, slice_num =3, slice_width= 500,
                slice_start=4000, slice_end= 20000-4000  )
    """

    arr = rotate_g12q_to_rectangle(g12q)
    m, n = arr.shape  # m should be 2*n-1
    age_edge, age_center = get_qedge(qstart=slice_start, qend=slice_end, qwidth=slice_width, noqs=slice_num)
    age_edge, age_center = np.int_(age_edge), np.int_(age_center)
    # print (age_edge, age_center)
    g2_aged = {}
    for i, age in enumerate(age_center):
        age_edges_0, age_edges_1 = age_edge[i * 2 : 2 * i + 2]
        g2i = arr[age_edges_0:age_edges_1].mean(axis=0)
        g2i_ = np.array(g2i)
        g2_aged[age] = g2i_[np.nonzero(g2i_)[0]]

    return g2_aged


def show_g12q_aged_g2(
    g12q,
    g2_aged,
    taus_aged=None,
    slice_width=10,
    timeperframe=1,
    vmin=1,
    vmax=1.25,
    save=True,
    uid="uid",
    path="",
    *argv,
    **kwargs
):
    """
    Octo 20, 2017, add taus_aged option

    Dec 16, 2015, Y.G.@CHX
    Plot one-time correlation function of different age with two correlation function
    Parameters:
        g12q: a 2-D array, one-q two correlation function, shape as ( imgs_length, imgs_length )
        tau_aged: a dict, taus for different age
        g2_aged: a dict, one time correlation function at different age
                 obtained by: for example,
                 g2_aged = get_aged_g2_from_g12q( g12q, slice_num =3, slice_width= 500,
                                                  slice_start=4000, slice_end= 20000-4000  )
                 the keys of dict is ages in unit of pixel
                 dict[key]:
                           a 1-D array, shape as ( imgs_length ),
                           a one-q one-time correlation function

    Options:
        slice_width: int,  each slice width in unit of pixel, for line width of a plot
        timeperframe: float, time per frame for axis unit
        vmin, float, matplot vmin
        vmax, float, matplot vmax

    Return:
         two plots, one for the two-time correlation, g12q,

    One example:
        show_g12q_aged_g2( g12q, g2_aged,timeperframe=1,vmin= 1, vmax= 1.22 )
    """

    age_center = np.array(list(sorted(g2_aged.keys())))
    print("the cut age centers are: " + str(age_center))
    age_center = np.int_(np.array(list(sorted(g2_aged.keys()))) / timeperframe) * 2  # in pixel
    M, N = g12q.shape

    # fig, ax = plt.subplots( figsize = (8,8) )

    figw = 10
    figh = 8
    fig = plt.figure(figsize=(figw, figh))

    # gs = gridspec.GridSpec(1, 2, width_ratios=[10, 8],height_ratios=[8,8]   )
    gs = gridspec.GridSpec(1, 2)
    ax = plt.subplot(gs[0])
    im = imshow(ax, g12q, origin="lower", cmap="viridis", norm=LogNorm(vmin, vmax), extent=[0, N, 0, N])

    # plt.gca().set_xticks(ticks)
    ticks = np.round(plt.gca().get_xticks() * timeperframe, 2)
    # print( ticks )
    ax.set_xticklabels(ticks)
    ax.set_yticklabels(ticks)
    # plt.xticks(ticks, fontsize=9)

    # ), extent=[0, g12q.shape[0]*timeperframe, 0, g12q.shape[0]*timeperframe ] )

    ax1 = plt.subplot(gs[1])
    linS1 = [[0] * len(age_center), np.int_(age_center - slice_width // 2)]
    linS2 = [[0] * len(age_center), np.int_(age_center + slice_width // 2)]
    linE1 = [np.int_(age_center - slice_width // 2), [0] * len(age_center)]
    linE2 = [np.int_(age_center + slice_width // 2), [0] * len(age_center)]
    linC = [[0] * len(age_center), np.int_(age_center)]

    for i in range(len(age_center)):
        ps = linS1[1][i]
        pe = linE1[0][i]
        if ps >= N:
            s0 = ps - N
            s1 = N
        else:
            s0 = 0
            s1 = ps
        e0 = s1
        e1 = s0
        # if pe>=N:e0=N;e1=pe - N
        # else:e0=pe;e1=0

        ps = linS2[1][i]
        pe = linE2[0][i]
        if ps >= N:
            S0 = ps - N
            S1 = N
        else:
            S0 = 0
            S1 = ps
        # if pe>=N:e0=N;E1=pe - N
        # else:E0=pe;E1=0
        E0 = S1
        E1 = S0

        ps = linC[1][i]
        if ps >= N:
            C0 = ps - N
            C1 = N
        else:
            C0 = 0
            C1 = ps
        # if pe>=N:e0=N;E1=pe - N
        # else:E0=pe;E1=0
        D0 = C1
        D1 = C0

        lined = slice_width / 2.0  # in data width
        linewidthc = (lined * (figh * 72.0 / N)) * 0.5
        # print( s0,e0, s1,e1, S0,E0, S1, E1)

        # lined= slice_width/2.  #in data width
        # linewidth=    (lined * (figh*72./N)) * 0.8
        linewidth = 1
        ax.plot([s0, e0], [s1, e1], linewidth=linewidth, ls="--", alpha=1, color=colors_array[i])
        ax.plot([S0, E0], [S1, E1], linewidth=linewidth, ls="--", alpha=1, color=colors_array[i])
        # print( i, [s0,e0],[s1,e1], [S0,E0],[S1,E1], colors_array[i]   )
        ax.plot([C0, D0], [C1, D1], linewidth=linewidthc, ls="-", alpha=0.0, color=colors_array[i])

    # ax.set_title(  '%s_frames'%(N)    )
    ax.set_title("%s_two_time" % uid)
    ax.set_xlabel(r"$t_1$ $(s)$", fontsize=18)
    ax.set_ylabel(r"$t_2$ $(s)$", fontsize=18)
    fig.colorbar(im)
    ax1.set_title("%s_aged_g2" % uid)
    ki = 0
    for i in sorted(g2_aged.keys()):
        # ax = fig.add_subplot(sx,sy,sn+1 )
        if taus_aged is None:
            gx = np.arange(len(g2_aged[i])) * timeperframe
        else:
            gx = taus_aged[i]
        # marker = next(markers)
        # print( g2_aged[i], marker )
        # print(i)
        ax1.plot(
            gx,
            g2_aged[i],
            marker="%s" % markers_array[ki],
            ls="-",
            color=colors_array[ki],
            label=r"$t_a= %.1f s$" % i,
        )
        # print( i, ki, colors_array[ki]  )
        ki += 1
        ax1.set_ylim(vmin, vmax)
        ax1.set_xlabel(r"$\tau $ $(s)$", fontsize=18)
        ax1.set_ylabel("g2")
        ax1.set_xscale("log")
    ax1.legend(fontsize="small", loc="best")
    if save:
        # fp = path + "uid= %s--Waterfall-"%uid + CurTime + '.png'
        fp = path + "%s_aged_g2" % uid + ".png"
        # print( fp )
        fig.savefig(fp, dpi=fig.dpi)

    # plt.show()


def plot_aged_g2(g2_aged, tau=None, timeperframe=1, ylim=None, xlim=None):
    """'A plot of g2 calculated from two-time"""
    fig = plt.figure(figsize=(8, 10))
    age_center = list(sorted(g2_aged.keys()))
    gs = gridspec.GridSpec(len(age_center), 1)
    for n, i in enumerate(age_center):
        ax = plt.subplot(gs[n])
        if tau is None:
            gx = np.arange(len(g2_aged[i])) * timeperframe
        else:
            gx = tau[i]
        marker = markers[n]
        c = colors[n]
        ax.plot(gx, g2_aged[i], "-%s" % marker, c=c, label=r"$age= %.1f s$" % (i * timeperframe))
        ax.set_xscale("log")
        ax.legend(fontsize="large", loc="best")
        ax.set_xlabel(r"$\tau $ $(s)$", fontsize=18)
        ax.set_ylabel("g2")
        if ylim is not None:
            ax.set_ylim(ylim)
        if xlim is not None:
            ax.set_ylim(xlim)


#####################################
# get fout-time


def get_tau_from_g12q(g12q, slice_num=6, slice_width=1, slice_start=None, slice_end=None):
    """
    Dec 16, 2015, Y.G.@CHX
    Get tau lines from two correlation function
    namely, get diag line of g12 as a function of ages

    Parameters:
        g12q: a 2-D array, one-q two correlation function, shape as ( imgs_length, imgs_length )

    Options:
        slice_num: int, the slice number of the diagonal of g12
        slice_width: int,  each slice width in unit of pixel
        slice start: int, can start from 0
        slice end: int, can end at  imgs_length -1


    Return:
        return: tau, a dict, tau lines
                 the keys of dict is tau(slice center) in unit of pixel
                 dict[key]:
                           a 1-D array, shape as ( tau_line-length ),

    One example:
        taus = get_aged_g2_from_g12q( g12q, slice_num =3, slice_width= 500,
                slice_start=4000, slice_end= 20000-4000  )
    """

    arr = rotate_g12q_to_rectangle(g12q)
    m, n = arr.shape  # m should be 2*n-1

    age_edge, age_center = get_qedge(qstart=slice_start, qend=slice_end, qwidth=slice_width, noqs=slice_num)
    age_edge, age_center = np.int_(age_edge), np.int_(age_center)
    # print (age_edge, age_center)
    tau = {}
    for i, age in enumerate(age_center):
        age_edges_0, age_edges_1 = age_edge[i * 2 : 2 * i + 2]
        # print (age_edges_0, age_edges_1)
        g2i = arr[:, age_edges_0:age_edges_1].mean(axis=1)
        g2i_ = np.array(g2i)
        tau[age] = g2i_[np.nonzero(g2i_)[0]]

    return tau


def show_g12q_taus(g12q, taus, slice_width=10, timeperframe=1, vmin=1, vmax=1.25):
    """
    Dec 16, 2015, Y.G.@CHX
    Plot tau-lines as a function of age with two correlation function


    Parameters:
        g12q: a 2-D array, one-q two correlation function, shape as ( imgs_length, imgs_length )
        tau, a dict, tau lines
                 the keys of dict is tau(slice center) in unit of pixel
                 dict[key]:
                           a 1-D array, shape as ( tau_line-length ),
                 obtained by: for example,
                         taus = get_tau_from_g12q( g12b_norm[:,:,0], slice_num = 5, slice_width=1,
                  slice_start=3, slice_end= 5000-1  ))


    Options:
        slice_width: int,  each slice width in unit of pixel, for line width of a plot
        timeperframe: float, time per frame for axis unit
        vmin, float, matplot vmin
        vmax, float, matplot vmax

    Return:
         two plots, one for tau lines~ages, g12q,

    One example:
        show_g12q_taus( g12b_norm[:,:,0], taus,  slice_width=50,
               timeperframe=1,vmin=1.01,vmax=1.55 )
    """

    age_center = list(taus.keys())
    print("the cut tau centers are: " + str(age_center))
    M, N = g12q.shape

    # fig, ax = plt.subplots( figsize = (8,8) )

    figw = 10
    figh = 10
    fig = plt.figure(figsize=(figw, figh))

    gs = gridspec.GridSpec(1, 2, width_ratios=[10, 8], height_ratios=[8, 8])
    ax = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    im = imshow(ax, g12q, origin="lower", cmap="viridis", norm=LogNorm(vmin=vmin, vmax=vmax), extent=[0, N, 0, N])

    linS = []
    linE = []
    linS.append(zip(np.int_(age_center) - 1, [0] * len(age_center)))
    linE.append(zip([N - 1] * len(age_center), N - np.int_(age_center)))
    for i, [ps, pe] in enumerate(zip(linS[0], linE[0])):
        lined = slice_width  # /2. *draw_scale_tau  #in data width
        linewidth = (lined * (figh * 72.0 / N)) * 0.8
        # print (ps,pe)
        ax.plot([ps[0], pe[0]], [ps[1], pe[1]], linewidth=linewidth)  # , color=   )

    ax.set_title("%s_frames" % (N))
    ax.set_xlabel(r"$t_1$ $(s)$", fontsize=18)
    ax.set_ylabel(r"$t_2$ $(s)$", fontsize=18)
    fig.colorbar(im)

    ax1.set_title("Tau_Cuts_in_G12")
    for i in sorted(taus.keys()):
        gx = np.arange(len(taus[i])) * timeperframe
        marker = next(markers)
        ax1.plot(gx, taus[i], "-%s" % marker, label=r"$tau= %.1f s$" % (i * timeperframe))
        ax1.set_ylim(vmin, vmax)
        ax1.set_xlabel(r"$t (s)$", fontsize=5)
        ax1.set_ylabel("g2")
        ax1.set_xscale("log")
    ax1.legend(fontsize="small", loc="best")
    # plt.show()


def histogram_taus(taus, hisbin=20, plot=True, timeperframe=1):
    """
    Dec 16, 2015, Y.G.@CHX
    Do histogram and plot of tau-lines


    Parameters:
       taus, a dict, tau lines
                 the keys of dict is tau(slice center) in unit of pixel
                 dict[key]:
                           a 1-D array, shape as ( tau_line-length ),
                 obtained by: for example,
                         taus = get_tau_from_g12q( g12b_norm[:,:,0], slice_num = 5, slice_width=1,
                  slice_start=3, slice_end= 5000-1  ))

    Options:
        bins: int,  bins number for the histogram
        plot: if True, show the histogram plot
        timeperframe: float, time per frame for axis unit


    Return:
         his:  a dict, his[key], the histogram of tau-lines
         if plot, plot the histogram  of tau-lines

    One example:
        his = histogram_taus(taus, hisbin=30, plot=True, timeperframe=timeperframe)
    """

    his = {}
    for key in list(taus.keys()):
        his[key] = np.histogram(taus[key], bins=hisbin)

    if plot:
        fig, ax1 = plt.subplots(figsize=(8, 8))
        ax1.set_title("Tau_histgram")
        for key in sorted(his.keys()):
            tx = 0.5 * (his[key][1][:-1] + his[key][1][1:])
            marker = next(markers)
            ax1.plot(tx, his[key][0], "-%s" % marker, label=r"$tau= %.1f s$" % (key * timeperframe))
            # ax1.set_ylim( 1.05,1.35 )
            ax1.set_xlim(1.05, 1.35)
            ax1.set_xlabel(r"$g_2$", fontsize=19)
            ax1.set_ylabel(r"histgram of g2 @ tau", fontsize=15)
            # ax1.set_xscale('log')
        ax1.legend(fontsize="large", loc="best")
        # plt.show()

    return his


#####################################
# get one-time
#####################################


def get_one_time_from_two_time_old(g12, norms=None, nopr=None):
    """
    Dec 16, 2015, Y.G.@CHX
    Get one-time correlation function from two correlation function
    namely, calculate the mean of each diag line of g12 to get one-time correlation fucntion

    Parameters:
        g12: a 3-D array, two correlation function, shape as ( imgs_length, imgs_length, q)

    Options:
        norms: if not None, a 2-D array, shape as ( imgs_length,   q), a normalization for further get one-time from two time, get by:  g12b_norm, g12b_not_norm, norms = auto_two_Array_g1_norm( imgsr, ring_mask, data_pixel = data_pixel )
        nopr: if not None, 1-D array, shape as [q], the number of interested pixel of each q


    Return:
        g2f12: a 2-D array, shape as ( imgs_length,  q),
                   a one-time correlation function

    One example:
        g2b_norm = get_one_time_from_two_time(g12b_norm,  norms=None, nopr=None )
        g2b_not_norm = get_one_time_from_two_time(g12b_not_norm, norms=norms, nopr=nopr)
    """

    m, n, noqs = g12.shape
    g2f12 = np.zeros([m, noqs])
    for q in range(noqs):
        y = g12[:, :, q]
        for tau in range(m):
            if norms is None:
                g2f12[tau, q] = np.nanmean(np.diag(y, k=int(tau)))
            else:
                yn = norms[:, q]
                yn1 = np.average(yn[tau:])
                yn2 = np.average(yn[: m - tau])
                g2f12[tau, q] = np.nanmean(np.diag(y, k=int(tau))) / (yn1 * yn2 * nopr[q])

    return g2f12


def get_one_time_from_two_time(g12, norms=None, nopr=None):
    """
    Dec 16, 2015, Y.G.@CHX
    Get one-time correlation function from two correlation function
    namely, calculate the mean of each diag line of g12 to get one-time correlation fucntion

    Parameters:
        g12: a 3-D array, two correlation function, shape as ( imgs_length, imgs_length, q)

    Options:
        norms: if not None, a 2-D array, shape as ( imgs_length,   q), a normalization for further get one-time from two time, get by:  g12b_norm, g12b_not_norm, norms = auto_two_Array_g1_norm( imgsr, ring_mask, data_pixel = data_pixel )
        nopr: if not None, 1-D array, shape as [q], the number of interested pixel of each q


    Return:
        g2f12: a 2-D array, shape as ( imgs_length,  q),
                   a one-time correlation function

    One example:
        g2b_norm = get_one_time_from_two_time(g12b_norm,  norms=None, nopr=None )
        g2b_not_norm = get_one_time_from_two_time(g12b_not_norm, norms=norms, nopr=nopr)
    """

    m, n, noqs = g12.shape
    if norms is None:
        g2f12 = np.array([np.nanmean(g12.diagonal(i), axis=1) for i in range(m)])
    else:
        g2f12 = np.zeros([m, noqs])
        for q in range(noqs):
            yn = norms[:, q]
            g2f12[i, q] = np.array(
                [
                    np.nanmean(g12[:, :, q].diagonal(i)) / (np.average(yn[i:]) * np.average(yn[: m - i]) * nopr[q])
                    for i in range(m)
                ]
            )
    return g2f12


def get_four_time_from_two_time(g12, g2=None, rois=None):
    """
    Dec 16, 2015, Y.G.@CHX
    Get four-time correlation function from two correlation function
    namely, calculate the deviation of each diag line of g12 to get four-time correlation fucntion
    TOBEDONE: deal with bad frames

    Parameters:
        g12: a 3-D array, two correlation function, shape as ( imgs_length, imgs_length, q)

    Options:
        g2: if not None, a 2-D array, shape as ( imgs_length,  q), or (tau, q)
            one-time correlation fucntion, for normalization of the four-time
        rois: if not None, a list, [x-slice-start, x-slice-end, y-slice-start, y-slice-end]

    Return:
        g4f12: a 2-D array, shape as ( imgs_length,  q),
                   a four-time correlation function

    One example:
        s1,s2 = 0,2000
        g4 = get_four_time_from_two_time( g12bm, g2b, roi=[s1,s2,s1,s2] )

    """
    m, n, noqs = g12.shape
    if g2 is not None:
        norm = (g2[0] - 1) ** 2
    else:
        norm = 1.0
    if rois is None:
        g4f12 = np.array([(np.nanstd(g12.diagonal(i), axis=1)) ** 2 / norm for i in range(m)])

    else:
        x1, x2, y1, y2 = rois
        g4f12 = np.array([(np.nanstd(g12[x1:x2, y1:y2, :].diagonal(i), axis=1)) ** 2 / norm for i in range(m)])

    return g4f12


######
def make_g12_mask(badframes_list, g12_shape):
    """
    Dec 16, 2015, Y.G.@CHX
    make g12 mask by badlines

    Parameters:
        badframes_list: list, contains the bad frame number, like [100, 155, 10000]
        g12_shape: the shape of one-q two correlation function, shape as ( imgs_length, imgs_length )
    Return:
        g12_mask: a 2-D array, shape as ( imgs_length, imgs_length )


    One example:
        g12_mask = make_g12_mask(bad_frames, g12b[:,:,0].shape)

    """

    m, n = g12_shape
    # g12_mask = np.ma.empty( ( m,n ) )
    g12_mask = np.ma.ones((m, n))
    g12_mask.mask = False
    for bdl in badframes_list:
        g12_mask.mask[:, bdl] = True
        g12_mask.mask[bdl, :] = True
    return g12_mask


def masked_g12(g12, badframes_list):
    """
    Dec 16, 2015, Y.G.@CHX
    make masked g12 with mask defined by badframes_list


    Parameters:
        g12: a 3-D array, two correlation function, shape as ( imgs_length, imgs_length, q)
        badframes_list: list, contains the bad frame number, like [100, 155, 10000]

    Return:
        g12m: a masked 3-D array, shape as same as g12, ( imgs_length, imgs_length, q )


    One example:
        g12m  =  masked_g12( g12b, bad_frames)

    """

    m, n, qs = g12.shape
    g12m = np.ma.empty_like(g12)
    g12_mask = make_g12_mask(badframes_list, g12[:, :, 0].shape)

    for i in range(qs):
        g12m[:, :, i] = g12[:, :, i] * g12_mask
    return g12m


def show_one_C12(
    C12, fig_ax=None, return_fig=False, interpolation="none", cmap="viridis", show_colorbar=True, *argv, **kwargs
):
    """
    plot one-q of two-time correlation function
    C12: two-time correlation function, with shape as [ time, time, qs]
    q_ind: if integer, for a SAXS q, the nth of q to be plotted
            if a list: for a GiSAXS [qz_ind, qr_ind]
    kwargs: support
        timeperframe: the time interval
        N1: the start frame(time)
        N2: the end frame(time)
        vmin/vmax: for plot
        title: if True, show the tile

    e.g.,
        show_C12(g12b, q_ind=1, N1=0, N2=500, vmin=1.05, vmax=1.07,  )

    """

    # strs =  [ 'timeperframe', 'N1', 'N2', 'vmin', 'vmax', 'title']

    if "uid" in kwargs:
        uid = kwargs["uid"]
    else:
        uid = "uid"

    shape = C12.shape

    if "timeperframe" in kwargs.keys():
        timeperframe = kwargs["timeperframe"]
    else:
        timeperframe = 1

    if "vmin" in kwargs.keys():
        vmin = kwargs["vmin"]
    else:
        vmin = 1
    if "vmax" in kwargs.keys():
        vmax = kwargs["vmax"]
    else:
        vmax = 1.05

    if "N1" in kwargs.keys():
        N1 = kwargs["N1"]
    else:
        N1 = 0

    if "N2" in kwargs.keys():
        N2 = kwargs["N2"]
    else:
        N2 = shape[0]
    if "title" in kwargs.keys():
        title = kwargs["title"]
    else:
        title = True

    data = C12[N1:N2, N1:N2]
    if fig_ax is None:
        if RUN_GUI:
            fig = Figure()
            ax = fig.add_subplot(111)
        else:
            fig, ax = plt.subplots()
    else:
        fig, ax = fig_ax
    im = imshow(
        ax,
        data,
        origin="lower",
        cmap=cmap,
        norm=LogNorm(vmin, vmax),
        extent=[0, data.shape[0] * timeperframe, 0, data.shape[0] * timeperframe],
        interpolation=interpolation,
    )
    if title:
        tit = "%s-[%s-%s] frames" % (uid, N1, N2)

        ax.set_title(tit)
    else:
        tit = ""
        # ax.set_title('%s-%s frames--Qth= %s'%(N1,N2,g12_num))
    ax.set_xlabel(r"$t_1$ $(s)$", fontsize=18)
    ax.set_ylabel(r"$t_2$ $(s)$", fontsize=18)
    if show_colorbar:
        fig.colorbar(im)

    save = False
    if "save" in kwargs:
        save = kwargs["save"]
    if save:
        path = kwargs["path"]
        # fp = path + 'Two-time--uid=%s'%(uid) + tit + CurTime + '.png'
        fp = path + "%s_Two_time" % (uid) + ".png"
        plt.savefig(fp, dpi=fig.dpi)

    if return_fig:
        return fig, ax, im


def show_C12(
    C12,
    fig_ax=None,
    q_ind=1,
    return_fig=False,
    interpolation="none",
    cmap="viridis",
    logs=True,
    qlabel=None,
    show_colorbar=True,
    *argv,
    **kwargs
):
    """
    plot one-q of two-time correlation function
    C12: two-time correlation function, with shape as [ time, time, qs]
    q_ind: if integer, for a SAXS q, the nth of q to be plotted, starting from 1,
            if a list: for a GiSAXS [qz_ind, qr_ind]
    kwargs: support
        timeperframe: the time interval
        N1: the start frame(time)
        N2: the end frame(time)
        vmin/vmax: for plot
        title: if True, show the tile

    e.g.,
        show_C12(g12b, q_ind=1, N1=0, N2=500, vmin=1.05, vmax=1.07,  )

    """

    # strs =  [ 'timeperframe', 'N1', 'N2', 'vmin', 'vmax', 'title']

    if "uid" in kwargs:
        uid = kwargs["uid"]
    else:
        uid = "uid"
    shape = C12.shape
    if (q_ind < 1) or (q_ind > shape[2]):
        raise Exceptions(
            "Error: qind starts from 1 (corresponding to python array index 0, but in the plot it will show as 1) to the max Q-length of two time funcs %s."
            % shape[2]
        )

    if isinstance(q_ind, int):
        C12_num = q_ind - 1
    else:
        qz_ind, qr_ind = q_ind - 1
        C12_num = qz_ind * num_qr + qr_ind

    if "timeperframe" in kwargs.keys():
        timeperframe = kwargs["timeperframe"]
    else:
        timeperframe = 1

    if "timeoffset" in kwargs.keys():  ### added timeoffset here
        timeoffset = kwargs["timeoffset"]
    else:
        timeoffset = 0

    if "vmin" in kwargs.keys():
        vmin = kwargs["vmin"]
    else:
        vmin = 1
    if "vmax" in kwargs.keys():
        vmax = kwargs["vmax"]
    else:
        vmax = 1.05

    if "N1" in kwargs.keys():
        N1 = kwargs["N1"]
        if N1 < 0:
            N1 = 0
    else:
        N1 = 0

    if "N2" in kwargs.keys():
        N2 = kwargs["N2"]
        if N2 > shape[0]:
            N2 = shape[0]
    else:
        N2 = shape[0]
    if "title" in kwargs.keys():
        title = kwargs["title"]
    else:
        title = True

    data = C12[N1:N2, N1:N2, C12_num]
    if fig_ax is None:
        if RUN_GUI:
            fig = Figure()
            ax = fig.add_subplot(111)
        else:
            fig, ax = plt.subplots()
    else:
        fig, ax = fig_ax

    # extent=[0, data.shape[0]*timeperframe, 0, data.shape[0]*timeperframe ]
    extent = np.array([N1, N2, N1, N2]) * timeperframe + timeoffset  ### added timeoffset to extend

    if logs:
        im = imshow(
            ax,
            data,
            origin="lower",
            cmap=cmap,
            norm=LogNorm(vmin, vmax),
            interpolation=interpolation,
            extent=extent,
        )
    else:
        im = imshow(
            ax, data, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation=interpolation, extent=extent
        )
    if qlabel is not None:
        if isinstance(q_ind, int):
            qstr = "Qth= %s-qval=%s" % (C12_num + 1, qlabel[C12_num])
    else:
        qstr = "Qth= %s" % (C12_num + 1)
    if title:
        if isinstance(q_ind, int):
            tit = "%s-[%s-%s] frames--" % (uid, N1, N2) + qstr
        else:
            tit = "%s-[%s-%s] frames--Qzth= %s--Qrth= %s" % (uid, N1, N2, qz_ind, qr_ind)
        ax.set_title(tit)
    else:
        tit = ""
        # ax.set_title('%s-%s frames--Qth= %s'%(N1,N2,g12_num))
    ax.set_xlabel(r"$t_1$ $(s)$", fontsize=18)
    ax.set_ylabel(r"$t_2$ $(s)$", fontsize=18)
    if show_colorbar:
        fig.colorbar(im)

    save = False
    if "save" in kwargs:
        save = kwargs["save"]
    if save:
        path = kwargs["path"]
        # fp = path + 'Two-time--uid=%s'%(uid) + tit + CurTime + '.png'
        fp = path + "%s_Two_time" % (uid) + ".png"
        plt.savefig(fp, dpi=fig.dpi)

    if return_fig:
        return fig, ax, im


class Exceptions(Exception):
    pass
