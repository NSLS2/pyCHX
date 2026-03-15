"""
Dec 10, 2015 Developed by Y.G.@CHX
yuzhang@bnl.gov
This module is for the SAXS XPCS analysis
"""

import os

from pandas import DataFrame
from scipy.special import erf

from pyCHX.chx_compress_analysis import (
    Multifile,
    compress_eigerdata,
    get_avg_imgc,
    get_each_ring_mean_intensityc,
    init_compress_eigerdata,
    mean_intensityc,
    read_compressed_eigerdata,
)
from pyCHX.chx_correlationc import Get_Pixel_Arrayc, auto_two_Arrayc, cal_g2c, get_pixelist_interp_iq
from pyCHX.chx_correlationp import cal_g2p
from pyCHX.chx_generic_functions import *
from pyCHX.chx_libs import RUN_GUI, Figure, colors, colors_, colors_copy, markers, markers_, markers_copy


def get_iq_invariant(qt, iqst):
    """Get integer( q**2 * iqst )
    iqst: shape should be time, q-length
    qt: shape as q-length
    return   q**2 * iqst, shape will be time length
    """
    return np.sum(iqst * qt**2, axis=1)


def plot_time_iq_invariant(
    time_stamp,
    invariant,
    pargs,
    save=True,
):
    fig, ax = plt.subplots()
    plot1D(
        x=time_stamp,
        y=invariant,
        xlabel="time (s)",
        ylabel="I(q)*Q^2",
        title="I(q)*Q^2 ~ time",
        m="o",
        c="b",
        ax=ax,
    )
    if save:
        path = pargs["path"]
        uid = pargs["uid"]

        save_arrays(
            np.vstack([time_stamp, np.array(invariant)]).T,
            label=["time", "Invariant"],
            filename="%s_iq_invariant.csv" % uid,
            path=path,
        )
        # fp = path + 'uid= %s--Iq~t-'%uid + CurTime + '.png'
        fp = path + "%s_iq_invariant" % uid + ".png"
        fig.savefig(fp, dpi=fig.dpi)


def plot_q2_iq(
    qt,
    iqst,
    time_stamp,
    pargs,
    ylim=[-0.001, 0.01],
    xlim=[0.007, 0.2],
    legend_size=4,
    save=True,
):
    fig, ax = plt.subplots()
    N = iqst.shape[0]
    for i in range(N):
        yi = iqst[i] * qt**2
        # time_labeli = 'time_%s s'%( round(  time_edge[i][0] * timeperframe, 3) )
        time_labeli = "time_%s s" % (round(time_stamp[i], 4))
        plot1D(
            x=qt,
            y=yi,
            legend=time_labeli,
            xlabel="Q (A-1)",
            ylabel="I(q)*Q^2",
            title="I(q)*Q^2 ~ time",
            m=markers[i],
            c=colors[i],
            ax=ax,
            ylim=ylim,
            xlim=xlim,
            legend_size=legend_size,
        )
    if save:
        path = pargs["path"]
        uid = pargs["uid"]
        fp = path + "%s_q2_iq" % uid + ".png"
        fig.savefig(fp, dpi=fig.dpi)


def recover_img_from_iq(qp, iq, center, mask):
    """YG. develop at CHX, 2017 July 18,
    Recover image a circular average
    """
    norm = get_pixelist_interp_iq(qp, iq, np.ones_like(mask), center)
    img_ = norm.reshape(mask.shape) * mask
    return img_


def get_cirucular_average_std(img, mask, setup_pargs, img_name="xx"):
    """YG. develop at CHX, 2017 July 18,
    Get the standard devation of tge circular average of img
    image-->I(q)-->image_mean--> (image- image_mean)**2 --> I(q) --> std = sqrt(I(q))
    """
    qp, iq, q = get_circular_average(img, mask, pargs=setup_pargs, save=False)
    center = setup_pargs["center"]
    img_ = (img - recover_img_from_iq(qp, iq, center, mask)) ** 2
    qp_, iq_, q_ = get_circular_average(img_, mask, pargs=setup_pargs, save=False)
    std = np.sqrt(iq_)
    return qp, iq, q, std


def get_delta_img(img, mask, setup_pargs, img_name="xx", plot=False):
    """YG. develop at CHX, 2017 July 18,
    Get the difference between img and image recovered from the circular average of img"""
    qp, iq, q = get_circular_average(img, mask, pargs=setup_pargs, save=False)
    center = setup_pargs["center"]
    img_ = recover_img_from_iq(qp, iq, center, mask)
    delta = img - img_ * img.mean() / img_.mean()
    if plot:
        show_img(delta, logs=True, aspect=1, cmap=cmap_albula, vmin=1e-5, vmax=10**1, image_name=img_name)
    return delta


def combine_ring_anglar_mask(ring_mask, ang_mask):
    """combine ring and anglar mask"""

    ring_max = ring_mask.max()
    ang_mask_ = np.zeros(ang_mask.shape)
    ind = np.where(ang_mask != 0)
    ang_mask_[ind] = ang_mask[ind] + 1e9  # add some large number to qr
    dumy_ring_mask = np.zeros(ring_mask.shape)
    dumy_ring_mask[ring_mask == 1] = 1
    dumy_ring_ang = dumy_ring_mask * ang_mask
    real_ang_lab = np.int_(np.unique(dumy_ring_ang)[1:]) - 1
    ring_ang = ring_mask * ang_mask_
    # print( real_ang_lab )

    ura = np.unique(ring_ang)[1:]
    ur = np.unique(ring_mask)[1:]
    ua = np.unique(ang_mask)[real_ang_lab]
    # print( np.unique( ring_mask )[1:], np.unique( ang_mask )[1:], np.unique( ring_ang )[1:] )

    ring_ang_ = np.zeros_like(ring_ang)
    newl = np.arange(1, len(ura) + 1)
    # newl = np.int_( real_ang_lab )
    # print( ura,  ur, ua  )
    # print( len(ura) )
    for i, label in enumerate(ura):
        # print (i, label)
        ring_ang_.ravel()[np.where(ring_ang.ravel() == label)[0]] = newl[i]
    # print( np.unique( ring_ang_ ), len( np.unique( ring_ang_ ) ) )
    return np.int_(ring_ang_)


def get_seg_from_ring_mask(inner_angle, outer_angle, num_angles, width_angle, center, ring_mask, qr_center):
    """YG. Jan 6, 2017
    A simple wrap function to get angle cut mask from ring_mask
    Parameter:
        inner_angle, outer_angle, num_angles, width_angle: to define the angle
        center: beam center
        ring_mask: two-d array
    Return:
        seg_mask: two-d array

    """
    widtha = (outer_angle - inner_angle) / (num_angles + 0.01)
    ang_mask, ang_center, ang_edges = get_angular_mask(
        ring_mask,
        inner_angle=inner_angle,
        outer_angle=outer_angle,
        width=widtha,
        num_angles=num_angles,
        center=center,
        flow_geometry=True,
    )
    # print( np.unique( ang_mask)[1:] )
    seg_mask = combine_ring_anglar_mask(ring_mask, ang_mask)
    qval_dict = get_qval_dict(qr_center=qr_center, qz_center=ang_center)
    return seg_mask, qval_dict


def get_seg_dict_from_ring_mask(inner_angle, outer_angle, num_angles, width_angle, center, ring_mask, qr_center):
    """YG. Jan 6, 2017
    A simple wrap function to get angle cut mask from ring_mask
    Parameter:
        inner_angle, outer_angle, num_angles, width_angle: to define the angle
        center: beam center
        ring_mask: two-d array
    Return:
        seg_mask: two-d array

    """
    widtha = (outer_angle - inner_angle) / (num_angles + 0.01)
    ang_mask, ang_center, ang_edges = get_angular_mask(
        np.ones_like(ring_mask),
        inner_angle=inner_angle,
        outer_angle=outer_angle,
        width=widtha,
        num_angles=num_angles,
        center=center,
        flow_geometry=True,
    )
    # print( np.unique( ang_mask)[1:] )
    seg_mask, good_ind = combine_two_roi_mask(ring_mask, ang_mask)
    qval_dict = get_qval_dict(qr_center=qr_center, qz_center=ang_center)
    # print( np.unique( seg_mask)[1:], good_ind )
    # print( list( qval_dict.keys()), good_ind , len(good_ind) )
    qval_dict_ = {i: qval_dict[k] for (i, k) in enumerate(good_ind)}
    return seg_mask, qval_dict_


def combine_two_roi_mask(ring_mask, ang_mask, pixel_num_thres=10):
    """combine two roi_mask into a new roi_mask
    pixel_num_thres: integer, the low limit pixel number in each roi of the combined mask,
                        i.e., if the pixel number in one roi of the combined mask smaller than pixel_num_thres,
                        that roi will be considered as bad one and be removed.
    e.g., ring_mask is a ring shaped mask, with unique index as (1,2)
          ang_mask is a angular shaped mask, with unique index as (1,2,3,4)
          the new mask will be ( 1,2,3,4 [for first ring];
                                 5,6,7,8 [for second ring];
                                 ...)

    """
    rf = np.ravel(ring_mask)
    af = np.ravel(ang_mask)
    ruiq = np.unique(ring_mask)
    auiq = np.unique(ang_mask)
    maxa = np.max(auiq)
    ring_mask_ = np.zeros_like(ring_mask)
    new_mask_ = np.zeros_like(ring_mask)
    new_mask_ = np.zeros_like(ring_mask)
    for i, ind in enumerate(ruiq[1:]):
        ring_mask_.ravel()[np.where(rf == ind)[0]] = maxa * i

    new_mask = (ring_mask_ + ang_mask) * np.array(ring_mask, dtype=bool) * np.array(ang_mask, dtype=bool)

    qind, pixelist = roi.extract_label_indices(new_mask)
    noqs = len(np.unique(qind))
    nopr = np.bincount(qind, minlength=(noqs + 1))[1:]
    # good_ind = np.unique( new_mask )[1:]
    good_ind = np.where(nopr >= pixel_num_thres)[0] + 1
    # print( good_ind )
    l = len(good_ind)

    new_ind = np.arange(1, l + 1)
    for i, gi in enumerate(good_ind):
        new_mask_.ravel()[np.where(new_mask.ravel() == gi)[0]] = new_ind[i]
    return new_mask_, good_ind - 1


def refine_qval_dict(qval_dict, roi_mask, new_mask, pixel_num_thres=10):
    """YG Dev@CHX 2019 May 29, refine qval_dict by applying a new mask
    qval_dict corresponding to the roi_mask, now with the new mask, there are some roi might have zero (less than
    pixel_num_thres number) pixel, so both the roi_mask and the qval_dict will be updated
    """
    new_roi_mask = np.zeros_like(roi_mask)
    roi_mask2 = roi_mask * new_mask
    qind, pixelist = roi.extract_label_indices(roi_mask2)
    noqs = len(np.unique(qind))
    nopr = np.bincount(qind, minlength=(noqs + 1))[1:]
    good_ind = np.where(nopr >= pixel_num_thres)[0] + 1
    l = len(good_ind)
    new_ind = np.arange(1, l + 1)
    for i, gi in enumerate(good_ind):
        new_roi_mask.ravel()[np.where(roi_mask2.ravel() == gi)[0]] = new_ind[i]
    qval_dict_ = {i: qval_dict[k - 1] for (i, k) in enumerate(good_ind)}
    return new_roi_mask, qval_dict_


def bin_1D(x, y, nx=None, min_x=None, max_x=None):
    """
    Bin the values in y based on their x-coordinates

    Parameters
    ----------
    x : array
        position
    y : array
        intensity
    nx : integer, optional
        number of bins to use defaults to default bin value
    min_x : float, optional
        Left edge of first bin defaults to minimum value of x
    max_x : float, optional
        Right edge of last bin defaults to maximum value of x

    Returns
    -------
    edges : array
        edges of bins, length nx + 1

    val : array
        sum of values in each bin, length nx

    count : array
        The number of counts in each bin, length nx
    """

    # handle default values
    if min_x is None:
        min_x = np.min(x)
    if max_x is None:
        max_x = np.max(x)
    if nx is None:
        nx = int(max_x - min_x)

    # print ( min_x, max_x, nx)

    # use a weighted histogram to get the bin sum
    bins = np.linspace(start=min_x, stop=max_x, num=nx + 1, endpoint=True)
    # print (x)
    # print (bins)
    val, _ = np.histogram(a=x, bins=bins, weights=y)
    # use an un-weighted histogram to get the counts
    count, _ = np.histogram(a=x, bins=bins)
    # return the three arrays
    return bins, val, count


def circular_average(
    image, calibrated_center, threshold=0, nx=None, pixel_size=(1, 1), min_x=None, max_x=None, mask=None
):
    """Circular average of the the image data
    The circular average is also known as the radial integration
    Parameters
    ----------
    image : array
        Image to compute the average as a function of radius
    calibrated_center : tuple
        The center of the image in pixel units
        argument order should be (row, col)
    threshold : int, optional
        Ignore counts above `threshold`
        default is zero
    nx : int, optional
        number of bins in x
        defaults is 100 bins
    pixel_size : tuple, optional
        The size of a pixel (in a real unit, like mm).
        argument order should be (pixel_height, pixel_width)
        default is (1, 1)
    min_x : float, optional number of pixels
        Left edge of first bin defaults to minimum value of x
    max_x : float, optional number of pixels
        Right edge of last bin defaults to maximum value of x
    Returns
    -------
    bin_centers : array
        The center of each bin in R. shape is (nx, )
    ring_averages : array
        Radial average of the image. shape is (nx, ).
    """
    radial_val = utils.radial_grid(calibrated_center, image.shape, pixel_size)

    if mask is not None:
        # maks = np.ones_like(  image )
        mask = np.array(mask, dtype=bool)
        binr = radial_val[mask]
        image_mask = np.array(image)[mask]

    else:
        binr = np.ravel(radial_val)
        image_mask = np.ravel(image)

    # if nx is None: #make a one-pixel width q
    #   nx = int( max_r - min_r)
    # if min_x is None:
    #    min_x= int( np.min( binr))
    #    min_x_= int( np.min( binr)/(np.sqrt(pixel_size[1]*pixel_size[0] )))
    # if max_x is None:
    #    max_x = int( np.max(binr ))
    #    max_x_ = int( np.max(binr)/(np.sqrt(pixel_size[1]*pixel_size[0] ))  )
    # if nx is None:
    #    nx = max_x_ - min_x_

    # binr_ = np.int_( binr /(np.sqrt(pixel_size[1]*pixel_size[0] )) )
    binr_ = binr / (np.sqrt(pixel_size[1] * pixel_size[0]))
    # print ( min_x, max_x, min_x_, max_x_, nx)
    bin_edges, sums, counts = bin_1D(binr_, image_mask, nx=nx, min_x=min_x, max_x=max_x)

    # print  (len( bin_edges), len( counts) )
    th_mask = counts > threshold

    # print  (len(th_mask) )
    ring_averages = sums[th_mask] / counts[th_mask]

    bin_centers = utils.bin_edges_to_centers(bin_edges)[th_mask]

    # print (len(  bin_centers ) )

    return bin_centers, ring_averages


def get_circular_average(
    avg_img,
    mask,
    pargs,
    show_pixel=True,
    min_x=None,
    max_x=None,
    nx=None,
    plot_=False,
    save=False,
    *argv,
    **kwargs
):
    """get a circular average of an image
    Parameters
    ----------

    avg_img: 2D-array, the image
    mask: 2D-array
    pargs: a dict, should contains
        center: the beam center in pixel
        Ldet: sample to detector distance
        lambda_: the wavelength
        dpix, the pixel size in mm. For Eiger1m/4m, the size is 75 um (0.075 mm)

    nx : int, optional
        number of bins in x
        defaults is 1500 bins

    plot_: a boolen type, if True, plot the one-D curve
    plot_qinpixel:a boolen type, if True, the x-axis of the one-D curve is q in pixel; else in real Q

    Returns
    -------
    qp: q in pixel
    iq: intensity of circular average
    q: q in real unit (A-1)


    """

    center, Ldet, lambda_, dpix = pargs["center"], pargs["Ldet"], pargs["lambda_"], pargs["dpix"]
    uid = pargs["uid"]
    qp, iq = circular_average(
        avg_img, center, threshold=0, nx=nx, pixel_size=(dpix, dpix), mask=mask, min_x=min_x, max_x=max_x
    )
    qp_ = qp * dpix
    #  convert bin_centers from r [um] to two_theta and then to q [1/px] (reciprocal space)
    two_theta = utils.radius_to_twotheta(Ldet, qp_)
    q = utils.twotheta_to_q(two_theta, lambda_)
    if plot_:
        if show_pixel:
            fig = plt.figure(figsize=(8, 6))
            ax1 = fig.add_subplot(111)
            # ax2 = ax1.twiny()
            ax1.semilogy(qp, iq, "-o")
            # ax1.semilogy(q,  iq , '-o')

            ax1.set_xlabel("q (pixel)")
            # ax1.set_xlabel('q ('r'$\AA^{-1}$)')
            # ax2.cla()
            ax1.set_ylabel("I(q)")
            title = ax1.set_title("uid= %s--Circular Average" % uid)

        else:
            fig = plt.figure(figsize=(8, 6))
            ax1 = fig.add_subplot(111)
            ax1.semilogy(q, iq, "-o")
            ax1.set_xlabel("q (" r"$\AA^{-1}$)")
            ax1.set_ylabel("I(q)")
            title = ax1.set_title("uid= %s--Circular Average" % uid)
            ax2 = None
        if "xlim" in kwargs.keys():
            ax1.set_xlim(kwargs["xlim"])
            x1, x2 = kwargs["xlim"]
            w = np.where((q >= x1) & (q <= x2))[0]
        if "ylim" in kwargs.keys():
            ax1.set_ylim(kwargs["ylim"])

        title.set_y(1.1)
        fig.subplots_adjust(top=0.85)
        path = pargs["path"]
        fp = path + "%s_q_Iq" % uid + ".png"
        fig.savefig(fp, dpi=fig.dpi)
    if save:
        path = pargs["path"]
        save_lists([q, iq], label=["q_A-1", "Iq"], filename="%s_q_Iq.csv" % uid, path=path)
    return qp, iq, q


def plot_circular_average(
    qp, iq, q, pargs, show_pixel=False, loglog=False, save=True, return_fig=False, *argv, **kwargs
):
    if RUN_GUI:
        fig = Figure()
        ax1 = fig.add_subplot(111)
    else:
        fig, ax1 = plt.subplots()

    uid = pargs["uid"]

    if show_pixel:
        if loglog:
            ax1.loglog(qp, iq, "-o")
        else:
            ax1.semilogy(qp, iq, "-o")
        ax1.set_xlabel("q (pixel)")
        ax1.set_ylabel("I(q)")
        title = ax1.set_title("%s_Circular Average" % uid)
    else:
        if loglog:
            ax1.loglog(qp, iq, "-o")
        else:
            ax1.semilogy(q, iq, "-o")
        ax1.set_xlabel("q (" r"$\AA^{-1}$)")
        ax1.set_ylabel("I(q)")
        title = ax1.set_title("%s_Circular Average" % uid)
        ax2 = None
    if "xlim" in kwargs.keys():
        xlim = kwargs["xlim"]
    else:
        xlim = [q.min(), q.max()]
    if "ylim" in kwargs.keys():
        ylim = kwargs["ylim"]
    else:
        ylim = [iq.min(), iq.max()]

    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)

    title.set_y(1.1)
    fig.subplots_adjust(top=0.85)
    if save:
        path = pargs["path"]
        fp = path + "%s_q_Iq" % uid + ".png"
        fig.savefig(fp, dpi=fig.dpi)
    if return_fig:
        return fig


def get_angular_average(avg_img, mask, pargs, min_r, max_r, nx=3600, plot_=False, save=False, *argv, **kwargs):
    """get a angular average of an image
    Parameters
    ----------

    avg_img: 2D-array, the image
    mask: 2D-array
    pargs: a dict, should contains
        center: the beam center in pixel
        Ldet: sample to detector distance
        lambda_: the wavelength
        dpix, the pixel size in mm. For Eiger1m/4m, the size is 75 um (0.075 mm)

    nx : int, optional
        number of bins in x
        defaults is 1500 bins

    plot_: a boolen type, if True, plot the one-D curve
    plot_qinpixel:a boolen type, if True, the x-axis of the one-D curve is q in pixel; else in real Q

    Returns
    -------
    ang: ang in degree
    iq: intensity of circular average



    """

    center, Ldet, lambda_, dpix = pargs["center"], pargs["Ldet"], pargs["lambda_"], pargs["dpix"]
    uid = pargs["uid"]

    angq, ang = angular_average(
        avg_img, calibrated_center=center, pixel_size=(dpix, dpix), nx=nx, min_r=min_r, max_r=max_r, mask=mask
    )

    if plot_:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.plot(angq, ang, "-o")
        ax.set_xlabel("angle (deg)")
        ax.set_ylabel("I(ang)")
        # ax.legend(loc = 'best')
        uid = pargs["uid"]
        title = ax.set_title("Uid= %s--t-I(Ang)" % uid)
        title.set_y(1.01)

        if save:
            # dt =datetime.now()
            # CurTime = '%s%02d%02d-%02d%02d-' % (dt.year, dt.month, dt.day,dt.hour,dt.minute)
            path = pargs["path"]
            uid = pargs["uid"]

            # fp = path + 'Uid= %s--Ang-Iq~t-'%uid + CurTime + '.png'
            fp = path + "uid=%s--Ang-Iq-t-" % uid + ".png"
            fig.savefig(fp, dpi=fig.dpi)

        # plt.show()

    return angq, ang


def angular_average(
    image,
    calibrated_center,
    threshold=0,
    nx=1500,
    pixel_size=(1, 1),
    min_r=None,
    max_r=None,
    min_x=None,
    max_x=None,
    mask=None,
):
    """Angular_average of the the image data

    Parameters
    ----------
    image : array
        Image to compute the average as a function of radius
    calibrated_center : tuple
        The center of the image in pixel units
        argument order should be (row, col)
    threshold : int, optional
        Ignore counts above `threshold`
        default is zero
    nx : int, optional
        number of bins in x
        defaults is 100 bins
    pixel_size : tuple, optional
        The size of a pixel (in a real unit, like mm).
        argument order should be (pixel_height, pixel_width)
        default is (1, 1)

    min_r: float, optional number of pixels
        The min r, e.g., the starting radius for angule average
    max_r:float, optional number of pixels
        The max r, e.g., the ending radius for angule average
        max_r - min_r gives the width of the angule average

    min_x : float, optional number of pixels
        Left edge of first bin defaults to minimum value of x
    max_x : float, optional number of pixels
        Right edge of last bin defaults to maximum value of x
    Returns
    -------
    bin_centers : array
        The center of each bin in degree shape is (nx, )
    ring_averages : array
        Radial average of the image. shape is (nx, ).
    """

    angle_val = utils.angle_grid(calibrated_center, image.shape, pixel_size)

    if min_r is None:
        min_r = 0
    if max_r is None:
        max_r = np.sqrt(
            (image.shape[0] - calibrated_center[0]) ** 2 + (image.shape[1] - calibrated_center[1]) ** 2
        )
    r_mask = make_ring_mask(calibrated_center, image.shape, min_r, max_r)

    if mask is not None:
        # maks = np.ones_like(  image )
        mask = np.array(mask * r_mask, dtype=bool)

        bina = angle_val[mask]
        image_mask = np.array(image)[mask]

    else:
        bina = np.ravel(angle_val)
        image_mask = np.ravel(image * r_mask)

    bin_edges, sums, counts = utils.bin_1D(bina, image_mask, nx, min_x=min_x, max_x=max_x)

    # print (counts)
    th_mask = counts > threshold
    ang_averages = sums[th_mask] / counts[th_mask]

    bin_centers = utils.bin_edges_to_centers(bin_edges)[th_mask]

    return bin_centers * 180 / np.pi, ang_averages


def get_t_iqc(FD, frame_edge, mask, pargs, nx=1500, plot_=False, save=False, show_progress=True, *argv, **kwargs):
    """Get t-dependent Iq

    Parameters
    ----------
    data_series:  a image series
    frame_edge: list, the ROI frame regions, e.g., [  [0,100], [200,400] ]
    mask:  a image mask

    nx : int, optional
        number of bins in x
        defaults is 1500 bins
    plot_: a boolen type, if True, plot the time~one-D curve with qp as x-axis
    Returns
    ---------
    qp: q in pixel
    iq: intensity of circular average
    q: q in real unit (A-1)

    """

    Nt = len(frame_edge)
    iqs = list(np.zeros(Nt))
    for i in range(Nt):
        t1, t2 = frame_edge[i]
        # print (t1,t2)
        avg_img = get_avg_imgc(FD, beg=t1, end=t2, sampling=1, plot_=False, show_progress=show_progress)
        qp, iqs[i], q = get_circular_average(avg_img, mask, pargs, nx=nx, plot_=False)

    if plot_:
        fig, ax = plt.subplots(figsize=(8, 6))
        for i in range(Nt):
            t1, t2 = frame_edge[i]
            ax.semilogy(q, iqs[i], label="frame: %s--%s" % (t1, t2))
            # ax.set_xlabel("q in pixel")
            ax.set_xlabel("Q " r"($\AA^{-1}$)")
            ax.set_ylabel("I(q)")

        if "xlim" in kwargs.keys():
            ax.set_xlim(kwargs["xlim"])
        if "ylim" in kwargs.keys():
            ax.set_ylim(kwargs["ylim"])

        ax.legend(
            loc="best",
        )

        uid = pargs["uid"]
        title = ax.set_title("uid= %s--t~I(q)" % uid)
        title.set_y(1.01)
        if save:
            # dt =datetime.now()
            # CurTime = '%s%02d%02d-%02d%02d-' % (dt.year, dt.month, dt.day,dt.hour,dt.minute)
            path = pargs["path"]
            uid = pargs["uid"]
            # fp = path + 'uid= %s--Iq~t-'%uid + CurTime + '.png'
            fp = path + "uid=%s--Iq-t-" % uid + ".png"
            fig.savefig(fp, dpi=fig.dpi)

            save_arrays(
                np.vstack([q, np.array(iqs)]).T,
                label=["q_A-1"] + ["Fram-%s-%s" % (t[0], t[1]) for t in frame_edge],
                filename="uid=%s-q-Iqt.csv" % uid,
                path=path,
            )

        # plt.show()

    return qp, np.array(iqs), q


def get_t_iqc_imstack(
    imgs, frame_edge, mask, pargs, nx=1500, plot_=False, save=False, show_progress=True, *argv, **kwargs
):
    """
    Get t-dependent Iq

    variant of get_t_iqc that takes an image stack like a dask array to calculate average images and then does the radial integration
    variant by LW 05/162024

    Parameters
    ----------
    imgs:  image stack like dask array
    frame_edge: list, the ROI frame regions, e.g., [  [0,100], [200,400] ]
    mask:  a image mask

    nx : int, optional
        number of bins in x
        defaults is 1500 bins
    plot_: a boolen type, if True, plot the time~one-D curve with qp as x-axis
    Returns
    ---------
    qp: q in pixel
    iq: intensity of circular average
    q: q in real unit (A-1)

    """

    Nt = len(frame_edge)
    iqs = list(np.zeros(Nt))
    for i in range(Nt):
        t1, t2 = frame_edge[i]
        # print (t1,t2)
        avg_img = np.average(imgs[t1:t2, :, :], axis=0)
        qp, iqs[i], q = get_circular_average(avg_img, mask, pargs, nx=nx, plot_=False)

    if plot_:
        fig, ax = plt.subplots(figsize=(8, 6))
        for i in range(Nt):
            t1, t2 = frame_edge[i]
            ax.semilogy(q, iqs[i], label="frame: %s--%s" % (t1, t2))
            # ax.set_xlabel("q in pixel")
            ax.set_xlabel("Q " r"($\AA^{-1}$)")
            ax.set_ylabel("I(q)")

        if "xlim" in kwargs.keys():
            ax.set_xlim(kwargs["xlim"])
        if "ylim" in kwargs.keys():
            ax.set_ylim(kwargs["ylim"])

        ax.legend(
            loc="best",
        )

        uid = pargs["uid"]
        title = ax.set_title("uid= %s--t~I(q)" % uid)
        title.set_y(1.01)
        if save:
            # dt =datetime.now()
            # CurTime = '%s%02d%02d-%02d%02d-' % (dt.year, dt.month, dt.day,dt.hour,dt.minute)
            path = pargs["path"]
            uid = pargs["uid"]
            # fp = path + 'uid= %s--Iq~t-'%uid + CurTime + '.png'
            fp = path + "uid=%s--Iq-t-" % uid + ".png"
            fig.savefig(fp, dpi=fig.dpi)

            save_arrays(
                np.vstack([q, np.array(iqs)]).T,
                label=["q_A-1"] + ["Fram-%s-%s" % (t[0], t[1]) for t in frame_edge],
                filename="uid=%s-q-Iqt.csv" % uid,
                path=path,
            )

        # plt.show()

    return qp, np.array(iqs), q


def plot_t_iqc(q, iqs, frame_edge, pargs, save=True, return_fig=False, legend_size=None, *argv, **kwargs):
    """Plot t-dependent Iq

    Parameters
    ----------
    q: q in real unit (A-1), one-D array
    frame_edge: list, the ROI frame regions, e.g., [  [0,100], [200,400] ]
    iqs: intensity of circular average, shape is [len(frame_edge), len(q)]
    pargs: a dict include data path, uid et.al info

    Returns
    ---------
    None
    """
    Nt = iqs.shape[0]
    if frame_edge is None:
        frame_edge = np.zeros(Nt, dtype=object)
        for i in range(Nt):
            frame_edge[i] = ["Edge_%i" % i, "Edge_%i" % (i + 1)]
    # Nt = len( frame_edge )
    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(Nt):
        t1, t2 = frame_edge[i]
        if np.any(iqs[i]):
            ax.semilogy(q, iqs[i], label="frame: %s--%s" % (t1, t2))

        # ax.set_xlabel("q in pixel")
        ax.set_xlabel("Q " r"($\AA^{-1}$)")
        ax.set_ylabel("I(q)")

    if "xlim" in kwargs.keys():
        ax.set_xlim(kwargs["xlim"])
    if "ylim" in kwargs.keys():
        ax.set_ylim(kwargs["ylim"])
    ax.legend(loc="best", fontsize=legend_size)
    uid = pargs["uid"]
    title = ax.set_title("%s--t~I(q)" % uid)
    title.set_y(1.01)
    if save:
        # dt =datetime.now()
        # CurTime = '%s%02d%02d-%02d%02d-' % (dt.year, dt.month, dt.day,dt.hour,dt.minute)
        path = pargs["path"]
        uid = pargs["uid"]
        # fp = path + 'uid= %s--Iq~t-'%uid + CurTime + '.png'
        fp = path + "%s_q_Iqt" % uid + ".png"
        fig.savefig(fp, dpi=fig.dpi)

        save_arrays(
            np.vstack([q, np.array(iqs)]).T,
            label=["q_A-1"] + ["Fram-%s-%s" % (t[0], t[1]) for t in frame_edge],
            filename="%s_q_Iqt" % uid,
            path=path,
        )
    if return_fig:
        return fig, ax
    # plt.show()


def get_distance(p1, p2):
    """Calc the distance between two point"""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def calc_q(L, a, wv):
    """calc_q(L,a,wv) - calculate the q value for length L, transverse
        distance a and wavelength wv.
    Use this to calculate the speckle size

    L - sample to detector distance (mm)
    a - pixel size transverse length from beam direction (mm)
    wv - wavelength
    Units of L and a should match and resultant q is in inverse units of wv.
    """
    theta = np.arctan2(a, L)
    q = 4 * np.pi * np.sin(theta / 2.0) / wv
    return q


def get_t_iq(data_series, frame_edge, mask, pargs, nx=1500, plot_=False, save=False, *argv, **kwargs):
    """Get t-dependent Iq

    Parameters
    ----------
    data_series:  a image series
    frame_edge: list, the ROI frame regions, e.g., [  [0,100], [200,400] ]
    mask:  a image mask

    nx : int, optional
        number of bins in x
        defaults is 1500 bins
    plot_: a boolen type, if True, plot the time~one-D curve with qp as x-axis

    Returns
    ---------
    qp: q in pixel
    iq: intensity of circular average
    q: q in real unit (A-1)

    """

    Nt = len(frame_edge)
    iqs = list(np.zeros(Nt))
    for i in range(Nt):
        t1, t2 = frame_edge[i]
        # print (t1,t2)
        avg_img = get_avg_img(data_series[t1:t2], sampling=1, plot_=False)
        qp, iqs[i], q = get_circular_average(avg_img, mask, pargs, nx=nx, plot_=False)

    if plot_:
        fig, ax = plt.subplots(figsize=(8, 6))
        for i in range(Nt):
            t1, t2 = frame_edge[i]
            ax.semilogy(q, iqs[i], label="frame: %s--%s" % (t1, t2))
            # ax.set_xlabel("q in pixel")
            ax.set_xlabel("Q " r"($\AA^{-1}$)")
            ax.set_ylabel("I(q)")

        if "xlim" in kwargs.keys():
            ax.set_xlim(kwargs["xlim"])
        if "ylim" in kwargs.keys():
            ax.set_ylim(kwargs["ylim"])

        ax.legend(loc="best")

        uid = pargs["uid"]
        title = ax.set_title("uid=%s--t-I(q)" % uid)
        title.set_y(1.01)
        if save:
            # dt =datetime.now()
            # CurTime = '%s%02d%02d-%02d%02d-' % (dt.year, dt.month, dt.day,dt.hour,dt.minute)
            path = pargs["path"]
            uid = pargs["uid"]
            # fp = path + 'Uid= %s--Iq~t-'%uid + CurTime + '.png'
            fp = path + "uid=%s--Iq-t-" % uid + ".png"
            fig.savefig(fp, dpi=fig.dpi)

        # plt.show()

    return qp, np.array(iqs), q


def get_t_ang(
    data_series,
    frame_edge,
    mask,
    center,
    pixel_size,
    min_r,
    max_r,
    pargs,
    nx=1500,
    plot_=False,
    save=False,
    *argv,
    **kwargs
):
    """Get t-dependent angule intensity

    Parameters
    ----------
    data_series:  a image series
    frame_edge: list, the ROI frame regions, e.g., [  [0,100], [200,400] ]
    mask:  a image mask

    pixel_size : tuple, optional
        The size of a pixel (in a real unit, like mm).
        argument order should be (pixel_height, pixel_width)
        default is (1, 1)
    center: the beam center in pixel
    min_r: float, optional number of pixels
        The min r, e.g., the starting radius for angule average
    max_r:float, optional number of pixels
        The max r, e.g., the ending radius for angule average

        max_r - min_r gives the width of the angule average

    nx : int, optional
        number of bins in x
        defaults is 1500 bins
    plot_: a boolen type, if True, plot the time~one-D curve with qp as x-axis

    Returns
    ---------
    qp: q in pixel
    iq: intensity of circular average
    q: q in real unit (A-1)

    """

    Nt = len(frame_edge)
    iqs = list(np.zeros(Nt))
    for i in range(Nt):
        t1, t2 = frame_edge[i]
        # print (t1,t2)
        avg_img = get_avg_img(data_series[t1:t2], sampling=1, plot_=False)
        qp, iqs[i] = angular_average(
            avg_img, center, pixel_size=pixel_size, nx=nx, min_r=min_r, max_r=max_r, mask=mask
        )

    if plot_:
        fig, ax = plt.subplots(figsize=(8, 8))
        for i in range(Nt):
            t1, t2 = frame_edge[i]
            # ax.semilogy(qp* 180/np.pi, iqs[i], label="frame: %s--%s"%( t1,t2) )
            ax.plot(qp, iqs[i], label="frame: %s--%s" % (t1, t2))
            ax.set_xlabel("angle (deg)")
            ax.set_ylabel("I(ang)")
        ax.legend(loc="best")
        uid = pargs["uid"]
        title = ax.set_title("Uid= %s--t-I(Ang)" % uid)
        title.set_y(1.01)

        if save:
            # dt =datetime.now()
            # CurTime = '%s%02d%02d-%02d%02d-' % (dt.year, dt.month, dt.day,dt.hour,dt.minute)
            path = pargs["path"]
            uid = pargs["uid"]
            # fp = path + 'Uid= %s--Ang-Iq~t-'%uid + CurTime + '.png'
            fp = path + "uid=%s--Ang-Iq-t-" % uid + ".png"
            fig.savefig(fp, dpi=fig.dpi)

        # plt.show()

    return qp, np.array(iqs)


def make_ring_mask(center, shape, min_r, max_r):
    """
    Make a ring mask.

    Parameters
    ----------
    center : tuple
        point in image where r=0; may be a float giving subpixel precision.
        Order is (rr, cc).
    shape: tuple
        Image shape which is used to determine the maximum extent of output
        pixel coordinates. Order is (rr, cc).

    min_r: float, optional number of pixels
        The min r, e.g., the starting radius of the ring
    max_r:float, optional number of pixels
        The max r, e.g., the ending radius of the ring
        max_r - min_r gives the width of the ring
    Returns
    -------
    ring_mask : array


    """
    r_val = utils.radial_grid(center, shape, [1.0, 1.0])
    r_mask = np.zeros_like(r_val, dtype=np.int32)
    r_mask[np.where((r_val > min_r) & (r_val < max_r))] = 1

    return r_mask


def _make_roi(coords, edges, shape):
    """Helper function to create ring rois and bar rois
    Parameters
    ----------
    coords : array
        shape is image shape
    edges : list
        List of tuples of inner (left or top) and outer (right or bottom)
        edges of each roi.
        e.g., edges=[(1, 2), (11, 12), (21, 22)]
    shape : tuple
        Shape of the image in which to create the ROIs
        e.g., shape=(512, 512)
    Returns
    -------
    label_array : array
        Elements not inside any ROI are zero; elements inside each
        ROI are 1, 2, 3, corresponding to the order they are
        specified in `edges`.
        Has shape=`image shape`
    """
    label_array = np.digitize(coords, edges, right=False)
    # Even elements of label_array are in the space between rings.
    label_array = (np.where(label_array % 2 != 0, label_array, 0) + 1) // 2
    return label_array.reshape(shape)


def angulars(edges, center, shape):
    """
    Draw annual (angluar-shaped) shaped regions of interest.
    Each ring will be labeled with an integer. Regions outside any ring will
    be filled with zeros.
    Parameters
    ----------
    edges: list
        giving the inner and outer angle in unit of radians
        e.g., [(1, 2), (11, 12), (21, 22)]
    center: tuple
        point in image where r=0; may be a float giving subpixel precision.
        Order is (rr, cc).
    shape: tuple
        Image shape which is used to determine the maximum extent of output
        pixel coordinates. Order is (rr, cc).
    Returns
    -------
    label_array : array
        Elements not inside any ROI are zero; elements inside each
        ROI are 1, 2, 3, corresponding to the order they are specified
        in edges.
    """
    edges = np.atleast_2d(np.asarray(edges)).ravel()
    if not 0 == len(edges) % 2:
        raise ValueError(
            "edges should have an even number of elements, " "giving inner, outer radii for each angular"
        )
    if not np.all(np.diff(edges) > 0):
        raise ValueError(
            "edges are expected to be monotonically increasing, "
            "giving inner and outer radii of each angular from "
            "r=0 outward"
        )

    angle_val = utils.angle_grid(center, shape).ravel()

    return _make_roi(angle_val, edges, shape)


def update_angular_mask_width_edge(edge, mask, center, roi_mask):
    """YG Dev@CHX May, 2019 primary developed for flow-geometry
    Update anglure mask using new edge
    Input:
       edge: the edge of the anglues
       mask: the mask of the image
       center: the beam center
       roi_mask: the roi mask
    Output:
       roi_mask: updated roi_mask (effective index starting from 1)
    """
    for i, (al, ah) in enumerate(edge):
        edge_ = np.array([[al, ah]])
        ang = angulars(np.radians(edge_), center, mask.shape) * mask
        w = np.ravel(ang) == 1
        np.ravel(roi_mask)[w] = i + 1
    return roi_mask


def fix_angle_mask_at_PN_180(edge, mask, center, roi_mask):
    """YG Dev@CHX May, 2019
    to fix the problem of making angluar mask at the angle edge around +/- 180
    Input:
       edge: the edge of the anglues
       mask: the mask of the image
       center: the beam center
       roi_mask: the roi mask
    Output:
       roi_mask: by fixing the edge effect (effective index starting from 1)
    """
    for i, (al, ah) in enumerate(edge):
        flag = True
        if al <= -180.0 and ah > -180:
            edge_ = np.array([[al + 360, 180]])
        elif al <= 180.0 and ah > 180:
            edge_ = np.array([[-180, ah - 360]])
        elif al <= -180.0 and ah < -180:
            edge_ = np.array([[al + 360, ah + 360]])
        elif al >= 180.0 and ah > 180:
            edge_ = np.array([[al - 360, ah - 360]])

        else:
            flag = False
        if flag:
            # print(i+1, al,ah, edge_)
            ang = angulars(np.radians(edge_), center, mask.shape) * mask
            w = np.ravel(ang) == 1
            # print(w)
            np.ravel(roi_mask)[w] = i + 1
    return roi_mask


def get_angular_mask(
    mask,
    inner_angle=0,
    outer_angle=360,
    width=None,
    edges=None,
    num_angles=12,
    center=None,
    dpix=[1, 1],
    flow_geometry=False,
    flow_angle=None,
    fix_180_angle=False,
    verbose=False,
):
    """
    mask: 2D-array
    inner_angle # the starting angle in unit of degree
    outer_angle #  the ending angle in unit of degree
    width       # width of each angle, in degree, default is None, there is no gap between the neighbour angle ROI
    edges: default, None. otherwise, give a customized angle edges
    num_angles    # number of angles

    center: the beam center in pixel
    dpix, the pixel size in mm. For Eiger1m/4m, the size is 75 um (0.075 mm)
    flow_geometry: if True, please give the flow angle. the map will be a mirror symmetry along the flow direction

    Returns
    -------
    ang_mask: a ring mask, np.array
    ang_center: ang in unit of degree
    ang_val: ang edges in degree

    """

    if flow_geometry:
        if verbose:
            print(
                """
For the flow geometry, please only define a quarter of the expected ROI.
The quarter ROI should start from around flow_angle - 90 to around the flow_angle
Otherwise, there will be somne errors.
The final ROI will have a center symmetry as well as a mirror symmetry along the flow direction.
An example for flow_angle=90 will be:
edges = roi.ring_edges( -10, 20, 2.5, 5) -->
                           array([[-10. ,  10. ],
                                   [ 12.5,  32.5],
                                   [ 35. ,  55. ],
                                   [ 57.5,  77.5],
                                   [ 80. , 100. ]])

                 """
            )

    if edges is None:
        if num_angles != 1:
            spacing = (outer_angle - inner_angle - num_angles * width) / (num_angles - 1)  # spacing between rings
        else:
            spacing = 0
        edges = roi.ring_edges(inner_angle, width, spacing, num_angles)

    # print (edges)
    angs = angulars(np.radians(edges), center, mask.shape)
    ang_center = np.average(edges, axis=1)
    ang_mask = angs * mask
    ang_mask = np.array(ang_mask, dtype=int)
    if flow_geometry:
        edges2 = edges - 180
        for edge_ in [edges2]:
            ang_mask = update_angular_mask_width_edge(edge_, mask, center, ang_mask)
            ang_mask = fix_angle_mask_at_PN_180(edge_, mask, center, ang_mask)
        if flow_angle is not None:
            edges3 = 2 * flow_angle - edges[:, ::-1]
            edges4 = 2 * flow_angle - edges[:, ::-1] - 180
            for edge_ in [edges3, edges4]:
                ang_mask = update_angular_mask_width_edge(edge_, mask, center, ang_mask)
                ang_mask = fix_angle_mask_at_PN_180(edge_, mask, center, ang_mask)
    else:
        # for i, edge_ in enumerate( edges ):
        # print(edge_)
        if fix_180_angle:
            ang_mask = fix_angle_mask_at_PN_180(edges, mask, center, ang_mask)
    labels, indices = roi.extract_label_indices(ang_mask)
    nopr = np.bincount(np.array(labels, dtype=int))[1:]
    if len(np.where(nopr == 0)[0] != 0):
        print("Some angs contain zero pixels. Please redefine the edges.")
    return ang_mask, ang_center, edges


def get_angular_mask_old(
    mask,
    inner_angle=0,
    outer_angle=360,
    width=None,
    edges=None,
    num_angles=12,
    center=None,
    dpix=[1, 1],
    flow_geometry=False,
    flow_angle=90,
):
    """
    mask: 2D-array
    inner_angle # the starting angle in unit of degree
    outer_angle #  the ending angle in unit of degree
    width       # width of each angle, in degree, default is None, there is no gap between the neighbour angle ROI
    edges: default, None. otherwise, give a customized angle edges
    num_angles    # number of angles

    center: the beam center in pixel
    dpix, the pixel size in mm. For Eiger1m/4m, the size is 75 um (0.075 mm)
    flow_geometry: if True, please give the flow angle. the map will be a mirror symmetry along the flow direction

    Returns
    -------
    ang_mask: a ring mask, np.array
    ang_center: ang in unit of degree
    ang_val: ang edges in degree

    """

    if flow_geometry:
        if edges is None:
            if inner_angle < 0:
                print("In this flow_geometry, the inner_angle should be larger than 0")
            if outer_angle > 180:
                print("In this flow_geometry, the out_angle should be smaller than 180")

    if edges is None:
        if num_angles != 1:
            spacing = (outer_angle - inner_angle - num_angles * width) / (num_angles - 1)  # spacing between rings
        else:
            spacing = 0
        edges = roi.ring_edges(inner_angle, width, spacing, num_angles)

    # print (edges)
    angs = angulars(np.radians(edges), center, mask.shape)
    ang_center = np.average(edges, axis=1)
    ang_mask = angs * mask
    ang_mask = np.array(ang_mask, dtype=int)

    if flow_geometry:
        outer_angle -= 180
        inner_angle -= 180
        edges2 = roi.ring_edges(inner_angle, width, spacing, num_angles)
        # print (edges)
        angs2 = angulars(np.radians(edges2), center, mask.shape)
        ang_mask2 = angs2 * mask
        ang_mask2 = np.array(ang_mask2, dtype=int)
        ang_mask += ang_mask2
    else:
        for i, (al, ah) in enumerate(edges):
            if al <= -180.0 and ah > -180:
                # print(i+1, al,ah)
                edge3 = np.array([[al + 360, 180]])
                ang3 = angulars(np.radians(edge3), center, mask.shape) * mask
                w = np.ravel(ang3) == 1
                # print(w)
                np.ravel(ang_mask)[w] = i + 1

    labels, indices = roi.extract_label_indices(ang_mask)
    nopr = np.bincount(np.array(labels, dtype=int))[1:]

    if len(np.where(nopr == 0)[0] != 0):
        # print (nopr)
        print("Some angs contain zero pixels. Please redefine the edges.")
    return ang_mask, ang_center, edges


def two_theta_to_radius(dist_sample, two_theta):
    """
    Converts scattering angle (2:math:`2\\theta`) to radius (from the calibrated center)
    with known detector to sample distance.

    Parameters
    ----------
    dist_sample : float
        distance from the sample to the detector (mm)

    two_theta : array
        An array of :math:`2\\theta` values

    Returns
    -------
    radius : array
        The L2 norm of the distance (mm) of each pixel from the calibrated center.
    """
    return np.tan(two_theta) * dist_sample


def get_ring_mask(
    mask,
    inner_radius=40,
    outer_radius=762,
    width=6,
    num_rings=12,
    edges=None,
    unit="pixel",
    pargs=None,
    return_q_in_pixel=False,
):
    # def get_ring_mask(  mask, inner_radius= 0.0020, outer_radius = 0.009, width = 0.0002, num_rings = 12,
    #                  edges=None, unit='pixel',pargs=None   ):
    """
    mask: 2D-array
    inner_radius #radius of the first ring
    outer_radius # radius of the last ring
    width       # width of each ring
    num_rings    # number of rings
    pargs: a dict, should contains
        center: the beam center in pixel
        Ldet: sample to detector distance
        lambda_: the wavelength, in unit of A
        dpix, the pixel size in mm. For Eiger1m/4m, the size is 75 um (0.075 mm)
        unit: if pixel, all the radius inputs are in unit of pixel
              else: should be in unit of A-1
    Returns
    -------
    ring_mask: a ring mask, np.array
    q_ring_center: q in real unit (A-1)
    q_ring_val: q edges in A-1

    """

    center, Ldet, lambda_, dpix = pargs["center"], pargs["Ldet"], pargs["lambda_"], pargs["dpix"]

    # spacing =  (outer_radius - inner_radius)/(num_rings-1) - 2    # spacing between rings
    # qc = np.int_( np.linspace( inner_radius,outer_radius, num_rings ) )
    # edges = np.zeros( [ len(qc), 2] )
    # if width%2:
    #   edges[:,0],edges[:,1] = qc - width//2,  qc + width//2 +1
    # else:
    #    edges[:,0],edges[:,1] = qc - width//2,  qc + width//2

    #  find the edges of the required rings
    if edges is None:
        if num_rings != 1:
            spacing = (outer_radius - inner_radius - num_rings * width) / (num_rings - 1)  # spacing between rings
        else:
            spacing = 0
        edges = roi.ring_edges(inner_radius, width, spacing, num_rings)

    if (unit == "pixel") or (unit == "p"):
        if not return_q_in_pixel:
            two_theta = utils.radius_to_twotheta(Ldet, edges * dpix)
            q_ring_val = utils.twotheta_to_q(two_theta, lambda_)
        else:
            q_ring_val = edges
            # print(edges)
    else:  # in unit of A-1
        two_theta = utils.q_to_twotheta(edges, lambda_)
        q_ring_val = edges
        edges = two_theta_to_radius(Ldet, two_theta) / dpix  # converto pixel

    q_ring_center = np.average(q_ring_val, axis=1)

    rings = roi.rings(edges, center, mask.shape)
    ring_mask = rings * mask
    ring_mask = np.array(ring_mask, dtype=int)

    labels, indices = roi.extract_label_indices(ring_mask)
    nopr = np.bincount(np.array(labels, dtype=int))[1:]

    if len(np.where(nopr == 0)[0] != 0):
        print(nopr)
        print("Some rings contain zero pixels. Please redefine the edges.")
    return ring_mask, q_ring_center, q_ring_val


def get_ring_anglar_mask(ring_mask, ang_mask, q_ring_center, ang_center):
    """get   ring_anglar mask"""

    ring_max = ring_mask.max()

    ang_mask_ = np.zeros(ang_mask.shape)
    ind = np.where(ang_mask != 0)
    ang_mask_[ind] = ang_mask[ind] + 1e9  # add some large number to qr

    dumy_ring_mask = np.zeros(ring_mask.shape)
    dumy_ring_mask[ring_mask == 1] = 1
    dumy_ring_ang = dumy_ring_mask * ang_mask
    real_ang_lab = np.int_(np.unique(dumy_ring_ang)[1:]) - 1

    ring_ang = ring_mask * ang_mask_

    # convert label_array_qzr to [1,2,3,...]
    ura = np.unique(ring_ang)[1:]

    ur = np.unique(ring_mask)[1:]
    ua = np.unique(ang_mask)[real_ang_lab]

    ring_ang_ = np.zeros_like(ring_ang)
    newl = np.arange(1, len(ura) + 1)
    # newl = np.int_( real_ang_lab )

    rc = [[q_ring_center[i]] * len(ua) for i in range(len(ur))]
    ac = list(ang_center[ua]) * len(ur)

    # rc =list( q_ring_center) * len( ua )
    # ac= [  [ ang_center[i]]*len( ur ) for i in range(len( ua ))  ]

    for i, label in enumerate(ura):
        # print (i, label)
        ring_ang_.ravel()[np.where(ring_ang.ravel() == label)[0]] = newl[i]

    return np.int_(ring_ang_), np.concatenate(np.array(rc)), np.array(ac)


def show_ring_ang_roi(data, rois, alpha=0.3, save=False, *argv, **kwargs):
    """
    May 16, 2016, Y.G.@CHX
    plot a saxs image with rois( a label array)

    Parameters:
        data: 2-D array, a gisaxs image
        rois:  2-D array, a label array


    Options:
        alpha:  transparency of the label array on top of data

    Return:
         a plot of a qzr map of a gisaxs image with rois( a label array)


    Examples:
        show_qzr_roi( avg_imgr, box_maskr, inc_x0, ticks)

    """

    # import matplotlib.pyplot as plt
    # import copy
    # import matplotlib.cm as mcm

    # cmap='viridis'
    # _cmap = copy.copy((mcm.get_cmap(cmap)))
    # _cmap.set_under('w', 0)

    avg_imgr, box_maskr = data, rois
    num_qzr = len(np.unique(box_maskr)) - 1
    fig, ax = plt.subplots(figsize=(8, 12))
    ax.set_title("ROI--Labeled Array on Data")
    im, im_label = show_label_array_on_image(
        ax,
        avg_imgr,
        box_maskr,
        imshow_cmap="viridis",
        cmap="Paired",
        alpha=alpha,
        vmin=0.01,
        vmax=30.0,
        origin="lower",
    )

    for i in range(1, num_qzr + 1):
        ind = np.where(box_maskr == i)[1]
        indz = np.where(box_maskr == i)[0]
        c = "%i" % i
        y_val = int(indz.mean())

        x_val = int(ind.mean())
        # print (xval, y)
        ax.text(x_val, y_val, c, va="center", ha="center")

        # print (x_val1,x_val2)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    if save:
        # dt =datetime.now()
        # CurTime = '%s%02d%02d-%02d%02d-' % (dt.year, dt.month, dt.day,dt.hour,dt.minute)
        path = kwargs["path"]
        if "uid" in kwargs:
            uid = kwargs["uid"]
        else:
            uid = "uid"
        # fp = path + "uid= %s--Waterfall-"%uid + CurTime + '.png'
        fp = path + "uid=%s--ROI-on-image-" % uid + ".png"
        fig.savefig(fp, dpi=fig.dpi)

    # ax.set_xlabel(r'$q_r$', fontsize=22)
    # ax.set_ylabel(r'$q_z$',fontsize=22)
    # plt.show()


def plot_qIq_with_ROI(
    q, iq, q_ring_center, q_ring_edge=None, logs=True, save=False, return_fig=False, *argv, **kwargs
):
    """Aug 6, 2016, Y.G.@CHX
    Update@2019, March to make a span plot with q_ring_edge
    plot q~Iq with interested q rings"""
    uid = "uid"
    if "uid" in kwargs.keys():
        uid = kwargs["uid"]
    if RUN_GUI:
        fig = Figure(figsize=(8, 6))
        axes = fig.add_subplot(111)
    else:
        fig, axes = plt.subplots(figsize=(8, 6))
    if logs:
        axes.semilogy(q, iq, "-o")
    else:
        axes.plot(q, iq, "-o")
    axes.set_title("%s--Circular Average with the Q ring values" % uid)
    axes.set_ylabel("I(q)")
    axes.set_xlabel("Q " r"($\AA^{-1}$)")
    if "xlim" in kwargs.keys():
        xlim = kwargs["xlim"]
    else:
        xlim = [q.min(), q.max()]
    if "ylim" in kwargs.keys():
        ylim = kwargs["ylim"]
    else:
        ylim = [iq.min(), iq.max()]
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if q_ring_edge is not None:
        for qe in q_ring_edge:
            p = axes.axvspan(qe[0], qe[1], facecolor="#2ca02c", alpha=0.5)
    else:
        num_rings = len(np.unique(q_ring_center))
        for i in range(num_rings):
            axes.axvline(q_ring_center[i])  # , linewidth = 5  )
    if save:
        # dt =datetime.now()
        # CurTime = '%s%02d%02d-%02d%02d-' % (dt.year, dt.month, dt.day,dt.hour,dt.minute)
        path = kwargs["path"]
        if "uid" in kwargs:
            uid = kwargs["uid"]
        else:
            uid = "uid"
        # fp = path + "uid= %s--Waterfall-"%uid + CurTime + '.png'
        fp = path + "%s_ROI_on_Iq" % uid + ".png"
        fig.savefig(fp, dpi=fig.dpi)
    # plt.show()
    if return_fig:
        return fig, axes


def get_each_ring_mean_intensity(
    data_series, ring_mask, sampling, timeperframe, plot_=True, save=False, *argv, **kwargs
):
    """
    get time dependent mean intensity of each ring
    """
    mean_int_sets, index_list = roi.mean_intensity(np.array(data_series[::sampling]), ring_mask)

    times = np.arange(len(data_series)) * timeperframe  # get the time for each frame
    num_rings = len(np.unique(ring_mask)[1:])
    if plot_:
        fig, ax = plt.subplots(figsize=(8, 8))
        uid = "uid"
        if "uid" in kwargs.keys():
            uid = kwargs["uid"]

        ax.set_title("%s--Mean intensity of each ring" % uid)
        for i in range(num_rings):
            ax.plot(mean_int_sets[:, i], label="Ring " + str(i + 1), marker="o", ls="-")
            ax.set_xlabel("Time")
            ax.set_ylabel("Mean Intensity")
        ax.legend(loc="best")

        if save:
            # dt =datetime.now()
            # CurTime = '%s%02d%02d-%02d%02d-' % (dt.year, dt.month, dt.day,dt.hour,dt.minute)
            path = kwargs["path"]
            # fp = path + "Uid= %s--Mean intensity of each ring-"%uid + CurTime + '.png'
            fp = path + "%s_Mean_intensity_of_each_ROI" % uid + ".png"

            fig.savefig(fp, dpi=fig.dpi)

        # plt.show()
    return times, mean_int_sets


# plot g2 results
def plot_saxs_rad_ang_g2(g2, taus, res_pargs=None, master_angle_plot=False, return_fig=False, *argv, **kwargs):
    """plot g2 results of segments with radius and angle partation ,

    g2: one-time correlation function
    taus: the time delays
    res_pargs, a dict, can contains
        uid/path/qr_center/qz_center/
    master_angle_plot: if True, plot angle first, then q
    kwargs: can contains
        vlim: [vmin,vmax]: for the plot limit of y, the y-limit will be [vmin * min(y), vmx*max(y)]
        ylim/xlim: the limit of y and x

    e.g.
    plot_saxs_rad_ang_g2( g2b, taus= np.arange( g2b.shape[0]) *timeperframe, q_ring_center = q_ring_center, ang_center=ang_center, vlim=[.99, 1.01] )

    """
    if res_pargs is not None:
        uid = res_pargs["uid"]
        path = res_pargs["path"]
        q_ring_center = res_pargs["q_ring_center"]
        num_qr = len(q_ring_center)
        ang_center = res_pargs["ang_center"]
        num_qa = len(ang_center)

    else:
        if "uid" in kwargs.keys():
            uid = kwargs["uid"]
        else:
            uid = "uid"
        if "path" in kwargs.keys():
            path = kwargs["path"]
        else:
            path = ""
        if "q_ring_center" in kwargs.keys():
            q_ring_center = kwargs["q_ring_center"]
            num_qr = len(q_ring_center)
        else:
            print("Please give q_ring_center")
        if "ang_center" in kwargs.keys():
            ang_center = kwargs["ang_center"]
            num_qa = len(ang_center)
        else:
            print("Please give ang_center")

    if master_angle_plot:
        first_var = num_qa
        sec_var = num_qr
    else:
        first_var = num_qr
        sec_var = num_qa

    for qr_ind in range(first_var):
        if RUN_GUI:
            fig = Figure(figsize=(10, 12))
        else:
            fig = plt.figure(figsize=(10, 12))
        # fig = plt.figure()
        if master_angle_plot:
            title_qr = "Angle= %.2f" % (ang_center[qr_ind]) + r"$^\circ$"
        else:
            title_qr = " Qr= %.5f  " % (q_ring_center[qr_ind]) + r"$\AA^{-1}$"

        plt.title("uid= %s:--->" % uid + title_qr, fontsize=20, y=1.1)
        # print (qz_ind,title_qz)
        # if num_qr!=1:plt.axis('off')
        plt.axis("off")
        sx = int(round(np.sqrt(sec_var)))
        if sec_var % sx == 0:
            sy = int(sec_var / sx)
        else:
            sy = int(sec_var / sx + 1)

        for sn in range(sec_var):
            ax = fig.add_subplot(sx, sy, sn + 1)
            ax.set_ylabel("g2")
            ax.set_xlabel(r"$\tau $ $(s)$", fontsize=16)
            if master_angle_plot:
                i = sn + qr_ind * num_qr
                title_qa = "%.5f  " % (q_ring_center[sn]) + r"$\AA^{-1}$"
            else:
                i = sn + qr_ind * num_qa
                title_qa = "%.2f" % (ang_center[sn]) + r"$^\circ$" + "( %d )" % (i)
            # title_qa = " Angle= " + '%.2f'%( ang_center[sn]) + r'$^\circ$' + '( %d )'%i

            # title_qa = '%.2f'%( ang_center[sn]) + r'$^\circ$' + '( %d )'%(i)
            # if num_qr==1:
            #    title = 'uid= %s:--->'%uid + title_qr + '__' +  title_qa
            # else:
            #    title = title_qa
            title = title_qa
            ax.set_title(title, y=1.1, fontsize=12)
            y = g2[:, i]
            ax.semilogx(taus, y, "-o", markersize=6)

            if "ylim" in kwargs:
                ax.set_ylim(kwargs["ylim"])
            elif "vlim" in kwargs:
                vmin, vmax = kwargs["vlim"]
                ax.set_ylim([min(y) * vmin, max(y[1:]) * vmax])
            else:
                pass

            if "xlim" in kwargs:
                ax.set_xlim(kwargs["xlim"])

        # dt =datetime.now()
        # CurTime = '%s%02d%02d-%02d%02d-' % (dt.year, dt.month, dt.day,dt.hour,dt.minute)

        # fp = path + 'g2--uid=%s-qr=%s'%(uid,q_ring_center[qr_ind]) + CurTime + '.png'
        fp = path + "uid=%s--g2-qr=%s" % (uid, q_ring_center[qr_ind]) + "-.png"
        plt.savefig(fp, dpi=fig.dpi)
        fig.set_tight_layout(True)
    if return_fig:
        return fig


############################################
##a good func to fit g2 for all types of geogmetries
############################################


def fit_saxs_rad_ang_g2(
    g2, res_pargs=None, function="simple_exponential", fit_range=None, master_angle_plot=False, *argv, **kwargs
):
    """
    Fit one-time correlation function

    The support functions include simple exponential and stretched/compressed exponential
    Parameters
    ----------
    g2: one-time correlation function for fit, with shape as [taus, qs]
    res_pargs: a dict, contains keys
        taus: the time delay, with the same length as g2
        q_ring_center:  the center of q rings, for the title of each sub-plot
        uid: unique id, for the title of plot

    function:
        'simple_exponential': fit by a simple exponential function, defined as
                    beta * np.exp(-2 * relaxation_rate * lags) + baseline
        'streched_exponential': fit by a streched exponential function, defined as
                    beta * (np.exp(-2 * relaxation_rate * lags))**alpha + baseline

    #fit_vibration:
    #    if True, will fit the g2 by a dumped sin function due to beamline mechnical oscillation

    Returns
    -------
    fit resutls:
        a dict, with keys as
        'baseline':
         'beta':
         'relaxation_rate':
    an example:
        result = fit_g2( g2, res_pargs, function = 'simple')
        result = fit_g2( g2, res_pargs, function = 'stretched')
    """

    if res_pargs is not None:
        uid = res_pargs["uid"]
        path = res_pargs["path"]
        q_ring_center = res_pargs["q_ring_center"]
        num_qr = len(q_ring_center)
        ang_center = res_pargs["ang_center"]
        num_qa = len(ang_center)
        taus = res_pargs["taus"]

    else:
        if "uid" in kwargs.keys():
            uid = kwargs["uid"]
        else:
            uid = "uid"
        if "path" in kwargs.keys():
            path = kwargs["path"]
        else:
            path = ""
        if "q_ring_center" in kwargs.keys():
            q_ring_center = kwargs["q_ring_center"]
            num_qr = len(q_ring_center)
        else:
            print("Please give q_ring_center")
        if "ang_center" in kwargs.keys():
            ang_center = kwargs["ang_center"]
            num_qa = len(ang_center)
        else:
            print("Please give ang_center")

    num_rings = g2.shape[1]
    beta = np.zeros(num_rings)  #  contrast factor
    rate = np.zeros(num_rings)  #  relaxation rate
    alpha = np.zeros(num_rings)  #  alpha
    baseline = np.zeros(num_rings)  #  baseline
    freq = np.zeros(num_rings)

    if function == "flow_para_function" or function == "flow_para":
        flow = np.zeros(num_rings)  #  baseline
    if "fit_variables" in kwargs:
        additional_var = kwargs["fit_variables"]
        _vars = [k for k in list(additional_var.keys()) if additional_var[k] is False]
    else:
        _vars = []

    # print (_vars)

    _guess_val = dict(beta=0.1, alpha=1.0, relaxation_rate=0.005, baseline=1.0)

    if "guess_values" in kwargs:
        guess_values = kwargs["guess_values"]
        _guess_val.update(guess_values)

    if function == "simple_exponential" or function == "simple":
        _vars = np.unique(_vars + ["alpha"])
        mod = Model(stretched_auto_corr_scat_factor)  # ,  independent_vars= list( _vars)   )

    elif function == "stretched_exponential" or function == "stretched":
        mod = Model(stretched_auto_corr_scat_factor)  # ,  independent_vars=  _vars)

    elif function == "stretched_vibration":
        mod = Model(stretched_auto_corr_scat_factor_with_vibration)  # ,  independent_vars=  _vars)

    elif function == "flow_para_function" or function == "flow_para":
        mod = Model(flow_para_function)  # ,  independent_vars=  _vars)

    else:
        print(
            "The %s is not supported.The supported functions include simple_exponential and stretched_exponential"
            % function
        )

    mod.set_param_hint("baseline", min=0.5, max=1.5)
    mod.set_param_hint("beta", min=0.0)
    mod.set_param_hint("alpha", min=0.0)
    mod.set_param_hint("relaxation_rate", min=0.0)
    if function == "flow_para_function" or function == "flow_para":
        mod.set_param_hint("flow_velocity", min=0)
    if function == "stretched_vibration":
        mod.set_param_hint("freq", min=0)
        mod.set_param_hint("amp", min=0)

    _beta = _guess_val["beta"]
    _alpha = _guess_val["alpha"]
    _relaxation_rate = _guess_val["relaxation_rate"]
    _baseline = _guess_val["baseline"]
    pars = mod.make_params(beta=_beta, alpha=_alpha, relaxation_rate=_relaxation_rate, baseline=_baseline)

    if function == "flow_para_function" or function == "flow_para":
        _flow_velocity = _guess_val["flow_velocity"]
        pars = mod.make_params(
            beta=_beta,
            alpha=_alpha,
            flow_velocity=_flow_velocity,
            relaxation_rate=_relaxation_rate,
            baseline=_baseline,
        )

    if function == "stretched_vibration":
        _freq = _guess_val["freq"]
        _amp = _guess_val["amp"]
        pars = mod.make_params(
            beta=_beta, alpha=_alpha, freq=_freq, amp=_amp, relaxation_rate=_relaxation_rate, baseline=_baseline
        )

    for v in _vars:
        pars["%s" % v].vary = False
    if master_angle_plot:
        first_var = num_qa
        sec_var = num_qr
    else:
        first_var = num_qr
        sec_var = num_qa

    for qr_ind in range(first_var):
        # fig = plt.figure(figsize=(10, 12))
        fig = plt.figure(figsize=(14, 8))
        # fig = plt.figure()
        if master_angle_plot:
            title_qr = "Angle= %.2f" % (ang_center[qr_ind]) + r"$^\circ$"
        else:
            title_qr = " Qr= %.5f  " % (q_ring_center[qr_ind]) + r"$\AA^{-1}$"

        # plt.title('uid= %s:--->'%uid + title_qr,fontsize=20, y =1.1)
        plt.axis("off")

        # sx = int(round(np.sqrt(  sec_var  )) )
        sy = 4
        # if sec_var%sx == 0:
        if sec_var % sy == 0:
            # sy = int(sec_var/sx)
            sx = int(sec_var / sy)
        else:
            # sy=int(sec_var/sx+1)
            sx = int(sec_var / sy + 1)

        for sn in range(sec_var):
            ax = fig.add_subplot(sx, sy, sn + 1)
            ax.set_ylabel(r"$g^($" + r"$^2$" + r"$^)$" + r"$(Q,$" + r"$\tau$" + r"$)$")
            ax.set_xlabel(r"$\tau $ $(s)$", fontsize=16)
            if master_angle_plot:
                i = sn + qr_ind * num_qr
                title_qa = "%.5f  " % (q_ring_center[sn]) + r"$\AA^{-1}$"
            else:
                i = sn + qr_ind * num_qa
                title_qa = "%.2f" % (ang_center[sn]) + r"$^\circ$" + "( %d )" % (i)

            title = title_qa
            ax.set_title(title, y=1.1)

            if fit_range is not None:
                y = g2[1:, i][fit_range[0] : fit_range[1]]
                lags = taus[1:][fit_range[0] : fit_range[1]]
            else:
                y = g2[1:, i]
                lags = taus[1:]

            result1 = mod.fit(y, pars, x=lags)

            # print ( result1.best_values)
            rate[i] = result1.best_values["relaxation_rate"]
            # rate[i] = 1e-16
            beta[i] = result1.best_values["beta"]

            # baseline[i] = 1.0
            baseline[i] = result1.best_values["baseline"]

            # print( result1.best_values['freq']  )

            if function == "simple_exponential" or function == "simple":
                alpha[i] = 1.0
            elif function == "stretched_exponential" or function == "stretched":
                alpha[i] = result1.best_values["alpha"]
            elif function == "stretched_vibration":
                alpha[i] = result1.best_values["alpha"]
                freq[i] = result1.best_values["freq"]

            if function == "flow_para_function" or function == "flow_para":
                flow[i] = result1.best_values["flow_velocity"]

            ax.semilogx(taus[1:], g2[1:, i], "ro")
            ax.semilogx(lags, result1.best_fit, "-b")

            txts = r"$\gamma$" + r"$ = %.3f$" % (1 / rate[i]) + r"$ s$"
            x = 0.25
            y0 = 0.75
            fontsize = 12
            ax.text(x=x, y=y0, s=txts, fontsize=fontsize, transform=ax.transAxes)
            txts = r"$\alpha$" + r"$ = %.3f$" % (alpha[i])
            # txts = r'$\beta$' + r'$ = %.3f$'%(beta[i]) +  r'$ s^{-1}$'
            ax.text(x=x, y=y0 - 0.1, s=txts, fontsize=fontsize, transform=ax.transAxes)

            txts = r"$baseline$" + r"$ = %.3f$" % (baseline[i])
            ax.text(x=x, y=y0 - 0.2, s=txts, fontsize=fontsize, transform=ax.transAxes)

            if function == "flow_para_function" or function == "flow_para":
                txts = r"$flow_v$" + r"$ = %.3f$" % (flow[i])
                ax.text(x=x, y=y0 - 0.3, s=txts, fontsize=fontsize, transform=ax.transAxes)

            if "ylim" in kwargs:
                ax.set_ylim(kwargs["ylim"])
            elif "vlim" in kwargs:
                vmin, vmax = kwargs["vlim"]
                ax.set_ylim([min(y) * vmin, max(y[1:]) * vmax])
            else:
                pass
            if "xlim" in kwargs:
                ax.set_xlim(kwargs["xlim"])

        fp = path + "uid=%s--g2--qr-%s--fit-" % (uid, q_ring_center[qr_ind]) + ".png"
        fig.savefig(fp, dpi=fig.dpi)
        fig.tight_layout()
        # plt.show()

    result = dict(beta=beta, rate=rate, alpha=alpha, baseline=baseline)
    if function == "flow_para_function" or function == "flow_para":
        result = dict(beta=beta, rate=rate, alpha=alpha, baseline=baseline, flow_velocity=flow)
    if function == "stretched_vibration":
        result = dict(beta=beta, rate=rate, alpha=alpha, baseline=baseline, freq=freq)

    return result


def save_seg_saxs_g2(g2, res_pargs, time_label=True, *argv, **kwargs):
    """
    Aug 8, 2016, Y.G.@CHX
    save g2 results,
       res_pargs should contain
           g2: one-time correlation function
           res_pargs: contions taus, q_ring_center values
           path:
           uid:

    """
    taus = res_pargs["taus"]
    qz_center = res_pargs["q_ring_center"]
    qr_center = res_pargs["ang_center"]
    path = res_pargs["path"]
    uid = res_pargs["uid"]

    df = DataFrame(np.hstack([(taus).reshape(len(g2), 1), g2]))
    columns = []
    columns.append("tau")

    for qz in qz_center:
        for qr in qr_center:
            columns.append([str(qz), str(qr)])

    df.columns = columns

    if time_label:
        dt = datetime.now()
        CurTime = "%s%02d%02d-%02d%02d-" % (dt.year, dt.month, dt.day, dt.hour, dt.minute)
        filename = os.path.join(path, "g2-%s-%s.csv" % (uid, CurTime))
    else:
        filename = os.path.join(path, "uid=%s--g2.csv" % (uid))
    df.to_csv(filename)
    print("The g2 of uid= %s is saved with filename as %s" % (uid, filename))


def linear_fit(x, y):
    D0 = np.polyfit(x, y, 1)
    gmfit = np.poly1d(D0)
    return D0, gmfit


def plot_gamma():
    """not work"""
    fig, ax = plt.subplots()
    ax.set_title("Uid= %s--Beta" % uid)
    ax.set_title("Uid= %s--Gamma" % uid)
    # ax.plot(  q_ring_center**2 , 1/rate, 'ro', ls='--')

    ax.loglog(q_ring_center, 1 / result["rate"], "ro", ls="--")
    # ax.set_ylabel('Log( Beta0 'r'$\beta$'"($s^{-1}$)")
    ax.set_ylabel("Log( Gamma )")
    ax.set_xlabel("$Log(q)$" r"($\AA^{-1}$)")
    # plt.show()


def multi_uids_saxs_flow_xpcs_analysis(
    uids,
    md,
    run_num=1,
    sub_num=None,
    good_start=10,
    good_end=None,
    force_compress=False,
    fit_vibration=True,
    fit=True,
    compress=True,
    para_run=False,
):
    """'Aug 16, 2016, YG@CHX-NSLS2
    Do SAXS-XPCS analysis for multi uid data
    uids: a list of uids to be analyzed
    md: metadata, should at least include
        mask: array, mask data
        data_dir: the path to save data, the result will be saved in data_dir/uid/...
        dpix:
        Ldet:
        lambda:
        timeperframe:
        center
    run_num: the run number
    sub_num: the number in each sub-run
    fit: if fit, do fit for g2 and show/save all fit plots
    compress: apply a compress algorithm

    Save g2/metadata/g2-fit plot/g2 q-rate plot/ of each uid in data_dir/uid/...
    return:
    g2s: a dictionary, {run_num: sub_num: g2_of_each_uid}
    taus,
    use_uids: return the valid uids
    """

    g2s = {}  # g2s[run_number][sub_seq]  =  g2 of each uid
    lag_steps = [0]
    useful_uids = {}
    if sub_num is None:
        sub_num = len(uids) // run_num

    mask = md["mask"]
    data_dir = md["data_dir"]
    # ring_mask = md['ring_mask']
    # q_ring_center = md['q_ring_center']

    seg_mask_v = md["seg_mask_v"]
    seg_mask_p = md["seg_mask_p"]
    rcen_p, acen_p = md["rcen_p"], md["acen_v"]
    rcen_v, acen_v = md["rcen_p"], md["acen_v"]

    lag_steps = [0]

    for run_seq in range(run_num):
        g2s[run_seq + 1] = {}
        useful_uids[run_seq + 1] = {}
        i = 0
        for sub_seq in range(0, sub_num):
            # good_end=good_end

            uid = uids[sub_seq + run_seq * sub_num]
            print("The %i--th uid to be analyzed is : %s" % (i, uid))
            try:
                detector = get_detector(db[uid])
                imgs = load_data(uid, detector, reverse=True)
            except:
                print("The %i--th uid: %s can not load data" % (i, uid))
                imgs = 0

            data_dir_ = os.path.join(data_dir, "%s/" % uid)
            os.makedirs(data_dir_, exist_ok=True)

            i += 1
            if imgs != 0:
                imgsa = apply_mask(imgs, mask)
                Nimg = len(imgs)
                md_ = imgs.md
                useful_uids[run_seq + 1][i] = uid
                g2s[run_seq + 1][i] = {}
                # if compress:
                filename = "/XF11ID/analysis/Compressed_Data" + "/uid_%s.cmp" % uid
                # update code here to use new pass uid to compress, 2016, Dec 3
                if False:
                    mask, avg_img, imgsum, bad_frame_list = compress_eigerdata(
                        imgs,
                        mask,
                        md_,
                        filename,
                        force_compress=force_compress,
                        bad_pixel_threshold=2.4e18,
                        nobytes=4,
                        para_compress=True,
                        num_sub=100,
                    )
                if True:
                    mask, avg_img, imgsum, bad_frame_list = compress_eigerdata(
                        uid,
                        mask,
                        md_,
                        filename,
                        force_compress=False,
                        bad_pixel_threshold=2.4e18,
                        nobytes=4,
                        para_compress=True,
                        num_sub=100,
                        dtypes="uid",
                        reverse=True,
                    )

                try:
                    md["Measurement"] = db[uid]["start"]["Measurement"]
                    # md['sample']=db[uid]['start']['sample']
                    # md['sample']= 'PS205000-PMMA-207000-SMMA3'
                    print(md["Measurement"])

                except:
                    md["Measurement"] = "Measurement"
                    md["sample"] = "sample"

                dpix = md["x_pixel_size"] * 1000.0  # in mm, eiger 4m is 0.075 mm
                lambda_ = md["incident_wavelength"]  # wavelegth of the X-rays in Angstroms
                Ldet = md["detector_distance"] * 1000  # detector to sample distance (mm)
                exposuretime = md["count_time"]
                acquisition_period = md["frame_time"]
                timeperframe = acquisition_period  # for g2
                # timeperframe = exposuretime#for visiblitly
                # timeperframe = 2  ## manual overwrite!!!! we apparently writing the wrong metadata....
                center = md["center"]

                setup_pargs = dict(
                    uid=uid,
                    dpix=dpix,
                    Ldet=Ldet,
                    lambda_=lambda_,
                    timeperframe=timeperframe,
                    center=center,
                    path=data_dir_,
                )

                md["avg_img"] = avg_img
                # plot1D( y = imgsum[ np.array( [i for i in np.arange( len(imgsum)) if i not in bad_frame_list])],
                #   title ='Uid= %s--imgsum'%uid, xlabel='Frame', ylabel='Total_Intensity', legend=''   )
                min_inten = 10

                # good_start = np.where( np.array(imgsum) > min_inten )[0][0]
                good_start = good_start

                if good_end is None:
                    good_end_ = len(imgs)
                else:
                    good_end_ = good_end
                FD = Multifile(filename, good_start, good_end_)

                good_start = max(good_start, np.where(np.array(imgsum) > min_inten)[0][0])
                print("With compression, the good_start frame number is: %s " % good_start)
                print("The good_end frame number is: %s " % good_end_)

                norm = None
                ###################

                # Do correlaton here
                for nconf, seg_mask in enumerate([seg_mask_v, seg_mask_p]):
                    if nconf == 0:
                        conf = "v"
                    else:
                        conf = "p"

                    rcen = md["rcen_%s" % conf]
                    acen = md["acen_%s" % conf]

                    if not para_run:
                        g2, lag_stepsv = cal_g2(
                            FD,
                            seg_mask,
                            bad_frame_list,
                            good_start,
                            num_buf=8,
                        )
                    else:
                        g2, lag_stepsv = cal_g2p(
                            FD, seg_mask, bad_frame_list, good_start, num_buf=8, imgsum=None, norm=norm
                        )

                    if len(lag_steps) < len(lag_stepsv):
                        lag_steps = lag_stepsv
                    taus = lag_steps * timeperframe
                    res_pargs = dict(
                        taus=taus,
                        q_ring_center=np.unique(rcen),
                        ang_center=np.unique(acen),
                        path=data_dir_,
                        uid=uid + "_1a_mq%s" % conf,
                    )
                    save_g2(g2, taus=taus, qr=rcen, qz=acen, uid=uid + "_1a_mq%s" % conf, path=data_dir_)

                    if nconf == 0:
                        g2s[run_seq + 1][i]["v"] = g2  # perpendular
                    else:
                        g2s[run_seq + 1][i]["p"] = g2  # parallel

                    if fit:
                        if False:
                            g2_fit_result, taus_fit, g2_fit = get_g2_fit(
                                g2,
                                res_pargs=res_pargs,
                                function="stretched_vibration",
                                vlim=[0.95, 1.05],
                                fit_variables={
                                    "baseline": True,
                                    "beta": True,
                                    "alpha": False,
                                    "relaxation_rate": True,
                                    "freq": fit_vibration,
                                    "amp": True,
                                },
                                fit_range=None,
                                guess_values={
                                    "baseline": 1.0,
                                    "beta": 0.05,
                                    "alpha": 1.0,
                                    "relaxation_rate": 0.01,
                                    "freq": 60,
                                    "amp": 0.1,
                                },
                            )

                        if nconf == 0:  # for vertical
                            function = "stretched"
                            g2_fit_result, taus_fit, g2_fit = get_g2_fit(
                                g2,
                                res_pargs=res_pargs,
                                function=function,
                                vlim=[0.95, 1.05],
                                fit_variables={
                                    "baseline": True,
                                    "beta": True,
                                    "alpha": False,
                                    "relaxation_rate": True,
                                },
                                fit_range=None,
                                guess_values={
                                    "baseline": 1.0,
                                    "beta": 0.05,
                                    "alpha": 1.0,
                                    "relaxation_rate": 0.01,
                                },
                            )
                        else:
                            function = "flow_para"
                            g2_fit_result, taus_fit, g2_fit = get_g2_fit(
                                g2,
                                res_pargs=res_pargs,
                                function=function,
                                vlim=[0.99, 1.05],
                                fit_range=None,
                                fit_variables={
                                    "baseline": True,
                                    "beta": True,
                                    "alpha": False,
                                    "relaxation_rate": True,
                                    "flow_velocity": True,
                                },
                                guess_values={
                                    "baseline": 1.0,
                                    "beta": 0.05,
                                    "alpha": 1.0,
                                    "relaxation_rate": 0.01,
                                    "flow_velocity": 1,
                                },
                            )

                        save_g2(
                            g2_fit,
                            taus=taus_fit,
                            qr=rcen,
                            qz=acen,
                            uid=uid + "_1a_mq%s" % conf + "_fit",
                            path=data_dir_,
                        )

                        res_pargs_fit = dict(
                            taus=taus,
                            q_ring_center=np.unique(rcen),
                            ang_center=[acen[0]],
                            path=data_dir_,
                            uid=uid + "_1a_mq%s" % conf + "_fit",
                        )

                        plot_g2(
                            g2,
                            res_pargs=res_pargs,
                            tau_2=taus_fit,
                            g2_2=g2_fit,
                            fit_res=g2_fit_result,
                            function=function,
                            master_plot="qz",
                            vlim=[0.95, 1.05],
                            geometry="ang_saxs",
                            append_name=conf + "_fit",
                        )

                        dfv = save_g2_fit_para_tocsv(
                            g2_fit_result, filename=uid + "_1a_mq" + conf + "_fit_para", path=data_dir_
                        )

                        fit_q_rate(
                            np.unique(rcen)[:],
                            dfv["relaxation_rate"],
                            power_variable=False,
                            uid=uid + "_" + conf + "_fit_rate",
                            path=data_dir_,
                        )

                        # psave_obj( fit_result, data_dir_ + 'uid=%s-g2-fit-para'%uid )
                    psave_obj(md, data_dir_ + "uid=%s-md" % uid)  # save the setup parameters

                FD = 0
                avg_img, imgsum, bad_frame_list = [0, 0, 0]
                md["avg_img"] = 0
                imgs = 0
                print("*" * 40)
                print()

    taus = taus
    return g2s, taus, useful_uids


def multi_uids_saxs_xpcs_analysis(
    uids,
    md,
    run_num=1,
    sub_num=None,
    good_start=10,
    good_end=None,
    force_compress=False,
    fit=True,
    compress=True,
    para_run=False,
):
    """'Aug 16, 2016, YG@CHX-NSLS2
    Do SAXS-XPCS analysis for multi uid data
    uids: a list of uids to be analyzed
    md: metadata, should at least include
        mask: array, mask data
        data_dir: the path to save data, the result will be saved in data_dir/uid/...
        dpix:
        Ldet:
        lambda:
        timeperframe:
        center
    run_num: the run number
    sub_num: the number in each sub-run
    fit: if fit, do fit for g2 and show/save all fit plots
    compress: apply a compress algorithm

    Save g2/metadata/g2-fit plot/g2 q-rate plot/ of each uid in data_dir/uid/...
    return:
    g2s: a dictionary, {run_num: sub_num: g2_of_each_uid}
    taus,
    use_uids: return the valid uids
    """

    g2s = {}  # g2s[run_number][sub_seq]  =  g2 of each uid
    lag_steps = [0]
    useful_uids = {}
    if sub_num is None:
        sub_num = len(uids) // run_num

    mask = md["mask"]
    data_dir = md["data_dir"]
    ring_mask = md["ring_mask"]
    q_ring_center = md["q_ring_center"]

    for run_seq in range(run_num):
        g2s[run_seq + 1] = {}
        useful_uids[run_seq + 1] = {}
        i = 0
        for sub_seq in range(0, sub_num):
            # good_end=good_end

            uid = uids[sub_seq + run_seq * sub_num]
            print("The %i--th uid to be analyzed is : %s" % (i, uid))
            try:
                detector = get_detector(db[uid])
                imgs = load_data(uid, detector, reverse=True)
            except:
                print("The %i--th uid: %s can not load data" % (i, uid))
                imgs = 0

            data_dir_ = os.path.join(data_dir, "%s/" % uid)
            os.makedirs(data_dir_, exist_ok=True)

            i += 1
            if imgs != 0:
                imgsa = apply_mask(imgs, mask)
                Nimg = len(imgs)
                md_ = imgs.md
                useful_uids[run_seq + 1][i] = uid
                if compress:
                    filename = "/XF11ID/analysis/Compressed_Data" + "/uid_%s.cmp" % uid
                    # update code here to use new pass uid to compress, 2016, Dec 3
                    if False:
                        mask, avg_img, imgsum, bad_frame_list = compress_eigerdata(
                            imgs,
                            mask,
                            md_,
                            filename,
                            force_compress=force_compress,
                            bad_pixel_threshold=2.4e18,
                            nobytes=4,
                            para_compress=True,
                            num_sub=100,
                        )
                    if True:
                        mask, avg_img, imgsum, bad_frame_list = compress_eigerdata(
                            uid,
                            mask,
                            md_,
                            filename,
                            force_compress=True,
                            bad_pixel_threshold=2.4e18,
                            nobytes=4,
                            para_compress=True,
                            num_sub=100,
                            dtypes="uid",
                            reverse=True,
                        )

                    try:
                        md["Measurement"] = db[uid]["start"]["Measurement"]
                        # md['sample']=db[uid]['start']['sample']
                        # md['sample']= 'PS205000-PMMA-207000-SMMA3'
                        print(md["Measurement"])

                    except:
                        md["Measurement"] = "Measurement"
                        md["sample"] = "sample"

                    dpix = md["x_pixel_size"] * 1000.0  # in mm, eiger 4m is 0.075 mm
                    lambda_ = md["incident_wavelength"]  # wavelegth of the X-rays in Angstroms
                    Ldet = md["detector_distance"] * 1000  # detector to sample distance (mm)
                    exposuretime = md["count_time"]
                    acquisition_period = md["frame_time"]
                    timeperframe = acquisition_period  # for g2
                    # timeperframe = exposuretime#for visiblitly
                    # timeperframe = 2  ## manual overwrite!!!! we apparently writing the wrong metadata....
                    center = md["center"]

                    setup_pargs = dict(
                        uid=uid,
                        dpix=dpix,
                        Ldet=Ldet,
                        lambda_=lambda_,
                        timeperframe=timeperframe,
                        center=center,
                        path=data_dir_,
                    )

                    md["avg_img"] = avg_img
                    # plot1D( y = imgsum[ np.array( [i for i in np.arange( len(imgsum)) if i not in bad_frame_list])],
                    #   title ='Uid= %s--imgsum'%uid, xlabel='Frame', ylabel='Total_Intensity', legend=''   )
                    min_inten = 10

                    # good_start = np.where( np.array(imgsum) > min_inten )[0][0]
                    good_start = good_start

                    if good_end is None:
                        good_end_ = len(imgs)
                    else:
                        good_end_ = good_end
                    FD = Multifile(filename, good_start, good_end_)

                    good_start = max(good_start, np.where(np.array(imgsum) > min_inten)[0][0])
                    print("With compression, the good_start frame number is: %s " % good_start)
                    print("The good_end frame number is: %s " % good_end_)

                    hmask = create_hot_pixel_mask(avg_img, 1e8)
                    qp, iq, q = get_circular_average(
                        avg_img,
                        mask * hmask,
                        pargs=setup_pargs,
                        nx=None,
                        plot_=False,
                        show_pixel=True,
                        xlim=[0.001, 0.05],
                        ylim=[0.0001, 500],
                    )

                    norm = get_pixelist_interp_iq(qp, iq, ring_mask, center)
                    if not para_run:
                        g2, lag_steps_ = cal_g2c(
                            FD, ring_mask, bad_frame_list, good_start, num_buf=8, imgsum=None, norm=norm
                        )
                    else:
                        g2, lag_steps_ = cal_g2p(
                            FD, ring_mask, bad_frame_list, good_start, num_buf=8, imgsum=None, norm=norm
                        )

                    if len(lag_steps) < len(lag_steps_):
                        lag_steps = lag_steps_

                    FD = 0
                    avg_img, imgsum, bad_frame_list = [0, 0, 0]
                    md["avg_img"] = 0
                    imgs = 0

                else:
                    sampling = 1000  # sampling should be one

                    # good_start = check_shutter_open( imgsra,  min_inten=5, time_edge = [0,10], plot_ = False )
                    good_start = good_start

                    good_series = apply_mask(imgsa[good_start:], mask)

                    imgsum, bad_frame_list = get_each_frame_intensity(
                        good_series, sampling=sampling, bad_pixel_threshold=1.2e8, plot_=False, uid=uid
                    )
                    bad_image_process = False

                    if len(bad_frame_list):
                        bad_image_process = True
                    print(bad_image_process)

                    g2, lag_steps_ = cal_g2(
                        good_series, ring_mask, bad_image_process, bad_frame_list, good_start, num_buf=8
                    )
                    if len(lag_steps) < len(lag_steps_):
                        lag_steps = lag_step_

                taus_ = lag_steps_ * timeperframe
                taus = lag_steps * timeperframe

                res_pargs = dict(taus=taus_, q_ring_center=q_ring_center, path=data_dir_, uid=uid)
                save_saxs_g2(g2, res_pargs)
                # plot_saxs_g2( g2, taus,  vlim=[0.95, 1.05], res_pargs=res_pargs)
                if fit:
                    fit_result = fit_saxs_g2(
                        g2,
                        res_pargs,
                        function="stretched",
                        vlim=[0.95, 1.05],
                        fit_variables={"baseline": True, "beta": True, "alpha": False, "relaxation_rate": True},
                        guess_values={"baseline": 1.0, "beta": 0.05, "alpha": 1.0, "relaxation_rate": 0.01},
                    )
                    fit_q_rate(
                        q_ring_center[:], fit_result["rate"][:], power_variable=False, uid=uid, path=data_dir_
                    )

                    psave_obj(fit_result, data_dir_ + "uid=%s-g2-fit-para" % uid)
                psave_obj(md, data_dir_ + "uid=%s-md" % uid)  # save the setup parameters

                g2s[run_seq + 1][i] = g2
                print("*" * 40)
                print()

    return g2s, taus, useful_uids


def plot_mul_g2(g2s, md):
    """
    Plot multi g2 functions generated by  multi_uids_saxs_xpcs_analysis
    Will create a large plot with q_number pannels
    Each pannel (for each q) will show a number (run number of g2 functions
    """

    q_ring_center = md["q_ring_center"]
    sids = md["sids"]
    useful_uids = md["useful_uids"]
    taus = md["taus"]
    run_num = md["run_num"]
    sub_num = md["sub_num"]
    uid_ = md["uid_"]

    fig = plt.figure(figsize=(12, 20))
    plt.title("uid= %s:--->" % uid_, fontsize=20, y=1.06)

    Nq = len(q_ring_center)
    if Nq != 1:
        plt.axis("off")
    sx = int(round(np.sqrt(Nq)))

    if Nq % sx == 0:
        sy = int(Nq / sx)
    else:
        sy = int(Nq / sx + 1)

    for sn in range(Nq):
        ax = fig.add_subplot(sx, sy, sn + 1)
        ax.set_ylabel(r"$g_2$" + "(" + r"$\tau$" + ")")
        ax.set_xlabel(r"$\tau $ $(s)$", fontsize=16)

        for run_seq in range(run_num):
            i = 0
            for sub_seq in range(0, sub_num):
                # print( run_seq, sub_seq )
                uid = useful_uids[run_seq + 1][sub_seq + 1]
                sid = sids[i]
                if i == 0:
                    title = r"$Q_r= $" + "%.5f  " % (q_ring_center[sn]) + r"$\AA^{-1}$"
                    ax.set_title(title, y=1.1, fontsize=12)
                y = g2s[run_seq + 1][sub_seq + 1][:, sn]
                len_tau = len(taus)
                len_g2 = len(y)
                len_ = min(len_tau, len_g2)

                # print ( len_tau, len(y))
                # ax.semilogx(taus[1:len_], y[1:len_], marker = '%s'%next(markers_), color='%s'%next(colors_),
                #            markersize=6, label = '%s'%sid)

                ax.semilogx(
                    taus[1:len_], y[1:len_], marker=markers[i], color=colors[i], markersize=6, label="%s" % sid
                )

                if sn == 0:
                    ax.legend(loc="best", fontsize=6)

                i = i + 1
    fig.set_tight_layout(True)


def get_QrQw_From_RoiMask(roi_mask, setup_pargs):
    """YG Dev Feb 4@CHX Get Q-center and Q-width fo transmission SAXS
    Input:
        roi_mask: int-type array, 2D roi mask, with q-index starting from 1
        setup_pargs: dict, at least with keys as
                     dpix (det pixel size),lamdba_( wavelength), center( beam center)
    Output:
        qr_cen: the q center of each ring
        qr_wid: the q width of each ring

    """
    qp_roi, iq_roi, q_roi = get_circular_average(roi_mask, np.array(roi_mask, dtype=bool), pargs=setup_pargs)
    Nmax = roi_mask.max()
    qr_cen = np.zeros(Nmax)
    qr_wid = np.zeros(Nmax)
    for i in range(1, 1 + Nmax):
        indi = np.where(iq_roi == i)[0]
        qind_s = q_roi[indi[0]]
        qind_e = q_roi[indi[-1]]
        # print(qind_s, qind_e)
        qr_cen[i - 1] = 0.5 * (qind_e + qind_s)
        qr_wid[i - 1] = qind_e - qind_s
    return qr_cen, qr_wid
