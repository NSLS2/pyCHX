"""
Dec 10, 2015 Developed by Y.G.@CHX
yuzhang@bnl.gov
This module is for the GiSAXS XPCS analysis
"""



from skbeam.core.accumulators.binned_statistic import BinnedStatistic1D, BinnedStatistic2D

from pyCHX.chx_compress import (
    Multifile,
    compress_eigerdata,
    get_avg_imgc,
    init_compress_eigerdata,
    read_compressed_eigerdata,
)
from pyCHX.chx_correlationc import cal_g2c
from pyCHX.chx_generic_functions import *
from pyCHX.chx_libs import colors, colors_, markers, markers_


def get_gisaxs_roi2(qr_edge, qz_edge, qr_map, qz_map, mask=None, qval_dict=None):
    """Y.G. 2019 Feb 12
    Get xpcs roi of gisaxs by giving Qr centers/edges, Qz centers/edges
    Parameters:
        qr_edge: list, e.g., [ [0.01,0.02],  [0.03,0.04] ].
                    each elment has two values for the start and end of one qr edge
        qz_edge: list, e.g., [ [0.01,0.02],  [0.03,0.04] ]
                   each elment has two values for the start and end of one qz edge
        qr_map: two-d array, the same shape as gisaxs frame, a qr map
        qz_map: two-d array, the same shape as gisaxs frame, a qz map
        mask: array, the scattering mask
        qval_dict: a dict, each key (a integer) with value as qr or (qr,qz) or (q//, q|-)
                    if not None, the new returned qval_dict will include the old one

    Return:
        roi_mask: array, the same shape as gisaxs frame, the label array of roi
        qval_dict, a dict, each key (a integer) with value as qr or (qr,qz) or (q//, q|-)
    """

    # qr_edge, qr_center = get_qedge( *Qr )
    # qz_edge, qz_center = get_qedge( *Qz )
    qr_edge, qz_edge = np.array(qr_edge), np.array(qz_edge)
    qr_center = 0.5 * (qr_edge[:, 0] + qr_edge[:, 1])
    qz_center = 0.5 * (qz_edge[:, 0] + qz_edge[:, 1])
    label_array_qz = get_qmap_label(qz_map, qz_edge)
    label_array_qr = get_qmap_label(qr_map, qr_edge)
    label_array_qzr, qzc, qrc = get_qzrmap(label_array_qz, label_array_qr, qz_center, qr_center)
    labels_qzr, indices_qzr = roi.extract_label_indices(label_array_qzr)
    labels_qz, indices_qz = roi.extract_label_indices(label_array_qz)
    labels_qr, indices_qr = roi.extract_label_indices(label_array_qr)
    if mask is None:
        mask = 1
    roi_mask = label_array_qzr * mask
    qval_dict = get_qval_dict(np.round(qr_center, 5), np.round(qz_center, 5), qval_dict=qval_dict)
    return roi_mask, qval_dict


def get_gisaxs_roi(Qr, Qz, qr_map, qz_map, mask=None, qval_dict=None):
    """Y.G. 2016 Dec 31
    Get xpcs roi of gisaxs
    Parameters:
        Qr: list, = [qr_start , qr_end, qr_width, qr_num], corresponding to qr start, qr end, qr width, qr number
        Qz: list, = [qz_start , qz_end, qz_width, qz_num], corresponding to qz start, qz end, qz width, qz number
        qr_map: two-d array, the same shape as gisaxs frame, a qr map
        qz_map: two-d array, the same shape as gisaxs frame, a qz map
        mask: array, the scattering mask
        qval_dict: a dict, each key (a integer) with value as qr or (qr,qz) or (q//, q|-)
                    if not None, the new returned qval_dict will include the old one

    Return:
        roi_mask: array, the same shape as gisaxs frame, the label array of roi
        qval_dict, a dict, each key (a integer) with value as qr or (qr,qz) or (q//, q|-)
    """

    qr_edge, qr_center = get_qedge(*Qr)
    qz_edge, qz_center = get_qedge(*Qz)
    label_array_qz = get_qmap_label(qz_map, qz_edge)
    label_array_qr = get_qmap_label(qr_map, qr_edge)
    label_array_qzr, qzc, qrc = get_qzrmap(label_array_qz, label_array_qr, qz_center, qr_center)
    labels_qzr, indices_qzr = roi.extract_label_indices(label_array_qzr)
    labels_qz, indices_qz = roi.extract_label_indices(label_array_qz)
    labels_qr, indices_qr = roi.extract_label_indices(label_array_qr)
    if mask is None:
        mask = 1
    roi_mask = label_array_qzr * mask
    qval_dict = get_qval_dict(np.round(qr_center, 5), np.round(qz_center, 5), qval_dict=qval_dict)
    return roi_mask, qval_dict


############
##developed at Octo 11, 2016
def get_qr(data, Qr, Qz, qr, qz, mask=None):
    """Octo 12, 2016, Y.G.@CHX
       plot one-d of I(q) as a function of qr for different qz

       data: a image/Eiger frame
       Qr: info for qr, = qr_start , qr_end, qr_width, qr_num
       Qz: info for qz, = qz_start,   qz_end,  qz_width , qz_num
       qr: qr-map
       qz: qz-map
       mask: a mask for qr-1d integration, default is None
       Return: qr_1d, a dataframe, with columns as qr1, qz1 (float value), qr2, qz2,....

    Examples:
        #to make two-qz, from 0.018 to 0.046, width as 0.008,
        qz_width = 0.008
        qz_start = 0.018 + qz_width/2
        qz_end = 0.046  -  qz_width/2
        qz_num= 2
        #to make one-qr, from 0.02 to 0.1, and the width is 0.1-0.012
        qr_width =  0.1-0.02
        qr_start =    0.02 + qr_width  /2
        qr_end =  0.01 -  qr_width  /2
        qr_num = 1
        Qr = [qr_start , qr_end, qr_width, qr_num]
        Qz=  [qz_start,   qz_end,  qz_width , qz_num ]
        new_mask[ :, 1020:1045] =0
        ticks = show_qzr_map(  qr,qz, inc_x0, data = avg_imgmr, Nzline=10,  Nrline=10   )
        qx, qy, qr, qz = convert_gisaxs_pixel_to_q( inc_x0, inc_y0,refl_x0,refl_y0, lamda=lamda, Lsd=Lsd )
        qr_1d = get_qr( avg_imgr, Qr, Qz, qr, qz, new_mask)

    """

    qr_start, qr_end, qr_width, qr_num = Qr
    qz_start, qz_end, qz_width, qz_num = Qz
    qr_edge, qr_center = get_qedge(qr_start, qr_end, qr_width, qr_num)
    qz_edge, qz_center = get_qedge(qz_start, qz_end, qz_width, qz_num)
    label_array_qr = get_qmap_label(qr, qr_edge)
    # qr_1d ={}
    # columns=[]

    for i, qzc_ in enumerate(qz_center):
        # print (i,qzc_)
        label_array_qz = get_qmap_label(qz, qz_edge[i * 2 : 2 * i + 2])
        # print (qzc_, qz_edge[i*2:2*i+2])
        label_array_qzr, qzc, qrc = get_qzrmap(label_array_qz, label_array_qr, qz_center, qr_center)
        # print (np.unique(label_array_qzr ))
        if mask is not None:
            label_array_qzr *= mask
        roi_pixel_num = np.sum(label_array_qzr, axis=0)
        qr_ = qr * label_array_qzr
        data_ = data * label_array_qzr
        qr_ave = np.sum(qr_, axis=0) / roi_pixel_num
        data_ave = np.sum(data_, axis=0) / roi_pixel_num
        qr_ave, data_ave = zip(*sorted(zip(*[qr_ave[~np.isnan(qr_ave)], data_ave[~np.isnan(data_ave)]])))
        if i == 0:
            N_interp = len(qr_ave)

        qr_ave_intp = np.linspace(np.min(qr_ave), np.max(qr_ave), N_interp)
        data_ave = np.interp(qr_ave_intp, qr_ave, data_ave)
        # columns.append( ['qr%s'%i, str(round(qzc_,4))] )
        if i == 0:
            df = np.hstack([(qr_ave_intp).reshape(N_interp, 1), data_ave.reshape(N_interp, 1)])
        else:
            df = np.hstack([df, (qr_ave_intp).reshape(N_interp, 1), data_ave.reshape(N_interp, 1)])
    # df = DataFrame( df  )
    # df.columns = np.concatenate( columns    )

    return df


########################
# get one-d of I(q) as a function of qr for different qz
#####################


def cal_1d_qr(
    data,
    Qr,
    Qz,
    qr,
    qz,
    inc_x0=None,
    mask=None,
    path=None,
    uid=None,
    setup_pargs=None,
    save=True,
    print_save_message=True,
):
    """Revised at July 18, 2017 by YG, to correct a divide by zero bug
        Dec 16, 2016, Y.G.@CHX
       calculate one-d of I(q) as a function of qr for different qz
       data: a dataframe
       Qr: info for qr, = qr_start , qr_end, qr_width, qr_num, the purpose of Qr is only for the defination of qr range (qr number does not matter)
       Qz: info for qz, = qz_start,   qz_end,  qz_width , qz_num
       qr: qr-map
       qz: qz-map
       inc_x0: x-center of incident beam
       mask: a mask for qr-1d integration
       setup_pargs: gives path, filename...

       Return: qr_1d, a dataframe, with columns as qr1, qz1 (float value), qz2,....
               Plot 1D cureve as a function of Qr for each Qz

    Examples:
        #to make two-qz, from 0.018 to 0.046, width as 0.008,
        qz_width = 0.008
        qz_start = 0.018 + qz_width/2
        qz_end = 0.046  -  qz_width/2
        qz_num= 2


        #to make one-qr, from 0.02 to 0.1, and the width is 0.1-0.012
        qr_width =  0.1-0.02
        qr_start =    0.02 + qr_width  /2
        qr_end =  0.01 -  qr_width  /2
        qr_num = 1

        Qr = [qr_start , qr_end, qr_width, qr_num]
        Qz=  [qz_start,   qz_end,  qz_width , qz_num ]
        new_mask[ :, 1020:1045] =0
        qx, qy, qr, qz = convert_gisaxs_pixel_to_q( inc_x0, inc_y0,refl_x0,refl_y0, lamda=lamda, Lsd=Lsd )

        qr_1d = get_1d_qr( avg_imgr, Qr, Qz, qr, qz, inc_x0,  new_mask)

    A plot example:
        plot1D( x= qr_1d['qr1'], y = qr_1d['0.0367'], logxy=True )
    """
    qr_start, qr_end, qr_width, qr_num = Qr
    qz_start, qz_end, qz_width, qz_num = Qz
    qr_edge, qr_center = get_qedge(qr_start, qr_end, qr_width, qr_num, verbose=False)
    qz_edge, qz_center = get_qedge(qz_start, qz_end, qz_width, qz_num, verbose=False)

    # print ('The qr_edge is:  %s\nThe qr_center is:  %s'%(qr_edge, qr_center))
    # print ('The qz_edge is:  %s\nThe qz_center is:  %s'%(qz_edge, qz_center))

    label_array_qr = get_qmap_label(qr, qr_edge)
    # qr_1d ={}
    columns = []
    for i, qzc_ in enumerate(qz_center):
        # print (i,qzc_)
        label_array_qz = get_qmap_label(qz, qz_edge[i * 2 : 2 * i + 2])
        # print (qzc_, qz_edge[i*2:2*i+2])
        label_array_qzr, qzc, qrc = get_qzrmap(label_array_qz, label_array_qr, qz_center, qr_center)
        # print (np.unique(label_array_qzr ))
        if mask is not None:
            label_array_qzr *= mask
        roi_pixel_num = np.sum(label_array_qzr, axis=0)
        # print( label_array_qzr )
        qr_ = qr * label_array_qzr
        data_ = data * label_array_qzr

        w = np.where(roi_pixel_num)
        qr_ave = np.zeros_like(roi_pixel_num, dtype=float)[w]
        data_ave = np.zeros_like(roi_pixel_num, dtype=float)[w]

        qr_ave = (np.sum(qr_, axis=0))[w] / roi_pixel_num[w]
        data_ave = (np.sum(data_, axis=0))[w] / roi_pixel_num[w]
        qr_ave, data_ave = zip(*sorted(zip(*[qr_ave[~np.isnan(qr_ave)], data_ave[~np.isnan(data_ave)]])))
        if i == 0:
            N_interp = len(qr_ave)
            columns.append(["qr"])
            # qr_1d[i]= qr_ave_intp
        qr_ave_intp = np.linspace(np.min(qr_ave), np.max(qr_ave), N_interp)
        data_ave = np.interp(qr_ave_intp, qr_ave, data_ave)
        # qr_1d[i]= [qr_ave_intp, data_ave]
        columns.append(["qz%s=%s" % (i, str(round(qzc_, 4)))])
        if i == 0:
            df = np.hstack([(qr_ave_intp).reshape(N_interp, 1), data_ave.reshape(N_interp, 1)])
        else:
            df = np.hstack([df, data_ave.reshape(N_interp, 1)])
    df = DataFrame(df)
    df.columns = np.concatenate(columns)

    if save:
        if path is None:
            path = setup_pargs["path"]
        if uid is None:
            uid = setup_pargs["uid"]
        filename = os.path.join(path, "%s_qr_1d.csv" % (uid))
        df.to_csv(filename)
        if print_save_message:
            print("The qr_1d is saved in %s with filename as %s_qr_1d.csv" % (path, uid))
    return df


def get_t_qrc(FD, frame_edge, Qr, Qz, qr, qz, mask=None, path=None, uid=None, save=True, *argv, **kwargs):
    """Get t-dependent qr

    Parameters
    ----------
    FD: a compressed imgs series handler
    frame_edge: list, the ROI frame regions, e.g., [  [0,100], [200,400] ]
    mask:  a image mask

    Returns
    ---------
    qrt_pds: dataframe, with columns as [qr, qz0_fra_from_beg1_to_end1,  qz0_fra_from_beg2_to_end2, ...
                                             qz1_fra_from_beg1_to_end1,  qz1_fra_from_beg2_to_end2, ...
                                        ...
                                        ]

    """

    Nt = len(frame_edge)
    iqs = list(np.zeros(Nt))
    qz_start, qz_end, qz_width, qz_num = Qz
    qz_edge, qz_center = get_qedge(qz_start, qz_end, qz_width, qz_num, verbose=False)
    # print('here')
    # qr_1d = np.zeros(   )

    if uid is None:
        uid = "uid"
    for i in range(Nt):
        # str(round(qz_center[j], 4 )
        t1, t2 = frame_edge[i]
        avg_imgx = get_avg_imgc(FD, beg=t1, end=t2, sampling=1, plot_=False)

        qrti = cal_1d_qr(avg_imgx, Qr, Qz, qr, qz, mask=mask, save=False)
        if i == 0:
            qrt_pds = np.zeros([len(qrti), 1 + Nt * qz_num])
            columns = np.zeros(1 + Nt * qz_num, dtype=object)
            columns[0] = "qr"
            qrt_pds[:, 0] = qrti["qr"]
        for j in range(qz_num):
            coli = qrti.columns[1 + j]
            qrt_pds[:, 1 + i + Nt * j] = qrti[coli]
            columns[1 + i + Nt * j] = coli + "_fra_%s_to_%s" % (t1, t2)

    qrt_pds = DataFrame(qrt_pds)
    qrt_pds.columns = columns
    if save:
        if path is None:
            path = setup_pargs["path"]
        if uid is None:
            uid = setup_pargs["uid"]
        filename = os.path.join(path, "%s_qrt_pds.csv" % (uid))
        qrt_pds.to_csv(filename)
        print("The qr~time is saved in %s with filename as %s_qrt_pds.csv" % (path, uid))
    return qrt_pds


def plot_qrt_pds(qrt_pds, frame_edge, qz_index=0, uid="uid", path="", fontsize=8, *argv, **kwargs):
    """Y.G. Jan 04, 2017
    plot t-dependent qr

        Parameters
        ----------
        qrt_pds: dataframe, with columns as [qr, qz0_fra_from_beg1_to_end1,  qz0_fra_from_beg2_to_end2, ...
                                                 qz1_fra_from_beg1_to_end1,  qz1_fra_from_beg2_to_end2, ...
                                            ...
                                            ]
        frame_edge: list, the ROI frame regions, e.g., [  [0,100], [200,400] ]

        qz_index, if = integer, e.g. =0, only plot the qr~t for qz0
                  if None, plot all qzs

    Returns

    """

    fig, ax = plt.subplots(figsize=(8, 6))
    cols = np.array(qrt_pds.columns)
    Nt = len(frame_edge)
    # num_qz = int(  (len( cols ) -1  ) /Nt )
    qr = qrt_pds["qr"]
    if qz_index is None:
        r = range(1, len(cols))
    else:
        r = range(1 + qz_index * Nt, 1 + (1 + qz_index) * Nt)
    for i in r:
        y = qrt_pds[cols[i]]
        ax.semilogy(qr, y, label=cols[i], marker=markers[i], color=colors[i], ls="-")
        # ax.set_xlabel("q in pixel")
        ax.set_xlabel(r"$Q_r$" + r"($\AA^{-1}$)")
        ax.set_ylabel("I(q)")

    if "xlim" in kwargs.keys():
        ax.set_xlim(kwargs["xlim"])
    if "ylim" in kwargs.keys():
        ax.set_ylim(kwargs["ylim"])

    ax.legend(loc="best", fontsize=fontsize)

    title = ax.set_title("%s_Iq_t" % uid)
    title.set_y(1.01)

    fp = path + "%s_Iq_t" % uid + ".png"
    fig.savefig(fp, dpi=fig.dpi)


def plot_t_qrc(qr_1d, frame_edge, save=False, pargs=None, fontsize=8, *argv, **kwargs):
    """plot t-dependent qr

        Parameters
        ----------
        qr_1d: array, with shape as time length, frame_edge
        frame_edge: list, the ROI frame regions, e.g., [  [0,100], [200,400] ]
        save: save the plot
        if save,  all the following paramters are given in argv
            {
            'path':
             'uid':  }

    Returns

    """

    fig, ax = plt.subplots(figsize=(8, 6))
    Nt = qr_1d.shape[1]
    q = qr_1d[:, 0]
    for i in range(Nt - 1):
        t1, t2 = frame_edge[i]
        ax.semilogy(q, qr_1d[:, i + 1], "o-", label="frame: %s--%s" % (t1, t2))
        # ax.set_xlabel("q in pixel")
        ax.set_xlabel(r"$Q_r$" + r"($\AA^{-1}$)")
        ax.set_ylabel("I(q)")

    if "xlim" in kwargs.keys():
        ax.set_xlim(kwargs["xlim"])
    if "ylim" in kwargs.keys():
        ax.set_ylim(kwargs["ylim"])

    ax.legend(loc="best", fontsize=fontsize)
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
            filename="uid=%s-q-Iqt" % uid,
            path=path,
        )


##########################################
###Functions for GiSAXS
##########################################


def make_gisaxs_grid(qr_w=10, qz_w=12, dim_r=100, dim_z=120):
    """Dec 16, 2015, Y.G.@CHX"""
    y, x = np.indices([dim_z, dim_r])
    Nr = int(dim_r / qp_w)
    Nz = int(dim_z / qz_w)
    noqs = Nr * Nz

    ind = 1
    for i in range(0, Nr):
        for j in range(0, Nz):
            y[qr_w * i : qr_w * (i + 1), qz_w * j : qz_w * (j + 1)] = ind
            ind += 1
    return y


###########################################
# for Q-map, convert pixel to Q
###########################################


def convert_Qmap(img, qx_map, qy_map=None, bins=None, rangeq=None, mask=None, statistic="sum"):
    """Y.G. Nov 3@CHX
    Convert a scattering image to a qmap by giving qx_map and qy_map
    Return converted qmap, x-coordinates and y-coordinates
    """
    if qy_map is not None:
        if rangeq is None:
            qx_min, qx_max = qx_map.min(), qx_map.max()
            qy_min, qy_max = qy_map.min(), qy_map.max()
            rangeq = [[qx_min, qx_max], [qy_min, qy_max]]
        if bins is None:
            bins = qx_map.shape
        if mask is not None:
            m = mask.ravel()
        else:
            m = None
        b2d = BinnedStatistic2D(
            qx_map.ravel(), qy_map.ravel(), statistic=statistic, bins=bins, mask=m, range=rangeq
        )
        remesh_data, xbins, ybins = b2d(img.ravel()), b2d.bin_centers[0], b2d.bin_centers[1]
    else:
        if rangeq is None:
            qx_min, qx_max = qx_map.min(), qx_map.max()
            rangeq = [qx_min, qx_max]
        if bins is None:
            bins = [qx_map.size]
        if mask is not None:
            m = mask.ravel()
        else:
            m = None
        b1d = BinnedStatistic1D(qx_map.ravel(), bins=bins, mask=m)
        remesh_data = b1d(img.ravel())
        xbins = b1d.bin_centers
        ybins = None
    return remesh_data, xbins, ybins


def get_refl_xy(inc_ang, inc_phi, inc_x0, inc_y0, pixelsize=[0.075, 0.075], Lsd=5000):
    """
    Input:
        inc_angle: deg,
        inc_phi: deg, by default, 0 ( if inc_x = ref_x )
        pixelsize: 0.075 mm for Eiger4M detector
        sample to detector distance: Lsd, in mm

    Output:
         reflected beam center x, y

    """
    px, py = pixelsize
    refl_y0 = np.tan(2 * np.radians(inc_ang)) * Lsd / (py) + inc_y0
    refl_x0 = inc_x0 - np.tan(np.radians(inc_phi)) * (refl_y0 - inc_y0) * py / px
    print("The reflection beam center is: [%.2f, %.2f] (pix)" % (refl_x0, refl_y0))
    return refl_x0, refl_y0


def get_alphaf_thetaf(
    inc_x0, inc_y0, inc_ang, inc_phi=0, pixelsize=[0.075, 0.075], Lsd=5000, dimx=2070.0, dimy=2167.0
):
    """Nov 19, 2018@SMI to get alphaf and thetaf for gi scattering
    Input:
        inc_angle: deg,
        inc_phi: deg, by default, 0 ( if inc_x = ref_x )
        pixelsize: 0.075 mm for Eiger4M detector
        sample to detector distance: Lsd, in mm
        detector image size: dimx = 2070,dimy=2167 for Eiger4M detector
    Output:
         reflected angle alphaf (outplane)
         reflected angle thetaf (inplane )

    """
    px, py = pixelsize
    y, x = np.indices([int(dimy), int(dimx)])
    alphai, thetai = np.radians(inc_ang), np.radians(inc_phi)
    alphaf = np.arctan2((y - inc_y0) * py, Lsd) - alphai
    thetaf = np.arctan2((x - inc_x0) * px, Lsd) / 2 - thetai
    # print( px, py, Lsd, dimy, dimx, alphai, thetai)
    return alphaf, thetaf


def convert_gisaxs_pixel_to_q2(
    inc_ang,
    alphaf,
    thetaf,
    phi=0,
    lamda=1.0,
    thetai=0.0,
):
    """
    Dec 16, 2015, Y.G.@CHX
    giving:
              incident_angle, (inc_ang), in deg
              alphaf,
              thetaf,
              the title angle (phi)
              wavelength: angstron

        get: q_parallel (qp), q_direction_z (qz)

    """
    pref = 2 * np.pi / lamda
    alphai = np.radians(inc_ang)
    thetai = np.radians(thetai)
    phi = np.radians(phi)

    qx = np.cos(alphaf) * np.cos(2 * thetaf) - np.cos(alphai) * np.cos(2 * thetai)
    qy_ = np.cos(alphaf) * np.sin(2 * thetaf) - np.cos(alphai) * np.sin(2 * thetai)
    qz_ = np.sin(alphaf) + np.sin(alphai)
    qy = qz_ * np.sin(phi) + qy_ * np.cos(phi)
    qz = qz_ * np.cos(phi) - qy_ * np.sin(phi)
    qr = np.sqrt(qx**2 + qy**2)
    return qx * pref, qy * pref, qr * pref, qz * pref


def get_incident_angles(inc_x0, inc_y0, refl_x0, refl_y0, pixelsize=[75, 75], Lsd=5.0):
    """
    Dec 16, 2015, Y.G.@CHX
    giving: incident beam center: bcenx,bceny
            reflected beam on detector: rcenx, rceny
            sample to detector distance: Lsd, in meters
            pixelsize: 75 um for Eiger4M detector
    get incident_angle (alphai), the title angle (phi)
    """
    if Lsd >= 1000:
        Lsd = Lsd / 1000.0

    px, py = pixelsize
    phi = np.arctan2((-refl_x0 + inc_x0) * px * 10 ** (-6), (refl_y0 - inc_y0) * py * 10 ** (-6))
    alphai = np.arctan2((refl_y0 - inc_y0) * py * 10 ** (-6), Lsd) / 2.0
    # thetai = np.arctan2(  (rcenx - bcenx)*px *10**(-6), Lsd   ) /2.  #??

    return alphai, phi


def get_reflected_angles(
    inc_x0, inc_y0, refl_x0, refl_y0, thetai=0.0, pixelsize=[75, 75], Lsd=5.0, dimx=2070.0, dimy=2167.0
):
    """Dec 16, 2015, Y.G.@CHX
    giving: incident beam center: bcenx,bceny
            reflected beam on detector: rcenx, rceny
            sample to detector distance: Lsd, in mm
            pixelsize: 75 um for Eiger4M detector
            detector image size: dimx = 2070,dimy=2167 for Eiger4M detector
    get  reflected angle alphaf (outplane)
         reflected angle thetaf (inplane )
    """
    # if Lsd>=1000:#it should be something wrong and the unit should be meter
    # convert Lsd from mm to m
    if Lsd >= 1000:
        Lsd = Lsd / 1000.0
    alphai, phi = get_incident_angles(inc_x0, inc_y0, refl_x0, refl_y0, pixelsize, Lsd)
    print("The incident_angle (alphai) is: %s" % (alphai * 180 / np.pi))
    px, py = pixelsize
    y, x = np.indices([int(dimy), int(dimx)])
    # alphaf = np.arctan2( (y-inc_y0)*py*10**(-6), Lsd )/2 - alphai
    alphaf = np.arctan2((y - inc_y0) * py * 10 ** (-6), Lsd) - alphai
    thetaf = np.arctan2((x - inc_x0) * px * 10 ** (-6), Lsd) / 2 - thetai
    return alphaf, thetaf, alphai, phi


def convert_gisaxs_pixel_to_q(
    inc_x0, inc_y0, refl_x0, refl_y0, pixelsize=[75, 75], Lsd=5.0, dimx=2070.0, dimy=2167.0, thetai=0.0, lamda=1.0
):
    """
    Dec 16, 2015, Y.G.@CHX
    giving: incident beam center: bcenx,bceny
                reflected beam on detector: rcenx, rceny
                sample to detector distance: Lsd, in meters
                pixelsize: 75 um for Eiger4M detector
                detector image size: dimx = 2070,dimy=2167 for Eiger4M detector
                wavelength: angstron

        get: q_parallel (qp), q_direction_z (qz)

    """
    alphaf, thetaf, alphai, phi = get_reflected_angles(
        inc_x0, inc_y0, refl_x0, refl_y0, thetai, pixelsize, Lsd, dimx, dimy
    )
    pref = 2 * np.pi / lamda
    qx = np.cos(alphaf) * np.cos(2 * thetaf) - np.cos(alphai) * np.cos(2 * thetai)
    qy_ = np.cos(alphaf) * np.sin(2 * thetaf) - np.cos(alphai) * np.sin(2 * thetai)
    qz_ = np.sin(alphaf) + np.sin(alphai)
    qy = qz_ * np.sin(phi) + qy_ * np.cos(phi)
    qz = qz_ * np.cos(phi) - qy_ * np.sin(phi)
    qr = np.sqrt(qx**2 + qy**2)
    return qx * pref, qy * pref, qr * pref, qz * pref


def get_qedge(qstart, qend, qwidth, noqs, verbose=True):
    """July 18, 2017 Revised by Y.G.@CHX,
    Add print info for noqs=1
    Dec 16, 2015, Y.G.@CHX
    DOCUMENT get_qedge( )
    give qstart,qend,qwidth,noqs
    return a qedge by giving the noqs, qstart,qend,qwidth.
           a qcenter, which is center of each qedge
    KEYWORD:  None"""
    import numpy as np

    if noqs != 1:
        spacing = (qend - qstart - noqs * qwidth) / (noqs - 1)  # spacing between rings
        qedges = (roi.ring_edges(qstart, qwidth, spacing, noqs)).ravel()
        qcenter = (qedges[::2] + qedges[1::2]) / 2
    else:
        spacing = 0
        qedges = (roi.ring_edges(qstart, qwidth, spacing, noqs)).ravel()
        # qedges =  np.array( [qstart, qend] )
        qcenter = [(qedges[1] + qedges[0]) / 2]
        if verbose:
            print("Since noqs=1, the qend is actually defined by qstart + qwidth.")
    return qedges, qcenter


def get_qedge2(
    qstart,
    qend,
    qwidth,
    noqs,
):
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
    return qedge, qcenter


###########################################
# for plot Q-map
###########################################


def get_qmap_label(qmap, qedge):
    import numpy as np

    """
    April 20, 2016, Y.G.@CHX
    give a qmap and qedge to bin the qmap into a label array
    """
    edges = np.atleast_2d(np.asarray(qedge)).ravel()
    label_array = np.digitize(qmap.ravel(), edges, right=False)
    label_array = np.int_(label_array)
    label_array = (np.where(label_array % 2 != 0, label_array, 0) + 1) // 2
    label_array = label_array.reshape(qmap.shape)
    return label_array


def get_qzrmap(label_array_qz, label_array_qr, qz_center, qr_center):
    """April 20, 2016, Y.G.@CHX, get   qzrmap"""
    qzmax = label_array_qz.max()
    label_array_qr_ = np.zeros(label_array_qr.shape)
    ind = np.where(label_array_qr != 0)
    label_array_qr_[ind] = label_array_qr[ind] + 1e4  # add some large number to qr
    label_array_qzr = label_array_qz * label_array_qr_

    # convert label_array_qzr to [1,2,3,...]
    uqzr = np.unique(label_array_qzr)[1:]

    uqz = np.unique(label_array_qz)[1:]
    uqr = np.unique(label_array_qr)[1:]
    # print (uqzr)
    label_array_qzr_ = np.zeros_like(label_array_qzr)
    newl = np.arange(1, len(uqzr) + 1)

    qzc = list(qz_center) * len(uqr)
    qrc = [[qr_center[i]] * len(uqz) for i in range(len(uqr))]

    for i, label in enumerate(uqzr):
        # print (i, label)
        label_array_qzr_.ravel()[np.where(label_array_qzr.ravel() == label)[0]] = newl[i]

    return np.int_(label_array_qzr_), np.array(qzc), np.concatenate(np.array(qrc))


def show_label_array_on_image(
    ax, image, label_array, cmap=None, norm=None, log_img=True, alpha=0.3, imshow_cmap="gray", **kwargs
):  # norm=LogNorm(),
    """
    This will plot the required ROI's(labeled array) on the image
    Additional kwargs are passed through to `ax.imshow`.
    If `vmin` is in kwargs, it is clipped to minimum of 0.5.
    Parameters
    ----------
    ax : Axes
        The `Axes` object to add the artist too
    image : array
        The image array
    label_array : array
        Expected to be an unsigned integer array.  0 is background,
        positive integers label region of interest
    cmap : str or colormap, optional
        Color map to use for plotting the label_array, defaults to 'None'
    imshow_cmap : str or colormap, optional
        Color map to use for plotting the image, defaults to 'gray'
    norm : str, optional
        Normalize scale data, defaults to 'Lognorm()'
    Returns
    -------
    im : AxesImage
        The artist added to the axes
    im_label : AxesImage
        The artist added to the axes
    """
    ax.set_aspect("equal")
    if log_img:
        norm=LogNorm(vmin=vmin, vmax=vmax)
        im = ax.imshow(image, cmap=imshow_cmap, interpolation="none", norm=norm)  # norm=norm,
    else:
        im = ax.imshow(image, cmap=imshow_cmap, interpolation="none", norm=norm, **kwargs)  # norm=norm,

    im_label = mpl_plot.show_label_array(
        ax, label_array, cmap=cmap, norm=norm, alpha=alpha, **kwargs
    )  # norm=norm,

    return im, im_label


def show_qz(qz):
    """Dec 16, 2015, Y.G.@CHX
    plot qz mape

    """

    fig, ax = plt.subplots()
    im = ax.imshow(qz, origin="lower", cmap="viridis", vmin=qz.min(), vmax=qz.max())
    fig.colorbar(im)
    ax.set_title("Q-z")
    # plt.show()


def show_qr(qr):
    """Dec 16, 2015, Y.G.@CHX
    plot qr mape

    """
    fig, ax = plt.subplots()
    im = ax.imshow(qr, origin="lower", cmap="viridis", vmin=qr.min(), vmax=qr.max())
    fig.colorbar(im)
    ax.set_title("Q-r")
    # plt.show()


def show_alphaf(
    alphaf,
):
    """Dec 16, 2015, Y.G.@CHX
    plot alphaf mape

    """

    fig, ax = plt.subplots()
    im = ax.imshow(alphaf * 180 / np.pi, origin="lower", cmap="viridis", vmin=-1, vmax=1.5)
    # im=ax.imshow(alphaf, origin='lower' ,cmap='viridis',norm= LogNorm(vmin=0.0001,vmax=2.00))
    fig.colorbar(im)
    ax.set_title("alphaf")
    # plt.show()


def get_1d_qr(
    data,
    Qr,
    Qz,
    qr,
    qz,
    inc_x0,
    mask=None,
    show_roi=True,
    ticks=None,
    alpha=0.3,
    loglog=False,
    save=True,
    setup_pargs=None,
):
    """Dec 16, 2015, Y.G.@CHX
       plot one-d of I(q) as a function of qr for different qz
       data: a dataframe
       Qr: info for qr, = qr_start , qr_end, qr_width, qr_num
       Qz: info for qz, = qz_start,   qz_end,  qz_width , qz_num
       qr: qr-map
       qz: qz-map
       inc_x0: x-center of incident beam
       mask: a mask for qr-1d integration
       show_roi: boolean, if ture, show the interest ROI
       ticks: ticks for the plot, = zticks, zticks_label, rticks, rticks_label
       alpha: transparency of ROI
       loglog: if True, plot in log-log scale
       setup_pargs: gives path, filename...

       Return: qr_1d, a dataframe, with columns as qr1, qz1 (float value), qr2, qz2,....
               Plot 1D cureve as a function of Qr for each Qz




    Examples:
        #to make two-qz, from 0.018 to 0.046, width as 0.008,
        qz_width = 0.008
        qz_start = 0.018 + qz_width/2
        qz_end = 0.046  -  qz_width/2
        qz_num= 2


        #to make one-qr, from 0.02 to 0.1, and the width is 0.1-0.012
        qr_width =  0.1-0.02
        qr_start =    0.02 + qr_width  /2
        qr_end =  0.01 -  qr_width  /2
        qr_num = 1

        Qr = [qr_start , qr_end, qr_width, qr_num]
        Qz=  [qz_start,   qz_end,  qz_width , qz_num ]
        new_mask[ :, 1020:1045] =0
        ticks = show_qzr_map(  qr,qz, inc_x0, data = avg_imgmr, Nzline=10,  Nrline=10   )
        qx, qy, qr, qz = convert_gisaxs_pixel_to_q( inc_x0, inc_y0,refl_x0,refl_y0, lamda=lamda, Lsd=Lsd )

        qr_1d = get_1d_qr( avg_imgr, Qr, Qz, qr, qz, inc_x0,  new_mask,  True, ticks, .8)

    A plot example:
        plot1D( x= qr_1d['qr1'], y = qr_1d['0.0367'], logxy=True )


    """

    qr_start, qr_end, qr_width, qr_num = Qr
    qz_start, qz_end, qz_width, qz_num = Qz
    qr_edge, qr_center = get_qedge(qr_start, qr_end, qr_width, qr_num)
    qz_edge, qz_center = get_qedge(qz_start, qz_end, qz_width, qz_num)

    print("The qr_edge is:  %s\nThe qr_center is:  %s" % (qr_edge, qr_center))
    print("The qz_edge is:  %s\nThe qz_center is:  %s" % (qz_edge, qz_center))
    label_array_qr = get_qmap_label(qr, qr_edge)

    if show_roi:
        label_array_qz0 = get_qmap_label(qz, qz_edge)
        label_array_qzr0, qzc0, qrc0 = get_qzrmap(label_array_qz0, label_array_qr, qz_center, qr_center)

        if mask is not None:
            label_array_qzr0 *= mask
        # data_ = data*label_array_qzr0
        show_qzr_roi(data, label_array_qzr0, inc_x0, ticks, alpha)

    fig, ax = plt.subplots()
    qr_1d = {}
    columns = []
    for i, qzc_ in enumerate(qz_center):
        # print (i,qzc_)
        label_array_qz = get_qmap_label(qz, qz_edge[i * 2 : 2 * i + 2])
        # print (qzc_, qz_edge[i*2:2*i+2])
        label_array_qzr, qzc, qrc = get_qzrmap(label_array_qz, label_array_qr, qz_center, qr_center)
        # print (np.unique(label_array_qzr ))
        if mask is not None:
            label_array_qzr *= mask
        roi_pixel_num = np.sum(label_array_qzr, axis=0)
        qr_ = qr * label_array_qzr
        data_ = data * label_array_qzr
        qr_ave = np.sum(qr_, axis=0) / roi_pixel_num
        data_ave = np.sum(data_, axis=0) / roi_pixel_num

        qr_ave, data_ave = zip(*sorted(zip(*[qr_ave[~np.isnan(qr_ave)], data_ave[~np.isnan(data_ave)]])))

        if i == 0:
            N_interp = len(qr_ave)

        qr_ave_intp = np.linspace(np.min(qr_ave), np.max(qr_ave), N_interp)
        data_ave = np.interp(qr_ave_intp, qr_ave, data_ave)

        qr_1d[i] = [qr_ave_intp, data_ave]
        columns.append(["qr%s" % i, str(round(qzc_, 4))])
        if loglog:
            ax.loglog(qr_ave_intp, data_ave, "--o", label="qz= %f" % qzc_, markersize=1)
        else:
            ax.plot(qr_ave_intp, data_ave, "--o", label="qz= %f" % qzc_)
        if i == 0:
            df = np.hstack([(qr_ave_intp).reshape(N_interp, 1), data_ave.reshape(N_interp, 1)])
        else:
            df = np.hstack([df, (qr_ave_intp).reshape(N_interp, 1), data_ave.reshape(N_interp, 1)])

    # ax.set_xlabel( r'$q_r$', fontsize=15)
    ax.set_xlabel(r"$q_r$" r"($\AA^{-1}$)", fontsize=18)
    ax.set_ylabel("$Intensity (a.u.)$", fontsize=18)
    ax.set_yscale("log")
    # ax.set_xscale('log')
    ax.set_xlim(qr.max(), qr.min())
    ax.legend(loc="best")

    df = DataFrame(df)
    df.columns = np.concatenate(columns)

    if save:
        # dt =datetime.now()
        # CurTime = '%s%02d%02d-%02d%02d-' % (dt.year, dt.month, dt.day,dt.hour,dt.minute)
        path = setup_pargs["path"]
        uid = setup_pargs["uid"]
        # filename = os.path.join(path, 'qr_1d-%s-%s.csv' % (uid,CurTime))
        filename = os.path.join(path, "uid=%s--qr_1d.csv" % (uid))
        df.to_csv(filename)
        print("The qr_1d is saved in %s with filename as uid=%s--qr_1d.csv" % (path, uid))

        # fp = path + 'Uid= %s--Circular Average'%uid + CurTime + '.png'
        fp = path + "uid=%s--qr_1d-" % uid + ".png"
        fig.savefig(fp, dpi=fig.dpi)
    # plt.show()
    return df


def plot_qr_1d_with_ROI(qr_1d, qr_center, loglog=False, save=True, uid="uid", path=""):
    """Dec 16, 2015, Y.G.@CHX
       plot one-d of I(q) as a function of qr with ROI
       qr_1d: a dataframe for qr_1d
       qr_center: the center of qr
       loglog: if True, plot in log-log scale
       Return:
               Plot 1D cureve with ROI
    A plot example:
        plot_1d_qr_with_ROI( df, qr_center,  loglog=False, save=True )

    """

    fig, ax = plt.subplots()
    Ncol = len(qr_1d.columns)
    Nqr = Ncol % 2
    qz_center = qr_1d.columns[1::1]  # qr_1d.columns[1::2]
    Nqz = len(qz_center)
    for i, qzc_ in enumerate(qz_center):
        x = qr_1d[qr_1d.columns[0]]
        y = qr_1d[qzc_]
        if loglog:
            ax.loglog(x, y, "--o", label="qz= %s" % qzc_, markersize=1)
        else:
            ax.plot(x, y, "--o", label="qz= %s" % qzc_)
    for qrc in qr_center:
        ax.axvline(qrc)  # , linewidth = 5  )

    # ax.set_xlabel( r'$q_r$', fontsize=15)
    ax.set_xlabel(r"$q_r$" r"($\AA^{-1}$)", fontsize=18)
    ax.set_ylabel("$Intensity (a.u.)$", fontsize=18)
    ax.set_yscale("log")
    # ax.set_xscale('log')
    ax.set_xlim(x.max(), x.min())
    ax.legend(loc="best")
    ax.set_title("%s_Qr_ROI" % uid)
    if save:
        fp = path + "%s_Qr_ROI" % uid + ".png"
        fig.savefig(fp, dpi=fig.dpi)
    # plt.show()


def interp_zeros(data):
    from scipy.interpolate import interp1d

    gf = data.ravel()
    (indice,) = gf.nonzero()
    start, stop = indice[0], indice[-1] + 1
    dx, dy = data.shape
    x = np.arange(dx * dy)
    f = interp1d(x[indice], gf[indice])
    gf[start:stop] = f(x[start:stop])
    return gf.reshape([dx, dy])


def get_qr_tick_label(qr, label_array_qr, inc_x0, interp=True):
    """
    Dec 16, 2015, Y.G.@CHX
    get zticks,zticks_label

    Parameters:

        qr:  2-D array, qr of a gisaxs image (data)
        label_array_qr: a labelled array of qr map, get by:
                        label_array_qr = get_qmap_label( qr, qz_edge)
    Options:
        interp: if True, make qz label round by np.round(data, 2)
        inc_x0: x-center of incident beam
    Return:
        rticks: list, r-tick positions in unit of pixel
        rticks_label: list, r-tick positions in unit of real space

    Examples:
        rticks,rticks_label = get_qr_tick_label( qr, label_array_qr)

    """

    rticks = []
    rticks_label = []
    num = len(np.unique(label_array_qr))
    for i in range(1, num):
        ind = np.sort(np.where(label_array_qr == i)[1])
        # tick = round( qr[label_array_qr==i].mean(),2)
        tick = qr[label_array_qr == i].mean()
        if ind[0] < inc_x0 and ind[-1] > inc_x0:  #
            # mean1 = int( (ind[np.where(ind < inc_x0)[0]]).mean() )
            # mean2 = int( (ind[np.where(ind > inc_x0)[0]]).mean() )

            mean1 = int((ind[np.where(ind < inc_x0)[0]])[0])
            mean2 = int((ind[np.where(ind > inc_x0)[0]])[0])
            rticks.append(mean1)
            rticks.append(mean2)
            rticks_label.append(tick)
            rticks_label.append(tick)
        else:
            # print('here')
            # mean = int( ind.mean() )
            mean = int(ind[0])
            # mean = int( (ind[0] +ind[-1])/2 )
            rticks.append(mean)
            rticks_label.append(tick)
            # print (rticks)
            # print (mean, tick)
    n = len(rticks)
    for i, rt in enumerate(rticks):
        if rt == 0:
            rticks[i] = n - i

    if interp:
        rticks = np.array(rticks)
        rticks_label = np.array(rticks_label)
        try:
            w = np.where(rticks <= inc_x0)[0]
            rticks1 = np.int_(np.interp(np.round(rticks_label[w], 3), rticks_label[w], rticks[w]))
            rticks_label1 = np.round(rticks_label[w], 3)
        except:
            rticks_label1 = []
        try:
            w = np.where(rticks > inc_x0)[0]
            rticks2 = np.int_(np.interp(np.round(rticks_label[w], 3), rticks_label[w], rticks[w]))
            rticks = np.append(rticks1, rticks2)
            rticks_label2 = np.round(rticks_label[w], 3)
        except:
            rticks_label2 = []

        rticks_label = np.append(rticks_label1, rticks_label2)

    return rticks, rticks_label


def get_qz_tick_label(qz, label_array_qz, interp=True):
    """
    Dec 16, 2015, Y.G.@CHX
    get zticks,zticks_label

    Parameters:

        qz:  2-D array, qz of a gisaxs image (data)
        label_array_qz: a labelled array of qz map, get by:
                        label_array_qz = get_qmap_label( qz, qz_edge)
        interp: if True, make qz label round by np.round(data, 2)

    Return:
        zticks: list, z-tick positions in unit of pixel
        zticks_label: list, z-tick positions in unit of real space

    Examples:
        zticks,zticks_label = get_qz_tick_label( qz, label_array_qz)

    """

    num = len(np.unique(label_array_qz))
    # zticks = np.array( [ int( np.where( label_array_qz==i )[0].mean() ) for i in range( 1,num ) ])
    zticks = np.array([int(np.where(label_array_qz == i)[0][0]) for i in range(1, num)])

    # zticks_label = np.array( [ round( qz[label_array_qz==i].mean(),4) for i in range( 1, num ) ])
    # zticks_label = np.array( [  qz[label_array_qz==i].mean() for i in range( 1, num ) ])
    zticks_label = np.array([qz[label_array_qz == i][0] for i in range(1, num)])

    if interp:
        zticks = np.int_(np.interp(np.round(zticks_label, 3), zticks_label, zticks))
        zticks_label = np.round(zticks_label, 3)
    return zticks, zticks_label


def get_qzr_map(qr, qz, inc_x0, Nzline=10, Nrline=10, interp=True, return_qrz_label=True, *argv, **kwargs):
    """
    Dec 31, 2016, Y.G.@CHX
    Calculate a qzr map of a gisaxs image (data) without plot

    Parameters:
        qr:  2-D array, qr of a gisaxs image (data)
        qz:  2-D array, qz of a gisaxs image (data)
        inc_x0:  the incident beam center x
    Options:
        Nzline: int, z-line number
        Nrline: int, r-line number

    Return:
    if return_qrz_label
        zticks: list, z-tick positions in unit of pixel
        zticks_label: list, z-tick positions in unit of real space
        rticks: list, r-tick positions in unit of pixel
        rticks_label: list, r-tick positions in unit of real space
    else: return the additional two below
        label_array_qr: qr label array with the same shpae as gisaxs image
        label_array_qz: qz label array with the same shpae as gisaxs image

    Examples:
        ticks = get_qzr_map(  qr, qz, inc_x0  )
    """
    qr_start, qr_end, qr_num = qr.min(), qr.max(), Nrline
    qz_start, qz_end, qz_num = qz.min(), qz.max(), Nzline
    qr_edge, qr_center = get_qedge(qr_start, qr_end, (qr_end - qr_start) / (qr_num + 100), qr_num)
    qz_edge, qz_center = get_qedge(qz_start, qz_end, (qz_end - qz_start) / (qz_num + 100), qz_num)

    label_array_qz = get_qmap_label(qz, qz_edge)
    label_array_qr = get_qmap_label(qr, qr_edge)

    labels_qz, indices_qz = roi.extract_label_indices(label_array_qz)
    labels_qr, indices_qr = roi.extract_label_indices(label_array_qr)
    num_qz = len(np.unique(labels_qz))
    num_qr = len(np.unique(labels_qr))
    zticks, zticks_label = get_qz_tick_label(qz, label_array_qz)
    # rticks,rticks_label  = get_qr_tick_label(label_array_qr,inc_x0)
    try:
        rticks, rticks_label = zip(*np.sort(zip(*get_qr_tick_label(qr, label_array_qr, inc_x0, interp=interp))))
    except:
        rticks, rticks_label = zip(*sorted(zip(*get_qr_tick_label(qr, label_array_qr, inc_x0, interp=interp))))
    # stride = int(len(zticks)/10)
    ticks = [zticks, zticks_label, rticks, rticks_label]
    if return_qrz_label:
        return zticks, zticks_label, rticks, rticks_label, label_array_qr, label_array_qz
    else:
        return zticks, zticks_label, rticks, rticks_label


def plot_qzr_map(qr, qz, inc_x0, ticks=None, data=None, uid="uid", path="", vmin=0.001, vmax=1e1, *argv, **kwargs):
    """
    Dec 31, 2016, Y.G.@CHX
    plot a qzr map of a gisaxs image (data)
    Parameters:
        qr:  2-D array, qr of a gisaxs image (data)
        qz:  2-D array, qz of a gisaxs image (data)
        inc_x0:  the incident beam center x

        ticks = [ zticks,zticks_label,rticks,rticks_label ], use ticks = get_qzr_map(  qr, qz, inc_x0  )  to get

            zticks: list, z-tick positions in unit of pixel
            zticks_label: list, z-tick positions in unit of real space
            rticks: list, r-tick positions in unit of pixel
            rticks_label: list, r-tick positions in unit of real space
            label_array_qr: qr label array with the same shpae as gisaxs image
            label_array_qz: qz label array with the same shpae as gisaxs image

    inc_x0:  the incident beam center x
    Options:
        data: 2-D array, a gisaxs image, if None, =qr+qz
        Nzline: int, z-line number
        Nrline: int, r-line number

    Return:
        None

    Examples:

        ticks = plot_qzr_map(  ticks, inc_x0, data = None, Nzline=10, Nrline= 10   )
        ticks = plot_qzr_map(  ticks, inc_x0, data = avg_imgmr, Nzline=10,  Nrline=10   )
    """

    import copy

    import matplotlib.cm as mcm
    import matplotlib.pyplot as plt

    if ticks is None:
        zticks, zticks_label, rticks, rticks_label, label_array_qr, label_array_qz = get_qzr_map(
            qr, qz, inc_x0, return_qrz_label=True
        )
    else:
        zticks, zticks_label, rticks, rticks_label, label_array_qr, label_array_qz = ticks

    cmap = "viridis"
    _cmap = copy.copy((mcm.get_cmap(cmap)))
    _cmap.set_under("w", 0)
    fig, ax = plt.subplots()
    if data is None:
        data = qr + qz
        im = ax.imshow(data, cmap="viridis", origin="lower")
    else:
        im = ax.imshow(data, cmap="viridis", origin="lower", norm=LogNorm(vmin=vmin, vmax=vmax))

    imr = ax.imshow(
        label_array_qr, origin="lower", cmap="viridis", vmin=0.5, vmax=None
    )  # ,interpolation='nearest',)
    imz = ax.imshow(
        label_array_qz, origin="lower", cmap="viridis", vmin=0.5, vmax=None
    )  # ,interpolation='nearest',)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.set_xlabel(r"$q_r$", fontsize=18)
    ax.set_ylabel(r"$q_z$", fontsize=18)

    stride = 1
    ax.set_yticks(zticks[::stride])
    yticks = zticks_label[::stride]
    ax.set_yticklabels(yticks, fontsize=7)
    # stride = int(len(rticks)/10)
    stride = 1
    ax.set_xticks(rticks[::stride])
    xticks = rticks_label[::stride]
    ax.set_xticklabels(xticks, fontsize=7)
    ax.set_title("%s_Qr_Qz_Map" % uid, y=1.03, fontsize=18)
    fp = path + "%s_Qr_Qz_Map" % (uid) + ".png"
    fig.savefig(fp, dpi=fig.dpi)


def show_qzr_map(qr, qz, inc_x0, data=None, Nzline=10, Nrline=10, interp=True, *argv, **kwargs):
    """
    Dec 16, 2015, Y.G.@CHX
    plot a qzr map of a gisaxs image (data)

    Parameters:
        qr:  2-D array, qr of a gisaxs image (data)
        qz:  2-D array, qz of a gisaxs image (data)
        inc_x0:  the incident beam center x

    Options:
        data: 2-D array, a gisaxs image, if None, =qr+qz
        Nzline: int, z-line number
        Nrline: int, r-line number


    Return:
        zticks: list, z-tick positions in unit of pixel
        zticks_label: list, z-tick positions in unit of real space
        rticks: list, r-tick positions in unit of pixel
        rticks_label: list, r-tick positions in unit of real space


    Examples:

        ticks = show_qzr_map(  qr, qz, inc_x0, data = None, Nzline=10, Nrline= 10   )
        ticks = show_qzr_map(  qr,qz, inc_x0, data = avg_imgmr, Nzline=10,  Nrline=10   )
    """

    import copy

    import matplotlib.cm as mcm
    import matplotlib.pyplot as plt

    cmap = "viridis"
    _cmap = copy.copy((mcm.get_cmap(cmap)))
    _cmap.set_under("w", 0)

    qr_start, qr_end, qr_num = qr.min(), qr.max(), Nrline
    qz_start, qz_end, qz_num = qz.min(), qz.max(), Nzline
    qr_edge, qr_center = get_qedge(qr_start, qr_end, (qr_end - qr_start) / (qr_num + 100), qr_num)
    qz_edge, qz_center = get_qedge(qz_start, qz_end, (qz_end - qz_start) / (qz_num + 100), qz_num)

    label_array_qz = get_qmap_label(qz, qz_edge)
    label_array_qr = get_qmap_label(qr, qr_edge)

    labels_qz, indices_qz = roi.extract_label_indices(label_array_qz)
    labels_qr, indices_qr = roi.extract_label_indices(label_array_qr)
    num_qz = len(np.unique(labels_qz))
    num_qr = len(np.unique(labels_qr))

    fig, ax = plt.subplots(figsize=(8, 14))

    if data is None:
        data = qr + qz
        im = ax.imshow(data, cmap="viridis", origin="lower")
    else:
        im = ax.imshow(data, cmap="viridis", origin="lower", norm=LogNorm(vmin=0.001, vmax=1e1))

    imr = ax.imshow(
        label_array_qr, origin="lower", cmap="viridis", vmin=0.5, vmax=None
    )  # ,interpolation='nearest',)
    imz = ax.imshow(
        label_array_qz, origin="lower", cmap="viridis", vmin=0.5, vmax=None
    )  # ,interpolation='nearest',)

    # caxr = fig.add_axes([0.88, 0.2, 0.03, .7])  #x,y, width, heigth
    # cba = fig.colorbar(im, cax=caxr    )
    # cba = fig.colorbar(im,  fraction=0.046, pad=0.04)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    # fig.colorbar(im, shrink =.82)
    # cba = fig.colorbar(im)

    ax.set_xlabel(r"$q_r$", fontsize=18)
    ax.set_ylabel(r"$q_z$", fontsize=18)

    zticks, zticks_label = get_qz_tick_label(qz, label_array_qz)
    # rticks,rticks_label  = get_qr_tick_label(label_array_qr,inc_x0)
    try:
        rticks, rticks_label = zip(*np.sort(zip(*get_qr_tick_label(qr, label_array_qr, inc_x0, interp=interp))))
    except:
        rticks, rticks_label = zip(*sorted(zip(*get_qr_tick_label(qr, label_array_qr, inc_x0, interp=interp))))
    # stride = int(len(zticks)/10)

    stride = 1
    ax.set_yticks(zticks[::stride])
    yticks = zticks_label[::stride]
    ax.set_yticklabels(yticks, fontsize=7)

    # stride = int(len(rticks)/10)
    stride = 1
    ax.set_xticks(rticks[::stride])
    xticks = rticks_label[::stride]
    ax.set_xticklabels(xticks, fontsize=7)

    if "uid" in kwargs:
        uid = kwargs["uid"]
    else:
        uid = "uid"

    ax.set_title("%s_Qr_Qz_Map" % uid, y=1.03, fontsize=18)

    save = False
    if "save" in kwargs:
        save = kwargs["save"]

    if save:
        path = kwargs["path"]
        fp = path + "%s_Qr_Qz_Map" % (uid) + ".png"
        fig.savefig(fp, dpi=fig.dpi)
    # plt.show()

    return zticks, zticks_label, rticks, rticks_label


def show_qzr_roi(
    data,
    rois,
    inc_x0,
    ticks,
    alpha=0.3,
    vmin=0.01,
    vmax=30.0,
    uid="uid",
    path="",
    save=False,
    return_fig=False,
    *argv,
    **kwargs
):
    """
    Dec 16, 2015, Y.G.@CHX
    plot a qzr map of a gisaxs image with rois( a label array)

    Parameters:
        data: 2-D array, a gisaxs image
        rois:  2-D array, a label array
        inc_x0:  the incident beam center x
        ticks:  zticks, zticks_label, rticks, rticks_label = ticks
            zticks: list, z-tick positions in unit of pixel
            zticks_label: list, z-tick positions in unit of real space
            rticks: list, r-tick positions in unit of pixel
            rticks_label: list, r-tick positions in unit of real space

    Options:
        alpha:  transparency of the label array on top of data

    Return:
         a plot of a qzr map of a gisaxs image with rois( a label array)


    Examples:
        show_qzr_roi( avg_imgr, box_maskr, inc_x0, ticks)

    """
    zticks, zticks_label, rticks, rticks_label = ticks
    avg_imgr, box_maskr = data, rois
    num_qzr = len(np.unique(box_maskr)) - 1

    # fig, ax = plt.subplots(figsize=(8,12))
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.set_title("%s_ROI--Labeled Array on Data" % uid)
    im, im_label = show_label_array_on_image(
        ax,
        avg_imgr,
        box_maskr,
        imshow_cmap="viridis",
        cmap="Paired",
        alpha=alpha,
        vmin=vmin,
        vmax=vmax,
        origin="lower",
    )

    for i in range(1, num_qzr + 1):
        ind = np.where(box_maskr == i)[1]
        indz = np.where(box_maskr == i)[0]
        c = "%i" % i
        y_val = int(indz.mean())

        # print (ind[0], ind[-1], inc_x0 )
        M, m = max(ind), min(ind)

        # if ind[0] < inc_x0 and ind[-1]>inc_x0:
        if m < inc_x0 and M > inc_x0:
            x_val1 = int((ind[np.where(ind < inc_x0)[0]]).mean())
            x_val2 = int((ind[np.where(ind > inc_x0)[0]]).mean())
            ax.text(x_val1, y_val, c, va="center", ha="center")
            ax.text(x_val2, y_val, c, va="center", ha="center")

        else:
            x_val = int(ind.mean())
            # print (xval, y)
            ax.text(x_val, y_val, c, va="center", ha="center")

        # print (x_val1,x_val2)

    # stride = int(len(zticks)/3)
    stride = 1
    ax.set_yticks(zticks[::stride])
    yticks = zticks_label[::stride]
    ax.set_yticklabels(yticks, fontsize=9)

    # stride = int(len(rticks)/3)
    stride = 1
    ax.set_xticks(rticks[::stride])
    xticks = rticks_label[::stride]
    ax.set_xticklabels(xticks, fontsize=9)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    ax.set_xlabel(r"$q_r$", fontsize=22)
    ax.set_ylabel(r"$q_z$", fontsize=22)

    fp = path + "%s_ROI_on_Image" % (uid) + ".png"
    if save:
        fig.savefig(fp, dpi=fig.dpi)
    if return_fig:
        return fig, ax


# plot g2 results


def plot_gisaxs_g2(g2, taus, res_pargs=None, one_plot=False, *argv, **kwargs):
    """Dec 16, 2015, Y.G.@CHX
     plot g2 results,
    g2: one-time correlation function
    taus: the time delays
    res_pargs, a dict, can contains
        uid/path/qr_center/qz_center/
    one_plot: if True, show all qz in one plot
    kwargs: can contains
        vlim: [vmin,vmax]: for the plot limit of y, the y-limit will be [vmin * min(y), vmx*max(y)]
        ylim/xlim: the limit of y and x

    e.g.
    plot_gisaxs_g2( g2b, taus= np.arange( g2b.shape[0]) *timeperframe, q_ring_center = q_ring_center, vlim=[.99, 1.01] )
    """

    if res_pargs is not None:
        uid = res_pargs["uid"]
        path = res_pargs["path"]
        qz_center = res_pargs["qz_center"]
        num_qz = len(qz_center)
        qr_center = res_pargs["qr_center"]
        num_qr = len(qr_center)

    else:
        if "uid" in kwargs.keys():
            uid = kwargs["uid"]
        else:
            uid = "uid"
        if "path" in kwargs.keys():
            path = kwargs["path"]
        else:
            path = ""
        if "qz_center" in kwargs.keys():
            qz_center = kwargs["qz_center"]
            num_qz = len(qz_center)
        else:
            print("Please give qz_center")
        if "qr_center" in kwargs.keys():
            qr_center = kwargs["qr_center"]
            num_qr = len(qr_center)
        else:
            print("Please give qr_center")

    if not one_plot:
        for qz_ind in range(num_qz):
            fig = plt.figure(figsize=(10, 12))
            # fig = plt.figure()
            title_qz = " Qz= %.5f  " % (qz_center[qz_ind]) + r"$\AA^{-1}$"
            plt.title("uid= %s:--->" % uid + title_qz, fontsize=20, y=1.1)
            # print (qz_ind,title_qz)
            if num_qz != 1:
                if num_qr != 1:
                    plt.axis("off")
            sx = int(round(np.sqrt(num_qr)))
            if num_qr % sx == 0:
                sy = int(num_qr / sx)
            else:
                sy = int(num_qr / sx + 1)
            for sn in range(num_qr):
                ax = fig.add_subplot(sx, sy, sn + 1)
                ax.set_ylabel("g2")
                ax.set_xlabel(r"$\tau $ $(s)$", fontsize=16)

                title_qr = " Qr= " + "%.5f  " % (qr_center[sn]) + r"$\AA^{-1}$"
                if num_qz == 1:
                    title = "uid= %s:--->" % uid + title_qz + "__" + title_qr
                else:
                    title = title_qr
                ax.set_title(title)

                y = g2[:, sn + qz_ind * num_qr]
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

            fp = path + "uid=%s--g2-qz=%s" % (uid, qz_center[qz_ind]) + ".png"
            fig.savefig(fp, dpi=fig.dpi)
            fig.tight_layout()
            # plt.show()

    else:
        if num_qz == 1:
            if num_qr == 1:
                fig = plt.figure(figsize=(8, 8))
            else:
                fig = plt.figure(figsize=(10, 12))
        else:
            fig = plt.figure(figsize=(10, 12))

        plt.title("uid= %s" % uid, fontsize=20, y=1.05)
        if num_qz != 1:
            if num_qr != 1:
                plt.axis("off")
        if num_qz == 1:
            if num_qr != 1:
                plt.axis("off")

        sx = int(round(np.sqrt(num_qr)))
        if num_qr % sx == 0:
            sy = int(num_qr / sx)
        else:
            sy = int(num_qr / sx + 1)

        for sn in range(num_qr):
            ax = fig.add_subplot(sx, sy, sn + 1)
            ax.set_ylabel("g2")
            ax.set_xlabel(r"$\tau $ $(s)$", fontsize=16)

            # title_qr = " Qr= " + '%.5f  '%( qr_center[sn]) + r'$\AA^{-1}$'
            title_qr = " Qr= " + "%.5s  " % (qr_center[sn]) + r"$\AA^{-1}$"
            title = title_qr
            ax.set_title(title)

            for qz_ind in range(num_qz):
                y = g2[:, sn + qz_ind * num_qr]
                if sn == 0:
                    title_qz = " Qz= %.5f  " % (qz_center[qz_ind]) + r"$\AA^{-1}$"
                    ax.semilogx(taus, y, "-o", markersize=6, label=title_qz)
                else:
                    ax.semilogx(taus, y, "-o", markersize=6, label="")

            if "ylim" in kwargs:
                ax.set_ylim(kwargs["ylim"])
            elif "vlim" in kwargs:
                vmin, vmax = kwargs["vlim"]
                ax.set_ylim([min(y) * vmin, max(y[1:]) * vmax])
            else:
                pass
            if "xlim" in kwargs:
                ax.set_xlim(kwargs["xlim"])
            if sn == 0:
                ax.legend(loc="best", fontsize=6)
        fp = path + "uid=%s--g2" % (uid) + ".png"
        fig.savefig(fp, dpi=fig.dpi)
        fig.tight_layout()
        # plt.show()


# plot g2 results


def plot_gisaxs_two_g2(g2, taus, g2b, tausb, res_pargs=None, one_plot=False, *argv, **kwargs):
    """Dec 16, 2015, Y.G.@CHX
     plot g2 results,
    g2: one-time correlation function from a multi-tau method
    g2b: another g2 from a two-time method
    taus: the time delays
    kwargs: can contains
        vlim: [vmin,vmax]: for the plot limit of y, the y-limit will be [vmin * min(y), vmx*max(y)]
        ylim/xlim: the limit of y and x

    e.g.
    plot_saxs_g2( g2b, taus= np.arange( g2b.shape[0]) *timeperframe, q_ring_center = q_ring_center, vlim=[.99, 1.01] )

    """

    if res_pargs is not None:
        uid = res_pargs["uid"]
        path = res_pargs["path"]
        qz_center = res_pargs["qz_center"]
        num_qz = len(qz_center)
        qr_center = res_pargs["qr_center"]
        num_qr = len(qr_center)

    else:
        if "uid" in kwargs.keys():
            uid = kwargs["uid"]
        else:
            uid = "uid"
        if "path" in kwargs.keys():
            path = kwargs["path"]
        else:
            path = ""
        if "qz_center" in kwargs.keys():
            qz_center = kwargs["qz_center"]
            num_qz = len(qz_center)
        else:
            print("Please give qz_center")
        if "qr_center" in kwargs.keys():
            qr_center = kwargs["qr_center"]
            num_qr = len(qr_center)
        else:
            print("Please give qr_center")

    if not one_plot:
        for qz_ind in range(num_qz):
            fig = plt.figure(figsize=(12, 10))
            # fig = plt.figure()
            title_qz = " Qz= %.5f  " % (qz_center[qz_ind]) + r"$\AA^{-1}$"
            plt.title("uid= %s:--->" % uid + title_qz, fontsize=20, y=1.1)
            # print (qz_ind,title_qz)
            if num_qz != 1:
                plt.axis("off")
            sx = int(round(np.sqrt(num_qr)))
            if num_qr % sx == 0:
                sy = int(num_qr / sx)
            else:
                sy = int(num_qr / sx + 1)
            for sn in range(num_qr):
                ax = fig.add_subplot(sx, sy, sn + 1)
                ax.set_ylabel("g2")
                ax.set_xlabel(r"$\tau $ $(s)$", fontsize=16)

                title_qr = " Qr= " + "%.5f  " % (qr_center[sn]) + r"$\AA^{-1}$"
                if num_qz == 1:
                    title = "uid= %s:--->" % uid + title_qz + "__" + title_qr
                else:
                    title = title_qr
                ax.set_title(title)

                y = g2b[:, sn + qz_ind * num_qr]
                ax.semilogx(tausb, y, "--r", markersize=6, label="by-two-time")

                # y2=g2[:, sn]
                y2 = g2[:, sn + qz_ind * num_qr]
                ax.semilogx(taus, y2, "o", markersize=6, label="by-multi-tau")

                if sn + qz_ind * num_qr == 0:
                    ax.legend(loc="best")
                if "ylim" in kwargs:
                    ax.set_ylim(kwargs["ylim"])
                elif "vlim" in kwargs:
                    vmin, vmax = kwargs["vlim"]
                    ax.set_ylim([min(y) * vmin, max(y[1:]) * vmax])
                else:
                    pass
                if "xlim" in kwargs:
                    ax.set_xlim(kwargs["xlim"])

            fp = path + "uid=%s--two-g2-qz=%s" % (uid, qz_center[qz_ind]) + ".png"
            fig.savefig(fp, dpi=fig.dpi)
            fig.tight_layout()
            # plt.show()

    else:
        fig = plt.figure(figsize=(12, 10))
        plt.title("uid= %s" % uid, fontsize=20, y=1.05)
        if num_qz != 1:
            if num_qr != 1:
                plt.axis("off")
        if num_qz == 1:
            if num_qr != 1:
                plt.axis("off")

        sx = int(round(np.sqrt(num_qr)))
        if num_qr % sx == 0:
            sy = int(num_qr / sx)
        else:
            sy = int(num_qr / sx + 1)

        for sn in range(num_qr):
            ax = fig.add_subplot(sx, sy, sn + 1)
            ax.set_ylabel("g2")
            ax.set_xlabel(r"$\tau $ $(s)$", fontsize=16)

            title_qr = " Qr= " + "%.5s  " % (qr_center[sn]) + r"$\AA^{-1}$"
            # title_qr = " Qr= " + '%.5f  '%( qr_center[sn]) + r'$\AA^{-1}$'
            title = title_qr
            ax.set_title(title)

            for qz_ind in range(num_qz):
                y = g2b[:, sn + qz_ind * num_qr]
                y2 = g2[:, sn + qz_ind * num_qr]
                title_qz = " Qz= %.5f  " % (qz_center[qz_ind]) + r"$\AA^{-1}$"
                label1 = ""
                label2 = ""

                if sn == 0:
                    label2 = title_qz

                elif sn == 1:
                    if qz_ind == 0:
                        label1 = "by-two-time"
                        label2 = "by-multi-tau"

                ax.semilogx(tausb, y, "-r", markersize=6, linewidth=4, label=label1)
                ax.semilogx(taus, y2, "o", markersize=6, label=label2)

            if "ylim" in kwargs:
                ax.set_ylim(kwargs["ylim"])
            elif "vlim" in kwargs:
                vmin, vmax = kwargs["vlim"]
                ax.set_ylim([min(y) * vmin, max(y[1:]) * vmax])
            else:
                pass
            if "xlim" in kwargs:
                ax.set_xlim(kwargs["xlim"])
            if (sn == 0) or (sn == 1):
                ax.legend(loc="best", fontsize=6)

        fp = path + "uid=%s--g2--two-g2-" % uid + ".png"

        fig.savefig(fp, dpi=fig.dpi)
        fig.tight_layout()
        # plt.show()


def save_gisaxs_g2(g2, res_pargs, time_label=False, taus=None, filename=None, *argv, **kwargs):
    """
    Aug 8, 2016, Y.G.@CHX
    save g2 results,
       res_pargs should contain
           g2: one-time correlation function
           res_pargs: contions taus, q_ring_center values
           path:
           uid:
    """

    if taus is None:
        taus = res_pargs["taus"]

    try:
        qz_center = res_pargs["qz_center"]
        qr_center = res_pargs["qr_center"]
    except:
        roi_label = res_pargs["roi_label"]

    path = res_pargs["path"]
    uid = res_pargs["uid"]

    df = DataFrame(np.hstack([(taus).reshape(len(g2), 1), g2]))
    columns = []
    columns.append("tau")

    try:
        for qz in qz_center:
            for qr in qr_center:
                columns.append([str(qz), str(qr)])
    except:
        columns.append([v for (k, v) in roi_label.items()])

    df.columns = columns

    if filename is None:
        if time_label:
            dt = datetime.now()
            CurTime = "%s%02d%02d-%02d%02d-" % (dt.year, dt.month, dt.day, dt.hour, dt.minute)
            filename = os.path.join(path, "g2-%s-%s.csv" % (uid, CurTime))
        else:
            filename = os.path.join(path, "uid=%s--g2.csv" % (uid))
    else:
        filename = os.path.join(path, filename)
    df.to_csv(filename)
    print("The correlation function of uid= %s is saved with filename as %s" % (uid, filename))


def stretched_auto_corr_scat_factor(x, beta, relaxation_rate, alpha=1.0, baseline=1):
    return beta * (np.exp(-2 * relaxation_rate * x)) ** alpha + baseline


def simple_exponential(x, beta, relaxation_rate, baseline=1):
    return beta * np.exp(-2 * relaxation_rate * x) + baseline


def fit_gisaxs_g2(g2, res_pargs, function="simple_exponential", one_plot=False, *argv, **kwargs):
    """
     July 20,2016, Y.G.@CHX
     Fit one-time correlation function

     The support functions include simple exponential and stretched/compressed exponential
     Parameters
     ----------
     g2: one-time correlation function for fit, with shape as [taus, qs]
     res_pargs: a dict, contains keys
         taus: the time delay, with the same length as g2
         q_ring_center:  the center of q rings, for the title of each sub-plot
         uid: unique id, for the title of plot
     kwargs:
         variables: if exist, should be a dict, like
                 { 'lags': True,  #always True
                    'beta', Ture, # usually True
                    'relaxation_rate': False, #always False
                    'alpha':False, #False for simple exponential, True for stretched/compressed
                    'baseline': True #sometimes be False, keep as 1
                  }

     function:
         'simple_exponential': fit by a simple exponential function, defined as
                     beta * np.exp(-2 * relaxation_rate * lags) + baseline
         'streched_exponential': fit by a streched exponential function, defined as
                     beta * (np.exp(-2 * relaxation_rate * lags))**alpha + baseline

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

    TO DO:
        add variables to options
    """

    taus = res_pargs["taus"]
    qz_center = res_pargs["qz_center"]
    num_qz = len(qz_center)
    qr_center = res_pargs["qr_center"]
    num_qr = len(qr_center)
    uid = res_pargs["uid"]
    path = res_pargs["path"]
    # uid=res_pargs['uid']

    num_rings = g2.shape[1]
    beta = np.zeros(num_rings)  #  contrast factor
    rate = np.zeros(num_rings)  #  relaxation rate
    alpha = np.zeros(num_rings)  #  alpha
    baseline = np.zeros(num_rings)  #  baseline

    if function == "simple_exponential" or function == "simple":
        _vars = np.unique(_vars + ["alpha"])
        mod = Model(stretched_auto_corr_scat_factor)  # ,  independent_vars= list( _vars)   )

    elif function == "stretched_exponential" or function == "stretched":
        mod = Model(stretched_auto_corr_scat_factor)  # ,  independent_vars=  _vars)

    else:
        print(
            "The %s is not supported.The supported functions include simple_exponential and stretched_exponential"
            % function
        )

    # mod.set_param_hint( 'beta', value = 0.05 )
    # mod.set_param_hint( 'alpha', value = 1.0 )
    # mod.set_param_hint( 'relaxation_rate', value = 0.005 )
    # mod.set_param_hint( 'baseline', value = 1.0, min=0.5, max= 1.5 )
    mod.set_param_hint("baseline", min=0.5, max=2.5)
    mod.set_param_hint("beta", min=0.0)
    mod.set_param_hint("alpha", min=0.0)
    mod.set_param_hint("relaxation_rate", min=0.0)

    if "fit_variables" in kwargs:
        additional_var = kwargs["fit_variables"]
        # print ( additional_var   )
        _vars = [k for k in list(additional_var.keys()) if additional_var[k] is False]
    else:
        _vars = []

    if "guess_values" in kwargs:
        if "beta" in list(kwargs["guess_values"].keys()):
            beta_ = kwargs["guess_values"]["beta"]
        else:
            beta_ = 0.05

        if "alpha" in list(kwargs["guess_values"].keys()):
            alpha_ = kwargs["guess_values"]["alpha"]
        else:
            alpha_ = 1.0
        if "relaxation_rate" in list(kwargs["guess_values"].keys()):
            relaxation_rate_ = kwargs["guess_values"]["relaxation_rate"]
        else:
            relaxation_rate_ = 0.005
        if "baseline" in list(kwargs["guess_values"].keys()):
            baseline_ = kwargs["guess_values"]["baseline"]
        else:
            baseline_ = 1.0
        pars = mod.make_params(beta=beta_, alpha=alpha_, relaxation_rate=relaxation_rate_, baseline=baseline_)
    else:
        pars = mod.make_params(beta=0.05, alpha=1.0, relaxation_rate=0.005, baseline=1.0)

    for v in _vars:
        pars["%s" % v].vary = False
        # print ( pars['%s'%v], pars['%s'%v].vary )
    result = {}

    if not one_plot:
        for qz_ind in range(num_qz):
            # fig = plt.figure(figsize=(10, 12))
            fig = plt.figure(figsize=(12, 10))
            # fig = plt.figure()
            title_qz = " Qz= %.5f  " % (qz_center[qz_ind]) + r"$\AA^{-1}$"
            plt.title("uid= %s:--->" % uid + title_qz, fontsize=20, y=1.1)
            # print (qz_ind,title_qz)
            if num_qz != 1:
                plt.axis("off")
            sx = int(round(np.sqrt(num_qr)))
            if num_qr % sx == 0:
                sy = int(num_qr / sx)
            else:
                sy = int(num_qr / sx + 1)
            for sn in range(num_qr):
                ax = fig.add_subplot(sx, sy, sn + 1)
                ax.set_ylabel("g2")
                ax.set_xlabel(r"$\tau $ $(s)$", fontsize=16)

                title_qr = " Qr= " + "%.5f  " % (qr_center[sn]) + r"$\AA^{-1}$"
                if num_qz == 1:
                    title = "uid= %s:--->" % uid + title_qz + "__" + title_qr
                else:
                    title = title_qr
                ax.set_title(title)

                i = sn + qz_ind * num_qr
                y = g2[1:, i]

                result1 = mod.fit(y, pars, x=taus[1:])

                # print ( result1.best_values)
                rate[i] = result1.best_values["relaxation_rate"]
                # rate[i] = 1e-16
                beta[i] = result1.best_values["beta"]

                # baseline[i] = 1.0
                baseline[i] = result1.best_values["baseline"]

                if function == "simple_exponential" or function == "simple":
                    alpha[i] = 1.0
                elif function == "stretched_exponential" or function == "stretched":
                    alpha[i] = result1.best_values["alpha"]

                ax.semilogx(taus[1:], y, "bo")
                ax.semilogx(taus[1:], result1.best_fit, "-r")

                if "ylim" in kwargs:
                    ax.set_ylim(kwargs["ylim"])
                elif "vlim" in kwargs:
                    vmin, vmax = kwargs["vlim"]
                    ax.set_ylim([min(y) * vmin, max(y[1:]) * vmax])
                else:
                    pass
                if "xlim" in kwargs:
                    ax.set_xlim(kwargs["xlim"])

                txts = r"$\tau$" + r"$ = %.3f$" % (1 / rate[i]) + r"$ s$"
                ax.text(x=0.02, y=0.55 + 0.3, s=txts, fontsize=14, transform=ax.transAxes)
                txts = r"$\alpha$" + r"$ = %.3f$" % (alpha[i])
                # txts = r'$\beta$' + r'$ = %.3f$'%(beta[i]) +  r'$ s^{-1}$'
                ax.text(x=0.02, y=0.45 + 0.3, s=txts, fontsize=14, transform=ax.transAxes)
                txts = r"$baseline$" + r"$ = %.3f$" % (baseline[i])
                ax.text(x=0.02, y=0.35 + 0.3, s=txts, fontsize=14, transform=ax.transAxes)

            result = dict(beta=beta, rate=rate, alpha=alpha, baseline=baseline)
            fp = path + "uid=%s--g2-qz=%s--fit" % (uid, qz_center[qz_ind]) + ".png"
            fig.savefig(fp, dpi=fig.dpi)
            fig.tight_layout()
            # plt.show()

    else:
        # fig = plt.figure(figsize=(10, 12))
        # fig = plt.figure(figsize=(12, 10))
        if num_qz == 1:
            if num_qr == 1:
                fig = plt.figure(figsize=(8, 8))
            else:
                fig = plt.figure(figsize=(10, 12))
        else:
            fig = plt.figure(figsize=(10, 12))

        plt.title("uid= %s" % uid, fontsize=20, y=1.05)
        if num_qz != 1:
            if num_qr != 1:
                plt.axis("off")

        sx = int(round(np.sqrt(num_qr)))
        if num_qr % sx == 0:
            sy = int(num_qr / sx)
        else:
            sy = int(num_qr / sx + 1)

        for sn in range(num_qr):
            ax = fig.add_subplot(sx, sy, sn + 1)
            ax.set_ylabel("g2")
            ax.set_xlabel(r"$\tau $ $(s)$", fontsize=16)

            # title_qr = " Qr= " + '%.5f  '%( qr_center[sn]) + r'$\AA^{-1}$'
            title_qr = " Qr= " + "%.5s  " % (qr_center[sn]) + r"$\AA^{-1}$"
            title = title_qr
            ax.set_title(title)

            for qz_ind in range(num_qz):
                i = sn + qz_ind * num_qr
                y = g2[1:, i]
                result1 = mod.fit(y, pars, x=taus[1:])
                # print ( result1.best_values)
                rate[i] = result1.best_values["relaxation_rate"]
                # rate[i] = 1e-16
                beta[i] = result1.best_values["beta"]
                # baseline[i] = 1.0
                baseline[i] = result1.best_values["baseline"]

                if function == "simple_exponential" or function == "simple":
                    alpha[i] = 1.0
                elif function == "stretched_exponential" or function == "stretched":
                    alpha[i] = result1.best_values["alpha"]

                if sn == 0:
                    title_qz = " Qz= %.5f  " % (qz_center[qz_ind]) + r"$\AA^{-1}$"
                    ax.semilogx(taus[1:], y, "o", markersize=6, label=title_qz)
                else:
                    ax.semilogx(taus[1:], y, "o", markersize=6, label="")

                ax.semilogx(taus[1:], result1.best_fit, "-r")

                # print(  result1.best_values['relaxation_rate'],  result1.best_values['beta'] )

                txts = r"$q_z$" + r"$_%s$" % qz_ind + r"$\tau$" + r"$ = %.3f$" % (1 / rate[i]) + r"$ s$"
                ax.text(x=0.02, y=0.55 + 0.3 - 0.1 * qz_ind, s=txts, fontsize=14, transform=ax.transAxes)

            if "ylim" in kwargs:
                ax.set_ylim(kwargs["ylim"])
            elif "vlim" in kwargs:
                vmin, vmax = kwargs["vlim"]
                ax.set_ylim([min(y) * vmin, max(y[1:]) * vmax])
            else:
                pass
            if "xlim" in kwargs:
                ax.set_xlim(kwargs["xlim"])
            if sn == 0:
                ax.legend(loc="best", fontsize=6)

        result = dict(beta=beta, rate=rate, alpha=alpha, baseline=baseline)
        fp = path + "uid=%s--g2--fit-" % (uid) + ".png"
        fig.savefig(fp, dpi=fig.dpi)
        fig.tight_layout()
        # plt.show()

        # dt =datetime.now()
        # CurTime = '%s%02d%02d-%02d%02d-' % (dt.year, dt.month, dt.day,dt.hour,dt.minute)
        # fp = path + 'g2--uid=%s-qz=%s-fit'%(uid,qz_center[qz_ind]) + CurTime + '.png'
        # fig.savefig( fp, dpi=fig.dpi)

    # result = dict( beta=beta, rate=rate, alpha=alpha, baseline=baseline )
    # fp = path + 'uid=%s--g2--fit-'%(uid)  + '.png'
    # fig.savefig( fp, dpi=fig.dpi)
    # fig.tight_layout()
    # plt.show()

    return result


# GiSAXS End
###############################


def get_each_box_mean_intensity(data_series, box_mask, sampling, timeperframe, plot_=True, *argv, **kwargs):
    """Dec 16, 2015, Y.G.@CHX
    get each box (ROI) mean intensity as a function of time


    """

    mean_int_sets, index_list = roi.mean_intensity(np.array(data_series[::sampling]), box_mask)
    try:
        N = len(data_series)
    except:
        N = data_series.length
    times = np.arange(N) * timeperframe  # get the time for each frame
    num_rings = len(np.unique(box_mask)[1:])
    if plot_:
        fig, ax = plt.subplots(figsize=(8, 8))
        uid = "uid"
        if "uid" in kwargs.keys():
            uid = kwargs["uid"]

        ax.set_title("uid= %s--Mean intensity of each box" % uid)
        for i in range(num_rings):
            ax.plot(times[::sampling], mean_int_sets[:, i], label="Box " + str(i + 1), marker="o", ls="-")
            ax.set_xlabel("Time")
            ax.set_ylabel("Mean Intensity")
        ax.legend()

        # fp = path + 'uid=%s--Mean intensity of each box-'%(uid)  + '.png'
        if "path" not in kwargs.keys():
            path = ""
        else:
            path = kwargs["path"]
        fp = path + "uid=%s--Mean-intensity-of-each-ROI-" % (uid) + ".png"
        fig.savefig(fp, dpi=fig.dpi)

        # plt.show()
    return times, mean_int_sets


def power_func(x, D0, power=2):
    return D0 * x**power


def fit_qr_qz_rate(qr, qz, rate, plot_=True, *argv, **kwargs):
    """
    Option:
        if power_variable = False, power =2 to fit q^2~rate,
                Otherwise, power is variable.
    """
    power_variable = False
    x = qr
    if "fit_range" in kwargs.keys():
        fit_range = kwargs["fit_range"]
    else:
        fit_range = None

    if "uid" in kwargs.keys():
        uid = kwargs["uid"]
    else:
        uid = "uid"

    if "path" in kwargs.keys():
        path = kwargs["path"]
    else:
        path = ""

    if fit_range is not None:
        y = rate[fit_range[0] : fit_range[1]]
        x = q[fit_range[0] : fit_range[1]]

    mod = Model(power_func)
    # mod.set_param_hint( 'power',   min=0.5, max= 10 )
    # mod.set_param_hint( 'D0',   min=0 )
    pars = mod.make_params(power=2, D0=1 * 10 ^ (-5))
    if power_variable:
        pars["power"].vary = True
    else:
        pars["power"].vary = False

    Nqr = len(qr)
    Nqz = len(qz)
    D0 = np.zeros(Nqz)
    power = 2  # np.zeros( Nqz )

    res = []
    for i, qz_ in enumerate(qz):
        try:
            y = np.array(rate["rate"][i * Nqr : (i + 1) * Nqr])
        except:
            y = np.array(rate[i * Nqr : (i + 1) * Nqr])

        # print( len(x), len(y) )
        _result = mod.fit(y, pars, x=x)
        res.append(_result)
        D0[i] = _result.best_values["D0"]
        # power[i] = _result.best_values['power']
        print("The fitted diffusion coefficient D0 is:  %.3e   A^2S-1" % D0[i])

    if plot_:
        fig, ax = plt.subplots()
        plt.title("Q%s-Rate--uid= %s_Fit" % (power, uid), fontsize=20, y=1.06)
        for i, qz_ in enumerate(qz):
            ax.plot(x**power, y, marker="o", label=r"$q_z=%.5f$" % qz_)
            ax.plot(x**power, res[i].best_fit, "-r")
            txts = r"$D0: %.3e$" % D0[i] + r" $A^2$" + r"$s^{-1}$"
            dy = 0.1
            ax.text(x=0.15, y=0.65 - dy * i, s=txts, fontsize=14, transform=ax.transAxes)
            legend = ax.legend(loc="best")

        ax.set_ylabel("Relaxation rate " r"$\gamma$" "($s^{-1}$)")
        ax.set_xlabel("$q^%s$" r"($\AA^{-2}$)" % power)

        dt = datetime.now()
        CurTime = "%s%02d%02d-%02d%02d-" % (dt.year, dt.month, dt.day, dt.hour, dt.minute)
        # fp = path + 'Q%s-Rate--uid=%s'%(power,uid) + CurTime + '--Fit.png'
        fp = path + "uid=%s--Q-Rate" % (uid) + "--fit-.png"
        fig.savefig(fp, dpi=fig.dpi)

        fig.tight_layout()
        # plt.show()

    return D0


# plot g4 results


def plot_gisaxs_g4(g4, taus, res_pargs=None, one_plot=False, *argv, **kwargs):
    """Dec 16, 2015, Y.G.@CHX
     plot g4 results,
    g4: four-time correlation function
    taus: the time delays
    res_pargs, a dict, can contains
        uid/path/qr_center/qz_center/
    kwargs: can contains
        vlim: [vmin,vmax]: for the plot limit of y, the y-limit will be [vmin * min(y), vmx*max(y)]
        ylim/xlim: the limit of y and x

    e.g.
    plot_gisaxs_g4( g4, taus= np.arange( g2b.shape[0]) *timeperframe, q_ring_center = q_ring_center, vlim=[.99, 1.01] )

    """

    if res_pargs is not None:
        uid = res_pargs["uid"]
        path = res_pargs["path"]
        qz_center = res_pargs["qz_center"]
        num_qz = len(qz_center)
        qr_center = res_pargs["qr_center"]
        num_qr = len(qr_center)

    else:
        if "uid" in kwargs.keys():
            uid = kwargs["uid"]
        else:
            uid = "uid"
        if "path" in kwargs.keys():
            path = kwargs["path"]
        else:
            path = ""
        if "qz_center" in kwargs.keys():
            qz_center = kwargs["qz_center"]
            num_qz = len(qz_center)
        else:
            print("Please give qz_center")
        if "qr_center" in kwargs.keys():
            qr_center = kwargs["qr_center"]
            num_qr = len(qr_center)
        else:
            print("Please give qr_center")

    if not one_plot:
        for qz_ind in range(num_qz):
            fig = plt.figure(figsize=(12, 10))
            # fig = plt.figure()
            title_qz = " Qz= %.5f  " % (qz_center[qz_ind]) + r"$\AA^{-1}$"
            plt.title("uid= %s:--->" % uid + title_qz, fontsize=20, y=1.1)
            # print (qz_ind,title_qz)
            if num_qz != 1:
                if num_qr != 1:
                    plt.axis("off")
            sx = int(round(np.sqrt(num_qr)))
            if num_qr % sx == 0:
                sy = int(num_qr / sx)
            else:
                sy = int(num_qr / sx + 1)
            for sn in range(num_qr):
                ax = fig.add_subplot(sx, sy, sn + 1)
                ax.set_ylabel("g4")
                ax.set_xlabel(r"$\tau $ $(s)$", fontsize=16)

                title_qr = " Qr= " + "%.5f  " % (qr_center[sn]) + r"$\AA^{-1}$"
                if num_qz == 1:
                    title = "uid= %s:--->" % uid + title_qz + "__" + title_qr
                else:
                    title = title_qr
                ax.set_title(title)

                y = g4[:, sn + qz_ind * num_qr]
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

            fp = path + "uid=%s--g4-qz=%s" % (uid, qz_center[qz_ind]) + ".png"
            fig.savefig(fp, dpi=fig.dpi)
            fig.tight_layout()
            # plt.show()

    else:
        fig = plt.figure(figsize=(12, 10))
        plt.title("uid= %s" % uid, fontsize=20, y=1.05)
        if num_qz != 1:
            if num_qr != 1:
                plt.axis("off")

        sx = int(round(np.sqrt(num_qr)))
        if num_qr % sx == 0:
            sy = int(num_qr / sx)
        else:
            sy = int(num_qr / sx + 1)

        for sn in range(num_qr):
            ax = fig.add_subplot(sx, sy, sn + 1)
            ax.set_ylabel("g4")
            ax.set_xlabel(r"$\tau $ $(s)$", fontsize=16)

            title_qr = " Qr= " + "%.5f  " % (qr_center[sn]) + r"$\AA^{-1}$"
            title = title_qr
            ax.set_title(title)

            for qz_ind in range(num_qz):
                y = g4[:, sn + qz_ind * num_qr]
                if sn == 0:
                    title_qz = " Qz= %.5f  " % (qz_center[qz_ind]) + r"$\AA^{-1}$"
                    ax.semilogx(taus, y, "-o", markersize=6, label=title_qz)
                else:
                    ax.semilogx(taus, y, "-o", markersize=6, label="")

            if "ylim" in kwargs:
                ax.set_ylim(kwargs["ylim"])
            elif "vlim" in kwargs:
                vmin, vmax = kwargs["vlim"]
                ax.set_ylim([min(y) * vmin, max(y[1:]) * vmax])
            else:
                pass
            if "xlim" in kwargs:
                ax.set_xlim(kwargs["xlim"])
            if sn == 0:
                ax.legend(loc="best", fontsize=6)
        fp = path + "uid=%s--g4-" % (uid) + ".png"
        fig.savefig(fp, dpi=fig.dpi)
        fig.tight_layout()
        # plt.show()


def multi_uids_gisaxs_xpcs_analysis(
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
    """'Sep 16, 2016, YG@CHX-NSLS2
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
    maskr = mask[::-1, :]
    data_dir = md["data_dir"]
    box_maskr = md["ring_mask"]
    qz_center = md["qz_center"]
    qr_center = md["qr_center"]

    for run_seq in range(run_num):
        g2s[run_seq + 1] = {}
        useful_uids[run_seq + 1] = {}
        i = 0
        for sub_seq in range(0, sub_num):
            uid = uids[sub_seq + run_seq * sub_num]
            print("The %i--th uid to be analyzed is : %s" % (i, uid))
            try:
                detector = get_detector(db[uid])
                imgs = load_data(uid, detector)
            except:
                print("The %i--th uid: %s can not load data" % (i, uid))
                imgs = 0

            data_dir_ = os.path.join(data_dir, "%s/" % uid)
            os.makedirs(data_dir_, exist_ok=True)
            i += 1
            if imgs != 0:
                Nimg = len(imgs)
                md_ = imgs.md
                useful_uids[run_seq + 1][i] = uid

                imgsr = reverse_updown(imgs)
                imgsra = apply_mask(imgsr, maskr)

                if compress:
                    filename = "/XF11ID/analysis/Compressed_Data" + "/uid_%s.cmp" % uid
                    maskr, avg_imgr, imgsum, bad_frame_list = compress_eigerdata(
                        imgsr,
                        maskr,
                        md_,
                        filename,
                        force_compress=force_compress,
                        bad_pixel_threshold=5e9,
                        nobytes=4,
                        para_compress=True,
                        num_sub=100,
                    )

                    try:
                        md["Measurement"] = db[uid]["start"]["Measurement"]
                        # md['sample']=db[uid]['start']['sample']
                        # print( md['Measurement'] )
                    except:
                        md["Measurement"] = "Measurement"
                        md["sample"] = "sample"

                    dpix = md["x_pixel_size"] * 1000.0  # in mm, eiger 4m is 0.075 mm
                    lambda_ = md["incident_wavelength"]  # wavelegth of the X-rays in Angstroms
                    Ldet = md["detector_distance"]
                    # detector to sample distance (mm), currently, *1000 for saxs, *1 for gisaxs
                    exposuretime = md["count_time"]
                    acquisition_period = md["frame_time"]
                    timeperframe = acquisition_period  # for g2
                    # timeperframe = exposuretime#for visiblitly
                    # timeperframe = 2  ## manual overwrite!!!! we apparently writing the wrong metadata....
                    setup_pargs = dict(
                        uid=uid, dpix=dpix, Ldet=Ldet, lambda_=lambda_, timeperframe=timeperframe, path=data_dir
                    )
                    md["avg_img"] = avg_imgr

                    min_inten = 0
                    # good_start = np.where( np.array(imgsum) > min_inten )[0][0]
                    # good_start = 0
                    # good_start = max(good_start, np.where( np.array(imgsum) > min_inten )[0][0] )

                    good_start = good_start
                    if good_end is None:
                        good_end_ = len(imgs)
                    else:
                        good_end_ = good_end
                    FD = Multifile(filename, good_start, good_end_)
                    good_start = max(good_start, np.where(np.array(imgsum) > min_inten)[0][0])
                    print("With compression, the good_start frame number is: %s " % good_start)
                    print("The good_end frame number is: %s " % good_end_)

                    if not para_run:
                        g2, lag_steps_ = cal_g2c(
                            FD, box_maskr, bad_frame_list, good_start, num_buf=8, imgsum=None, norm=None
                        )
                    else:
                        g2, lag_steps_ = cal_g2p(
                            FD, box_maskr, bad_frame_list, good_start, num_buf=8, imgsum=None, norm=None
                        )

                    if len(lag_steps) < len(lag_steps_):
                        lag_steps = lag_steps_

                else:
                    sampling = 1000  # sampling should be one

                    # good_start = check_shutter_open( imgsra,  min_inten=5, time_edge = [0,10], plot_ = False )
                    good_start = 0
                    good_series = apply_mask(imgsar[good_start:], maskr)
                    imgsum, bad_frame_list = get_each_frame_intensity(
                        good_series, sampling=sampling, bad_pixel_threshold=1.2e8, plot_=False, uid=uid
                    )
                    bad_image_process = False

                    if len(bad_frame_list):
                        bad_image_process = True
                    print(bad_image_process)

                    g2, lag_steps_ = cal_g2(
                        good_series, box_maskr, bad_image_process, bad_frame_list, good_start, num_buf=8
                    )
                    if len(lag_steps) < len(lag_steps_):
                        lag_steps = lag_step_

                taus_ = lag_steps_ * timeperframe
                taus = lag_steps * timeperframe
                res_pargs = dict(taus=taus_, qz_center=qz_center, qr_center=qr_center, path=data_dir_, uid=uid)
                save_gisaxs_g2(g2, res_pargs)
                # plot_gisaxs_g2( g2, taus,  vlim=[0.95, 1.1], res_pargs=res_pargs, one_plot=True)

                if fit:
                    fit_result = fit_gisaxs_g2(
                        g2,
                        res_pargs,
                        function="stretched",
                        vlim=[0.95, 1.1],
                        fit_variables={"baseline": True, "beta": True, "alpha": False, "relaxation_rate": True},
                        guess_values={"baseline": 1.229, "beta": 0.05, "alpha": 1.0, "relaxation_rate": 0.01},
                        one_plot=True,
                    )

                    fit_qr_qz_rate(qr_center, qz_center, fit_result, power_variable=False, uid=uid, path=data_dir_)

                psave_obj(md, data_dir_ + "uid=%s-md" % uid)  # save the setup parameters

                g2s[run_seq + 1][i] = g2

                print("*" * 40)
                print()

    return g2s, taus, useful_uids
