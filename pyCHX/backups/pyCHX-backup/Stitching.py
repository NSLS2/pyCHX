import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import PIL
from scipy.signal import savgol_filter as sf

from pyCHX.chx_generic_functions import plot1D, show_img
from pyCHX.DataGonio import convert_Qmap


def get_base_all_filenames(inDir, base_filename_cut_length=-7):
    """YG Sep 26, 2017 @SMI
    Get base filenames and their related all filenames
    Input:
      inDir, str, input data dir
       base_filename_cut_length: to which length the base name is unique
    Output:
      dict: keys,  base filename
            vales, all realted filename
    """
    from os import listdir
    from os.path import isfile, join

    tifs = np.array([f for f in listdir(inDir) if isfile(join(inDir, f))])
    tifsc = list(tifs.copy())
    utifs = np.sort(np.unique(np.array([f[:base_filename_cut_length] for f in tifs])))[::-1]
    files = {}
    for uf in utifs:
        files[uf] = []
        i = 0
        reName = []
        for i in range(len(tifsc)):
            if uf in tifsc[i]:
                files[uf].append(inDir + tifsc[i])
                reName.append(tifsc[i])
        for fn in reName:
            tifsc.remove(fn)
    return files


def check_stitch_two_imgs(img1, img2, overlap_width, badpixel_width=10):
    """YG Check the stitched two imgs
    For SMI, overlap_width = 85  #for calibration data--Kevin gives one, 4  degree
             overlap_width = 58  #for TWD data, 5 degree step
    For SMI:
        d0 =  np.array(  PIL.Image.open(infiles[0]).convert('I') ).T
        d1 = np.array(  PIL.Image.open(infiles[1]).convert('I') ).T
    Example:

    img1 = np.array(  PIL.Image.open(infiles[0]).convert('I') ).T
    img2 = np.array(  PIL.Image.open(infiles[1]).convert('I') ).T
    img12 = check_stitch_two_imgs( img1, img2, overlap_width=58, badpixel_width =10 )
    show_img(img12[200:800, 120:250], logs = False,   cmap = cmap_vge_hdr, vmin=0.8*img12.min(),
          vmax= 1.2*img12.max(), aspect=1,image_name = 'Check_Overlap')

    """
    w = overlap_width
    ow = badpixel_width
    M, N = img1.shape[0], img1.shape[1]
    d0 = img1
    d1 = img2
    d01 = np.zeros([M, N * 2 - w * (2 - 1)])
    d01[:, :N] = d0
    i = 1
    a1, a2, b1, b2 = N * i - w * (i - 1) - ow, N * (i + 1) - w * i, w - ow, N
    d01[:, a1:a2] = d1[:, b1:b2]
    return d01


def Correct_Overlap_Images_Intensities(
    infiles, window_length=101, polyorder=5, overlap_width=58, badpixel_width=10
):
    """YG Correct WAXS Images intensities by using overlap area intensity
     Image intensity keep same for the first image
     Other image intensity is scaled by a pixel-width intensity array, which is averaged in the overlap area and then smoothed by
     scipy.signal import savgol_filter with parameters as  window_length=101, polyorder=5,

     from scipy.signal import savgol_filter as sf

     Return: data: array, stitched image with corrected intensity
            dataM: dict, each value is the image with correted intensity
            scale: scale for each image, the first scale=1 by defination
            scale_smooth: smoothed scale

    Exampe:
    data, dataM, scale,scale_smooth = Correct_Overlap_Images_Intensities( infiles, window_length=101, polyorder=5,
                                        overlap_width=58, badpixel_width =10  )

    show_img(data, logs = True,   vmin=0.8* data.min(), vmax= 1.2*data.max()
          cmap = cmap_vge_hdr,   aspect=1,  image_name = 'Merged_Sca_Img')


     fig = plt.figure()# figsize=[2,8])
     for i in range(len(infiles)):
         #print(i)
         ax = fig.add_subplot(1,8, i+1)
         d = process.load(  infiles[i]  )
         show_img( dataM[i], logs = True, show_colorbar= False,show_ticks =False,
                  ax= [fig, ax], image_name= '%02d'%(i+1), cmap = cmap_vge_hdr, vmin=100, vmax=2e3,
                 aspect=1, save=True, path=ResDir)



    """

    w = overlap_width
    ow = badpixel_width
    Nf = len(infiles)
    dataM = {}

    for i in range(len(infiles)):
        d = np.array(PIL.Image.open(infiles[i]).convert("I")).T / 1.0
        if i == 0:
            M, N = d.shape[0], d.shape[1]
            data = np.zeros([M, N * Nf - w * (Nf - 1)])
            data[:, :N] = d
            scale = np.zeros([len(infiles), M])
            scale_smooth = np.zeros([len(infiles), M])
            overlap_int = np.zeros([2 * len(infiles) - 1, M])
            overlap_int[0] = np.average(d[:, N - w : N - ow], axis=1)
            scale[0] = 1
            scale_smooth[0] = 1
            dataM[0] = d
        else:
            a1, a2, b1, b2 = N * i - w * (i - 1) - ow, N * (i + 1) - w * i, w - ow, N
            overlap_int[2 * i - 1] = np.average(d[:, 0 : w - ow], axis=1)
            overlap_int[2 * i] = np.average(d[:, N - w : N - ow], axis=1)
            scale[i] = overlap_int[2 * i - 2] / overlap_int[2 * i - 1] * scale[i - 1]
            scale_smooth[i] = sf(
                scale[i],
                window_length=window_length,
                polyorder=polyorder,
                deriv=0,
                delta=1.0,
                axis=-1,
                mode="mirror",
                cval=0.0,
            )
            data[:, a1:a2] = d[:, b1:b2] * np.repeat(scale_smooth[i], b2 - b1, axis=0).reshape([M, b2 - b1])
            dataM[i] = np.zeros_like(dataM[i - 1])
            dataM[i][:, 0 : w - ow] = dataM[i - 1][:, N - w : N - ow]
            dataM[i][:, w - ow :] = data[:, a1:a2]
    return data, dataM, scale, scale_smooth


def check_overlap_scaling_factor(scale, scale_smooth, i=1, filename=None, save=False):
    """check_overlap_scaling_factor( scale,scale_smooth, i=1 )"""
    fig, ax = plt.subplots()
    plot1D(scale[i], m="o", c="k", ax=ax, title="Scale_averaged_line_intensity_%s" % i, ls="", legend="Data")
    plot1D(
        scale_smooth[i],
        ax=ax,
        title="Scale_averaged_line_intensity_%s" % i,
        m="",
        c="r",
        ls="-",
        legend="Smoothed",
    )
    if save:
        fig.savefig(filename)


def stitch_WAXS_in_Qspace(dataM, phis, calibration, dx=0, dy=22, dz=0, dq=0.015, mask=None):
    """YG Octo 11, 2017 stitch waxs scattering images in qspace
    dataM: the data (with corrected intensity), dict format (todolist, make array also avialable)
    phis: for SMI, the rotation angle around z-aixs
    For SMI
    dx=  0  #in pixel unit
    dy = 22  #in pixel unit
    dz = 0
    calibration: class, for calibration

    Return: Intensity_map, qxs, qzs

    Example:
    phis = np.array( [get_phi(infile,
                phi_offset=4.649, phi_start=1.0, phi_spacing=5.0,) for infile in infiles]     )  # For TWD data

    calibration = CalibrationGonio(wavelength_A=0.619920987) # 20.0 keV
    #calibration.set_image_size( data.shape[1], data.shape[0] )
    calibration.set_image_size(195, height=1475) # Pilatus300kW vertical
    calibration.set_pixel_size(pixel_size_um=172.0)
    calibration.set_beam_position(97.0, 1314.0)
    calibration.set_distance(0.275)

    Intensity_map, qxs, qzs = stitch_WAXS_in_Qspace( dataM, phis, calibration)
    #Get center of the qmap
    bx,by = np.argmin( np.abs(qxs) ), np.argmin( np.abs(qzs) )
    print( bx, by )

    """

    q_range = get_qmap_range(calibration, phis[0], phis[-1])
    if q_range[0] > 0:
        q_range[0] = 0
    if q_range[2] > 0:
        q_range[2] = 0
    qx_min = q_range[0]
    qx_max = q_range[1]
    qxs = np.arange(q_range[0], q_range[1], dq)
    qzs = np.arange(q_range[2], q_range[3], dq)
    QXs, QZs = np.meshgrid(qxs, qzs)
    num_qx = len(qxs)
    qz_min = q_range[2]
    qz_max = q_range[3]
    num_qz = len(qzs)
    phi = phis[0]

    Intensity_map = np.zeros((len(qzs), len(qxs)))
    count_map = np.zeros((len(qzs), len(qxs)))
    # Intensity_mapN = np.zeros( (8, len(qzs), len(qxs)) )
    for i in range(len(phis)):
        dM = np.rot90(dataM[i].T)
        D = dM.ravel()
        phi = phis[i]
        calibration.set_angles(det_phi_g=phi, det_theta_g=0.0, offset_x=dx, offset_y=dy, offset_z=dz)
        calibration.clear_maps()
        QZ = calibration.qz_map().ravel()  # [pixel_list]
        QX = calibration.qx_map().ravel()  # [pixel_list]
        bins = [num_qz, num_qx]
        rangeq = [[qz_min, qz_max], [qx_min, qx_max]]
        # Nov 7,2017 using new func to qmap
        remesh_data, zbins, xbins = convert_Qmap(dM, QZ, QX, bins=bins, range=rangeq, mask=mask)
        # Normalize by the binning
        num_per_bin, zbins, xbins = convert_Qmap(np.ones_like(dM), QZ, QX, bins=bins, range=rangeq, mask=mask)

        # remesh_data, zbins, xbins = np.histogram2d(QZ, QX, bins=bins, range=rangeq, normed=False, weights=D)
        # Normalize by the binning
        # num_per_bin, zbins, xbins = np.histogram2d(QZ, QX, bins=bins, range=rangeq, normed=False, weights=None)
        Intensity_map += remesh_data
        count_map += num_per_bin
        # Intensity_mapN[i]     = np.nan_to_num( remesh_data/num_per_bin     )
    Intensity_map = np.nan_to_num(Intensity_map / count_map)
    return Intensity_map, qxs, qzs


def plot_qmap_in_folder(inDir):
    """YG. Sep 27@SMI
    Plot Qmap data from inDir, which contains qmap data and extent data
    """
    import pickle as cpl

    from pyCHX.chx_generic_functions import show_img
    from pyCHX.chx_libs import cmap_vge_hdr, plt

    fp = get_base_all_filenames(inDir, base_filename_cut_length=-10)
    print(
        "There will %s samples and totally %s files to be analyzed."
        % (len(fp.keys()), len(np.concatenate(list(fp.values()))))
    )
    for k in list(fp.keys()):
        for s in fp[k]:
            if "npy" in s:
                d = np.load(s)  # * qmask
            if "pkl" in s:
                xs, zs = cpl.load(open(s, "rb"))
        show_img(
            d,
            logs=False,
            show_colorbar=True,
            show_ticks=True,
            xlabel="$q_x \, (\AA^{-1})$",
            ylabel="$q_z \, (\AA^{-1})$",
            cmap=cmap_vge_hdr,  # vmin= np.min(d), vmax = np.max(d),
            aspect=1,
            vmin=-1,
            vmax=np.max(d) * 0.5,
            extent=[xs[0], xs[-1], zs[0], zs[-1]],
            image_name=k[:-1],
            path=inDir,
            save=True,
        )
        plt.close("all")


def get_qmap_range(calibration, phi_min, phi_max):
    """YG Sep 27@SMI
    Get q_range, [ qx_start, qx_end, qz_start, qz_end ] for SMI WAXS qmap
            (only rotate around z-axis, so det_theta_g=0.,actually being the y-axis for beamline conventional defination)
            based on calibration on Sep 22,  offset_x= 0,  offset_y= 22
    Input:
        calibration: class, See SciAnalysis.XSAnalysis.DataGonio.CalibrationGonio
        phi_min: min of phis
        phi_max: max of phis
    Output:
        qrange: np.array([ qx_start, qx_end, qz_start, qz_end     ])
    """
    calibration.set_angles(det_phi_g=phi_max, det_theta_g=0.0, offset_x=0, offset_y=22)
    calibration._generate_qxyz_maps()
    qx_end = np.max(calibration.qx_map_data)
    qz_start = np.min(calibration.qz_map_data)
    qz_end = np.max(calibration.qz_map_data)

    calibration.set_angles(det_phi_g=phi_min, det_theta_g=0.0, offset_x=0, offset_y=22)
    calibration._generate_qxyz_maps()
    qx_start = np.min(calibration.qx_map_data)
    return np.array([qx_start, qx_end, qz_start, qz_end])


def get_phi(filename, phi_offset=0, phi_start=4.5, phi_spacing=4.0, polarity=-1, ext="_WAXS.tif"):
    pattern_re = "^.+\/?([a-zA-Z0-9_]+_)(\d\d\d\d\d\d)(\%s)$" % ext
    # print( pattern_re )
    # pattern_re='^.+\/?([a-zA-Z0-9_]+_)(\d\d\d)(\.tif)$'
    phi_re = re.compile(pattern_re)
    phi_offset = phi_offset
    m = phi_re.match(filename)

    if m:
        idx = float(m.groups()[1])
        # print(idx)
        # phi_c = polarity*( phi_offset + phi_start + (idx-1)*phi_spacing )
        phi_c = polarity * (phi_offset + phi_start + (idx - 0) * phi_spacing)

    else:
        print("ERROR: File {} doesn't match phi_re".format(filename))
        phi_c = 0.0

    return phi_c


############For CHX beamline


def get_qmap_qxyz_range(
    calibration,
    det_theta_g_lim,
    det_phi_g_lim,
    sam_phi_lim,
    sam_theta_lim,
    sam_chi_lim,
    offset_x=0,
    offset_y=0,
    offset_z=0,
):
    """YG Nov 8, 2017@CHX
    Get q_range, [ qx_start, qx_end, qz_start, qz_end ] for SMI WAXS qmap
            (only rotate around z-axis, so det_theta_g=0.,actually being the y-axis for beamline conventional defination)
            based on calibration on Sep 22,  offset_x= 0,  offset_y= 22
    Input:
        calibration: class, See SciAnalysis.XSAnalysis.DataGonio.CalibrationGonio
        phi_min: min of phis
        phi_max: max of phis
    Output:
        qrange: np.array([ qx_start, qx_end, qz_start, qz_end     ])
    """

    i = 0
    calibration.set_angles(
        det_theta_g=det_theta_g_lim[i],
        det_phi_g=det_phi_g_lim[i],
        sam_phi=sam_phi_lim[i],
        sam_theta=sam_theta_lim[i],
        sam_chi=sam_chi_lim[i],
        offset_x=offset_x,
        offset_y=offset_y,
        offset_z=offset_z,
    )
    calibration._generate_qxyz_maps()
    qx_start = np.min(calibration.qx_map_data)
    qy_start = np.min(calibration.qy_map_data)
    qz_start = np.min(calibration.qz_map_data)

    i = 1

    calibration.set_angles(
        det_theta_g=det_theta_g_lim[i],
        det_phi_g=det_phi_g_lim[i],
        sam_phi=sam_phi_lim[i],
        sam_theta=sam_theta_lim[i],
        sam_chi=sam_chi_lim[i],
        offset_x=offset_x,
        offset_y=offset_y,
        offset_z=offset_z,
    )

    calibration._generate_qxyz_maps()
    qx_end = np.min(calibration.qx_map_data)
    qy_end = np.min(calibration.qy_map_data)
    qz_end = np.min(calibration.qz_map_data)

    return np.array([qx_start, qx_end]), np.array([qy_start, qy_end]), np.array([qz_start, qz_end])


def stitch_WAXS_in_Qspace_CHX(
    data,
    angle_dict,
    calibration,
    vary_angle="phi",
    qxlim=None,
    qylim=None,
    qzlim=None,
    det_theta_g=0,
    det_phi_g=0.0,
    sam_phi=0,
    sam_theta=0,
    sam_chi=0,
    dx=0,
    dy=0,
    dz=0,
    dq=0.0008,
):
    """YG Octo 11, 2017 stitch waxs scattering images in qspace
    dataM: the data (with corrected intensity), dict format (todolist, make array also avialable)
    phis: for SMI, the rotation angle around z-aixs
    For SMI
    dx=  0  #in pixel unit
    dy = 22  #in pixel unit
    dz = 0
    calibration: class, for calibration

    Return: Intensity_map, qxs, qzs

    Example:
    phis = np.array( [get_phi(infile,
                phi_offset=4.649, phi_start=1.0, phi_spacing=5.0,) for infile in infiles]     )  # For TWD data

    calibration = CalibrationGonio(wavelength_A=0.619920987) # 20.0 keV
    #calibration.set_image_size( data.shape[1], data.shape[0] )
    calibration.set_image_size(195, height=1475) # Pilatus300kW vertical
    calibration.set_pixel_size(pixel_size_um=172.0)
    calibration.set_beam_position(97.0, 1314.0)
    calibration.set_distance(0.275)

    Intensity_map, qxs, qzs = stitch_WAXS_in_Qspace( dataM, phis, calibration)
    #Get center of the qmap
    bx,by = np.argmin( np.abs(qxs) ), np.argmin( np.abs(qzs) )
    print( bx, by )
    """
    qx_min, qx_max = qxlim[0], qxlim[1]
    qy_min, qy_max = qylim[0], qylim[1]
    qz_min, qz_max = qzlim[0], qzlim[1]

    qxs = np.arange(qxlim[0], qxlim[1], dq)
    qys = np.arange(qylim[0], qylim[1], dq)
    qzs = np.arange(qzlim[0], qzlim[1], dq)

    QXs, QYs = np.meshgrid(qxs, qys)
    QZs, QYs = np.meshgrid(qzs, qys)
    QZs, QXs = np.meshgrid(qzs, qxs)

    num_qx = len(qxs)
    num_qy = len(qys)
    num_qz = len(qzs)

    Intensity_map_XY = np.zeros((len(qxs), len(qys)))
    count_map_XY = np.zeros((len(qxs), len(qys)))

    Intensity_map_ZY = np.zeros((len(qzs), len(qys)))
    count_map_ZY = np.zeros((len(qzs), len(qys)))

    Intensity_map_ZX = np.zeros((len(qzs), len(qxs)))
    count_map_ZX = np.zeros((len(qzs), len(qxs)))

    N = len(data)
    N = len(angle_dict[vary_angle])
    print(N)
    # Intensity_mapN = np.zeros( (8, len(qzs), len(qxs)) )
    for i in range(N):
        dM = data[i]
        D = dM.ravel()
        sam_phi = angle_dict[vary_angle][i]
        print(i, sam_phi)
        calibration.set_angles(
            det_theta_g=det_theta_g,
            det_phi_g=det_phi_g,
            sam_phi=sam_phi,
            sam_theta=sam_theta,
            sam_chi=sam_chi,
            offset_x=dx,
            offset_y=dy,
            offset_z=dz,
        )
        calibration.clear_maps()
        calibration._generate_qxyz_maps()

        QZ = calibration.qz_map_lab_data.ravel()  # [pixel_list]
        QX = calibration.qx_map_lab_data.ravel()  # [pixel_list]
        QY = calibration.qy_map_lab_data.ravel()  # [pixel_list]

        bins_xy = [num_qx, num_qy]
        bins_zy = [num_qz, num_qy]
        bins_zx = [num_qz, num_qx]

        rangeq_xy = [[qx_min, qx_max], [qy_min, qy_max]]
        rangeq_zy = [[qz_min, qz_max], [qy_min, qy_max]]
        rangeq_zx = [[qz_min, qz_max], [qx_min, qx_max]]
        print(rangeq_xy, rangeq_zy, rangeq_zx)

        remesh_dataxy, xbins, ybins = np.histogram2d(
            QX, QY, bins=bins_xy, range=rangeq_xy, normed=False, weights=D
        )
        # Normalize by the binning
        num_per_binxy, xbins, ybins = np.histogram2d(
            QX, QY, bins=bins_xy, range=rangeq_xy, normed=False, weights=None
        )
        Intensity_map_XY += remesh_dataxy
        count_map_XY += num_per_binxy

        remesh_datazy, zbins, ybins = np.histogram2d(
            QZ, QY, bins=bins_zy, range=rangeq_zy, normed=False, weights=D
        )
        # Normalize by the binning
        num_per_binzy, zbins, ybins = np.histogram2d(
            QZ, QY, bins=bins_zy, range=rangeq_zy, normed=False, weights=None
        )
        Intensity_map_ZY += remesh_datazy
        count_map_ZY += num_per_binzy

        remesh_datazx, zbins, xbins = np.histogram2d(
            QZ, QX, bins=bins_zx, range=rangeq_zx, normed=False, weights=D
        )
        # Normalize by the binning
        num_per_binzx, zbins, xbins = np.histogram2d(
            QZ, QX, bins=bins_zx, range=rangeq_zx, normed=False, weights=None
        )
        Intensity_map_ZX += remesh_datazx
        count_map_ZX += num_per_binzx

        # Intensity_mapN[i]     = np.nan_to_num( remesh_data/num_per_bin     )
    Intensity_map_XY = np.nan_to_num(Intensity_map_XY / count_map_XY)
    Intensity_map_ZY = np.nan_to_num(Intensity_map_ZY / count_map_ZY)
    Intensity_map_ZX = np.nan_to_num(Intensity_map_ZX / count_map_ZX)

    return Intensity_map_XY, Intensity_map_ZY, Intensity_map_ZX, qxs, qys, qzs
