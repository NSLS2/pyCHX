### This enables local import of pyCHX for testing

import pickle as cpk

import historydict

# from pyCHX.chx_handlers import use_dask, use_pims
from chx_handlers import use_dask, use_pims

# from pyCHX.chx_libs import (
from chx_libs import (
    EigerHandler,
    Javascript,
    LogNorm,
    Model,
    cmap_albula,
    cmap_vge,
    datetime,
    db,
    getpass,
    h5py,
    multi_tau_lags,
    np,
    os,
    pims,
    plt,
    random,
    roi,
    time,
    tqdm,
    utils,
    warnings,
)
from eiger_io.fs_handler import EigerImages
from skimage.draw import line, line_aa, polygon

# changes to current version of chx_packages.py
# added load_dask_data in generic_functions


use_pims(db)  # use pims for importing eiger data, register_handler 'AD_EIGER2' and 'AD_EIGER'

# from pyCHX.chx_compress import (
from chx_compress import (
    MultifileBNLCustom,
    combine_binary_files,
    compress_eigerdata,
    create_compress_header,
    get_eigerImage_per_file,
    init_compress_eigerdata,
    para_compress_eigerdata,
    para_segment_compress_eigerdata,
    read_compressed_eigerdata,
    segment_compress_eigerdata,
)

# from pyCHX.chx_compress_analysis import (
from chx_compress_analysis import (
    Multifile,
    cal_each_ring_mean_intensityc,
    cal_waterfallc,
    compress_eigerdata,
    get_avg_imgc,
    get_each_frame_intensityc,
    get_each_ring_mean_intensityc,
    get_time_edge_avg_img,
    mean_intensityc,
    plot_each_ring_mean_intensityc,
    plot_waterfallc,
    read_compressed_eigerdata,
)

# from pyCHX.chx_correlationc import Get_Pixel_Arrayc, auto_two_Arrayc, cal_g2c, get_pixelist_interp_iq
from chx_correlationc import Get_Pixel_Arrayc, auto_two_Arrayc, cal_g2c, get_pixelist_interp_iq

# from pyCHX.chx_correlationp import _one_time_process_errorp, auto_two_Arrayp, cal_g2p, cal_GPF, get_g2_from_ROI_GPF
from chx_correlationp import _one_time_process_errorp, auto_two_Arrayp, cal_g2p, cal_GPF, get_g2_from_ROI_GPF

# from pyCHX.chx_crosscor import CrossCorrelator2, run_para_ccorr_sym
from chx_crosscor import CrossCorrelator2, run_para_ccorr_sym

# from pyCHX.chx_generic_functions import (
from chx_generic_functions import (
    R_2,
    RemoveHot,
    apply_mask,
    average_array_withNan,
    check_bad_uids,
    check_lost_metadata,
    check_ROI_intensity,
    check_shutter_open,
    combine_images,
    copy_data,
    create_cross_mask,
    create_fullImg_with_box,
    create_hot_pixel_mask,
    create_multi_rotated_rectangle_mask,
    create_polygon_mask,
    create_rectangle_mask,
    create_ring_mask,
    create_seg_ring,
    create_time_slice,
    create_user_folder,
    delete_data,
    extract_data_from_file,
    filter_roi_mask,
    find_bad_pixels,
    find_bad_pixels_FD,
    find_good_xpcs_uids,
    find_index,
    find_uids,
    fit_one_peak_curve,
    get_averaged_data_from_multi_res,
    get_avg_img,
    get_bad_frame_list,
    get_base_all_filenames,
    get_cross_point,
    get_current_pipeline_filename,
    get_current_pipeline_fullpath,
    get_curve_turning_points,
    get_detector,
    get_detectors,
    get_each_frame_intensity,
    get_echos,
    get_eigerImage_per_file,
    get_fit_by_two_linear,
    get_fra_num_by_dose,
    get_g2_fit_general,
    get_image_edge,
    get_image_with_roi,
    get_img_from_iq,
    get_last_uids,
    get_mass_center_one_roi,
    get_max_countc,
    get_meta_data,
    get_multi_tau_lag_steps,
    get_non_uniform_edges,
    get_print_uids,
    get_q_rate_fit_general,
    get_qval_dict,
    get_qval_qwid_dict,
    get_roi_mask_qval_qwid_by_shift,
    get_roi_nr,
    get_series_g2_taus,
    get_SG_norm,
    get_sid_filenames,
    get_sid_filenames_v2,
    get_sid_filenames_v3,
    get_today_date,
    get_touched_qwidth,
    get_waxs_beam_center,
    lin2log_g2,
    linear_fit,
    load_dask_data,
    load_data,
    load_mask,
    load_pilatus,
    ls_dir,
    mask_badpixels,
    mask_exclude_badpixel,
    move_beamstop,
    pad_length,
    pload_obj,
    plot1D,
    plot_fit_two_linear_fit,
    plot_g2_general,
    plot_q_g2fitpara_general,
    plot_q_rate_fit_general,
    plot_q_rate_general,
    plot_xy_with_fit,
    plot_xy_x2,
    print_dict,
    psave_obj,
    read_dict_csv,
    refine_roi_mask,
    reverse_updown,
    ring_edges,
    run_time,
    save_array_to_tiff,
    save_arrays,
    save_current_pipeline,
    save_dict_csv,
    save_g2_fit_para_tocsv,
    save_g2_general,
    save_lists,
    save_oavs_tifs,
    save_oavs_tifs_v2,
    sgolay2d,
    shift_mask,
    show_img,
    show_ROI_on_image,
    shrink_image,
    trans_data_to_pd,
    update_qval_dict,
    update_roi_mask,
    validate_uid,
)

# from pyCHX.chx_olog import Attachment, LogEntry, update_olog_id, update_olog_uid, update_olog_uid_with_file
# from chx_olog import Attachment, LogEntry, update_olog_id, update_olog_uid, update_olog_uid_with_file

# from pyCHX.chx_outlier_detection import (
from chx_outlier_detection import is_outlier, outlier_mask

# from pyCHX.chx_specklecp import (
from chx_specklecp import (
    get_binned_his_std,
    get_contrast,
    get_his_std_from_pds,
    get_xsvs_fit,
    plot_g2_contrast,
    plot_xsvs_fit,
    save_bin_his_std,
    save_KM,
    xsvsc,
    xsvsp,
)

# from pyCH.chx_xpcs_xsvs_jupyter_V1 import(
from chx_xpcs_xsvs_jupyter_V1 import (
    compress_multi_uids,
    do_compress_on_line,
    get_fra_num_by_dose,
    get_iq_from_uids,
    get_series_g2_from_g12,
    get_series_one_time_mulit_uids,
    get_t_iqc_uids,
    get_two_time_mulit_uids,
    get_uids_by_range,
    get_uids_in_time_period,
    plot_dose_g2,
    plot_entries_from_csvlist,
    plot_entries_from_uids,
    plot_t_iqc_uids,
    plot_t_iqtMq2,
    realtime_xpcs_analysis,
    run_xpcs_xsvs_single,
    wait_data_acquistion_finish,
    wait_func,
)

# from pyCHX.Create_Report import (
from Create_Report import (
    create_multi_pdf_reports_for_uids,
    create_one_pdf_reports_for_uids,
    create_pdf_report,
    export_xpcs_results_to_h5,
    extract_xpcs_results_from_h5,
    make_pdf_report,
)

# from pyCHX.DataGonio import qphiavg
from DataGonio import qphiavg

# from pyCHX.SAXS import (
from SAXS import (
    fit_form_factor,
    fit_form_factor2,
    form_factor_residuals_bg_lmfit,
    form_factor_residuals_lmfit,
    get_form_factor_fit_lmfit,
    poly_sphere_form_factor_intensity,
    show_saxs_qmap,
)

# from pyCHX.Two_Time_Correlation_Function import (
from Two_Time_Correlation_Function import (
    get_aged_g2_from_g12,
    get_aged_g2_from_g12q,
    get_four_time_from_two_time,
    get_one_time_from_two_time,
    rotate_g12q_to_rectangle,
    show_C12,
)

# from pyCHX.XPCS_GiSAXS import (
from XPCS_GiSAXS import (
    cal_1d_qr,
    convert_gisaxs_pixel_to_q,
    fit_qr_qz_rate,
    get_1d_qr,
    get_each_box_mean_intensity,
    get_gisaxs_roi,
    get_qedge,
    get_qmap_label,
    get_qr_tick_label,
    get_qzr_map,
    get_qzrmap,
    get_reflected_angles,
    get_t_qrc,
    multi_uids_gisaxs_xpcs_analysis,
    plot_gisaxs_g4,
    plot_gisaxs_two_g2,
    plot_qr_1d_with_ROI,
    plot_qrt_pds,
    plot_qzr_map,
    plot_t_qrc,
    show_qzr_map,
    show_qzr_roi,
)

# from pyCHX.XPCS_SAXS import (
from XPCS_SAXS import (
    cal_g2,
    combine_two_roi_mask,
    create_hot_pixel_mask,
    get_angular_mask,
    get_circular_average,
    get_cirucular_average_std,
    get_each_ring_mean_intensity,
    get_QrQw_From_RoiMask,
    get_ring_mask,
    get_seg_from_ring_mask,
    get_t_iq,
    get_t_iqc,
    get_t_iqc_imstack,
    multi_uids_saxs_xpcs_analysis,
    plot_circular_average,
    plot_qIq_with_ROI,
    plot_t_iqc,
    recover_img_from_iq,
    save_lists,
)
