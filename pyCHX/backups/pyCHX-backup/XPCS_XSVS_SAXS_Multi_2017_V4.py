# python XPCS_XSVS_SAXS_Multi_2017_V4.py


from pyCHX.chx_packages import *
from pyCHX.chx_xpcs_xsvs_jupyter import run_xpcs_xsvs_single


def XPCS_XSVS_SAXS_Multi(
    start_time,
    stop_time,
    run_pargs,
    suf_ids=None,
    uid_average="Au50_7p5PEGX1_vs_slow_120116",
):
    scat_geometry = run_pargs["scat_geometry"]
    force_compress = run_pargs["force_compress"]
    para_compress = run_pargs["para_compress"]
    run_fit_form = run_pargs["run_fit_form"]
    run_waterfall = run_pargs["run_waterfall"]
    run_t_ROI_Inten = run_pargs["run_t_ROI_Inten"]
    # run_fit_g2 = run_pargs['run_fit_g2'],
    fit_g2_func = run_pargs["fit_g2_func"]
    run_one_time = run_pargs["run_one_time"]
    run_two_time = run_pargs["run_two_time"]
    run_four_time = run_pargs["run_four_time"]
    run_xsvs = run_pargs["run_xsvs"]
    ###############################################################
    if scat_geometry != "saxs":  # to be done for other types
        run_xsvs = False
    ###############################################################
    att_pdf_report = run_pargs["att_pdf_report"]
    show_plot = run_pargs["show_plot"]
    CYCLE = run_pargs["CYCLE"]
    mask_path = run_pargs["mask_path"]
    mask_name = run_pargs["mask_name"]
    good_start = run_pargs["good_start"]
    use_imgsum_norm = run_pargs["use_imgsum_norm"]

    mask = load_mask(mask_path, mask_name, plot_=False, image_name="%s_mask" % mask_name, reverse=True)
    # mask *= pixel_mask
    mask[:, 2069] = 0  # False  #Concluded from the previous results
    # np.save(  data_dir + 'mask', mask)
    show_img(mask, image_name="%s_mask" % uid_average, save=True, path=data_dir)
    mask_load = mask.copy()

    username = getpass.getuser()
    data_dir0 = os.path.join("/XF11ID/analysis/", run_pargs["CYCLE"], username, "Results/")
    os.makedirs(data_dir0, exist_ok=True)
    print("Results from this analysis will be stashed in the directory %s" % data_dir0)
    data_dir = os.path.join(data_dir0, uid_average + "/")
    os.makedirs(data_dir, exist_ok=True)
    uid_average = "uid=" + uid_average

    if suf_ids is None:
        sids, uids, fuids = find_uids(start_time, stop_time)
    else:
        sids, uids, fuids = suf_ids
    print(uids)
    uid = uids[0]

    data_dir_ = data_dir
    uid_ = uid_average
    ### For Load results

    multi_res = {}
    for uid, fuid in zip(guids, fuids):
        multi_res[uid] = extract_xpcs_results_from_h5(
            filename="uid=%s_Res.h5" % fuid, import_dir=data_dir0 + uid + "/"
        )
    # Get and Plot Averaged Data

    mkeys = list(multi_res.keys())
    uid = uid_average
    setup_pargs["uid"] = uid
    avg_img = get_averaged_data_from_multi_res(multi_res, keystr="avg_img")
    imgsum = get_averaged_data_from_multi_res(multi_res, keystr="imgsum")
    if scat_geometry == "saxs":
        q_saxs = get_averaged_data_from_multi_res(multi_res, keystr="q_saxs")
        iq_saxs = get_averaged_data_from_multi_res(multi_res, keystr="iq_saxs")
        qt = get_averaged_data_from_multi_res(multi_res, keystr="qt")
        iqst = get_averaged_data_from_multi_res(multi_res, keystr="iqst")
    elif scat_geometry == "gi_saxs":
        qr_1d_pds = get_averaged_data_from_multi_res(multi_res, keystr="qr_1d_pds")
        qr_1d_pds = trans_data_to_pd(qr_1d_pds, label=qr_1d_pds_label)
    if run_waterfall:
        wat = get_averaged_data_from_multi_res(multi_res, keystr="wat")
    if run_t_ROI_Inten:
        times_roi = get_averaged_data_from_multi_res(multi_res, keystr="times_roi")
        mean_int_sets = get_averaged_data_from_multi_res(multi_res, keystr="mean_int_sets")

    if run_one_time:
        g2 = get_averaged_data_from_multi_res(multi_res, keystr="g2")
        taus = get_averaged_data_from_multi_res(multi_res, keystr="taus")
        g2_pds = save_g2_general(
            g2,
            taus=taus,
            qr=np.array(list(qval_dict.values()))[:, 0],
            uid=uid + "_g2.csv",
            path=data_dir,
            return_res=True,
        )
        g2_fit_result, taus_fit, g2_fit = get_g2_fit_general(
            g2,
            taus,
            function=fit_g2_func,
            vlim=[0.95, 1.05],
            fit_range=None,
            fit_variables={"baseline": True, "beta": True, "alpha": False, "relaxation_rate": True},
            guess_values={
                "baseline": 1.0,
                "beta": 0.05,
                "alpha": 1.0,
                "relaxation_rate": 0.01,
            },
        )
        g2_fit_paras = save_g2_fit_para_tocsv(g2_fit_result, filename=uid + "_g2_fit_paras.csv", path=data_dir)

    if run_two_time:
        g12b = get_averaged_data_from_multi_res(multi_res, keystr="g12b", different_length=True)
        g2b = get_averaged_data_from_multi_res(multi_res, keystr="g2b")
        tausb = get_averaged_data_from_multi_res(multi_res, keystr="tausb")

        g2b_pds = save_g2_general(
            g2b,
            taus=tausb,
            qr=np.array(list(qval_dict.values()))[:, 0],
            qz=None,
            uid=uid + "_g2b.csv",
            path=data_dir,
            return_res=True,
        )
        g2_fit_resultb, taus_fitb, g2_fitb = get_g2_fit_general(
            g2b,
            tausb,
            function=fit_g2_func,
            vlim=[0.95, 1.05],
            fit_range=None,
            fit_variables={"baseline": True, "beta": True, "alpha": False, "relaxation_rate": True},
            guess_values={
                "baseline": 1.0,
                "beta": 0.05,
                "alpha": 1.0,
                "relaxation_rate": 0.01,
            },
        )

        g2b_fit_paras = save_g2_fit_para_tocsv(g2_fit_resultb, filename=uid + "_g2b_fit_paras.csv", path=data_dir)

    if run_four_time:
        g4 = get_averaged_data_from_multi_res(multi_res, keystr="g4")
        taus4 = get_averaged_data_from_multi_res(multi_res, keystr="taus4")
        g4_pds = save_g2_general(
            g4,
            taus=taus4,
            qr=np.array(list(qval_dict.values()))[:, 0],
            qz=None,
            uid=uid + "_g4.csv",
            path=data_dir,
            return_res=True,
        )

    if run_xsvs:
        contrast_factorL = get_averaged_data_from_multi_res(
            multi_res, keystr="contrast_factorL", different_length=False
        )
        times_xsvs = get_averaged_data_from_multi_res(multi_res, keystr="times_xsvs", different_length=False)
        cont_pds = save_arrays(
            contrast_factorL,
            label=times_xsvs,
            filename="%s_contrast_factorL.csv" % uid,
            path=data_dir,
            return_res=True,
        )
        if False:
            spec_kmean = get_averaged_data_from_multi_res(multi_res, keystr="spec_kmean")
            spec_pds = get_averaged_data_from_multi_res(multi_res, keystr="spec_pds", different_length=False)
            times_xsvs = get_averaged_data_from_multi_res(multi_res, keystr="times_xsvs", different_length=False)
            spec_his, spec_std = get_his_std_from_pds(spec_pds, his_shapes=None)
            ML_val, KL_val, K_ = get_xsvs_fit(
                spec_his,
                spec_kmean,
                spec_std,
                max_bins=2,
                varyK=False,
            )
            contrast_factorL = get_contrast(ML_val)
            spec_km_pds = save_KM(
                spec_kmean,
                KL_val,
                ML_val,
                qs=q_ring_center,
                level_time=times_xsvs,
                uid=uid_average,
                path=data_dir_average,
            )
            plot_xsvs_fit(
                spec_his,
                ML_val,
                KL_val,
                K_mean=spec_kmean,
                spec_std=spec_std,
                xlim=[0, 15],
                vlim=[0.9, 1.1],
                uid=uid_average,
                qth=None,
                logy=True,
                times=times_xsvs,
                q_ring_center=q_ring_center,
                path=data_dir,
            )

    if scat_geometry == "saxs":
        show_saxs_qmap(
            avg_img,
            setup_pargs,
            width=600,
            vmin=0.1,
            vmax=np.max(avg_img * 0.1),
            logs=True,
            image_name="%s_img_avg" % uid,
            save=True,
        )
        plot_circular_average(
            q_saxs,
            iq_saxs,
            q_saxs,
            pargs=setup_pargs,
            xlim=[q_saxs.min(), q_saxs.max()],
            ylim=[iq_saxs.min(), iq_saxs.max()],
        )
        plot_qIq_with_ROI(
            q_saxs,
            iq_saxs,
            qr,
            logs=True,
            uid=uid,
            xlim=[q_saxs.min(), q_saxs.max()],
            ylim=[iq_saxs.min(), iq_saxs.max()],
            save=True,
            path=data_dir,
        )
        plot1D(
            y=imgsum,
            title="%s_img_sum_t" % uid,
            xlabel="Frame",
            colors="b",
            ylabel="Total_Intensity",
            legend="imgsum",
            save=True,
            path=data_dir,
        )
        plot_t_iqc(
            qt,
            iqst,
            frame_edge=None,
            pargs=setup_pargs,
            xlim=[qt.min(), qt.max()],
            ylim=[iqst.min(), iqst.max()],
            save=True,
        )
        show_ROI_on_image(
            avg_img,
            roi_mask,
            center,
            label_on=False,
            rwidth=700,
            alpha=0.9,
            save=True,
            path=data_dir,
            uid=uid,
            vmin=np.min(avg_img),
            vmax=np.max(avg_img),
        )

    elif scat_geometry == "gi_saxs":
        show_img(
            avg_img,
            vmin=0.1,
            vmax=np.max(avg_img * 0.1),
            logs=True,
            image_name=uidstr + "_img_avg",
            save=True,
            path=data_dir,
        )
        plot_qr_1d_with_ROI(
            qr_1d_pds,
            qr_center=np.unique(np.array(list(qval_dict.values()))[:, 0]),
            loglog=False,
            save=True,
            uid=uidstr,
            path=data_dir,
        )
        show_qzr_roi(avg_img, roi_mask, inc_x0, ticks, alpha=0.5, save=True, path=data_dir, uid=uidstr)

    if run_waterfall:
        plot_waterfallc(
            wat, qth_interest, aspect=None, vmax=np.max(wat), uid=uid, save=True, path=data_dir, beg=good_start
        )
    if run_t_ROI_Inten:
        plot_each_ring_mean_intensityc(times_roi, mean_int_sets, uid=uid, save=True, path=data_dir)

    if run_one_time:
        plot_g2_general(
            g2_dict={1: g2, 2: g2_fit},
            taus_dict={1: taus, 2: taus_fit},
            vlim=[0.95, 1.05],
            qval_dict=qval_dict,
            fit_res=g2_fit_result,
            geometry=scat_geometry,
            filename=uid + "_g2",
            path=data_dir,
            function=fit_g2_func,
            ylabel="g2",
            append_name="_fit",
        )

        D0, qrate_fit_res = get_q_rate_fit_general(
            qval_dict, g2_fit_paras["relaxation_rate"], geometry=scat_geometry
        )
        plot_q_rate_fit_general(
            qval_dict,
            g2_fit_paras["relaxation_rate"],
            qrate_fit_res,
            geometry=scat_geometry,
            uid=uid,
            path=data_dir,
        )

    if run_two_time:
        show_C12(
            g12b,
            q_ind=qth_interest,
            N1=0,
            N2=min(len(imgsa), 1000),
            vmin=1.01,
            vmax=1.25,
            timeperframe=timeperframe,
            save=True,
            path=data_dir,
            uid=uid,
        )
        plot_g2_general(
            g2_dict={1: g2b, 2: g2_fitb},
            taus_dict={1: tausb, 2: taus_fitb},
            vlim=[0.95, 1.05],
            qval_dict=qval_dict,
            fit_res=g2_fit_resultb,
            geometry=scat_geometry,
            filename=uid + "_g2",
            path=data_dir,
            function=fit_g2_func,
            ylabel="g2",
            append_name="_b_fit",
        )

    if run_two_time and run_one_time:
        plot_g2_general(
            g2_dict={1: g2, 2: g2b},
            taus_dict={1: taus, 2: tausb},
            vlim=[0.95, 1.05],
            qval_dict=qval_dict,
            g2_labels=["from_one_time", "from_two_time"],
            geometry=scat_geometry,
            filename=uid + "_g2_two_g2",
            path=data_dir,
            ylabel="g2",
        )
    if run_four_time:
        plot_g2_general(
            g2_dict={1: g4},
            taus_dict={1: taus4},
            vlim=[0.95, 1.05],
            qval_dict=qval_dict,
            fit_res=None,
            geometry=scat_geometry,
            filename=uid + "_g4",
            path=data_dir,
            ylabel="g4",
        )

    if run_xsvs:
        plot_g2_contrast(
            contrast_factorL,
            g2,
            times_xsvs,
            taus,
            qr,
            vlim=[0.8, 2.0],
            qth=qth_interest,
            uid=uid,
            path=data_dir,
            legend_size=14,
        )
        plot_g2_contrast(
            contrast_factorL,
            g2,
            times_xsvs,
            taus,
            qr,
            vlim=[0.8, 1.2],
            qth=None,
            uid=uid,
            path=data_dir,
            legend_size=4,
        )

    md = multi_res[mkeys[0]]["md"]
    md["uid"] = uid
    md["suid"] = uid
    md["Measurement"] = uid
    md["beg"] = None
    md["end"] = None
    md["bad_frame_list"] = "unknown"
    md["metadata_file"] = data_dir + "md.csv-&-md.pkl"
    psave_obj(md, data_dir + "%s_md" % uid)  # save the setup parameters
    save_dict_csv(md, data_dir + "%s_md.csv" % uid, "w")

    Exdt = {}
    if scat_geometry == "gi_saxs":
        for k, v in zip(
            ["md", "roi_mask", "qval_dict", "avg_img", "mask", "pixel_mask", "imgsum", "qr_1d_pds"],
            [md, roi_mask, qval_dict, avg_img, mask, pixel_mask, imgsum, qr_1d_pds],
        ):
            Exdt[k] = v
    elif scat_geometry == "saxs":
        for k, v in zip(
            [
                "md",
                "q_saxs",
                "iq_saxs",
                "iqst",
                "qt",
                "roi_mask",
                "qval_dict",
                "avg_img",
                "mask",
                "pixel_mask",
                "imgsum",
                "bad_frame_list",
            ],
            [
                md,
                q_saxs,
                iq_saxs,
                iqst,
                qt,
                roi_mask,
                qval_dict,
                avg_img,
                mask,
                pixel_mask,
                imgsum,
            ],
        ):
            Exdt[k] = v

    if run_waterfall:
        Exdt["wat"] = wat
    if run_t_ROI_Inten:
        Exdt["times_roi"] = times_roi
        Exdt["mean_int_sets"] = mean_int_sets
    if run_one_time:
        for k, v in zip(["taus", "g2", "g2_fit_paras"], [taus, g2, g2_fit_paras]):
            Exdt[k] = v
    if run_two_time:
        for k, v in zip(["tausb", "g2b", "g2b_fit_paras", "g12b"], [tausb, g2b, g2b_fit_paras, g12b]):
            Exdt[k] = v
    if run_four_time:
        for k, v in zip(["taus4", "g4"], [taus4, g4]):
            Exdt[k] = v
    if run_xsvs:
        for k, v in zip(
            ["spec_kmean", "spec_pds", "times_xsvs", "spec_km_pds", "contrast_factorL"],
            [spec_kmean, spec_pds, times_xsvs, spec_km_pds, contrast_factorL],
        ):
            Exdt[k] = v

        contr_pds = save_arrays(
            Exdt["contrast_factorL"],
            label=Exdt["times_xsvs"],
            filename="%s_contr.csv" % uid,
            path=data_dir,
            return_res=True,
        )

    export_xpcs_results_to_h5(uid + "_Res.h5", data_dir, export_dict=Exdt)
    # extract_dict = extract_xpcs_results_from_h5( filename = uid + '_Res.h5', import_dir = data_dir )
    ## Create PDF report for each uid
    pdf_out_dir = data_dir
    pdf_filename = "XPCS_Analysis_Report_for_%s%s.pdf" % (uid_average, pdf_version)
    if run_xsvs:
        pdf_filename = "XPCS_XSVS_Analysis_Report_for_%s%s.pdf" % (uid_average, pdf_version)

    # pdf_filename = "XPCS_XSVS_Analysis_Report_for_uid=%s%s.pdf"%(uid_average,'_2')
    make_pdf_report(
        data_dir,
        uid_average,
        pdf_out_dir,
        pdf_filename,
        username,
        run_fit_form,
        run_one_time,
        run_two_time,
        run_four_time,
        run_xsvs,
        report_type=scat_geometry,
    )
    ### Attach each g2 result to the corresponding olog entry
    if att_pdf_report:
        os.environ["HTTPS_PROXY"] = "https://proxy:8888"
        os.environ["no_proxy"] = "cs.nsls2.local,localhost,127.0.0.1"
        pname = pdf_out_dir + pdf_filename
        atch = [Attachment(open(pname, "rb"))]
        try:
            update_olog_uid(uid=fuids[-1], text="Add XPCS Averaged Analysis PDF Report", attachments=atch)
        except:
            print(
                "I can't attach this PDF: %s due to a duplicated filename. Please give a different PDF file."
                % pname
            )

    print(fuids[-1])

    # The End!


if False:
    start_time, stop_time = "2016-12-1  16:30:00", "2016-12-1  16:31:50"  # for 10 nm, 20, for test purpose
    suf_ids = find_uids(start_time, stop_time)
    sp = "test"
    uid_averages = [sp + "_vs_test1_120116", sp + "_vs_test2_120116", sp + "_vs_test3_120116"]

    run_pargs = dict(
        scat_geometry="saxs",
        # scat_geometry = 'gi_saxs',
        force_compress=False,  # True, #False, #True,#False,
        para_compress=True,
        run_fit_form=False,
        run_waterfall=True,  # False,
        run_t_ROI_Inten=True,
        # run_fit_g2 = True,
        fit_g2_func="stretched",
        run_one_time=True,  # False,
        run_two_time=True,  # False,
        run_four_time=False,  # True, #False,
        run_xsvs=True,
        att_pdf_report=True,
        show_plot=False,
        CYCLE="2016_3",
        # if scat_geometry == 'gi_saxs':
        # mask_path = '/XF11ID/analysis/2016_3/masks/',
        # mask_name =  'Nov16_4M-GiSAXS_mask.npy',
        # elif scat_geometry == 'saxs':
        mask_path="/XF11ID/analysis/2016_3/masks/",
        mask_name="Nov28_4M_SAXS_mask.npy",
        good_start=5,
        #####################################for saxs
        uniformq=True,
        inner_radius=0.005,  # 0.005 for 50 nmAu/SiO2, 0.006, #for 10nm/coralpor
        outer_radius=0.04,  # 0.04 for 50 nmAu/SiO2, 0.05, #for 10nm/coralpor
        num_rings=12,
        gap_ring_number=6,
        number_rings=1,
        ############################for gi_saxs
        # inc_x0 = 1473,
        # inc_y0 = 372,
        # refl_x0 = 1473,
        # refl_y0 = 730,
        qz_start=0.025,
        qz_end=0.04,
        qz_num=3,
        gap_qz_num=1,
        # qz_width = ( qz_end - qz_start)/(qz_num +1),
        qr_start=0.0025,
        qr_end=0.07,
        qr_num=14,
        gap_qr_num=5,
        definde_second_roi=True,
        qz_start2=0.04,
        qz_end2=0.050,
        qz_num2=1,
        gap_qz_num2=1,
        qr_start2=0.002,
        qr_end2=0.064,
        qr_num2=10,
        gap_qr_num2=5,
        # qcenters = [ 0.00235,0.00379,0.00508,0.00636,0.00773, 0.00902] #in A-1
        # width = 0.0002
        qth_interest=1,  # the intested single qth
        use_sqnorm=False,
        use_imgsum_norm=True,
        pdf_version="_1",  # for pdf report name
    )

    step = 1
    Nt = len(uid_averages)
    for i in range(Nt):
        t0 = time.time()
        suf_idsi = (
            suf_ids[0][i * step : (i + 1) * step],
            suf_ids[1][i * step : (i + 1) * step],
            suf_ids[2][i * step : (i + 1) * step],
        )
        XPCS_XSVS_SAXS_Multi(0, 0, run_pargs=run_pargs, suf_ids=suf_idsi, uid_average=uid_averages[i])

    run_time(t0)
