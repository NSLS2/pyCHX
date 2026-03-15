import pandas as pds

from pyCHX.chx_libs import colors, markers
from pyCHX.chx_packages import *

# from pyCHX.chx_generic_functions import get_short_long_labels_from_qval_dict
# RUN_GUI = False
# from pyCHX.chx_libs import markers


# from IPython import get_ipython
# ip = get_ipython()
# ip.run_line_magic(
#     "run", "/nsls2/data/chx/shared/CHX_Software/packages/environment_management/chx_analysis_setup.ipynb"
# )


def get_t_iqc_uids(uid_list, setup_pargs, slice_num=10, slice_width=1):
    """Get Iq at different time edge (difined by slice_num and slice_width) for a list of uids
    Input:
        uid_list: list of string (uid)
        setup_pargs: dict, for caculation of Iq, the key of this dict should include
                        'center': beam center
                        'dpix': pixel size
                        'lambda_': X-ray wavelength
        slice_num: slice number of the time edge
        slice_edge: the width of the time_edge
    Output:
        qs: dict, with uid as key, with value as q values
        iqsts:dict, with uid as key, with value as iq values
        tstamp:dict, with uid as key, with value as time values

    """
    iqsts = {}
    tstamp = {}
    qs = {}
    label = []
    for uid in uid_list:
        md = get_meta_data(uid)
        luid = md["uid"]
        timeperframe = md["cam_acquire_period"]
        N = md["cam_num_images"]
        filename = "/XF11ID/analysis/Compressed_Data" + "/uid_%s.cmp" % luid
        good_start = 5
        FD = Multifile(filename, good_start, N)
        Nimg = FD.end - FD.beg
        time_edge = create_time_slice(Nimg, slice_num=slice_num, slice_width=slice_width, edges=None)
        time_edge = np.array(time_edge) + good_start
        # print( time_edge )
        tstamp[uid] = time_edge[:, 0] * timeperframe
        qpt, iqsts[uid], qt = get_t_iqc(FD, time_edge, None, pargs=setup_pargs, nx=1500)
        qs[uid] = qt

    return qs, iqsts, tstamp


def plot_t_iqtMq2(qt, iqst, tstamp, ax=None, perf=""):
    """plot q2~Iq at differnt time"""
    if ax is None:
        fig, ax = plt.subplots()
    q = qt
    for i in range(iqst.shape[0]):
        yi = iqst[i] * q**2
        time_labeli = perf + "time_%s s" % (round(tstamp[i], 3))
        plot1D(
            x=q,
            y=yi,
            legend=time_labeli,
            xlabel="Q (A-1)",
            ylabel="I(q)*Q^2",
            title="I(q)*Q^2 ~ time",
            m=markers[i],
            c=colors[i],
            ax=ax,
            ylim=[-0.001, 0.005],
        )  # , xlim=[0.007,0.1] )


def plot_t_iqc_uids(qs, iqsts, tstamps):
    """plot q2~Iq at differnt time for a uid list"""
    keys = list(qs.keys())
    fig, ax = plt.subplots()
    for uid in keys:
        qt = qs[uid]
        iqst = iqsts[uid]
        tstamp = tstamps[uid]
        plot_t_iqtMq2(qt, iqst, tstamp, ax=ax, perf=uid + "_")


def plot_entries_from_csvlist(
    csv_list,
    uid_list,
    inDir,
    key="g2",
    qth=1,
    legend_size=8,
    yshift=0.01,
    ymulti=1,
    xlim=None,
    ylim=None,
    uid_length=None,
    legend=None,
    fp_fulluid=True,
):
    """
    YG Feb2, 2018, make yshift be also a list

    YG June 9, 2017@CHX
     YG Sep 29, 2017@CHX.
    plot enteries for a list csvs
    Input:
        csv_list: list, a list of uid (string)
        inDir: string, imported folder for saved analysis results
        key: string, plot entry, surport
            'g2' for one-time,
            'iq' for q~iq
            'mean_int_sets' for mean intensity of each roi as a function of frame
            TODOLIST:#also can plot the following
                dict_keys(['qt', 'imgsum', 'qval_dict_v', 'bad_frame_list', 'iqst',
                'times_roi', 'iq_saxs', 'g2', 'mask', 'g2_uids', 'taus_uids',
                'g2_fit_paras', 'mean_int_sets', 'roi_mask', 'qval_dict', 'taus',
                'pixel_mask', 'avg_img', 'qval_dict_p', 'q_saxs', 'md'])
        qth: integer, the intesrest q number
        yshift: float, values of shift in y direction
        xlim: [x1,x2], for plot x limit
        ylim: [y1,y2], for plot y limit
    Output:
        show the plot
    Example:
    uid_list = ['5492b9', '54c5e0']
    plot_entries_from_uids( uid_list, inDir, yshift = 0.01, key= 'g2', ylim=[1, 1.2])
    """

    uid_dict = {}
    fig, ax = plt.subplots()
    for uid in uid_list:
        if uid_length is not None:
            uid_ = uid[:uid_length]
        else:
            uid_ = uid
        # print(uid_)
        uid_dict[uid_] = get_meta_data(uid)["uid"]
    # for i, u in enumerate( list( uid_dict.keys() )):

    for i, fp in enumerate(list(csv_list)):
        u = uid_list[i]  # print(u)
        inDiru = inDir + u + "/"
        if fp_fulluid:
            inDiru = inDir + uid_dict[u] + "/"
        else:
            inDiru = inDir + u + "/"
        d = pds.read_csv(inDiru + fp)
        # print(d)

        if key == "g2":
            taus = d["tau"][1:]
            col = d.columns[qth + 1]
            # print( qth+1, col )
            y = d[col][1:]
            if legend is None:
                leg = u
            else:
                leg = "uid=%s-->" % u + legend[i]
            if isinstance(yshift, list):
                yshift_ = yshift[i]
                ii = i + 1
            else:
                yshift_ = yshift
                ii = i
            plot1D(
                x=taus,
                y=y + yshift_ * ii,
                c=colors[i],
                m=markers[i],
                ax=ax,
                logx=True,
                legend=leg,
                xlabel="t (sec)",
                ylabel="g2",
                legend_size=legend_size,
            )
            title = "Q = %s" % (col)
            ax.set_title(title)
        elif key == "imgsum":
            y = total_res[key]
            plot1D(
                y=d + yshift_ * ii,
                c=colors[i],
                m=markers[i],
                ax=ax,
                logx=False,
                legend=u,
                xlabel="Frame",
                ylabel="imgsum",
            )

        elif key == "iq":
            x = total_res["q_saxs"]
            y = total_res["iq_saxs"]
            plot1D(
                x=x,
                y=y * ymulti[i] + yshift_ * ii,
                c=colors[i],
                m=markers[i],
                ax=ax,
                logx=False,
                logy=True,
                legend=u,
                xlabel="Q " r"($\AA^{-1}$)",
                ylabel="I(q)",
            )

        else:
            d = total_res[key][:, qth]
            plot1D(
                x=np.arange(len(d)),
                y=d + yshift_ * ii,
                c=colors[i],
                m=markers[i],
                ax=ax,
                logx=False,
                legend=u,
                xlabel="xx",
                ylabel=key,
            )
    if key == "mean_int_sets":
        ax.set_xlabel("frame ")
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    return fig, ax


def plot_entries_from_uids(
    uid_list,
    inDir,
    key="g2",
    qth=1,
    legend_size=8,
    yshift=0.01,
    ymulti=1,
    xlim=None,
    ylim=None,
    legend=None,
    uid_length=None,
    filename_list=None,
    fp_fulluid=False,
    fp_append=None,
):  # ,title=''  ):
    """
    YG Feb2, 2018, make yshift be also a list

    YG June 9, 2017@CHX
     YG Sep 29, 2017@CHX.
    plot enteries for a list uids
    Input:
        uid_list: list, a list of uid (string)
        inDir: string, imported folder for saved analysis results
        key: string, plot entry, surport
            'g2' for one-time,
            'iq' for q~iq
            'mean_int_sets' for mean intensity of each roi as a function of frame
            TODOLIST:#also can plot the following
                dict_keys(['qt', 'imgsum', 'qval_dict_v', 'bad_frame_list', 'iqst',
                'times_roi', 'iq_saxs', 'g2', 'mask', 'g2_uids', 'taus_uids',
                'g2_fit_paras', 'mean_int_sets', 'roi_mask', 'qval_dict', 'taus',
                'pixel_mask', 'avg_img', 'qval_dict_p', 'q_saxs', 'md'])
        qth: integer, the intesrest q number
        yshift: float, values of shift in y direction
        xlim: [x1,x2], for plot x limit
        ylim: [y1,y2], for plot y limit
    Output:
        show the plot
    Example:
    uid_list = ['5492b9', '54c5e0']
    plot_entries_from_uids( uid_list, inDir, yshift = 0.01, key= 'g2', ylim=[1, 1.2])
    """

    uid_dict = {}
    fig, ax = plt.subplots()
    for uid in uid_list:
        if uid_length is not None:
            uid_ = uid[:uid_length]
        else:
            uid_ = uid
        # print(uid_)
        uid_dict[uid_] = get_meta_data(uid)["uid"]
    # for i, u in enumerate( list( uid_dict.keys() )):
    for i, u in enumerate(list(uid_list)):
        # print(u)
        if isinstance(yshift, list):
            yshift_ = yshift[i]
            ii = i + 1
        else:
            yshift_ = yshift
            ii = i
        if uid_length is not None:
            u = u[:uid_length]
        inDiru = inDir + u + "/"
        if fp_fulluid:
            inDiru = inDir + uid_dict[u] + "/"
        else:
            inDiru = inDir + u + "/"
        if filename_list is None:
            if fp_append is not None:
                filename = "uid=%s%s_Res.h5" % (uid_dict[u], fp_append)
            else:
                filename = "uid=%s_Res.h5" % uid_dict[u]
        else:
            filename = filename_list[i]
        total_res = extract_xpcs_results_from_h5(filename=filename, import_dir=inDiru, exclude_keys=["g12b"])
        if key == "g2":
            d = total_res[key][1:, qth]
            taus = total_res["taus"][1:]
            if legend is None:
                leg = u
            else:
                leg = "uid=%s-->" % u + legend[i]
            plot1D(
                x=taus,
                y=d + yshift_ * ii,
                c=colors[i],
                m=markers[i],
                ax=ax,
                logx=True,
                legend=leg,
                xlabel="t (sec)",
                ylabel="g2",
                legend_size=legend_size,
            )
            title = "Q = %s" % (total_res["qval_dict"][qth])
            ax.set_title(title)
        elif key == "imgsum":
            d = total_res[key]
            plot1D(
                y=d + yshift_ * ii,
                c=colors[i],
                m=markers[i],
                ax=ax,
                logx=False,
                legend=u,
                xlabel="Frame",
                ylabel="imgsum",
            )

        elif key == "iq":

            x = total_res["q_saxs"]
            y = total_res["iq_saxs"]
            plot1D(
                x=x,
                y=y * ymulti[i] + yshift_ * ii,
                c=colors[i],
                m=markers[i],
                ax=ax,
                logx=False,
                logy=True,
                legend=u,
                xlabel="Q " r"($\AA^{-1}$)",
                ylabel="I(q)",
            )

        else:
            d = total_res[key][:, qth]
            plot1D(
                x=np.arange(len(d)),
                y=d + yshift_ * ii,
                c=colors[i],
                m=markers[i],
                ax=ax,
                logx=False,
                legend=u,
                xlabel="xx",
                ylabel=key,
            )
    if key == "mean_int_sets":
        ax.set_xlabel("frame ")
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    return fig, ax


####################################################################################################
##For real time analysis##
#################################################################################################


def get_iq_from_uids(uids, mask, setup_pargs):
    """Y.G. developed July 17, 2017 @CHX
    Get q-Iq of a uids dict, each uid could corrrespond one frame or a time seriers
    uids: dict, val: meaningful decription, key: a list of uids
    mask: bool-type 2D array
    setup_pargs: dict, at least should contains, the following paramters for calculation of I(q)

         'Ldet': 4917.50495,
         'center': [988, 1120],
         'dpix': 0.075000003562308848,
         'exposuretime': 0.99998999,
         'lambda_': 1.2845441,
         'path': '/XF11ID/analysis/2017_2/yuzhang/Results/Yang_Pressure/',

    """
    Nuid = len(np.concatenate(np.array(list(uids.values()))))
    label = np.zeros([Nuid + 1], dtype=object)
    img_data = {}  # np.zeros( [ Nuid, avg_img.shape[0], avg_img.shape[1]])

    n = 0
    for k in list(uids.keys()):
        for uid in uids[k]:

            uidstr = "uid=%s" % uid
            sud = get_sid_filenames(db[uid])
            # print(sud)
            md = get_meta_data(uid)
            imgs = load_data(uid, md["detector"], reverse=True)
            md.update(imgs.md)
            Nimg = len(imgs)
            if Nimg != 1:
                filename = "/XF11ID/analysis/Compressed_Data" + "/uid_%s.cmp" % sud[1]
                mask0, avg_img, imgsum, bad_frame_list = compress_eigerdata(
                    imgs,
                    mask,
                    md,
                    filename,
                    force_compress=False,
                    para_compress=True,
                    bad_pixel_threshold=1e14,
                    bins=1,
                    num_sub=100,
                    num_max_para_process=500,
                    with_pickle=True,
                )
            else:
                avg_img = imgs[0]
            show_img(
                avg_img,
                vmin=0.00001,
                vmax=1e1,
                logs=True,
                aspect=1,  # save_format='tif',
                image_name=uidstr + "_img_avg",
                save=True,
                path=setup_pargs["path"],
                cmap=cmap_albula,
            )

            setup_pargs["uid"] = uidstr

            qp_saxs, iq_saxs, q_saxs = get_circular_average(avg_img, mask, pargs=setup_pargs, save=True)
            if n == 0:
                iqs = np.zeros([len(q_saxs), Nuid + 1])
                iqs[:, 0] = q_saxs
                label[0] = "q"
            img_data[k + "_" + uid] = avg_img
            iqs[:, n + 1] = iq_saxs
            label[n + 1] = k + "_" + uid
            n += 1
            plot_circular_average(
                qp_saxs,
                iq_saxs,
                q_saxs,
                pargs=setup_pargs,
                xlim=[q_saxs.min(), q_saxs.max() * 0.9],
                ylim=[iq_saxs.min(), iq_saxs.max()],
            )
    if "filename" in list(setup_pargs.keys()):
        filename = setup_pargs["filename"]
    else:
        filename = "qIq.csv"
    pd = save_arrays(iqs, label=label, dtype="array", filename=filename, path=setup_pargs["path"], return_res=True)
    return pd, img_data


def wait_func(wait_time=2):
    print("Waiting %s secdons for upcoming data..." % wait_time)
    time.sleep(wait_time)
    # print( 'Starting to do something here...')


def wait_data_acquistion_finish(uid, wait_time=2, max_try_num=3):
    """check the completion of a data uid acquistion
    Parameter:
       uid:
       wait_time:  the waiting step in unit of second
       check_func: the function to check the completion
       max_try_num: the maximum number for waiting
    Return:
       True: completion
       False: not completion (include waiting time exceeds the max_wait_time)

    """
    FINISH = False
    Fake_FINISH = True
    w = 0
    sleep_time = 0
    while not FINISH:
        try:
            get_meta_data(uid)
            FINISH = True
            print("The data acquistion finished.")
            print("Starting to do something here...")
        except:
            wait_func(wait_time=wait_time)
            w += 1
            print("Try number: %s" % w)
            if w > max_try_num:
                print("There could be something going wrong with data acquistion.")
                print("Force to terminate after %s tries." % w)
                FINISH = True
                Fake_FINISH = False
            sleep_time += wait_time
    return FINISH * Fake_FINISH  # , sleep_time


def get_uids_by_range(start_uidth=-1, end_uidth=0):
    """Y.G. Dec 22, 2016
    A wrap funciton to find uids by giving start and end uid number, i.e. -10, -1
    Return:
    uids: list, uid with 8 character length
    fuids: list, uid with full length

    """
    hdrs = list([db[n] for n in range(start_uidth, end_uidth)])
    if len(hdrs) != 0:
        print("Totally %s uids are found." % (len(hdrs)))

    uids = []  # short uid
    fuids = []  # full uid
    for hdr in hdrs:
        fuid = hdr["start"]["uid"]
        uids.append(fuid[:8])
        fuids.append(fuid)
    uids = uids[::-1]
    fuids = fuids[::-1]
    return np.array(uids), np.array(fuids)


def get_uids_in_time_period(start_time, stop_time):
    """Y.G. Dec 22, 2016
    A wrap funciton to find uids by giving start and end time
    Return:
    uids: list, uid with 8 character length
    fuids: list, uid with full length

    """
    hdrs = list(db(start_time=start_time, stop_time=stop_time))
    if len(hdrs) != 0:
        print("Totally %s uids are found." % (len(hdrs)))

    uids = []  # short uid
    fuids = []  # full uid
    for hdr in hdrs:
        fuid = hdr["start"]["uid"]
        uids.append(fuid[:8])
        fuids.append(fuid)
    uids = uids[::-1]
    fuids = fuids[::-1]
    return np.array(uids), np.array(fuids)


def do_compress_on_line(start_time, stop_time, mask_dict=None, mask=None, wait_time=2, max_try_num=3):
    """Y.G. Mar 10, 2017
    Do on-line compress by giving start time and stop time
    Parameters:
        mask_dict: a dict, e.g., {mask1: mask_array1, mask2:mask_array2}
        wait_time: search interval time
        max_try_num: for each found uid, will try max_try_num*wait_time seconds
    Return:
        running time
    """

    t0 = time.time()
    uids, fuids = get_uids_in_time_period(start_time, stop_time)
    print(fuids)
    if len(fuids):
        for uid in fuids:
            print("*" * 50)
            print("Do compress for %s now..." % uid)
            if db[uid]["start"]["plan_name"] == "count":
                finish = wait_data_acquistion_finish(uid, wait_time, max_try_num)
                if finish:
                    try:
                        md = get_meta_data(uid)
                        compress_multi_uids(
                            [uid],
                            mask=mask,
                            mask_dict=mask_dict,
                            force_compress=False,
                            para_compress=True,
                            bin_frame_number=1,
                        )

                        update_olog_uid(uid=md["uid"], text="Data are on-line sparsified!", attachments=None)
                    except:
                        print("There are something wrong with this data: %s..." % uid)
            print("*" * 50)
    return time.time() - t0


def realtime_xpcs_analysis(
    start_time, stop_time, run_pargs, md_update=None, wait_time=2, max_try_num=3, emulation=False, clear_plot=False
):
    """Y.G. Mar 10, 2017
    Do on-line xpcs by giving start time and stop time
    Parameters:
        run_pargs: all the run control parameters, including giving roi_mask
        md_update: if not None, a dict, will update all the found uid metadata by this md_update
                    e.g,
                        md['beam_center_x'] = 1012
                        md['beam_center_y']= 1020
                        md['det_distance']= 16718.0
        wait_time: search interval time
        max_try_num: for each found uid, will try max_try_num*wait_time seconds
        emulation: if True, it will only check dataset and not do real analysis
    Return:
        running time
    """

    t0 = time.time()
    uids, fuids = get_uids_in_time_period(start_time, stop_time)
    # print( fuids )
    if len(fuids):
        for uid in fuids:
            print("*" * 50)
            # print('Do compress for %s now...'%uid)
            print("Starting analysis for %s now..." % uid)
            if db[uid]["start"]["plan_name"] == "count" or db[uid]["start"]["plan_name"] == "manual_count":
                # if db[uid]['start']['dtype'] =='xpcs':
                finish = wait_data_acquistion_finish(uid, wait_time, max_try_num)
                if finish:
                    try:
                        md = get_meta_data(uid)
                        ##corect some metadata
                        if md_update is not None:
                            md.update(md_update)
                            # if 'username' in list(md.keys()):
                            # try:
                            #    md_cor['username'] = md_update['username']
                            # except:
                            #    md_cor = None
                        # uid = uid[:8]
                        # print(md_cor)
                        if not emulation:
                            # suid=uid[:6]
                            run_xpcs_xsvs_single(
                                uid, run_pargs=run_pargs, md_cor=None, return_res=False, clear_plot=clear_plot
                            )
                        # update_olog_uid( uid= md['uid'], text='Data are on-line sparsified!',attachments=None)
                    except:
                        print("There are something wrong with this data: %s..." % uid)
            else:
                print("\nThis is not a XPCS series. We will simiply ignore it.")
            print("*" * 50)

    # print( 'Sleep 10 sec here!!!')
    # time.sleep(10)

    return time.time() - t0


####################################################################################################
##compress multi uids, sequential compress for uids, but for each uid, can apply parallel compress##
#################################################################################################
def compress_multi_uids(
    uids,
    mask,
    mask_dict=None,
    force_compress=False,
    para_compress=True,
    bin_frame_number=1,
    reverse=True,
    rot90=False,
    use_local_disk=True,
):
    """Compress time series data for a set of uids
    Parameters:
        uids: list, a list of uid
        mask: bool array, mask array
        force_compress: default is False, just load the compresssed data;
                    if True, will compress it to overwrite the old compressed data
        para_compress: apply the parallel compress algorithm
        bin_frame_number:
    Return:
        None, save the compressed data in, by default, /XF11ID/analysis/Compressed_Data with filename as
              '/uid_%s.cmp' uid is the full uid string

    e.g.,  compress_multi_uids( uids, mask, force_compress= False,  bin_frame_number=1 )

    """
    for uid in uids:
        print("UID: %s is in processing..." % uid)
        if validate_uid(uid):
            md = get_meta_data(uid)
            imgs = load_data(uid, md["detector"], reverse=reverse, rot90=rot90)
            sud = get_sid_filenames(db[uid])
            for pa in sud[2]:
                if "master.h5" in pa:
                    data_fullpath = pa
            print(imgs, data_fullpath)
            if mask_dict is not None:
                mask = mask_dict[md["detector"]]
                print("The detecotr is: %s" % md["detector"])
            md.update(imgs.md)
            if not use_local_disk:
                cmp_path = "/nsls2/xf11id1/analysis/Compressed_Data"
            else:
                cmp_path = "/tmp_data/compressed"
            cmp_path = "/nsls2/xf11id1/analysis/Compressed_Data"
            if bin_frame_number == 1:
                cmp_file = "/uid_%s.cmp" % md["uid"]
            else:
                cmp_file = "/uid_%s_bined--%s.cmp" % (md["uid"], bin_frame_number)
            filename = cmp_path + cmp_file
            mask, avg_img, imgsum, bad_frame_list = compress_eigerdata(
                imgs,
                mask,
                md,
                filename,
                force_compress=force_compress,
                para_compress=para_compress,
                bad_pixel_threshold=1e14,
                reverse=reverse,
                rot90=rot90,
                bins=bin_frame_number,
                num_sub=100,
                num_max_para_process=500,
                with_pickle=True,
                direct_load_data=use_local_disk,
                data_path=data_fullpath,
            )

    print("Done!")


####################################################################################################
##get_two_time_mulit_uids, sequential cal for uids, but apply parallel for each uid ##
#################################################################################################


def get_two_time_mulit_uids(
    uids,
    roi_mask,
    norm=None,
    bin_frame_number=1,
    path=None,
    force_generate=False,
    md=None,
    imgs=None,
    direct_load_data=False,
    compress_path=None,
):
    """Calculate two time correlation by using auto_two_Arrayc func for a set of uids,
        if the two-time resutls are already created, by default (force_generate=False), just pass
    Parameters:
        uids: list, a list of uid
        roi_mask: bool array, roi mask array
        norm: the normalization array
        path: string, where to save the two time
        force_generate: default, False, if the two-time resutls are already created, just pass
                        if True, will force to calculate two-time no matter exist or not

    Return:
        None, save the two-time in as  path + uid + 'uid=%s_g12b'%uid

    e.g.,
        get_two_time_mulit_uids( guids, roi_mask,  norm= norm,bin_frame_number=1,
                        path= data_dir,force_generate=False )

    """

    qind, pixelist = roi.extract_label_indices(roi_mask)
    for uid in uids:
        print("UID: %s is in processing..." % uid)
        if not direct_load_data:
            md = get_meta_data(uid)
            imgs = load_data(uid, md["detector"], reverse=True)
        else:
            pass
        N = len(imgs)
        # print( N )
        if compress_path is None:
            compress_path = "/XF11ID/analysis/Compressed_Data/"
        if bin_frame_number == 1:
            filename = "%s" % compress_path + "uid_%s.cmp" % md["uid"]
        else:
            filename = "%s" % compress_path + "uid_%s_bined--%s.cmp" % (md["uid"], bin_frame_number)

        FD = Multifile(filename, 0, N // bin_frame_number)
        # print( FD.beg, FD.end)
        uid_ = md["uid"]
        os.makedirs(path + uid_ + "/", exist_ok=True)
        filename = path + uid_ + "/" + "uid=%s_g12b" % uid
        doit = True
        if not force_generate:
            if os.path.exists(filename + ".npy"):
                doit = False
                print("The two time correlation function for uid=%s is already calculated. Just pass..." % uid)
        if doit:
            data_pixel = Get_Pixel_Arrayc(FD, pixelist, norm=norm).get_data()
            g12b = auto_two_Arrayc(data_pixel, roi_mask, index=None)
            np.save(filename, g12b)
            del g12b
            print("The two time correlation function for uid={} is saved as {}.".format(uid, filename))


def get_series_g2_from_g12(
    g12b, fra_num_by_dose=None, dose_label=None, good_start=0, log_taus=True, num_bufs=8, time_step=1
):
    """
    Get a series of one-time function from two-time by giving noframes
    Parameters:
        g12b: a two time function
        good_start: the start frame number
        fra_num_by_dose: a list, correlation number starting from index 0,
                if this number is larger than g12b length, will give a warning message, and
                will use g12b length to replace this number
                by default is None, will = [ g12b.shape[0] ]
        dose_label: the label of each dose, also is the keys of returned g2, lag
        log_taus: if true, will only return a g2 with the correponding tau values
                    as calculated by multi-tau defined taus
    Return:

        g2_series, a dict, with keys as dose_label (corrected on if warning message is given)
        lag_steps, the corresponding lags

    """
    g2 = {}
    lag_steps = {}
    L, L, qs = g12b.shape
    if fra_num_by_dose is None:
        fra_num_by_dose = [L]
    if dose_label is None:
        dose_label = fra_num_by_dose
    fra_num_by_dose = sorted(fra_num_by_dose)
    dose_label = sorted(dose_label)
    for i, good_end in enumerate(fra_num_by_dose):
        key = round(dose_label[i], 3)
        # print( good_end )
        if good_end > L:
            warnings.warn(
                "Warning: the dose value is too large, and please check the maxium dose in this data set and give a smaller dose value. We will use the maxium dose of the data."
            )
            good_end = L
        if not log_taus:
            g2[key] = get_one_time_from_two_time(g12b[good_start:good_end, good_start:good_end, :])
        else:
            # print(  good_end,  num_bufs )
            lag_step = get_multi_tau_lag_steps(good_end, num_bufs)
            lag_step = lag_step[lag_step < good_end - good_start]
            # print( len(lag_steps ) )
            lag_steps[key] = lag_step * time_step
            g2[key] = get_one_time_from_two_time(g12b[good_start:good_end, good_start:good_end, :])[lag_step]

    return lag_steps, g2


def get_fra_num_by_dose(exp_dose, exp_time, att=1, dead_time=2):
    """
    Calculate the frame number to be correlated by giving a X-ray exposure dose

    Paramters:
        exp_dose: a list, the exposed dose, e.g., in unit of exp_time(ms)*N(fram num)*att( attenuation)
        exp_time: float, the exposure time for a xpcs time sereies
        dead_time: dead time for the fast shutter reponse time, CHX = 2ms
    Return:
        noframes: the frame number to be correlated, exp_dose/( exp_time + dead_time )
    e.g.,

    no_dose_fra = get_fra_num_by_dose(  exp_dose = [ 3.34* 20, 3.34*50, 3.34*100, 3.34*502, 3.34*505 ],
                                   exp_time = 1.34, dead_time = 2)

    --> no_dose_fra  will be array([ 20,  50, 100, 502, 504])
    """
    return np.int_(np.array(exp_dose) / (exp_time + dead_time) / att)


def get_series_one_time_mulit_uids(
    uids,
    qval_dict,
    trans=None,
    good_start=0,
    path=None,
    exposure_dose=None,
    dead_time=0,
    num_bufs=8,
    save_g2=True,
    md=None,
    imgs=None,
    direct_load_data=False,
):
    """Calculate a dose depedent series of one time correlations from two time
    Parameters:
        uids: list, a list of uid
        trans: list, same length as uids, the transmission list
        exposure_dose: list, a list x-ray exposure dose;
                by default is None, namely,  = [ max_frame_number ],
                can be [3.34  334, 3340] in unit of ms,  in unit of exp_time(ms)*N(fram num)*att( attenuation)
        path: string, where to load the two time, if None, ask for it
                    the real g12 path is two_time_path + uid + '/'
        qval_dict: the dictionary for q values
    Return:
        taus_uids, with keys as uid, and
                taus_uids[uid] is also a dict, with keys as dose_frame
        g2_uids, with keys as uid, and
                g2_uids[uid] is also a dict, with keys as  dose_frame
        will also save g2 results to the 'path'
    """

    if path is None:
        print("Please calculate two time function first by using get_two_time_mulit_uids function.")
    else:
        taus_uids = {}
        g2_uids = {}
        for i, uid in enumerate(uids):
            print("UID: %s is in processing..." % uid)
            if not direct_load_data:
                md = get_meta_data(uid)
                imgs = load_data(uid, md["detector"], reverse=True)
                # print(md)
                detectors = md["detector"]
                if isinstance(detectors, list):
                    if len(detectors) > 1:
                        if "_image" in md["detector"]:
                            pref = md["detector"][:-5]
                        else:
                            pref = md["detector"]
                        for k in [
                            "beam_center_x",
                            "beam_center_y",
                            "cam_acquire_time",
                            "cam_acquire_period",
                            "cam_num_images",
                            "wavelength",
                            "det_distance",
                            "photon_energy",
                        ]:
                            md[k] = md[pref + "%s" % k]

            else:
                pass
            N = len(imgs)
            if exposure_dose is None:
                exposure_dose = [N]
            try:
                g2_path = path + uid + "/"
                g12b = np.load(g2_path + "uid=%s_g12b.npy" % uid)
            except:
                g2_path = path + md["uid"] + "/"
                g12b = np.load(g2_path + "uid=%s_g12b.npy" % uid)
            try:
                exp_time = float(md["cam_acquire_time"])  # *1000 #from second to ms
            except:
                exp_time = float(md["exposure time"])  # * 1000  #from second to ms
            if trans is None:
                try:
                    transi = md["transmission"]
                except:
                    transi = [1]
            else:
                transi = trans[i]
            fra_num_by_dose = get_fra_num_by_dose(
                exp_dose=exposure_dose, exp_time=exp_time, dead_time=dead_time, att=transi
            )

            print("uid: %s--> fra_num_by_dose: %s" % (uid, fra_num_by_dose))

            taus_uid, g2_uid = get_series_g2_from_g12(
                g12b,
                fra_num_by_dose=fra_num_by_dose,
                dose_label=exposure_dose,
                good_start=good_start,
                num_bufs=num_bufs,
                time_step=exp_time,
            )  # md['cam_acquire_period'] )
            g2_uids["uid_%03d=%s" % (i, uid)] = g2_uid
            taus_uids["uid_%03d=%s" % (i, uid)] = taus_uid
            if save_g2:
                for k in list(g2_uid.keys()):
                    # print(k)
                    uid_ = uid + "_fra_%s_%s" % (good_start, k)
                    save_g2_general(
                        g2_uid[k],
                        taus=taus_uid[k],
                        qr=np.array(list(qval_dict.values()))[:, 0],
                        uid=uid_ + "_g2.csv",
                        path=g2_path,
                        return_res=False,
                    )
        return taus_uids, g2_uids


def plot_dose_g2(
    taus_uids,
    g2_uids,
    qval_dict,
    qth_interest=None,
    ylim=[0.95, 1.05],
    vshift=0.1,
    fit_res=None,
    geometry="saxs",
    filename="dose" + "_g2",
    legend_size=None,
    path=None,
    function=None,
    g2_labels=None,
    ylabel="g2_dose",
    append_name="_dose",
    return_fig=False,
):
    """Plot a does-dependent g2
    taus_uids, dict, with format as {uid1: { dose1: tau_1, dose2: tau_2...}, uid2: ...}
    g2_uids, dict, with format as {uid1: { dose1: g2_1, dose2: g2_2...}, uid2: ...}
    qval_dict: a dict of qvals
    vshift: float, vertical shift value of different dose of g2

    """

    uids = sorted(list(taus_uids.keys()))
    # print( uids )
    dose = sorted(list(taus_uids[uids[0]].keys()))
    if qth_interest is None:
        g2_dict = {}
        taus_dict = {}
        if g2_labels is None:
            g2_labels = []
        for i in range(len(dose)):
            g2_dict[i + 1] = []
            taus_dict[i + 1] = []
            # print ( i )
            for j in range(len(uids)):
                # print( uids[i] , dose[j])
                g2_dict[i + 1].append(g2_uids[uids[j]][dose[i]] + vshift * i)
                taus_dict[i + 1].append(taus_uids[uids[j]][dose[i]])
                if j == 0:
                    g2_labels.append("Dose_%s" % dose[i])

        plot_g2_general(
            g2_dict,
            taus_dict,
            ylim=[ylim[0], ylim[1] + vshift * len(dose)],
            qval_dict=qval_dict,
            fit_res=None,
            geometry=geometry,
            filename=filename,
            path=path,
            function=function,
            ylabel=ylabel,
            g2_labels=g2_labels,
            append_name=append_name,
        )

    else:
        fig, ax = plt.subplots()
        q = qval_dict[qth_interest - 1][0]
        j = 0
        for uid in uids:
            # uid = uids[0]
            # print( uid )
            dose_list = sorted(list(taus_uids["%s" % uid].keys()))
            # print( dose_list )
            for i, dose in enumerate(dose_list):
                dose = float(dose)
                if j == 0:
                    legend = "dose_%s" % round(dose, 2)
                else:
                    legend = ""

                # print( markers[i], colors[i] )

                plot1D(
                    x=taus_uids["%s" % uid][dose_list[i]],
                    y=g2_uids["%s" % uid][dose_list[i]][:, qth_interest] + i * vshift,
                    logx=True,
                    ax=ax,
                    legend=legend,
                    m=markers[i],
                    c=colors[i],
                    lw=3,
                    title="%s_Q=%s" % (uid, q) + r"$\AA^{-1}$",
                    legend_size=legend_size,
                )
                ylabel = "g2--Dose (trans*exptime_sec)"
            j += 1

        ax.set_ylabel(r"$%s$" % ylabel + "(" + r"$\tau$" + ")")
        ax.set_xlabel(r"$\tau $ $(s)$", fontsize=16)
        ax.set_ylim(ylim)
        if return_fig:
            return fig, ax
    # return taus_dict, g2_dict


def run_xpcs_xsvs_single(uid, run_pargs, md_cor=None, return_res=False, reverse=True, clear_plot=False):
    """Y.G. Dec 22, 2016
       Run XPCS XSVS analysis for a single uid
       Parameters:
           uid: unique id
           run_pargs: dict, control run type and setup parameters, such as q range et.al.
           reverse:,True, revserse the image upside down
       Return:
       save analysis result to csv/png/h5 files
       return_res: if true, return a dict, containing g2,g4,g12,contrast et.al. depending on the run type
       An example for the run_pargs:

    run_pargs=  dict(
                    scat_geometry = 'gi_saxs'  #suport 'saxs', 'gi_saxs', 'ang_saxs' (for anisotropics saxs or flow-xpcs)
                    force_compress =  True,#False,
                    para_compress = True,
                    run_fit_form = False,
                    run_waterfall = True,#False,
                    run_t_ROI_Inten = True,
                    #run_fit_g2 = True,
                    fit_g2_func = 'stretched',
                    run_one_time = True,#False,
                    run_two_time = True,#False,
                    run_four_time = False,
                    run_xsvs=True,
                    att_pdf_report = True,
                    show_plot = False,

                    CYCLE = '2016_3',
                    mask_path = '/XF11ID/analysis/2016_3/masks/',
                    mask_name = 'Nov28_4M_SAXS_mask.npy',
                    good_start   = 5,

                    uniformq = True,
                    inner_radius= 0.005, #0.005 for 50 nm, 0.006, #for 10nm/coralpor
                    outer_radius = 0.04, #0.04 for 50 nm, 0.05, #for 10nm/coralpor
                    num_rings = 12,
                    gap_ring_number = 6,
                    number_rings= 1,
                    #qcenters = [ 0.00235,0.00379,0.00508,0.00636,0.00773, 0.00902] #in A-1
                    #width = 0.0002
                    qth_interest = 1, #the intested single qth
                    use_sqnorm = False,
                    use_imgsum_norm = True,

                    pdf_version = '_1' #for pdf report name
                  )

    md_cor: if not None, will update the metadata with md_cor

    """

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
    try:
        run_dose = run_pargs["run_dose"]
    except:
        run_dose = False
    ###############################################################
    if scat_geometry == "gi_saxs":  # to be done for other types
        run_xsvs = False
    ###############################################################

    ###############################################################
    if scat_geometry == "ang_saxs":
        run_xsvs = False
        run_waterfall = False
        run_two_time = False
        run_four_time = False
        run_t_ROI_Inten = False
    ###############################################################
    if "bin_frame" in list(run_pargs.keys()):
        bin_frame = run_pargs["bin_frame"]
        bin_frame_number = run_pargs["bin_frame_number"]
    else:
        bin_frame = False
    if not bin_frame:
        bin_frame_number = 1

    att_pdf_report = run_pargs["att_pdf_report"]
    show_plot = run_pargs["show_plot"]
    CYCLE = run_pargs["CYCLE"]
    mask_path = run_pargs["mask_path"]
    mask_name = run_pargs["mask_name"]
    good_start = run_pargs["good_start"]
    use_imgsum_norm = run_pargs["use_imgsum_norm"]
    try:
        use_sqnorm = run_pargs["use_sqnorm"]
    except:
        use_sqnorm = False
    try:
        inc_x0 = run_pargs["inc_x0"]
        inc_y0 = run_pargs["inc_y0"]
    except:
        inc_x0 = None
        inc_y0 = None

    # for different scattering geogmetry, we only need to change roi_mask
    # and qval_dict
    qval_dict = run_pargs["qval_dict"]
    if scat_geometry != "ang_saxs":
        roi_mask = run_pargs["roi_mask"]
        qind, pixelist = roi.extract_label_indices(roi_mask)
        noqs = len(np.unique(qind))
        nopr = np.bincount(qind, minlength=(noqs + 1))[1:]

    else:
        roi_mask_p = run_pargs["roi_mask_p"]
        qval_dict_p = run_pargs["qval_dict_p"]
        roi_mask_v = run_pargs["roi_mask_v"]
        qval_dict_v = run_pargs["qval_dict_v"]

    if scat_geometry == "gi_saxs":
        refl_x0 = run_pargs["refl_x0"]
        refl_y0 = run_pargs["refl_y0"]
        Qr, Qz, qr_map, qz_map = run_pargs["Qr"], run_pargs["Qz"], run_pargs["qr_map"], run_pargs["qz_map"]

    taus = None
    g2 = None
    tausb = None
    g2b = None
    g12b = None
    taus4 = None
    g4 = None
    times_xsv = None
    contrast_factorL = None
    qth_interest = run_pargs["qth_interest"]
    pdf_version = run_pargs["pdf_version"]

    try:
        username = run_pargs["username"]
    except:
        username = getpass.getuser()

    data_dir0 = os.path.join("/XF11ID/analysis/", CYCLE, username, "Results/")
    os.makedirs(data_dir0, exist_ok=True)
    print("Results from this analysis will be stashed in the directory %s" % data_dir0)
    # uid = (sys.argv)[1]
    print("*" * 40)
    print("*" * 5 + "The processing uid is: %s" % uid + "*" * 5)
    print("*" * 40)
    suid = uid  # [:6]
    data_dir = os.path.join(data_dir0, "%s/" % suid)
    os.makedirs(data_dir, exist_ok=True)
    print("Results from this analysis will be stashed in the directory %s" % data_dir)
    md = get_meta_data(uid)
    uidstr = "uid=%s" % uid[:6]
    imgs = load_data(uid, md["detector"], reverse=reverse)
    md.update(imgs.md)
    Nimg = len(imgs)
    if md_cor is not None:
        md.update(md_cor)

    if inc_x0 is not None:
        md["beam_center_x"] = inc_x0
    if inc_y0 is not None:
        md["beam_center_y"] = inc_y0

    # print( run_pargs )
    # print( run_pargs['inc_x0'],run_pargs['inc_y0'] )
    # print(  inc_x0, inc_y0 )

    if md["detector"] == "eiger1m_single_image":
        Chip_Mask = np.load("/XF11ID/analysis/2017_1/masks/Eiger1M_Chip_Mask.npy")
    elif md["detector"] == "eiger4m_single_image" or md["detector"] == "image":
        Chip_Mask = np.array(np.load("/XF11ID/analysis/2017_1/masks/Eiger4M_chip_mask.npy"), dtype=bool)
        BadPix = np.load("/XF11ID/analysis/2018_1/BadPix_4M.npy")
        Chip_Mask.ravel()[BadPix] = 0
    elif md["detector"] == "eiger500K_single_image":
        Chip_Mask = 1  # to be defined the chip mask
    else:
        Chip_Mask = 1
    # show_img(Chip_Mask)

    center = [int(md["beam_center_y"]), int(md["beam_center_x"])]  # beam center [y,x] for python image

    pixel_mask = 1 - np.int_(np.array(imgs.md["pixel_mask"], dtype=bool))
    print("The data are: %s" % imgs)

    if False:
        print_dict(
            md,
            [
                "suid",
                "number of images",
                "uid",
                "scan_id",
                "start_time",
                "stop_time",
                "sample",
                "Measurement",
                "acquire period",
                "exposure time",
                "det_distanc",
                "beam_center_x",
                "beam_center_y",
            ],
        )
    ## Overwrite Some Metadata if Wrong Input
    dpix, lambda_, Ldet, exposuretime, timeperframe, center = check_lost_metadata(
        md, Nimg, inc_x0=inc_x0, inc_y0=inc_y0, pixelsize=7.5 * 10 * (-5)
    )

    print("The beam center is: %s" % center)

    timeperframe *= bin_frame_number

    setup_pargs = dict(
        uid=uidstr,
        dpix=dpix,
        Ldet=Ldet,
        lambda_=lambda_,
        exposuretime=exposuretime,
        timeperframe=timeperframe,
        center=center,
        path=data_dir,
    )
    # print_dict( setup_pargs )

    mask = load_mask(mask_path, mask_name, plot_=False, image_name=uidstr + "_mask", reverse=reverse)
    mask *= pixel_mask
    if md["detector"] == "eiger4m_single_image":
        mask[:, 2069] = 0  # False  #Concluded from the previous results
    show_img(mask, image_name=uidstr + "_mask", save=True, path=data_dir)
    mask_load = mask.copy()
    imgsa = apply_mask(imgs, mask)

    img_choice_N = 2
    img_samp_index = random.sample(range(len(imgs)), img_choice_N)
    avg_img = get_avg_img(imgsa, img_samp_index, plot_=False, uid=uidstr)

    if avg_img.max() == 0:
        print("There are no photons recorded for this uid: %s" % uid)
        print("The data analysis should be terminated! Please try another uid.")

    else:
        if scat_geometry != "saxs":
            show_img(
                avg_img,
                vmin=0.1,
                vmax=np.max(avg_img * 0.1),
                logs=True,
                image_name=uidstr + "_%s_frames_avg" % img_choice_N,
                save=True,
                path=data_dir,
            )
        else:
            show_saxs_qmap(
                avg_img,
                setup_pargs,
                width=400,
                show_pixel=False,
                vmin=0.1,
                vmax=np.max(avg_img),
                logs=True,
                image_name=uidstr + "_%s_frames_avg" % img_choice_N,
            )

        compress = True
        photon_occ = len(np.where(avg_img)[0]) / (imgsa[0].size)
        # compress =  photon_occ < .4  #if the photon ocupation < 0.5, do compress
        print("The non-zeros photon occupation is %s." % (photon_occ))
        print("Will " + "Always " + ["NOT", "DO"][compress] + " apply compress process.")
        # good_start = 5  #make the good_start at least 0
        t0 = time.time()
        filename = "/XF11ID/analysis/Compressed_Data" + "/uid_%s.cmp" % md["uid"]
        mask, avg_img, imgsum, bad_frame_list = compress_eigerdata(
            imgs,
            mask,
            md,
            filename,
            force_compress=force_compress,
            para_compress=para_compress,
            bad_pixel_threshold=1e14,
            bins=bin_frame_number,
            num_sub=100,
            num_max_para_process=500,
            with_pickle=True,
        )
        min_inten = 10
        good_start = max(good_start, np.where(np.array(imgsum) > min_inten)[0][0])
        print("The good_start frame number is: %s " % good_start)
        FD = Multifile(filename, good_start, len(imgs))
        # FD = Multifile(filename, good_start, 100)
        uid_ = uidstr + "_fra_%s_%s" % (FD.beg, FD.end)
        print(uid_)
        plot1D(
            y=imgsum[np.array([i for i in np.arange(good_start, len(imgsum)) if i not in bad_frame_list])],
            title=uidstr + "_imgsum",
            xlabel="Frame",
            ylabel="Total_Intensity",
            legend="imgsum",
        )
        run_time(t0)

        mask = mask * Chip_Mask

        # %system   free && sync && echo 3 > /proc/sys/vm/drop_caches && free
        ## Get bad frame list by a polynominal fit
        bad_frame_list = get_bad_frame_list(
            imgsum,
            fit=True,
            plot=True,
            polyfit_order=30,
            scale=5.5,
            good_start=good_start,
            uid=uidstr,
            path=data_dir,
        )
        print("The bad frame list length is: %s" % len(bad_frame_list))

        ### Creat new mask by masking the bad pixels and get new avg_img
        if False:
            mask = mask_exclude_badpixel(bp, mask, md["uid"])
            avg_img = get_avg_imgc(FD, sampling=1, bad_frame_list=bad_frame_list)

        show_img(
            avg_img,
            vmin=0.001,
            vmax=np.max(avg_img),
            logs=True,
            aspect=1,  # save_format='tif',
            image_name=uidstr + "_img_avg",
            save=True,
            path=data_dir,
            cmap=cmap_albula,
        )

        imgsum_y = imgsum[np.array([i for i in np.arange(len(imgsum)) if i not in bad_frame_list])]
        imgsum_x = np.arange(len(imgsum_y))
        save_lists(
            [imgsum_x, imgsum_y], label=["Frame", "Total_Intensity"], filename=uidstr + "_img_sum_t", path=data_dir
        )
        plot1D(
            y=imgsum_y,
            title=uidstr + "_img_sum_t",
            xlabel="Frame",
            ylabel="Total_Intensity",
            legend="imgsum",
            save=True,
            path=data_dir,
        )

        ############for SAXS and ANG_SAXS (Flow_SAXS)
        if scat_geometry == "saxs" or scat_geometry == "ang_saxs":

            # show_saxs_qmap( avg_img, setup_pargs, width=600, vmin=.1, vmax=np.max(avg_img*.1), logs=True,
            #   image_name= uidstr + '_img_avg',  save=True)
            # np.save(  data_dir + 'uid=%s--img-avg'%uid, avg_img)

            # try:
            #    hmask = create_hot_pixel_mask( avg_img, threshold = 1000, center=center, center_radius= 600)
            # except:
            #    hmask=1
            hmask = 1
            qp_saxs, iq_saxs, q_saxs = get_circular_average(
                avg_img * Chip_Mask, mask * hmask * Chip_Mask, pargs=setup_pargs, save=True
            )

            plot_circular_average(
                qp_saxs,
                iq_saxs,
                q_saxs,
                pargs=setup_pargs,
                xlim=[q_saxs.min(), q_saxs.max()],
                ylim=[iq_saxs.min(), iq_saxs.max()],
            )

            # pd = trans_data_to_pd( np.where( hmask !=1),
            #        label=[md['uid']+'_hmask'+'x', md['uid']+'_hmask'+'y' ], dtype='list')

            # pd.to_csv('/XF11ID/analysis/Commissioning/eiger4M_badpixel.csv', mode='a'  )

            # mask =np.array( mask * hmask, dtype=bool)
            # show_img( mask )

            if run_fit_form:
                form_res = fit_form_factor(
                    q_saxs,
                    iq_saxs,
                    guess_values={"radius": 2500, "sigma": 0.05, "delta_rho": 1e-10},
                    fit_range=[0.0001, 0.015],
                    fit_variables={"radius": T, "sigma": T, "delta_rho": T},
                    res_pargs=setup_pargs,
                    xlim=[0.0001, 0.015],
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
                uid=uidstr,
                vmin=np.min(avg_img),
                vmax=np.max(avg_img),
            )

            qr = np.array([qval_dict[k][0] for k in list(qval_dict.keys())])
            plot_qIq_with_ROI(
                q_saxs,
                iq_saxs,
                qr,
                logs=True,
                uid=uidstr,
                xlim=[q_saxs.min(), q_saxs.max()],
                ylim=[iq_saxs.min(), iq_saxs.max()],
                save=True,
                path=data_dir,
            )

            if scat_geometry != "ang_saxs":
                Nimg = FD.end - FD.beg
                time_edge = create_time_slice(N=Nimg, slice_num=3, slice_width=1, edges=None)
                time_edge = np.array(time_edge) + good_start
                # print( time_edge )
                qpt, iqst, qt = get_t_iqc(FD, time_edge, mask * Chip_Mask, pargs=setup_pargs, nx=1500)
                plot_t_iqc(
                    qt,
                    iqst,
                    time_edge,
                    pargs=setup_pargs,
                    xlim=[qt.min(), qt.max()],
                    ylim=[iqst.min(), iqst.max()],
                    save=True,
                )

        elif scat_geometry == "gi_waxs":
            # roi_mask[badpixel] = 0
            qr = np.array([qval_dict[k][0] for k in list(qval_dict.keys())])
            show_ROI_on_image(
                avg_img, roi_mask, label_on=True, alpha=0.5, save=True, path=data_dir, uid=uidstr
            )  # , vmin=1, vmax=15)

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
            ticks_ = get_qzr_map(qr_map, qz_map, inc_x0, Nzline=10, Nrline=10)
            ticks = ticks_[:4]
            plot_qzr_map(qr_map, qz_map, inc_x0, ticks=ticks_, data=avg_img, uid=uidstr, path=data_dir)
            show_qzr_roi(avg_img, roi_mask, inc_x0, ticks, alpha=0.5, save=True, path=data_dir, uid=uidstr)
            qr_1d_pds = cal_1d_qr(avg_img, Qr, Qz, qr_map, qz_map, inc_x0, setup_pargs=setup_pargs)
            plot_qr_1d_with_ROI(
                qr_1d_pds,
                qr_center=np.unique(np.array(list(qval_dict.values()))[:, 0]),
                loglog=False,
                save=True,
                uid=uidstr,
                path=data_dir,
            )

            Nimg = FD.end - FD.beg
            time_edge = create_time_slice(N=Nimg, slice_num=3, slice_width=1, edges=None)
            time_edge = np.array(time_edge) + good_start
            qrt_pds = get_t_qrc(FD, time_edge, Qr, Qz, qr_map, qz_map, path=data_dir, uid=uidstr)
            plot_qrt_pds(qrt_pds, time_edge, qz_index=0, uid=uidstr, path=data_dir)

        ##############################
        ##the below works for all the geometries
        ########################################
        if scat_geometry != "ang_saxs":
            roi_inten = check_ROI_intensity(
                avg_img, roi_mask, ring_number=qth_interest, uid=uidstr, save=True, path=data_dir
            )
        if scat_geometry == "saxs" or scat_geometry == "gi_saxs" or scat_geometry == "gi_waxs":
            if run_waterfall:
                wat = cal_waterfallc(FD, roi_mask, qindex=qth_interest, save=True, path=data_dir, uid=uidstr)
            if run_waterfall:
                plot_waterfallc(
                    wat,
                    qindex=qth_interest,
                    aspect=None,
                    vmax=np.max(wat),
                    uid=uidstr,
                    save=True,
                    path=data_dir,
                    beg=FD.beg,
                )
        ring_avg = None

        if run_t_ROI_Inten:
            times_roi, mean_int_sets = cal_each_ring_mean_intensityc(
                FD, roi_mask, timeperframe=None, multi_cor=True
            )
            plot_each_ring_mean_intensityc(times_roi, mean_int_sets, uid=uidstr, save=True, path=data_dir)
            roi_avg = np.average(mean_int_sets, axis=0)

        uid_ = uidstr + "_fra_%s_%s" % (FD.beg, FD.end)
        lag_steps = None

        if use_sqnorm:
            norm = get_pixelist_interp_iq(qp_saxs, iq_saxs, roi_mask, center)
        else:
            norm = None

        define_good_series = False
        if define_good_series:
            FD = Multifile(filename, beg=good_start, end=Nimg)
            uid_ = uidstr + "_fra_%s_%s" % (FD.beg, FD.end)
            print(uid_)

        if "g2_fit_variables" in list(run_pargs.keys()):
            g2_fit_variables = run_pargs["g2_fit_variables"]
        else:
            g2_fit_variables = {"baseline": True, "beta": True, "alpha": False, "relaxation_rate": True}

        if "g2_guess_values" in list(run_pargs.keys()):
            g2_guess_values = run_pargs["g2_guess_values"]
        else:
            g2_guess_values = {
                "baseline": 1.0,
                "beta": 0.05,
                "alpha": 1.0,
                "relaxation_rate": 0.01,
            }

        if "g2_guess_limits" in list(run_pargs.keys()):
            g2_guess_limits = run_pargs["g2_guess_limits"]
        else:
            g2_guess_limits = dict(baseline=[1, 2], alpha=[0, 2], beta=[0, 1], relaxation_rate=[0.001, 5000])

        if run_one_time:
            if use_imgsum_norm:
                imgsum_ = imgsum
            else:
                imgsum_ = None
            if scat_geometry != "ang_saxs":
                t0 = time.time()
                g2, lag_steps = cal_g2p(
                    FD, roi_mask, bad_frame_list, good_start, num_buf=8, num_lev=None, imgsum=imgsum_, norm=norm
                )
                run_time(t0)
                taus = lag_steps * timeperframe
                g2_pds = save_g2_general(
                    g2,
                    taus=taus,
                    qr=np.array(list(qval_dict.values()))[:, 0],
                    uid=uid_ + "_g2.csv",
                    path=data_dir,
                    return_res=True,
                )
                g2_fit_result, taus_fit, g2_fit = get_g2_fit_general(
                    g2,
                    taus,
                    function=fit_g2_func,
                    vlim=[0.95, 1.05],
                    fit_range=None,
                    fit_variables=g2_fit_variables,
                    guess_values=g2_guess_values,
                    guess_limits=g2_guess_limits,
                )

                g2_fit_paras = save_g2_fit_para_tocsv(
                    g2_fit_result, filename=uid_ + "_g2_fit_paras.csv", path=data_dir
                )

                # if run_one_time:
                # plot_g2_general( g2_dict={1:g2}, taus_dict={1:taus},vlim=[0.95, 1.05], qval_dict = qval_dict, fit_res= None,
                #                geometry='saxs',filename=uid_+'--g2',path= data_dir,   ylabel='g2')

                plot_g2_general(
                    g2_dict={1: g2, 2: g2_fit},
                    taus_dict={1: taus, 2: taus_fit},
                    vlim=[0.95, 1.05],
                    qval_dict=qval_dict,
                    fit_res=g2_fit_result,
                    geometry=scat_geometry,
                    filename=uid_ + "_g2",
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
                    uid=uid_,
                    path=data_dir,
                )

            else:
                t0 = time.time()
                g2_v, lag_steps_v = cal_g2p(
                    FD, roi_mask_v, bad_frame_list, good_start, num_buf=8, num_lev=None, imgsum=imgsum_, norm=norm
                )
                g2_p, lag_steps_p = cal_g2p(
                    FD, roi_mask_p, bad_frame_list, good_start, num_buf=8, num_lev=None, imgsum=imgsum_, norm=norm
                )
                run_time(t0)

                taus_v = lag_steps_v * timeperframe
                g2_pds_v = save_g2_general(
                    g2_v,
                    taus=taus_v,
                    qr=np.array(list(qval_dict_v.values()))[:, 0],
                    uid=uid_ + "_g2v.csv",
                    path=data_dir,
                    return_res=True,
                )

                taus_p = lag_steps_p * timeperframe
                g2_pds_p = save_g2_general(
                    g2_p,
                    taus=taus_p,
                    qr=np.array(list(qval_dict_p.values()))[:, 0],
                    uid=uid_ + "_g2p.csv",
                    path=data_dir,
                    return_res=True,
                )

                fit_g2_func_v = "stretched"  # for vertical
                g2_fit_result_v, taus_fit_v, g2_fit_v = get_g2_fit_general(
                    g2_v,
                    taus_v,
                    function=fit_g2_func_v,
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
                g2_fit_paras_v = save_g2_fit_para_tocsv(
                    g2_fit_result_v, filename=uid_ + "_g2_fit_paras_v.csv", path=data_dir
                )

                fit_g2_func_p = "flow_para"  # for parallel
                g2_fit_result_p, taus_fit_p, g2_fit_p = get_g2_fit_general(
                    g2_p,
                    taus_p,
                    function=fit_g2_func_p,
                    vlim=[0.95, 1.05],
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
                g2_fit_paras_p = save_g2_fit_para_tocsv(
                    g2_fit_result_p, filename=uid_ + "_g2_fit_paras_p.csv", path=data_dir
                )

                plot_g2_general(
                    g2_dict={1: g2_v, 2: g2_fit_v},
                    taus_dict={1: taus_v, 2: taus_fit_v},
                    vlim=[0.95, 1.05],
                    qval_dict=qval_dict_v,
                    fit_res=g2_fit_result_v,
                    geometry=scat_geometry,
                    filename=uid_ + "_g2_v",
                    path=data_dir,
                    function=fit_g2_func_v,
                    ylabel="g2_v",
                    append_name="_fit",
                )

                plot_g2_general(
                    g2_dict={1: g2_p, 2: g2_fit_p},
                    taus_dict={1: taus_p, 2: taus_fit_p},
                    vlim=[0.95, 1.05],
                    qval_dict=qval_dict_p,
                    fit_res=g2_fit_result_p,
                    geometry=scat_geometry,
                    filename=uid_ + "_g2_p",
                    path=data_dir,
                    function=fit_g2_func_p,
                    ylabel="g2_p",
                    append_name="_fit",
                )

                combine_images(
                    [data_dir + uid_ + "_g2_v_fit.png", data_dir + uid_ + "_g2_p_fit.png"],
                    data_dir + uid_ + "_g2_fit.png",
                    outsize=(2000, 2400),
                )

                D0_v, qrate_fit_res_v = get_q_rate_fit_general(
                    qval_dict_v, g2_fit_paras_v["relaxation_rate"], geometry=scat_geometry
                )
                plot_q_rate_fit_general(
                    qval_dict_v,
                    g2_fit_paras_v["relaxation_rate"],
                    qrate_fit_res_v,
                    geometry=scat_geometry,
                    uid=uid_ + "_vert",
                    path=data_dir,
                )

                D0_p, qrate_fit_res_p = get_q_rate_fit_general(
                    qval_dict_p, g2_fit_paras_p["relaxation_rate"], geometry=scat_geometry
                )
                plot_q_rate_fit_general(
                    qval_dict_p,
                    g2_fit_paras_p["relaxation_rate"],
                    qrate_fit_res_p,
                    geometry=scat_geometry,
                    uid=uid_ + "_para",
                    path=data_dir,
                )

                combine_images(
                    [data_dir + uid_ + "_vert_Q_Rate_fit.png", data_dir + uid_ + "_para_Q_Rate_fit.png"],
                    data_dir + uid_ + "_Q_Rate_fit.png",
                    outsize=(2000, 2400),
                )

        # For two-time
        data_pixel = None
        if run_two_time:

            data_pixel = Get_Pixel_Arrayc(FD, pixelist, norm=norm).get_data()
            t0 = time.time()
            g12b = auto_two_Arrayc(data_pixel, roi_mask, index=None)
            if run_dose:
                np.save(data_dir + "uid=%s_g12b" % uid, g12b)

            if lag_steps is None:
                num_bufs = 8
                noframes = FD.end - FD.beg
                num_levels = int(np.log(noframes / (num_bufs - 1)) / np.log(2) + 1) + 1
                tot_channels, lag_steps, dict_lag = multi_tau_lags(num_levels, num_bufs)
                max_taus = lag_steps.max()
                lag_steps = lag_steps[lag_steps < Nimg - good_start]

            run_time(t0)

            show_C12(
                g12b,
                q_ind=qth_interest,
                N1=FD.beg,
                N2=min(FD.end, 5000),
                vmin=0.99,
                vmax=1.3,
                timeperframe=timeperframe,
                save=True,
                cmap=cmap_albula,
                path=data_dir,
                uid=uid_,
            )

            # print('here')
            # show_C12(g12b, q_ind= 3, N1= 5, N2=min(5000,5000), vmin=.8, vmax=1.31, cmap=cmap_albula,
            # timeperframe= timeperframe,save=False, path= data_dir,  uid  =  uid_ +'_' + k)
            max_taus = Nimg
            t0 = time.time()
            # g2b = get_one_time_from_two_time(g12b)[:max_taus]
            g2b = get_one_time_from_two_time(g12b)[lag_steps]

            tausb = lag_steps * timeperframe
            run_time(t0)

            # tausb = np.arange( g2b.shape[0])[:max_taus] *timeperframe
            g2b_pds = save_g2_general(
                g2b,
                taus=tausb,
                qr=np.array(list(qval_dict.values()))[:, 0],
                qz=None,
                uid=uid_ + "_g2b.csv",
                path=data_dir,
                return_res=True,
            )

            g2_fit_resultb, taus_fitb, g2_fitb = get_g2_fit_general(
                g2b,
                tausb,
                function=fit_g2_func,
                vlim=[0.95, 1.05],
                fit_range=None,
                fit_variables=g2_fit_variables,
                guess_values=g2_guess_values,
                guess_limits=g2_guess_limits,
            )

            g2b_fit_paras = save_g2_fit_para_tocsv(
                g2_fit_resultb, filename=uid_ + "_g2b_fit_paras.csv", path=data_dir
            )

            D0b, qrate_fit_resb = get_q_rate_fit_general(
                qval_dict, g2b_fit_paras["relaxation_rate"], fit_range=None, geometry=scat_geometry
            )

            # print( qval_dict,   g2b_fit_paras['relaxation_rate'],  qrate_fit_resb )
            plot_q_rate_fit_general(
                qval_dict,
                g2b_fit_paras["relaxation_rate"],
                qrate_fit_resb,
                geometry=scat_geometry,
                uid=uid_ + "_two_time",
                path=data_dir,
            )

            plot_g2_general(
                g2_dict={1: g2b, 2: g2_fitb},
                taus_dict={1: tausb, 2: taus_fitb},
                vlim=[0.95, 1.05],
                qval_dict=qval_dict,
                fit_res=g2_fit_resultb,
                geometry=scat_geometry,
                filename=uid_ + "_g2",
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
                    filename=uid_ + "_g2_two_g2",
                    path=data_dir,
                    ylabel="g2",
                )

        # Four Time Correlation

        if run_four_time:  # have to run one and two first
            t0 = time.time()
            g4 = get_four_time_from_two_time(g12b, g2=g2b)[:max_taus]
            run_time(t0)

            taus4 = np.arange(g4.shape[0]) * timeperframe
            g4_pds = save_g2_general(
                g4,
                taus=taus4,
                qr=np.array(list(qval_dict.values()))[:, 0],
                qz=None,
                uid=uid_ + "_g4.csv",
                path=data_dir,
                return_res=True,
            )
            plot_g2_general(
                g2_dict={1: g4},
                taus_dict={1: taus4},
                vlim=[0.95, 1.05],
                qval_dict=qval_dict,
                fit_res=None,
                geometry=scat_geometry,
                filename=uid_ + "_g4",
                path=data_dir,
                ylabel="g4",
            )

        if run_dose:
            get_two_time_mulit_uids(
                [uid], roi_mask, norm=norm, bin_frame_number=bin_frame_number, path=data_dir0, force_generate=False
            )
            N = len(imgs)
            try:
                tr = md["transmission"]
            except:
                tr = 1
            if "dose_frame" in list(run_pargs.keys()):
                dose_frame = run_pargs["dose_frame"]
            else:
                dose_frame = np.int_([N / 8, N / 4, N / 2, 3 * N / 4, N * 0.99])
                # N/32, N/16, N/8, N/4 ,N/2, 3*N/4, N*0.99
            exposure_dose = tr * exposuretime * dose_frame
            taus_uids, g2_uids = get_series_one_time_mulit_uids(
                [uid],
                qval_dict,
                good_start=good_start,
                path=data_dir0,
                exposure_dose=exposure_dose,
                num_bufs=8,
                save_g2=False,
                dead_time=0,
                trans=[tr],
            )

            plot_dose_g2(
                taus_uids,
                g2_uids,
                ylim=[0.95, 1.2],
                vshift=0.00,
                qval_dict=qval_dict,
                fit_res=None,
                geometry=scat_geometry,
                filename="%s_dose_analysis" % uid_,
                path=data_dir,
                function=None,
                ylabel="g2_Dose",
                g2_labels=None,
                append_name="",
            )

        # Speckel Visiblity
        if run_xsvs:
            max_cts = get_max_countc(FD, roi_mask)
            qind, pixelist = roi.extract_label_indices(roi_mask)
            noqs = len(np.unique(qind))
            nopr = np.bincount(qind, minlength=(noqs + 1))[1:]
            # time_steps = np.array( utils.geometric_series(2,   len(imgs)   ) )
            time_steps = [0, 1]  # only run the first two levels
            num_times = len(time_steps)
            times_xsvs = exposuretime + (2 ** (np.arange(len(time_steps))) - 1) * timeperframe
            print("The max counts are: %s" % max_cts)

            ### Do historam
            if roi_avg is None:
                times_roi, mean_int_sets = cal_each_ring_mean_intensityc(
                    FD,
                    roi_mask,
                    timeperframe=None,
                )
                roi_avg = np.average(mean_int_sets, axis=0)

            t0 = time.time()
            spec_bins, spec_his, spec_std = xsvsp(
                FD,
                np.int_(roi_mask),
                norm=None,
                max_cts=int(max_cts + 2),
                bad_images=bad_frame_list,
                only_two_levels=True,
            )
            spec_kmean = np.array([roi_avg * 2**j for j in range(spec_his.shape[0])])
            run_time(t0)

            run_xsvs_all_lags = False
            if run_xsvs_all_lags:
                times_xsvs = exposuretime + lag_steps * acquisition_period
                if data_pixel is None:
                    data_pixel = Get_Pixel_Arrayc(FD, pixelist, norm=norm).get_data()
                t0 = time.time()
                spec_bins, spec_his, spec_std, spec_kmean = get_binned_his_std(
                    data_pixel, np.int_(ro_mask), lag_steps
                )
                run_time(t0)
            spec_pds = save_bin_his_std(
                spec_bins, spec_his, spec_std, filename=uid_ + "_spec_res.csv", path=data_dir
            )

            ML_val, KL_val, K_ = get_xsvs_fit(
                spec_his,
                spec_kmean,
                spec_std,
                max_bins=2,
                varyK=False,
            )

            # print( 'The observed average photon counts are: %s'%np.round(K_mean,4))
            # print( 'The fitted average photon counts are: %s'%np.round(K_,4))
            print(
                "The difference sum of average photon counts between fit and data are: %s"
                % np.round(abs(np.sum(spec_kmean[0, :] - K_)), 4)
            )
            print("#" * 30)
            qth = 10
            print("The fitted M for Qth= %s are: %s" % (qth, ML_val[qth]))
            print(K_[qth])
            print("#" * 30)

            plot_xsvs_fit(
                spec_his,
                ML_val,
                KL_val,
                K_mean=spec_kmean,
                spec_std=spec_std,
                xlim=[0, 10],
                vlim=[0.9, 1.1],
                uid=uid_,
                qth=qth_interest,
                logy=True,
                times=times_xsvs,
                q_ring_center=qr,
                path=data_dir,
            )

            plot_xsvs_fit(
                spec_his,
                ML_val,
                KL_val,
                K_mean=spec_kmean,
                spec_std=spec_std,
                xlim=[0, 15],
                vlim=[0.9, 1.1],
                uid=uid_,
                qth=None,
                logy=True,
                times=times_xsvs,
                q_ring_center=qr,
                path=data_dir,
            )

            ### Get contrast
            contrast_factorL = get_contrast(ML_val)
            spec_km_pds = save_KM(
                spec_kmean, KL_val, ML_val, qs=qr, level_time=times_xsvs, uid=uid_, path=data_dir
            )
            # print( spec_km_pds )

            plot_g2_contrast(
                contrast_factorL,
                g2,
                times_xsvs,
                taus,
                qr,
                vlim=[0.8, 1.2],
                qth=qth_interest,
                uid=uid_,
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
                uid=uid_,
                path=data_dir,
                legend_size=4,
            )

        md["mask_file"] = mask_path + mask_name
        md["mask"] = mask
        md["NOTEBOOK_FULL_PATH"] = None
        md["good_start"] = good_start
        md["bad_frame_list"] = bad_frame_list
        md["avg_img"] = avg_img
        md["roi_mask"] = roi_mask

        if scat_geometry == "gi_saxs":
            md["Qr"] = Qr
            md["Qz"] = Qz
            md["qval_dict"] = qval_dict
            md["beam_center_x"] = inc_x0
            md["beam_center_y"] = inc_y0
            md["beam_refl_center_x"] = refl_x0
            md["beam_refl_center_y"] = refl_y0

        elif scat_geometry == "saxs" or "gi_waxs":
            md["qr"] = qr
            # md['qr_edge'] = qr_edge
            md["qval_dict"] = qval_dict
            md["beam_center_x"] = center[1]
            md["beam_center_y"] = center[0]

        elif scat_geometry == "ang_saxs":
            md["qval_dict_v"] = qval_dict_v
            md["qval_dict_p"] = qval_dict_p
            md["beam_center_x"] = center[1]
            md["beam_center_y"] = center[0]

        md["beg"] = FD.beg
        md["end"] = FD.end
        md["metadata_file"] = data_dir + "md.csv-&-md.pkl"
        psave_obj(md, data_dir + "uid=%s_md" % uid[:6])  # save the setup parameters
        # psave_obj(  md, data_dir + 'uid=%s_md'%uid ) #save the setup parameters
        save_dict_csv(md, data_dir + "uid=%s_md.csv" % uid, "w")

        Exdt = {}
        if scat_geometry == "gi_saxs":
            for k, v in zip(
                [
                    "md",
                    "roi_mask",
                    "qval_dict",
                    "avg_img",
                    "mask",
                    "pixel_mask",
                    "imgsum",
                    "bad_frame_list",
                    "qr_1d_pds",
                ],
                [md, roi_mask, qval_dict, avg_img, mask, pixel_mask, imgsum, bad_frame_list, qr_1d_pds],
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
                    bad_frame_list,
                ],
            ):
                Exdt[k] = v
        elif scat_geometry == "gi_waxs":
            for k, v in zip(
                ["md", "roi_mask", "qval_dict", "avg_img", "mask", "pixel_mask", "imgsum", "bad_frame_list"],
                [md, roi_mask, qval_dict, avg_img, mask, pixel_mask, imgsum, bad_frame_list],
            ):
                Exdt[k] = v
        elif scat_geometry == "ang_saxs":
            for k, v in zip(
                [
                    "md",
                    "q_saxs",
                    "iq_saxs",
                    "roi_mask_v",
                    "roi_mask_p",
                    "qval_dict_v",
                    "qval_dict_p",
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
                    roi_mask_v,
                    roi_mask_p,
                    qval_dict_v,
                    qval_dict_p,
                    avg_img,
                    mask,
                    pixel_mask,
                    imgsum,
                    bad_frame_list,
                ],
            ):
                Exdt[k] = v

        if run_waterfall:
            Exdt["wat"] = wat
        if run_t_ROI_Inten:
            Exdt["times_roi"] = times_roi
            Exdt["mean_int_sets"] = mean_int_sets
        if run_one_time:
            if scat_geometry != "ang_saxs":
                for k, v in zip(["taus", "g2", "g2_fit_paras"], [taus, g2, g2_fit_paras]):
                    Exdt[k] = v
            else:
                for k, v in zip(["taus_v", "g2_v", "g2_fit_paras_v"], [taus_v, g2_v, g2_fit_paras_v]):
                    Exdt[k] = v
                for k, v in zip(["taus_p", "g2_p", "g2_fit_paras_p"], [taus_p, g2_p, g2_fit_paras_p]):
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

        export_xpcs_results_to_h5("uid=%s_Res.h5" % md["uid"], data_dir, export_dict=Exdt)
        # extract_dict = extract_xpcs_results_from_h5( filename = 'uid=%s_Res.h5'%md['uid'], import_dir = data_dir )
        # Creat PDF Report
        pdf_out_dir = os.path.join("/XF11ID/analysis/", CYCLE, username, "Results/")
        pdf_filename = "XPCS_Analysis_Report_for_uid=%s%s.pdf" % (uid, pdf_version)
        if run_xsvs:
            pdf_filename = "XPCS_XSVS_Analysis_Report_for_uid=%s%s.pdf" % (uid, pdf_version)
        # pdf_filename

        print(data_dir, uid[:6], pdf_out_dir, pdf_filename, username)

        make_pdf_report(
            data_dir,
            uid[:6],
            pdf_out_dir,
            pdf_filename,
            username,
            run_fit_form,
            run_one_time,
            run_two_time,
            run_four_time,
            run_xsvs,
            run_dose=run_dose,
            report_type=scat_geometry,
        )
        ## Attach the PDF report to Olog
        if att_pdf_report:
            os.environ["HTTPS_PROXY"] = "https://proxy:8888"
            os.environ["no_proxy"] = "cs.nsls2.local,localhost,127.0.0.1"
            pname = pdf_out_dir + pdf_filename
            atch = [Attachment(open(pname, "rb"))]
            try:
                update_olog_uid(uid=md["uid"], text="Add XPCS Analysis PDF Report", attachments=atch)
            except:
                print(
                    "I can't attach this PDF: %s due to a duplicated filename. Please give a different PDF file."
                    % pname
                )

        if show_plot:
            plt.show()
        # else:
        #    plt.close('all')
        if clear_plot:
            plt.close("all")
        if return_res:
            res = {}
            if scat_geometry == "saxs":
                for k, v in zip(
                    [
                        "md",
                        "q_saxs",
                        "iq_saxs",
                        "iqst",
                        "qt",
                        "avg_img",
                        "mask",
                        "imgsum",
                        "bad_frame_list",
                        "roi_mask",
                        "qval_dict",
                    ],
                    [md, q_saxs, iq_saxs, iqst, qt, avg_img, mask, imgsum, bad_frame_list, roi_mask, qval_dict],
                ):
                    res[k] = v

            elif scat_geometry == "ang_saxs":
                for k, v in zip(
                    [
                        "md",
                        "q_saxs",
                        "iq_saxs",
                        "roi_mask_v",
                        "roi_mask_p",
                        "qval_dict_v",
                        "qval_dict_p",
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
                        roi_mask_v,
                        roi_mask_p,
                        qval_dict_v,
                        qval_dict_p,
                        avg_img,
                        mask,
                        pixel_mask,
                        imgsum,
                        bad_frame_list,
                    ],
                ):
                    res[k] = v

            elif scat_geometry == "gi_saxs":
                for k, v in zip(
                    [
                        "md",
                        "roi_mask",
                        "qval_dict",
                        "avg_img",
                        "mask",
                        "pixel_mask",
                        "imgsum",
                        "bad_frame_list",
                        "qr_1d_pds",
                    ],
                    [md, roi_mask, qval_dict, avg_img, mask, pixel_mask, imgsum, bad_frame_list, qr_1d_pds],
                ):
                    res[k] = v

            elif scat_geometry == "gi_waxs":
                for k, v in zip(
                    ["md", "roi_mask", "qval_dict", "avg_img", "mask", "pixel_mask", "imgsum", "bad_frame_list"],
                    [md, roi_mask, qval_dict, avg_img, mask, pixel_mask, imgsum, bad_frame_list],
                ):
                    res[k] = v

            if run_waterfall:
                res["wat"] = wat
            if run_t_ROI_Inten:
                res["times_roi"] = times_roi
                res["mean_int_sets"] = mean_int_sets
            if run_one_time:
                if scat_geometry != "ang_saxs":
                    res["g2"] = g2
                    res["taus"] = taus
                else:
                    res["g2_p"] = g2_p
                    res["taus_p"] = taus_p
                    res["g2_v"] = g2_v
                    res["taus_v"] = taus_v

            if run_two_time:
                res["tausb"] = tausb
                res["g12b"] = g12b
                res["g2b"] = g2b
            if run_four_time:
                res["g4"] = g4
                res["taus4"] = taus4
            if run_xsvs:
                res["spec_kmean"] = spec_kmean
                res["spec_pds"] = spec_pds
                res["contrast_factorL"] = contrast_factorL
                res["times_xsvs"] = times_xsvs
            return res


# uid = '3ff4ee'
# run_xpcs_xsvs_single( uid,  run_pargs )
