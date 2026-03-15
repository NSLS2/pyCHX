################################
######Movie_maker###############
################################


def read_imgs(inDir):
    """Give image folder: inDir
    Get a pims.sequences,
    e.g. inDir= '/home/.../*.png'
    """
    from pims import ImageSequence as Images

    return Images(inDir)


def select_regoin(
    img,
    vert,
    keep_shape=True,
    qmask=None,
):
    """Get a pixellist by a rectangular region
     defined by
     verts e.g. xs,xe,ys,ye = vert #x_start, x_end, y_start,y_end
     (dimy, dimx,) = img.shape
    Giving cut postion, start, end, width"""
    import numpy as np

    xs, xe, ys, ye = vert
    if keep_shape:
        img_ = np.zeros_like(img)
        # img_= np.zeros( [dimy,dimx])

        try:
            img_[ys:ye, xs:xe] = True
        except:
            img_[ys:ye, xs:xe, :] = True
        pixellist_ = np.where(img_.ravel())[0]
        # pixellist_ =  img_.ravel()
        if qmask is not None:
            b = np.where(qmask.flatten() == False)[0]
            pixellist_ = np.intersect1d(pixellist_, b)
        # imgx = img[pixellist_]
        # imgx = imgx.reshape( xe-xs, ye-ys)
        imgx = img_.ravel()
        imgx[pixellist_] = img.ravel()[pixellist_]
        imgx = imgx.reshape(img.shape)

    else:
        try:
            imgx = img[ys:ye, xs:xe]
        except:
            imgx = img[ys:ye, xs:xe, :]

    return imgx


def save_png_series(
    imgs, ROI=None, logs=True, outDir=None, uid=None, vmin=None, vmax=None, cmap="viridis", dpi=100
):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LogNorm

    """
    save a series of images in a format of png

    Parameters
    ----------
    imgs : array
        image data array for the movie
        dimensions are: [num_img][num_rows][num_cols]
    ROI: e.g. xs,xe,ys,ye = vert #x_start, x_end, y_start,y_end
    outDir: the output path
    vmin/vmax: for image contrast
    cmap: the color for plot
    dpi: resolution

    Returns
    -------
    save png files

    """
    if uid == None:
        uid = "uid"
    num_frame = 0
    for img in imgs:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if ROI is None:
            i0 = img
            asp = 1.0
        else:
            i0 = select_regoin(
                img,
                ROI,
                keep_shape=False,
            )
            xs, xe, ys, ye = ROI
            asp = (ye - ys) / float(xe - xs)
        ax.set_aspect("equal")

        if not logs:
            im = ax.imshow(
                i0, origin="lower", cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax
            )  # vmin=0,vmax=1,
        else:
            im = ax.imshow(i0, origin="lower", cmap=cmap, interpolation="nearest", norm=LogNorm(vmin, vmax))
        # ttl = ax.text(.75, .2, '', transform = ax.transAxes, va='center', color='white', fontsize=18)
        # fig.set_size_inches( [5., 5 * asp] )
        # plt.tight_layout()
        fname = outDir + "uid_%s-frame-%s.png" % (uid, num_frame)
        num_frame += 1
        plt.savefig(fname, dpi=None)


def movie_maker(
    imgs,
    num_frames=None,
    ROI=None,
    interval=20,
    fps=15,
    real_interval=1.0,
    movie_name="movie.mp4",
    outDir=None,
    movie_writer="ffmpeg",
    logs=True,
    show_text_on_image=False,
    vmin=None,
    vmax=None,
    cmap="viridis",
    dpi=100,
):
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LogNorm

    """
    Make a movie by give a image series

    Parameters
    ----------
    imgs : array
        image data array for the movie
        dimensions are: [num_img][num_rows][num_cols]
    ROI: e.g. xs,xe,ys,ye = vert #x_start, x_end, y_start,y_end

    num_frames : int
        number of frames in the array

    interval : int, optional
        delay between frames

    movie_name : str, optional
        name of the movie to save

    movie_writer : str, optional
        movie writer

    fps : int, optional
        Frame rate for movie.

    real_interval:
       the real time interval between each frame in unit of ms
    outDir: the output path
    vmin/vmax: for image contrast
    cmap: the color for plot
    dpi: resolution

    Returns
    -------
    #ani :
    #    movie

    """

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    if ROI is None:
        i0 = imgs[0]
        asp = 1.0

    else:
        i0 = select_regoin(
            imgs[0],
            ROI,
            keep_shape=False,
        )
        xs, xe, ys, ye = ROI
        asp = (ye - ys) / float(xe - xs)

    ax.set_aspect("equal")
    # print( cmap, vmin, vmax )

    if not logs:
        im = ax.imshow(i0, origin="lower", cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
    else:
        im = ax.imshow(i0, origin="lower", cmap=cmap, interpolation="nearest", norm=LogNorm(vmin, vmax))

    # ttl = ax.text(.75, .2, '', transform = ax.transAxes, va='center', color='white', fontsize=18)
    ttl = ax.text(0.75, 0.2, "", transform=ax.transAxes, va="center", color="black", fontsize=18)
    # print asp
    # fig.set_size_inches( [5., 5 * asp] )

    plt.tight_layout()

    if num_frames is None:
        num_frames = len(imgs)

    def update_img(n):
        if ROI is None:
            ign = imgs[n]
        else:
            ign = select_regoin(
                imgs[n],
                ROI,
                keep_shape=False,
            )
        im.set_data(ign)
        if show_text_on_image:
            if real_interval >= 10:
                ttl.set_text("%s s" % (n * real_interval / 1000.0))
            elif real_interval < 10:
                ttl.set_text("%s ms" % (n * real_interval))
            # im.set_text(n)
            # print (n)

    ani = animation.FuncAnimation(fig, update_img, num_frames, interval=interval)
    writer = animation.writers[movie_writer](fps=fps)

    if outDir is not None:
        movie_name = outDir + movie_name
    ani.save(movie_name, writer=writer, dpi=dpi)
    # return ani
