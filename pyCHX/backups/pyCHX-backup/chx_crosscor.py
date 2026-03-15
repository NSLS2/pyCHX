# Develop new version
# Original from #/XF11ID/analysis/Analysis_Pipelines/Develop/chxanalys/chxanalys/chx_correlation.py
# ######################################################################
# Let's change from mask's to indices
########################################################################

"""
This module is for functions specific to spatial correlation in order to tackle the motion of speckles
"""
from __future__ import absolute_import, division, print_function

from collections import namedtuple

import numpy as np
from scipy.signal import fftconvolve
from skbeam.core.roi import extract_label_indices

# from __future__ import absolute_import, division, print_function
from skbeam.core.utils import multi_tau_lags

# for a convenient status bar
try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterator):
        return iterator


from scipy.fftpack.helper import next_fast_len


def get_cor_region(cor, cij, qid, fitw):
    """YG developed@CHX July/2019, Get a rectangle region of the cor class by giving center and width"""
    ceni = cor.centers[qid]
    x1, x2, y1, y2 = max(0, ceni[0] - fitw), ceni[0] + fitw, max(0, ceni[1] - fitw), ceni[1] + fitw
    return cij[qid][x1:x2, y1:y2]


def direct_corss_cor(im1, im2):
    """YG developed@CHX July/2019, directly calculate the cross correlation of two images
    Input:
        im1: the first image
        im2: the second image
    Return:
        The cross correlation
    """
    sx, sy = im1.shape
    Nx, Ny = sx // 2, sy // 2
    C = np.zeros([2 * Nx, 2 * Ny])
    for i in range(-Nx, Nx):
        for j in range(-Ny, Ny):
            if i == 0:
                if j == 0:
                    d1 = im1[:, :]
                    d2 = im2[:, :]
                elif j < 0:
                    d1 = im1[:j, :]
                    d2 = im2[-j:, :]
                else:  ##j>0
                    d1 = im1[j:, :]
                    d2 = im2[:-j, :]
            elif i < 0:
                if j == 0:
                    d1 = im1[:, :i]
                    d2 = im2[:, -i:]
                elif j < 0:
                    d1 = im1[:j, :i]
                    d2 = im2[-j:, -i:]
                else:  ##j>0
                    d1 = im1[j:, :i]
                    d2 = im2[:-j, -i:]
            else:  # i>0:
                if j == 0:
                    d1 = im1[:, i:]
                    d2 = im2[:, :-i]
                elif j < 0:
                    d1 = im1[:j, i:]
                    d2 = im2[-j:, :-i]
                else:  ##j>0
                    d1 = im1[j:, i:]
                    d2 = im2[:-j, :-i]
            # print(i,j)
            C[i + Nx, j + Ny] = np.sum(d1 * d2) / (np.average(d1) * np.average(d2) * d1.size)
    return C.T


class CrossCorrelator2:
    """
    Compute a 1D or 2D cross-correlation on data.
    This uses a mask, which may be binary (array of 0's and 1's),
    or a list of non-negative integer id's to compute cross-correlations
    separately on.
    The symmetric averaging scheme introduced here is inspired by a paper
    from Schatzel, although the implementation is novel in that it
    allows for the usage of arbitrary masks. [1]_
    Examples
    --------
    >> ccorr = CrossCorrelator(mask.shape, mask=mask)
    >> # correlated image
    >> cimg = cc(img1)
    or, mask may may be ids
    >> cc = CrossCorrelator(ids)
    #(where ids is same shape as img1)
    >> cc1 = cc(img1)
    >> cc12 = cc(img1, img2)
    # if img2 shifts right of img1, point of maximum correlation is shifted
    # right from correlation center
    References
    ----------
    .. [1] Schatzel, Klaus, Martin Drewel, and Sven Stimac. "Photon
           correlation measurements at large lag times: improving
           statistical accuracy." Journal of Modern Optics 35.4 (1988):
           711-718.
    """

    # TODO : when mask is None, don't compute a mask, submasks
    def __init__(self, shape, mask=None, normalization=None, progress_bar=True):
        """
        Prepare the spatial correlator for various regions specified by the
        id's in the image.
        Parameters
        ----------
        shape : 1 or 2-tuple
            The shape of the incoming images or curves. May specify 1D or
            2D shapes by inputting a 1 or 2-tuple
        mask : 1D or 2D np.ndarray of int, optional
            Each non-zero integer represents unique bin. Zero integers are
            assumed to be ignored regions. If None, creates a mask with
            all points set to 1
        normalization: string or list of strings, optional
            These specify the normalization and may be any of the
            following:
                'regular' : divide by pixel number
                'symavg' : use symmetric averaging
            Defaults to ['regular'] normalization
        Delete argument wrap as not used. See fftconvolve as this
        expands arrays to get complete convolution, IE no need
        to expand images of subregions.
        """
        if normalization is None:
            normalization = ["regular"]
        elif not isinstance(normalization, list):
            normalization = list([normalization])
        self.normalization = normalization
        self.progress_bar = progress_bar
        if mask is None:  # we can do this easily now.
            mask = np.ones(shape)

        # initialize subregion information for the correlations
        # first find indices of subregions and sort them by subregion id
        pii, pjj = np.where(mask)
        bind = mask[pii, pjj]
        ord = np.argsort(bind)
        bind = bind[ord]
        pii = pii[ord]
        pjj = pjj[ord]  # sort them all

        # make array of pointers into position arrays
        pos = np.append(0, 1 + np.where(np.not_equal(bind[1:], bind[:-1]))[0])
        pos = np.append(pos, len(bind))
        self.pos = pos
        self.ids = bind[pos[:-1]]
        self.nids = len(self.ids)
        sizes = np.array(
            [
                [
                    pii[pos[i] : pos[i + 1]].min(),
                    pii[pos[i] : pos[i + 1]].max(),
                    pjj[pos[i] : pos[i + 1]].min(),
                    pjj[pos[i] : pos[i + 1]].max(),
                ]
                for i in range(self.nids)
            ]
        )
        self.pii = pii
        self.pjj = pjj
        self.offsets = sizes[:, 0:3:2].copy()
        # WE now have two sets of positions of the subregions
        # (pii-offsets[0],pjj-offsets[1]) in subregion and (pii,pjj) in
        # images. pos is a pointer such that (pos[i]:pos[i+1])
        # are the indices in the position arrays of subregion i.

        self.sizes = 1 + (np.diff(sizes)[:, [0, 2]]).copy()  # make sizes be for regions
        centers = np.array(self.sizes.copy()) // 2
        self.centers = centers
        if len(self.ids) == 1:
            self.centers = self.centers[0, :]

    def __call__(self, img1, img2=None, normalization=None, check_res=False):
        """Run the cross correlation on an image/curve or against two
            images/curves
        Parameters
        ----------
        img1 : 1D or 2D np.ndarray
            The image (or curve) to run the cross correlation on
        img2 : 1D or 2D np.ndarray
            If not set to None, run cross correlation of this image (or
            curve) against img1. Default is None.
        normalization : string or list of strings
            normalization types. If not set, use internally saved
            normalization parameters
        Returns
        -------
        ccorrs : 1d or 2d np.ndarray
            An image of the correlation. The zero correlation is
            located at shape//2 where shape is the 1 or 2-tuple
            shape of the array
        """
        progress_bar = self.progress_bar
        if normalization is None:
            normalization = self.normalization

        if img2 is None:
            self_correlation = True
        else:
            self_correlation = False

        ccorrs = list()

        pos = self.pos
        # loop over individual regions
        if progress_bar:
            R = tqdm(range(self.nids))
        else:
            R = range(self.nids)
        for reg in R:
            # for reg in tqdm(range(self.nids)): #for py3.5
            ii = self.pii[pos[reg] : pos[reg + 1]]
            jj = self.pjj[pos[reg] : pos[reg + 1]]
            i = ii.copy() - self.offsets[reg, 0]
            j = jj.copy() - self.offsets[reg, 1]
            # set up size for fft with padding
            shape = 2 * self.sizes[reg, :] - 1
            fshape = [next_fast_len(int(d)) for d in shape]
            # fslice = tuple([slice(0, int(sz)) for sz in shape])

            submask = np.zeros(self.sizes[reg, :])
            submask[i, j] = 1
            mma1 = np.fft.rfftn(submask, fshape)  # for mask
            # do correlation by ffts
            maskcor = np.fft.irfftn(mma1 * mma1.conj(), fshape)  # [fslice])
            # print(reg, maskcor)
            # maskcor = _centered(np.fft.fftshift(maskcor), self.sizes[reg,:]) #make smaller??
            maskcor = _centered(maskcor, self.sizes[reg, :])  # make smaller??
            # choose some small value to threshold
            maskcor *= maskcor > 0.5
            tmpimg = np.zeros(self.sizes[reg, :])
            tmpimg[i, j] = img1[ii, jj]
            im1 = np.fft.rfftn(tmpimg, fshape)  # image 1
            if self_correlation:
                # ccorr = np.real(np.fft.ifftn(im1 * im1.conj(), fshape)[fslice])
                ccorr = np.fft.irfftn(im1 * im1.conj(), fshape)  # [fslice])
                # ccorr = np.fft.fftshift(ccorr)
                ccorr = _centered(ccorr, self.sizes[reg, :])
            else:
                ndim = img1.ndim
                tmpimg2 = np.zeros_like(tmpimg)
                tmpimg2[i, j] = img2[ii, jj]
                im2 = np.fft.rfftn(tmpimg2, fshape)  # image 2
                ccorr = np.fft.irfftn(im1 * im2.conj(), fshape)  # [fslice])
                # ccorr = _centered(np.fft.fftshift(ccorr), self.sizes[reg,:])
                ccorr = _centered(ccorr, self.sizes[reg, :])
                # print('here')

                ###check here
                if check_res:
                    if reg == 0:
                        self.norm = maskcor
                        self.ck = ccorr.copy()
                        #    print(ccorr.max())
                        self.tmp = tmpimg
                        self.fs = fshape
                    ###end the check

            # now handle the normalizations
            if "symavg" in normalization:
                mim1 = np.fft.rfftn(tmpimg * submask, fshape)
                Icorr = np.fft.irfftn(mim1 * mma1.conj(), fshape)  # [fslice])
                # Icorr = _centered(np.fft.fftshift(Icorr), self.sizes[reg,:])
                Icorr = _centered(Icorr, self.sizes[reg, :])
                # do symmetric averaging
                if self_correlation:
                    Icorr2 = np.fft.irfftn(mma1 * mim1.conj(), fshape)  # [fslice])
                    # Icorr2 = _centered(np.fft.fftshift(Icorr2), self.sizes[reg,:])
                    Icorr2 = _centered(Icorr2, self.sizes[reg, :])
                else:
                    mim2 = np.fft.rfftn(tmpimg2 * submask, fshape)
                    Icorr2 = np.fft.irfftn(mma1 * mim2.conj(), fshape)
                    # Icorr2 = _centered(np.fft.fftshift(Icorr2), self.sizes[reg,:])
                    Icorr2 = _centered(Icorr2, self.sizes[reg, :])
                # there is an extra condition that Icorr*Icorr2 != 0
                w = np.where(np.abs(Icorr * Icorr2) > 0)  # DO WE NEED THIS (use i,j).
                ccorr[w] *= maskcor[w] / Icorr[w] / Icorr2[w]
                # print 'size:',tmpimg.shape,Icorr.shape
                if check_res:
                    if reg == 0:
                        self.ckn = ccorr.copy()
            if "regular" in normalization:
                # only run on overlapping regions for correlation
                w = np.where(maskcor > 0.5)

                if self_correlation:
                    ccorr[w] /= maskcor[w] * np.average(tmpimg[w]) ** 2
                else:
                    ccorr[w] /= maskcor[w] * np.average(tmpimg[w]) * np.average(tmpimg2[w])
                    if check_res:
                        if reg == 0:
                            self.ckn = ccorr.copy()
                    #    print('here')
                    #    print( np.average(tmpimg[w]) )
                    #    print(  maskcor[w] )
                    #    print(  ccorr.max(), maskcor[w], np.average(tmpimg[w]), np.average(tmpimg2[w]) )
            ccorrs.append(ccorr)

        if len(ccorrs) == 1:
            ccorrs = ccorrs[0]

        return ccorrs


def _centered(img, sz):
    n = sz // 2
    # ind=np.r_[-n[0]:0,0:sz[0]-n[0]]
    img = np.take(img, np.arange(-n[0], sz[0] - n[0]), 0, mode="wrap")
    # ind=np.r_[-n[1]:0,0:sz[1]-n[1]]
    img = np.take(img, np.arange(-n[1], sz[1] - n[1]), 1, mode="wrap")
    return img


##define a custmoized fftconvolve

########################################################################################
# modifided version from signaltools.py in scipy (Mark March 2017)
# Author: Travis Oliphant
# 1999 -- 2002


import threading
import warnings

# from . import sigtools
import numpy as np
from numpy import (
    allclose,
    angle,
    arange,
    argsort,
    array,
    asarray,
    atleast_1d,
    atleast_2d,
    cast,
    dot,
    exp,
    expand_dims,
    iscomplexobj,
    isscalar,
    mean,
    ndarray,
    newaxis,
    ones,
    pi,
    poly,
    polyadd,
    polyder,
    polydiv,
    polymul,
    polysub,
    polyval,
    prod,
    product,
    r_,
    ravel,
    real_if_close,
    reshape,
    roots,
    sort,
    sum,
    take,
    transpose,
    unique,
    where,
    zeros,
    zeros_like,
)
from numpy.fft import irfftn, rfftn
from numpy.lib import NumpyVersion
from scipy import linalg
from scipy.fftpack import fft, fft2, fftfreq, fftn, ifft, ifft2, ifftn, ifftshift

# from ._arraytools import axis_slice, axis_reverse, odd_ext, even_ext, const_ext

_rfft_mt_safe = NumpyVersion(np.__version__) >= "1.9.0.dev-e24486e"

_rfft_lock = threading.Lock()


def fftconvolve_new(in1, in2, mode="full"):
    """Convolve two N-dimensional arrays using FFT.

    Convolve `in1` and `in2` using the fast Fourier transform method, with
    the output size determined by the `mode` argument.

    This is generally much faster than `convolve` for large arrays (n > ~500),
    but can be slower when only a few output values are needed, and can only
    output float arrays (int or object array inputs will be cast to float).

    Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`;from scipy.signal import fftconvolve
        if sizes of `in1` and `in2` are not equal then `in1` has to be the
        larger array.get_window
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:

        ``full``
           The output is the full discrete linear convolution
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.

    Returns
    -------
    out : array
        An N-dimensional array containing a subset of the discrete linear
        convolution of `in1` with `in2`.

    Examples
    --------
    Autocorrelation of white noise is an impulse.  (This is at least 100 times
    as fast as `convolve`.)

    >>> from scipy import signal
    >>> sig = np.random.randn(1000)
    >>> autocorr = signal.fftconvolve(sig, sig[::-1], mode='full')

    >>> import matplotlib.pyplot as plt
    >>> fig, (ax_orig, ax_mag) = plt.subplots(2, 1)
    >>> ax_orig.plot(sig)
    >>> ax_orig.set_title('White noise')
    >>> ax_mag.plot(np.arange(-len(sig)+1,len(sig)), autocorr)
    >>> ax_mag.set_title('Autocorrelation')
    >>> fig.tight_layout()
    >>> fig.show()

    Gaussian blur implemented using FFT convolution.  Notice the dark borders
    around the image, due to the zero-padding beyond its boundaries.
    The `convolve2d` function allows for other types of image boundaries,
    but is far slower.

    >>> from scipy import misc
    >>> lena = misc.lena()
    >>> kernel = np.outer(signal.gaussian(70, 8), signal.gaussian(70, 8))
    >>> blurred = signal.fftconvolve(lena, kernel, mode='same')

    >>> fig, (ax_orig, ax_kernel, ax_blurred) = plt.subplots(1, 3)
    >>> ax_orig.imshow(lena, cmap='gray')
    >>> ax_orig.set_title('Original')
    >>> ax_orig.set_axis_off()
    >>> ax_kernel.imshow(kernel, cmap='gray')
    >>> ax_kernel.set_title('Gaussian kernel')
    >>> ax_kernel.set_axis_off()
    >>> ax_blurred.imshow(blurred, cmap='gray')
    >>> ax_blurred.set_title('Blurred')
    >>> ax_blurred.set_axis_off()
    >>> fig.show()

    """
    in1 = asarray(in1)
    in2 = asarray(in2)

    if in1.ndim == in2.ndim == 0:  # scalar inputs
        return in1 * in2
    elif not in1.ndim == in2.ndim:
        raise ValueError("in1 and in2 should have the same dimensionality")
    elif in1.size == 0 or in2.size == 0:  # empty arrays
        return array([])

    s1 = array(in1.shape)
    s2 = array(in2.shape)
    complex_result = np.issubdtype(in1.dtype, np.complex) or np.issubdtype(in2.dtype, np.complex)
    shape = s1 + s2 - 1

    if mode == "valid":
        _check_valid_mode_shapes(s1, s2)

    # Speed up FFT by padding to optimal size for FFTPACK
    # expand by at least twice+1
    fshape = [_next_regular(int(d)) for d in shape]
    fslice = tuple([slice(0, int(sz)) for sz in shape])
    # Pre-1.9 NumPy FFT routines are not threadsafe.  For older NumPys, make
    # sure we only call rfftn/irfftn from one thread at a time.
    if not complex_result and (_rfft_mt_safe or _rfft_lock.acquire(False)):
        try:
            ret = irfftn(rfftn(in1, fshape) * rfftn(in2, fshape), fshape)[fslice].copy()
        finally:
            if not _rfft_mt_safe:
                _rfft_lock.release()
    else:
        # If we're here, it's either because we need a complex result, or we
        # failed to acquire _rfft_lock (meaning rfftn isn't threadsafe and
        # is already in use by another thread).  In either case, use the
        # (threadsafe but slower) SciPy complex-FFT routines instead.
        ret = ifftn(fftn(in1, fshape) * fftn(in2, fshape))[fslice].copy()
        if not complex_result:
            ret = ret.real

    if mode == "full":
        return ret
    elif mode == "same":
        return _centered(ret, s1)
    elif mode == "valid":
        return _centered(ret, s1 - s2 + 1)
    else:
        raise ValueError("Acceptable mode flags are 'valid'," " 'same', or 'full'.")


def _cross_corr1(img1, img2=None):
    """Compute the cross correlation of one (or two) images.
    Parameters
    ----------
    img1 : np.ndarray
        the image or curve to cross correlate
    img2 : 1d or 2d np.ndarray, optional
        If set, cross correlate img1 against img2.  A shift of img2
        to the right of img1 will lead to a shift of the point of
        highest correlation to the right.
        Default is set to None
    """
    ndim = img1.ndim

    if img2 is None:
        img2 = img1

    if img1.shape != img2.shape:
        errorstr = "Image shapes don't match. "
        errorstr += "(img1 : {},{}; img2 : {},{})".format(*img1.shape, *img2.shape)
        raise ValueError(errorstr)

    # need to reverse indices for second image
    # fftconvolve(A,B) = FFT^(-1)(FFT(A)*FFT(B))
    # but need FFT^(-1)(FFT(A(x))*conj(FFT(B(x)))) = FFT^(-1)(A(x)*B(-x))
    reverse_index = tuple([slice(None, None, -1) for i in range(ndim)])
    imgc = fftconvolve(img1, img2[reverse_index], mode="same")

    return imgc


class CrossCorrelator1:
    """
    Compute a 1D or 2D cross-correlation on data.
    This uses a mask, which may be binary (array of 0's and 1's),
    or a list of non-negative integer id's to compute cross-correlations
    separately on.
    The symmetric averaging scheme introduced here is inspired by a paper
    from Schätzel, although the implementation is novel in that it
    allows for the usage of arbitrary masks. [1]_
    Examples
    --------
    >> ccorr = CrossCorrelator(mask.shape, mask=mask)
    >> # correlated image
    >> cimg = cc(img1)
    or, mask may may be ids
    >> cc = CrossCorrelator(ids)
    #(where ids is same shape as img1)
    >> cc1 = cc(img1)
    >> cc12 = cc(img1, img2)
    # if img2 shifts right of img1, point of maximum correlation is shifted
    # right from correlation center
    References
    ----------
    .. [1] Schätzel, Klaus, Martin Drewel, and Sven Stimac. “Photon
           correlation measurements at large lag times: improving
           statistical accuracy.” Journal of Modern Optics 35.4 (1988):
           711-718.
    """

    # TODO : when mask is None, don't compute a mask, submasks
    def __init__(self, shape, mask=None, normalization=None):
        """
        Prepare the spatial correlator for various regions specified by the
        id's in the image.
        Parameters
        ----------
        shape : 1 or 2-tuple
            The shape of the incoming images or curves. May specify 1D or
            2D shapes by inputting a 1 or 2-tuple
        mask : 1D or 2D np.ndarray of int, optional
            Each non-zero integer represents unique bin. Zero integers are
            assumed to be ignored regions. If None, creates a mask with
            all points set to 1
        normalization: string or list of strings, optional
            These specify the normalization and may be any of the
            following:
                'regular' : divide by pixel number
                'symavg' : use symmetric averaging
            Defaults to ['regular'] normalization
        Delete argument wrap as not used. See fftconvolve as this
        expands arrays to get complete convolution, IE no need
        to expand images of subregions.
        """
        if normalization is None:
            normalization = ["regular"]
        elif not isinstance(normalization, list):
            normalization = list([normalization])
        self.normalization = normalization

        if mask is None:  # we can do this easily now.
            mask = np.ones(shape)

        # initialize subregions information for the correlation
        # first find indices of subregions and sort them by subregion id
        pii, pjj = np.where(mask)
        bind = mask[pii, pjj]
        ord = np.argsort(bind)
        bind = bind[ord]
        pii = pii[ord]
        pjj = pjj[ord]  # sort them all

        # make array of pointers into position arrays
        pos = np.append(0, 1 + np.where(np.not_equal(bind[1:], bind[:-1]))[0])
        pos = np.append(pos, len(bind))
        self.pos = pos
        self.ids = bind[pos[:-1]]
        self.nids = len(self.ids)
        sizes = np.array(
            [
                [
                    pii[pos[i] : pos[i + 1]].min(),
                    pii[pos[i] : pos[i + 1]].max(),
                    pjj[pos[i] : pos[i + 1]].min(),
                    pjj[pos[i] : pos[i + 1]].max(),
                ]
                for i in range(self.nids)
            ]
        )
        # make indices for subregions arrays and their sizes
        pi = pii.copy()
        pj = pjj.copy()
        for i in range(self.nids):
            pi[pos[i] : pos[i + 1]] -= sizes[i, 0]
            pj[pos[i] : pos[i + 1]] -= sizes[i, 2]
        self.pi = pi
        self.pj = pj
        self.pii = pii
        self.pjj = pjj
        sizes = 1 + (np.diff(sizes)[:, [0, 2]])  # make sizes be for regions
        self.sizes = sizes.copy()  # the shapes of each correlation
        # WE now have two sets of positions of the subregions (pi,pj) in subregion
        # and (pii,pjj) in images. pos is a pointer such that (pos[i]:pos[i+1])
        # is the indices in the position arrays of subregion i.

        # Making a list of arrays holding the masks for each id. Ideally, mask
        # is binary so this is one element to quickly index original images
        self.submasks = list()
        self.centers = list()
        # the positions of each axes of each correlation
        self.positions = list()
        self.maskcorrs = list()
        # regions where the correlations are not zero
        self.pxlst_maskcorrs = list()

        # basically saving bunch of mask related stuff like indexing etc, just
        # to save some time when actually computing the cross correlations
        for id in range(self.nids):
            submask = np.zeros(self.sizes[id, :])
            submask[pi[pos[id] : pos[id + 1]], pj[pos[id] : pos[id + 1]]] = 1
            self.submasks.append(submask)

            maskcorr = _cross_corr1(submask)
            # quick fix for             #if self.wrap is False:
            #    submask = _expand_image1(submask)finite numbers should be integer so
            # choose some small value to threshold
            maskcorr *= maskcorr > 0.5
            self.maskcorrs.append(maskcorr)
            self.pxlst_maskcorrs.append(maskcorr > 0)
            # centers are shape//2 as performed by fftshift
            center = np.array(maskcorr.shape) // 2
            self.centers.append(np.array(maskcorr.shape) // 2)
            if mask.ndim == 1:
                self.positions.append(np.arange(maskcorr.shape[0]) - center[0])
            elif mask.ndim == 2:
                self.positions.append(
                    [np.arange(maskcorr.shape[0]) - center[0], np.arange(maskcorr.shape[1]) - center[1]]
                )

        if len(self.ids) == 1:
            self.positions = self.positions[0]
            self.centers = self.centers[0]

    def __call__(self, img1, img2=None, normalization=None, desc="cc"):
        """Run the cross correlation on an image/curve or against two
            images/curves
        Parameters
        ----------
        img1 : 1D or 2D np.ndarray
            The image (or curve) to run the cross correlation on
        img2 : 1D or 2D np.ndarray
            If not set to None, run cross correlation of this image (or
            curve) against img1. Default is None.
        normalization : string or list of strings
            normalization types. If not set, use internally saved
            normalization parameters
        Returns
        -------
        ccorrs : 1d or 2d np.ndarray
            An image of the correlation. The zero correlation is
            located at shape//2 where shape is the 1 or 2-tuple
            shape of the array
        """
        if normalization is None:
            normalization = self.normalization

        if img2 is None:
            self_correlation = True
            # img2 = img1
        else:
            self_correlation = False

        ccorrs = list()
        rngiter = tqdm(range(self.nids), desc=desc)

        pos = self.pos
        for reg in rngiter:
            i = self.pi[pos[reg] : pos[reg + 1]]
            j = self.pj[pos[reg] : pos[reg + 1]]
            ii = self.pii[pos[reg] : pos[reg + 1]]
            jj = self.pjj[pos[reg] : pos[reg + 1]]
            tmpimg = np.zeros(self.sizes[reg, :])
            tmpimg[i, j] = img1[ii, jj]
            if not self_correlation:
                tmpimg2 = np.zeros_like(tmpimg)
                tmpimg2[i, j] = img2[ii, jj]

            if self_correlation:
                ccorr = _cross_corr1(tmpimg)
            else:
                ccorr = _cross_corr1(tmpimg, tmpimg2)
            # now handle the normalizations
            if "symavg" in normalization:
                # do symmetric averaging
                Icorr = _cross_corr1(tmpimg * self.submasks[reg], self.submasks[reg])
                if self_correlation:
                    Icorr2 = _cross_corr1(self.submasks[reg], tmpimg * self.submasks[reg])
                else:
                    Icorr2 = _cross_corr1(self.submasks[reg], tmpimg2 * self.submasks[reg])
                # there is an extra condition that Icorr*Icorr2 != 0
                w = np.where(np.abs(Icorr * Icorr2) > 0)  # DO WE NEED THIS (use i,j).
                ccorr[w] *= self.maskcorrs[reg][w] / Icorr[w] / Icorr2[w]

            if "regular" in normalization:
                # only run on overlapping regions for correlation
                w = self.pxlst_maskcorrs[reg]  # NEED THIS?

                if self_correlation:
                    ccorr[w] /= self.maskcorrs[reg][w] * np.average(tmpimg[w]) ** 2
                else:
                    ccorr[w] /= self.maskcorrs[reg][w] * np.average(tmpimg[w]) * np.average(tmpimg2[w])
            ccorrs.append(ccorr)

        if len(ccorrs) == 1:
            ccorrs = ccorrs[0]

        return ccorrs


##for parallel
from multiprocessing import Pool

import dill

from pyCHX.chx_compress import apply_async, map_async


def run_para_ccorr_sym(ccorr_sym, FD, nstart=0, nend=None, imgsum=None, img_norm=None):
    """
    example:
    ccorr_sym = CrossCorrelator2(roi_mask.shape, mask=roi_mask, normalization='symavg')
    img_norm = get_img_from_iq( qp_saxs, iq_saxs, roi_mask.shape, center)

    """

    if nend is None:
        nend = FD.end - 1
    if nend > FD.end - 1:
        nend = FD.end - 1
    N = nend - nstart
    if imgsum is None:
        imgsum = np.ones(N)
    if img_norm is None:
        img_norm = 1.0
    inputs = range(N)
    pool = Pool(processes=len(inputs))
    print("Starting assign the tasks...")
    results = {}
    for i in tqdm(range(nstart, nend)):
        # img1 = FD.rdframe(i)
        # img2 = FD.rdframe(i+1)
        results[i] = apply_async(
            pool,
            ccorr_sym,
            (FD.rdframe(i) / (imgsum[i] * img_norm), FD.rdframe(1 + i) / (imgsum[i + 1] * img_norm)),
        )
    pool.close()
    print("Starting running the tasks...")
    res = [results[k].get() for k in tqdm(list(sorted(results.keys())))]

    for i in inputs:
        if i == 0:
            cc = res[i]
            Nc = len(cc)
        else:
            cci = res[i]
            for j in range(Nc):
                cc[j] += cci[j]

    for i in range(Nc):
        cc[i] = cc[i] / N

    del results
    del res

    return cc
