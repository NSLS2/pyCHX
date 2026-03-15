# simple brute force multitau
# from pyCHX.chx_generic_functions import average_array_withNan
import numpy as np
import skbeam.core.roi as roi
from numpy.fft import fft, ifft
from tqdm import tqdm


def fit_one_peak_curve(x, y, fit_range):
    """YG Dev@Aug 10, 2019 fit a curve with a single Lorentzian shape
    Parameters:
        x: one-d array, x-axis data
        y: one-d array, y-axis data
        fit_range: [x1, x2], a list of index, to define the x-range for fit
    Return:
        center:  float, center of the peak
        center_std: float, error bar of center in the fitting
        fwhm:  float, full width at half max intensity of the peak, 2*sigma
        fwhm_std:float, error bar of the full width at half max intensity of the peak
        xf: the x in the fit
        out: the fitting class resutled from lmfit

    """
    from lmfit.models import LinearModel, LorentzianModel

    peak = LorentzianModel()
    background = LinearModel()
    model = peak + background
    x1, x2 = xrange
    xf = x[x1:x2]
    yf = y[x1:x2]
    model.set_param_hint("slope", value=5)
    model.set_param_hint("intercept", value=0)
    model.set_param_hint("center", value=0.005)
    model.set_param_hint("amplitude", value=0.1)
    model.set_param_hint("sigma", value=0.003)
    # out=model.fit(yf, x=xf)#, method='nelder')
    out = model.fit(yf, x=xf, method="leastsq")
    cen = out.params["center"].value
    cen_std = out.params["center"].stderr
    wid = out.params["sigma"].value * 2
    wid_std = out.params["sigma"].stderr * 2
    return cen, cen_std, wid, wid_std, xf, out


def plot_xy_with_fit(x, y, xf, out, xlim=[1e-3, 0.01], xlabel="q (" r"$\AA^{-1}$)", ylabel="I(q)", filename=None):
    """YG Dev@Aug 10, 2019 to plot x,y with fit,
    currently this code is dedicated to plot q-Iq with fit and show the fittign parameter, peak pos, peak wid"""

    yf2 = out.model.eval(params=out.params, x=xf)
    fig, ax = plt.subplots()
    plot1D(x=x, y=y, ax=ax, m="o", ls="", c="k", legend="data")
    plot1D(x=xf, y=yf2, ax=ax, m="", ls="-", c="r", legend="fit", logy=True)
    ax.set_xlim(xlim)
    # ax.set_ylim( 0.1, 4)
    # ax.set_title(uid+'--t=%.2f'%tt)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    txts = r"peak" + r" = %.5f +/- %.5f " % (cen, cen_std)
    ax.text(x=0.02, y=0.2, s=txts, fontsize=14, transform=ax.transAxes)
    txts = r"wid" + r" = %.4f +/- %.4f" % (wid, wid_std)
    # txts = r'$\beta$' + r'$ = %.3f$'%(beta[i]) +  r'$ s^{-1}$'
    ax.text(x=0.02, y=0.1, s=txts, fontsize=14, transform=ax.transAxes)
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    return ax


#############For APD detector
def get_pix_g2_fft(time_inten):
    """YG Dev@CHX 2018/12/4 get g2 for oneD intensity
        g2 = G/(P*F)
        G = <I(t) * I(t+dt)>
        P =  <I(t)>
        F = <I(t+dt)>
    Input:
        time_inten: 1d-array,
                    a time dependent intensity for one pixel
    Return:
        G/(P*F)
    """
    G = get_pix_g2_G(time_inten)
    P, F = get_pix_g2_PF(time_inten)
    return G / (P * F)


def get_pix_g2_G(time_inten):
    """YG Dev@CHX 2018/12/4 get G for oneD intensity
        g2 = G/(P*F)
        G = <I(t) * I(t+dt)>
        P =  <I(t)>
        F = <I(t+dt)>
    Input:
        time_inten: 1d-array,
                    a time dependent intensity for one pixel
    Return:
        G
    """
    L = len(time_inten)
    norm = np.arange(L, 0, -1)
    return np.correlate(time_inten, time_inten, mode="full")[L - 1 :] / norm


def get_pix_g2_PF(time_inten):
    """YG Dev@CHX 2018/12/4 get past and future intensity in the normalization of g2
        g2 = G/(P*F)
        G = <I(t) * I(t+dt)>
        P =  <I(t)>
        F = <I(t+dt)>
    Input:
        time_inten: 1d-array,
                    a time dependent intensity for one pixel
    Return:
        P, F
    """

    cum_sum = np.cumsum(time_inten)[::-1]
    cum_Norm = np.arange(time_inten.shape[0], 0, -1)
    P = cum_sum / cum_Norm

    cum_sum2 = np.cumsum(time_inten[::-1], axis=0)[::-1]
    cum_Norm2 = np.arange(time_inten.shape[0], 0, -1)
    F = cum_sum2 / cum_Norm2
    return P, F


###################


def get_ab_correlation(a, b):
    """YG 2018/11/05/ derived from pandas.frame corrwith method
    Get correlation of two one-d array,  formula-->
        A = ( ( a-a.mean() ) * (b-b.mean()) ).sum
        B = ( len(a)-1 ) * a.std() * b.std()
        Cor = A/B
    Input:
        a: one-d  array
        b: one-d   array
    Output:
        c: one-d array, correlation

    """
    a = np.array(a)
    b = np.array(b)
    return ((a - a.mean()) * (b - b.mean())).sum() / ((len(a) - 1) * a.std() * b.std())


def get_oneQ_g2_fft(time_inten_oneQ, axis=0):
    """YG Dev@CHX 2018/10/15 get g2 for one Q by giving time_inten for that Q
        g2 = G/(P*F)
        G = <I(t) * I(t+dt)>
        P =  <I(t)>
        F = <I(t+dt)>
    Input:
        time_inten_oneQ: 2d-array,  shape=[time, pixel number in the ROI],
                    a time dependent intensity for a list of pixels
                     (   the equivilent pixels belongs to one Q    )
    Return:
        G/(P*F)
    """
    L = time_inten_oneQ.shape[0]
    P, F = get_g2_PF(time_inten_oneQ)
    G = auto_correlation_fft(time_inten_oneQ, axis=axis)
    G2f = np.average(G, axis=1) / L
    Pf = np.average(P, axis=1)
    Ff = np.average(F, axis=1)
    g2f = G2f / (Pf * Ff)
    return g2f


def get_g2_PF(time_inten):
    """YG Dev@CHX 2018/10/15 get past and future intensity in the normalization of g2
        g2 = G/(P*F)
        G = <I(t) * I(t+dt)>
        P =  <I(t)>
        F = <I(t+dt)>
    Input:
        time_inten: 2d-array, shape=[time, pixel number in the ROI],
                    a time dependent intensity for a list of pixels
    Return:
        P, F
    """

    cum_sum = np.cumsum(time_inten, axis=0)[::-1]
    cum_Norm = np.arange(time_inten.shape[0], 0, -1)
    P = cum_sum / cum_Norm[:, np.newaxis]

    cum_sum2 = np.cumsum(time_inten[::-1], axis=0)[::-1]
    cum_Norm2 = np.arange(time_inten.shape[0], 0, -1)
    F = cum_sum2 / cum_Norm2[:, np.newaxis]
    return P, F


def auto_correlation_fft_padding_zeros(a, axis=-1):
    """Y.G. Dev@CHX, 2018/10/15 Do autocorelation of ND array by fft
    Math:
        Based on auto_cor(arr) = ifft(  fft( arr ) * fft(arr[::-1]) )
        In numpy form
                 auto_cor(arr) = ifft(
                             fft( arr, n=2N-1, axis=axis ) ##padding enough zeros
                                                           ## for axis
                             * np.conjugate(               ## conju for reverse array
                             fft(arr , n=2N-1, axis=axis) )
                             )  #do reverse fft
    Input:
        a: 2d array,   shape=[time, pixel number in the ROI],
                    a time dependent intensity for a list of pixels
        axis: the axis for doing autocor
    Output:
        return: the autocorrelation array
            if a is one-d and two-d, return the cor with the same length of a
            else: return the cor with full length defined by 2*N-1

    """
    a = np.asarray(a)
    M = a.shape
    N = M[axis]
    # print(M, N, 2*N-1)
    cor = np.real(
        ifft(fft(a, n=N * 2 - 1, axis=axis) * np.conjugate(fft(a, n=N * 2 - 1, axis=axis)), n=N * 2 - 1, axis=axis)
    )

    if len(M) == 1:
        return cor[:N]
    elif len(M) == 2:
        if axis == -1 or axis == 1:
            return cor[:, :N]
        elif axis == 0 or axis == -2:
            return cor[:N]
    else:
        return cor


def auto_correlation_fft(a, axis=-1):
    """Y.G. Dev@CHX, 2018/10/15 Do autocorelation of ND array by fft
    Math:
        Based on auto_cor(arr) = ifft(  fft( arr ) * fft(arr[::-1]) )
        In numpy form
                 auto_cor(arr) = ifft(
                             fft( arr, n=2N-1, axis=axis ) ##padding enough zeros
                                                           ## for axis
                             * np.conjugate(               ## conju for reverse array
                             fft(arr , n=2N-1, axis=axis) )
                             )  #do reverse fft
    Input:
        a: 2d array,   shape=[time, pixel number in the ROI],
                    a time dependent intensity for a list of pixels
        axis: the axis for doing autocor
    Output:
        return: the autocorrelation array
            if a is one-d and two-d, return the cor with the same length of a
            else: return the cor with full length defined by 2*N-1

    """
    a = np.asarray(a)
    cor = np.real(ifft(fft(a, axis=axis) * np.conjugate(fft(a, axis=axis)), axis=axis))
    return cor


def multitau(Ipix, bind, lvl=12, nobuf=8):
    """
    tt,g2=multitau(Ipix,bind,lvl=12,nobuf=8)
       Ipix is matrix of no-frames by no-pixels
       bind is bin indicator, one per pixel. All pixels
       in the same bin have the same value of bind (bin indicator).
       This number of images save at each level is nobf(8) and the
       number of level is lvl(12).
     returns:
     tt is the offsets of each g2 point.
                 tt*timeperstep is time of points
         g2 is max(bind)+1,time steps.
            plot(tt[1:],g2[1:,i]) will plot each g2.
    """
    # if num_lev is None:
    #    num_lev = int(np.log( noframes/(num_buf-1))/np.log(2) +1) +1
    # print(nobuf,nolvl)
    nobins = bind.max() + 1
    nobufov2 = nobuf // 2
    # t0=time.time()
    noperbin = np.bincount(bind)
    G2 = np.zeros((nobufov2 * (1 + lvl), nobins))
    tt = np.zeros(nobufov2 * (1 + lvl))
    dII = Ipix.copy()
    t = np.bincount(bind, np.mean(dII, axis=0)) ** 2 / noperbin
    G2[0, :] = np.bincount(bind, np.mean(dII * dII, axis=0)) / t
    for j in np.arange(1, nobuf):
        tt[j] = j
        print(j, tt[j])
        # t=noperbin*np.bincount(bind,np.mean(dII,axis=0))**2
        t = (
            np.bincount(bind, np.mean(dII[j:, :], axis=0))
            * np.bincount(bind, np.mean(dII[:-j, :], axis=0))
            / noperbin
        )
        G2[j, :] = np.bincount(bind, np.mean(dII[j:, :] * dII[:-j, :], axis=0)) / t
    for l in tqdm(np.arange(1, lvl), desc="Calcuate g2..."):
        nn = dII.shape[0] // 2 * 2  # make it even
        dII = (dII[0:nn:2, :] + dII[1:nn:2, :]) / 2.0  # sum in pairs
        nn = nn // 2
        if nn < nobuf:
            break
        for j in np.arange(nobufov2, min(nobuf, nn)):
            ind = nobufov2 + nobufov2 * l + (j - nobufov2)
            tt[ind] = 2**l * j
            t = (
                np.bincount(bind, np.mean(dII[j:, :], axis=0))
                * np.bincount(bind, np.mean(dII[:-j, :], axis=0))
                / noperbin
            )
            G2[ind, :] = np.bincount(bind, np.mean(dII[j:, :] * dII[:-j, :], axis=0)) / t
    # print(ind)
    # print(time.time()-t0)
    return (tt[: ind + 1], G2[: ind + 1, :])


def average_array_withNan(array, axis=0, mask=None):
    """YG. Jan 23, 2018
    Average array invovling np.nan along axis

    Input:
        array: ND array, actually should be oneD or twoD at this stage..TODOLIST for ND
        axis: the average axis
        mask: bool, same shape as array, if None, will mask all the nan values
    Output:
        avg: averaged array along axis
    """
    shape = array.shape
    if mask is None:
        mask = np.isnan(array)
        # mask = np.ma.masked_invalid(array).mask
    array_ = np.ma.masked_array(array, mask=mask)
    try:
        sums = np.array(np.ma.sum(array_[:, :], axis=axis))
    except:
        sums = np.array(np.ma.sum(array_[:], axis=axis))

    cts = np.sum(~mask, axis=axis)
    # print(cts)
    return sums / cts


def autocor_for_pix_time(pix_time_data, dly_dict, pixel_norm=None, frame_norm=None, multi_tau_method=True):
    """YG Feb 20, 2018@CHX
    Do correlation for pixel_time type data with tau as defined as dly
    Input:
        pix_time_data: 2D array, shape as [number of frame (time), pixel list (each as one q) ]
        dly_dict: the taus dict, (use multi step e.g.)
        multi_tau_method: if True, using multi_tau_method to average data for higher level
        pixel_norm: if not None, should be array with shape as pixel number
        frame_norm: if not None, should be array with shape as frame number

    return:
        g2: with shape as [ pixel_list number, dly number ]

    """
    Nt, Np = pix_time_data.shape
    Ntau = len(np.concatenate(list(dly_dict.values())))
    G2 = np.zeros([Ntau, Np])
    Gp = np.zeros([Ntau, Np])
    Gf = np.zeros([Ntau, Np])
    # mask_pix = np.isnan(pix_time_data)
    # for tau_ind, tau in tqdm( enumerate(dly), desc= 'Calcuate g2...'  ):
    tau_ind = 0
    # if multi_tau_method:
    pix_time_datac = pix_time_data.copy()
    # else:
    if pixel_norm is not None:
        pix_time_datac /= pixel_norm
    if frame_norm is not None:
        pix_time_datac /= frame_norm

    for tau_lev, tau_key in tqdm(enumerate(list(dly_dict.keys())), desc="Calcuate g2..."):
        # print(tau_key)
        taus = dly_dict[tau_key]
        if multi_tau_method:
            if tau_lev > 0:
                nobuf = len(dly_dict[1])
                nn = pix_time_datac.shape[0] // 2 * 2  # make it even
                pix_time_datac = (pix_time_datac[0:nn:2, :] + pix_time_datac[1:nn:2, :]) / 2.0  # sum in pairs
                nn = nn // 2
                if nn < nobuf:
                    break
                # print(nn)
                # if(nn<1): break
            else:
                nn = pix_time_datac.shape[0]
        for tau in taus:
            if multi_tau_method:
                IP = pix_time_datac[: nn - tau // 2**tau_lev, :]
                IF = pix_time_datac[tau // 2**tau_lev : nn, :]
                # print( tau_ind, nn ,  tau//2**tau_lev, tau_lev+1,tau, IP.shape)
            else:
                IP = pix_time_datac[: Nt - tau, :]
                IF = pix_time_datac[tau:Nt, :]
                # print( tau_ind,   tau_lev+1,tau, IP.shape)

            # IP_mask = mask_pix[: Nt - tau,:]
            # IF_mask = mask_pix[tau: Nt,: ]
            # IPF_mask  = IP_mask | IF_mask
            # IPFm = average_array_withNan(IP*IF, axis = 0, )#mask= IPF_mask )
            # IPm =  average_array_withNan(IP, axis = 0, )#   mask= IP_mask )
            # IFm =  average_array_withNan(IF, axis = 0 , )#  mask= IF_mask )
            G2[tau_ind] = average_array_withNan(
                IP * IF,
                axis=0,
            )  # IPFm
            Gp[tau_ind] = average_array_withNan(
                IP,
                axis=0,
            )  # IPm
            Gf[tau_ind] = average_array_withNan(
                IF,
                axis=0,
            )  # IFm
            tau_ind += 1
    # for i in range(G2.shape[0]-1, 0, -1):
    #    if np.isnan(G2[i,0]):
    #        gmax = i
    gmax = tau_ind
    return G2[:gmax, :], Gp[:gmax, :], Gf[:gmax, :]


def autocor_xytframe(self, n):
    """Do correlation for one xyt frame--with data name as n"""

    data = read_xyt_frame(n)  # load data
    N = len(data)
    crl = correlate(data, data, "full")[N - 1 :]
    FN = arange(1, N + 1, dtype=float)[::-1]
    IP = cumsum(data)[::-1]
    IF = cumsum(data[::-1])[::-1]

    return crl / (IP * IF) * FN


###################For Fit

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import leastsq

# duplicate my curfit function from yorick, except use sigma and not w
# notice the main feature is an adjust list.


def curfit(x, y, a, sigy=None, function_name=None, adj=None):
    a = np.array(a)
    if adj is None:
        adj = np.arange(len(a), dtype="int32")
    if function_name is None:
        function_name = funct
    # print( a, adj, a[adj] )
    # print(x,y,a)
    afit, cv, idt, m, ie = leastsq(_residuals, a[adj], args=(x, y, sigy, a, adj, function_name), full_output=True)
    a[adj] = afit
    realcv = np.identity(afit.size)
    realcv[np.ix_(adj, adj)] = cv
    nresids = idt["fvec"]
    chisq = np.sum(nresids**2) / (len(y) - len(adj))
    # print( cv )
    # yfit=y-yfit*sigy
    sigmaa = np.zeros(len(a))
    sigmaa[adj] = np.sqrt(np.diag(cv))
    return (chisq, a, sigmaa, nresids)


# hidden residuals for leastsq
def _residuals(p, x, y, sigy, pall, adj, fun):
    # print(p, pall, adj )
    pall[adj] = p
    # print(pall)
    if sigy is None:
        return y - fun(x, pall)
    else:
        return (y - fun(x, pall)) / sigy


# print out fit result nicely
def fitpr(chisq, a, sigmaa, title=None, lbl=None):
    """nicely print out results of a fit"""
    # get fitted results.
    if lbl == None:
        lbl = []
        for i in xrange(a.size):
            lbl.append("A%(#)02d" % {"#": i})
    # print resuls of a fit.
    if title != None:
        print(title)
    print("   chisq=%(c).4f" % {"c": chisq})
    for i in range(a.size):
        print("     %(lbl)8s =%(m)10.4f +/- %(s).4f" % {"lbl": lbl[i], "m": a[i], "s": sigmaa[i]})


# easy plot for fit
def fitplot(x, y, sigy, yfit, pl=plt):
    pl.plot(x, yfit)
    pl.errorbar(x, y, fmt="o", yerr=sigy)


# define a default function, a straight line.
def funct(x, p):
    return p[0] * x + p[1]


# A 1D Gaussian
def Gaussian(x, p):
    """
    One-D Gaussian Function
    xo, amplitude, sigma, offset  = p


    """
    xo, amplitude, sigma, offset = p
    g = offset + amplitude * 1.0 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-1 / 2.0 * (x - xo) ** 2 / sigma**2)
    return g


###########For ellipse shaped sectors by users
def elps_r(a, b, theta):
    """
    Returns the radius of an ellipse with semimajor/minor axes a/b
    at angle theta (in radians)"""
    return a * b / np.sqrt(((b * np.cos(theta)) ** 2) + ((a * np.sin(theta)) ** 2))


def place_in_interval(val, interval_list):
    """
    For a sorted list interval_list, returns the bin index val belongs to (0-indexed)
    Returns -1 if outside of bins"""
    if val < interval_list[0] or val >= interval_list[-1]:
        return -1
    else:
        return np.argmax(val < interval_list) - 1


def gen_elps_sectors(a, b, r_min, r_n, th_n, c_x, c_y, th_min=0, th_max=360):
    """
    Returns a list of x/y coordinates of ellipsoidal sectors ROI. Cuts th_max - th_min degrees into
    th_n number of angular bins, and r_n number of radial bins, starting from r_min*r to r where
    r is the radius of the ellipsoid at that angle. The ROIs are centered around c_x, c_y. Defaults to 360 deg.

    Example:

        roi_mask = np.zeros_like( avg_img , dtype = np.int32)
        sectors = gen_elps_sectors(110,55,0.2,5,24,579,177)
        for ii,sector in enumerate(sectors):
                roi_mask[sector[1],sector[0]] = ii + 1

    """
    th_list = np.linspace(th_min, th_max, th_n + 1)
    r_list = np.linspace(r_min, 1, r_n + 1)
    regions_list = [
        [[np.array([], dtype=np.int_), np.array([], dtype=np.int_)] for _ in range(r_n)] for _ in range(th_n)
    ]
    w = int(np.ceil(a * 2))
    h = int(np.ceil(b * 2))
    x_offset = c_x - w // 2
    y_offset = c_y - h // 2
    for ii in range(w):
        cur_x = ii - (w - 1) // 2
        for jj in range(h):
            cur_y = jj - (h - 1) // 2
            cur_theta = np.arctan2(cur_y, cur_x) % (np.pi * 2)
            cur_r = np.sqrt(cur_x**2 + cur_y**2)
            cur_elps_r = elps_r(a, b, cur_theta)
            cur_r_list = r_list * cur_elps_r
            cur_theta = np.rad2deg(cur_theta)  # Convert to degrees to compare with th_list
            r_ind = place_in_interval(cur_r, cur_r_list)
            th_ind = place_in_interval(cur_theta, th_list)
            if (r_ind != -1) and (th_ind != -1):
                regions_list[th_ind][r_ind][0] = np.append(regions_list[th_ind][r_ind][0], ii + x_offset)
                regions_list[th_ind][r_ind][1] = np.append(regions_list[th_ind][r_ind][1], jj + y_offset)
    sectors = []
    for th_reg_list in regions_list:
        for sector in th_reg_list:
            sectors += [(sector[0], sector[1])]
    return sectors
