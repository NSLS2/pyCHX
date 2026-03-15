"""
Sep 10 Developed by Y.G.@CHX
yuzhang@bnl.gov
This module is for the static SAXS analysis, such as fit form factor
"""

# import numpy as np
from lmfit import Model, Parameter, Parameters, fit_report, minimize, report_fit
from scipy.optimize import curve_fit, least_squares, leastsq
from scipy.special import gamma, gammaln

from pyCHX.chx_generic_functions import find_index, plot1D, show_img

# import matplotlib as mpl
# import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm
from pyCHX.chx_libs import *


def mono_sphere_form_factor_intensity(x, radius, delta_rho=100, fit_func="G"):
    """
    Input:
        x/q: in A-1, array or a value
        radius/R: in A

        delta_rho: Scattering Length Density(SLD) difference between solvent and the scatter, A-2
    Output:
        The form factor intensity of the mono dispersed scatter
    """
    q = x
    R = radius
    qR = q * R
    volume = (4.0 / 3.0) * np.pi * (R**3)
    prefactor = 36 * np.pi * ((delta_rho * volume) ** 2) / (4 * np.pi)
    P = (np.sin(qR) - qR * np.cos(qR)) ** 2 / (qR**6)
    P *= prefactor
    P = P.real
    return P


def gaussion(x, u, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-((x - u) ** 2) / (2 * (sigma**2)))


def Schultz_Zimm(x, u, sigma):
    """http://sasfit.ingobressler.net/manual/Schultz-Zimm
    See also The size distribution of ‘gold standard’ nanoparticles
        Anal Bioanal Chem (2009) 395:1651–1660
        DOI 10.1007/s00216-009-3049-5
    """
    k = 1.0 / (sigma) ** 2
    return 1.0 / u * (x / u) ** (k - 1) * k**k * np.exp(-k * x / u) / gamma(k)


def distribution_func(radius=1.0, sigma=0.1, num_points=20, spread=3, func="G"):
    """
    radius: the central radius
    sigma: sqrt root of variance in percent
    """

    if 1 - spread * sigma <= 0:
        spread = (1 - sigma) / sigma - 1
    # print( num_points  )
    x, rs = np.linspace(
        radius - radius * spread * sigma, radius + radius * spread * sigma, int(num_points), retstep=True
    )
    # print(x)
    if func == "G":
        func = gaussion
    elif func == "S":
        func = Schultz_Zimm

    return x, rs, func(x, radius, radius * sigma)


def poly_sphere_form_factor_intensity(
    x, radius, sigma=0.1, delta_rho=1.00, background=0, num_points=20, spread=5, fit_func="G"
):
    """
    Input:
        x/q: in A-1, array or a value
        radius/R: in A
        sigma:sqrt root of variance in percent
        delta_rho: Scattering Length Density(SLD) difference between solvent and the scatter, A-2
        fit_func: G: Guassian;S:  Flory–Schulz distribution
    Output:
        The form factor intensity of the polydispersed scatter
    """
    q = x
    R = radius
    if not hasattr(q, "__iter__"):
        q = np.array([q])
    v = np.zeros((len(q)))
    if sigma == 0:
        v = mono_sphere_form_factor_intensity(q, R, delta_rho)
    else:
        r, rs, wt = distribution_func(radius=R, sigma=sigma, num_points=num_points, spread=spread, func=fit_func)
        for i, Ri in enumerate(r):
            # print(Ri, wt[i],delta_rho, rs)
            v += mono_sphere_form_factor_intensity(q, Ri, delta_rho) * wt[i] * rs
    return v + background  # * delta_rho


def poly_sphere_form_factor_intensity_q2(
    x, radius, sigma=0.1, delta_rho=1, fit_func="G"
):  # , scale=1, baseline=0):
    """
    Input:
        x/q: in A-1, array or a value
        radius/R: in A
        sigma:sqrt root of variance in percent
        delta_rho: Scattering Length Density(SLD) difference between solvent and the scatter, A-2
    Output:
        The form factor intensity of the polydispersed scatter
    """

    return poly_sphere_form_factor_intensity(x, radius, sigma, delta_rho, fit_func) * x**2  # * scale + baseline


def find_index_old(x, x0, tolerance=None):
    # find the position of P in a list (plist) with tolerance

    N = len(x)
    i = 0
    position = None
    if tolerance == None:
        tolerance = (x[1] - x[0]) / 2.0
    if x0 > max(x):
        position = len(x) - 1
    elif x0 < min(x):
        position = 0
    else:
        for item in x:
            if abs(item - x0) <= tolerance:
                position = i
                # print 'Found Index!!!'
                break
            i += 1
    return position


def form_factor_residuals(p, iq, q, num_points=20, spread=5, fit_func="G", form_model="poly_sphere"):
    """Residuals for fit iq by spheical form factor using leastsq.
    p: parameters for radius, sigma, delta_rho, background

    """

    radius, sigma, delta_rho, background = p
    fiq = poly_sphere_form_factor_intensity(
        q,
        radius=radius,
        sigma=sigma,
        delta_rho=delta_rho,
        background=background,
        num_points=num_points,
        spread=spread,
        fit_func=fit_func,
    )
    radius, sigma, delta_rho, background = abs(radius), abs(sigma), abs(delta_rho), abs(background)
    err = np.log(iq / fiq)
    return np.sqrt(np.abs(err))


def form_factor_residuals_bg(
    p, iq, q, num_points=20, spread=5, fit_func="G", form_model="poly_sphere"
):  # , qpower=-4.0,   ):
    """Residuals for fit iq by spheical form factor using leastsq.
    p: parameters for radius, sigma, delta_rho, background
    """

    radius, sigma, delta_rho, background, q4_scale, qpower = p
    bk = q4_scale * q ** (qpower) + background
    fiq = (
        poly_sphere_form_factor_intensity(
            q,
            radius=radius,
            sigma=sigma,
            delta_rho=delta_rho,
            background=0,
            num_points=num_points,
            spread=spread,
            fit_func=fit_func,
        )
        + bk
    )
    radius, sigma, delta_rho, background, q4_scale = (
        abs(radius),
        abs(sigma),
        abs(delta_rho),
        abs(background),
        abs(q4_scale),
    )
    err = np.log(iq / fiq)  # iq- (fiq + bk  )
    return np.sqrt(np.abs(err))


def form_factor_residuals_lmfit(p, iq, q, num_points=20, spread=5, fit_func="G", form_model="poly_sphere"):
    """Residuals for fit iq by spheical form factor using leastsq.
    p: parameters for radius, sigma, delta_rho, background
    """
    radius, sigma, delta_rho, background = (
        p["radius"],
        p["sigma"],
        p["delta_rho"],
        p["background"],
    )

    fiq = poly_sphere_form_factor_intensity(
        q,
        radius=radius,
        sigma=sigma,
        delta_rho=delta_rho,
        background=background,
        num_points=num_points,
        spread=spread,
        fit_func=fit_func,
    )
    err = np.log(iq / fiq)
    return err


def form_factor_residuals_bg_lmfit(p, iq, q, num_points=20, spread=5, fit_func="G", form_model="poly_sphere"):
    """Residuals for fit iq by spheical form factor using leastsq.
    p: parameters for radius, sigma, delta_rho, background
    """
    radius, sigma, delta_rho, background, qpower_scale, qpower = (
        p["radius"],
        p["sigma"],
        p["delta_rho"],
        p["background"],
        p["qpower_scale"],
        p["qpower"],
    )
    bk = qpower_scale * q ** (qpower) + background
    fiq = (
        poly_sphere_form_factor_intensity(
            q,
            radius=radius,
            sigma=sigma,
            delta_rho=delta_rho,
            background=0,
            num_points=num_points,
            spread=spread,
            fit_func=fit_func,
        )
        + bk
    )
    err = np.log(iq / fiq)
    return err


def get_form_factor_fit_lmfit(
    q,
    iq,
    guess_values,
    guess_limit=None,
    fit_range=None,
    fit_variables=None,
    function="poly_sphere",
    fit_func="G",
    qpower_bg=False,
    num_points=20,
    spread=5,
    *argv,
    **kwargs
):
    """
    YG Dev@CHX 2019/8/1

    Fit form factor
    The support fitting functions include
                poly_sphere (poly_sphere_form_factor_intensity),
                mono_sphere (mono_sphere_form_factor_intensity)
    Parameters
    ----------
    q: q vector
    iq: form factor
    qpower_bg: if True, consider a q**(-power) background in the fitting
    guess_values:a dict, contains keys
        radius: the initial guess of spherecentral radius
        sigma: the initial guess of sqrt root of variance in percent

    function:
        mono_sphere (mono_sphere_form_factor_intensity): fit by mono dispersed sphere model
        poly_sphere (poly_sphere_form_factor_intensity): fit by poly dispersed sphere model

    Returns
    -------
    fit resutls:
         radius
         sigma
    an example:
        result = fit_form_factor( q, iq,  res_pargs=None,function='poly_sphere'
    """
    # print(q4_bg)
    if fit_range is not None:
        x1, x2 = fit_range
        q1 = find_index(q, x1, tolerance=None)
        q2 = find_index(q, x2, tolerance=None)
    else:
        q1 = 0
        q2 = len(q)

    q_ = q[q1:q2]
    iq_ = iq[q1:q2]
    pars = Parameters()
    for var in list(guess_values.keys()):
        pars.add(var, value=guess_values[var])
    if not qpower_bg:
        mod = form_factor_residuals_lmfit
    else:
        # print('here')
        mod = form_factor_residuals_bg_lmfit

    if guess_limit is not None:
        for var in list(guess_limit.keys()):
            m, M = guess_limit[var]
            pars[var].min = m
            pars[var].max = M
    if fit_variables is not None:
        for var in list(fit_variables.keys()):
            pars[var].vary = fit_variables[var]
    # print( pars )
    result = minimize(
        mod, pars, args=(iq_, q_), kws={"num_points": num_points, "spread": spread, "fit_func": fit_func}
    )
    fitp = {}
    fitpe = {}
    rp = result.params
    for var in list(rp.keys()):
        fitp[var] = rp[var].value
        fitpe[var] = rp[var].stderr
    if not qpower_bg:
        radius, sigma, delta_rho, background = (
            fitp["radius"],
            fitp["sigma"],
            fitp["delta_rho"],
            fitp["background"],
        )
        fitq = poly_sphere_form_factor_intensity(
            q_,
            radius=radius,
            sigma=sigma,
            delta_rho=delta_rho,
            background=background,
            num_points=num_points,
            spread=spread,
            fit_func=fit_func,
        )
    else:
        radius, sigma, delta_rho, background, qpower_scale, qpower = (
            fitp["radius"],
            fitp["sigma"],
            fitp["delta_rho"],
            fitp["background"],
            fitp["qpower_scale"],
            fitp["qpower"],
        )
        bk = qpower_scale * q_ ** (qpower) + background
        fitq = (
            poly_sphere_form_factor_intensity(
                q_,
                radius=radius,
                sigma=sigma,
                delta_rho=delta_rho,
                background=0,
                num_points=num_points,
                spread=spread,
                fit_func=fit_func,
            )
            + bk
        )

    # yf= result.model.eval(params=result.params, x=q_)
    # print( result.best_values )
    return fitp, fitpe, q_, fitq, result


def get_form_factor_fit2(
    q,
    iq,
    guess_values,
    fit_range=None,
    fit_variables=None,
    function="poly_sphere",
    fit_func="G",
    q4_bg=False,
    num_points=20,
    spread=5,
    bounds=None,
    *argv,
    **kwargs
):
    """
    Fit form factor
    The support fitting functions include
                poly_sphere (poly_sphere_form_factor_intensity),
                mono_sphere (mono_sphere_form_factor_intensity)
    Parameters
    ----------
    q: q vector
    iq: form factor
    q4_bg: if True, consider a q**(-4) background in the fitting
    guess_values:a dict, contains keys
        radius: the initial guess of spherecentral radius
        sigma: the initial guess of sqrt root of variance in percent

    function:
        mono_sphere (mono_sphere_form_factor_intensity): fit by mono dispersed sphere model
        poly_sphere (poly_sphere_form_factor_intensity): fit by poly dispersed sphere model

    Returns
    -------
    fit resutls:
         radius
         sigma
    an example:
        result = fit_form_factor( q, iq,  res_pargs=None,function='poly_sphere'
    """
    # print(q4_bg)
    if fit_range is not None:
        x1, x2 = fit_range
        q1 = find_index(q, x1, tolerance=None)
        q2 = find_index(q, x2, tolerance=None)
    else:
        q1 = 0
        q2 = len(q)

    q_ = q[q1:q2]
    iq_ = iq[q1:q2]
    if not q4_bg:
        fit_funcs = form_factor_residuals
        radius, sigma, delta_rho, background = (
            guess_values["radius"],
            guess_values["sigma"],
            guess_values["delta_rho"],
            guess_values["background"],
        )
        p = [radius, sigma, delta_rho, background]
        pfit, pcov, infodict, errmsg, success = leastsq(
            fit_funcs,
            [p],
            args=(iq_, q_, num_points, spread, fit_func, function),
            full_output=1,
            ftol=1.49012e-38,
            xtol=1.49012e-10,
            factor=100,
        )
    else:
        # print('here')
        fit_funcs = form_factor_residuals_bg
        radius, sigma, delta_rho, background, q4_scale, qpower = (
            guess_values["radius"],
            guess_values["sigma"],
            guess_values["delta_rho"],
            guess_values["background"],
            guess_values["q4_scale"],
            guess_values["qpower"],
        )
        p = [radius, sigma, delta_rho, background, q4_scale, qpower]
        if bounds is None:
            bounds = (-np.inf, np.inf)
        print(p)  # , qpower)
        pfit, pcov, infodict, errmsg, success = leastsq(
            fit_funcs,
            [p],
            args=(iq_, q_, num_points, spread, fit_func, function),
            ftol=1.49012e-38,
            xtol=1.49012e-10,
        )

    # print(q4_bg)
    # resL = leastsq( fit_funcs, [ p ], args=( iq_, q_,  num_points, spread, fit_func, function ),
    #                          full_output=1, ftol=1.49012e-38, xtol=1.49012e-10, factor=100)

    # radius, sigma, delta_rho, background = np.abs(pfit)
    if not q4_bg:
        radius, sigma, delta_rho, background = pfit
        fitq = poly_sphere_form_factor_intensity(
            q_,
            radius=radius,
            sigma=sigma,
            delta_rho=delta_rho,
            background=background,
            num_points=num_points,
            spread=spread,
            fit_func=fit_func,
        )
    else:
        radius, sigma, delta_rho, background, q4_scale, qpower = pfit
        bk = q4_scale * q_ ** (qpower) + background
        fitq = (
            poly_sphere_form_factor_intensity(
                q_,
                radius=radius,
                sigma=sigma,
                delta_rho=delta_rho,
                background=0,
                num_points=num_points,
                spread=spread,
                fit_func=fit_func,
            )
            + bk
        )

    if (len(iq_) > len(p)) and pcov is not None:
        s_sq = (fit_funcs(pfit, iq_, q_, num_points, spread, fit_func, function)).sum() / (len(iq_) - len(p))
        pcov = pcov * s_sq
    else:
        pcov = np.inf
    # print(pcov)
    error = []
    for i in range(len(pfit)):
        try:
            error.append(np.absolute(pcov[i][i]) ** 0.5)
        except:
            error.append(None)
    pfit_leastsq = pfit
    perr_leastsq = np.array(error)

    return pfit_leastsq, perr_leastsq, q_, fitq  # , resL


def get_form_factor_fit(
    q,
    iq,
    guess_values,
    fit_range=None,
    fit_variables=None,
    function="poly_sphere",
    fit_func="G",
    fit_power=0,
    *argv,
    **kwargs
):
    """
    Fit form factor for GUI

    The support fitting functions include
                poly_sphere (poly_sphere_form_factor_intensity),
                mono_sphere (mono_sphere_form_factor_intensity)
    Parameters
    ----------
    q: q vector
    iq: form factor

    guess_values:a dict, contains keys
        radius: the initial guess of spherecentral radius
        sigma: the initial guess of sqrt root of variance in percent

    function:
        mono_sphere (mono_sphere_form_factor_intensity): fit by mono dispersed sphere model
        poly_sphere (poly_sphere_form_factor_intensity): fit by poly dispersed sphere model

    Returns
    -------
    fit resutls:
         radius
         sigma
    an example:
        result = fit_form_factor( q, iq,  res_pargs=None,function='poly_sphere'
    """

    if function == "poly_sphere":
        mod = Model(poly_sphere_form_factor_intensity)  # _q2 )
    elif function == "mono_sphere":
        mod = Model(mono_sphere_form_factor_intensity)
    else:
        print("The %s is not supported.The supported functions include poly_sphere and mono_sphere" % function)

    if fit_range is not None:
        x1, x2 = fit_range
        q1 = find_index(q, x1, tolerance=None)
        q2 = find_index(q, x2, tolerance=None)
    else:
        q1 = 0
        q2 = len(q)

    q_ = q[q1:q2]
    iq_ = iq[q1:q2]

    _r = guess_values["radius"]
    _sigma = guess_values["sigma"]
    _delta_rho = guess_values["delta_rho"]
    _background = guess_values["background"]
    # _scale = guess_values['scale']
    # _baseline = guess_values['baseline']

    mod.set_param_hint("radius", min=_r / 10, max=_r * 10)
    mod.set_param_hint("sigma", min=_sigma / 10, max=_sigma * 10)
    # mod.set_param_hint( 'scale',   min= _scale/1E3, max= _scale*1E3  )
    # mod.set_param_hint( 'baseline',   min= 0  )
    # mod.set_param_hint( 'delta_rho',   min= 0  )
    # mod.set_param_hint( 'delta_rho',   min= _delta_rho/1E6, max= _delta_rho*1E6  )
    pars = mod.make_params(
        radius=_r, sigma=_sigma, delta_rho=_delta_rho, background=_background
    )  # scale= _scale, baseline =_baseline  )

    if fit_variables is not None:
        for var in list(fit_variables.keys()):
            pars[var].vary = fit_variables[var]
    # pars['delta_rho'].vary =False
    # fit_power = 0
    result = mod.fit(iq_ * q_**fit_power, pars, x=q_)  # , fit_func=fit_func )
    if function == "poly_sphere":
        sigma = result.best_values["sigma"]
    elif function == "mono_sphere":
        sigma = 0
    r = result.best_values["radius"]
    # scale =  result.best_values['scale']
    # baseline = result.best_values['baseline']
    delta_rho = result.best_values["delta_rho"]
    print(result.best_values)
    return result, q_


def plot_form_factor_with_fit(q, iq, q_, result, fit_power=0, res_pargs=None, return_fig=False, *argv, **kwargs):
    if res_pargs is not None:
        uid = res_pargs["uid"]
        path = res_pargs["path"]
    else:
        if "uid" in kwargs.keys():
            uid = kwargs["uid"]
        else:
            uid = "uid"
        if "path" in kwargs.keys():
            path = kwargs["path"]
        else:
            path = ""

    # fig = Figure()
    # ax = fig.add_subplot(111)
    fig, ax = plt.subplots()

    title_qr = "form_factor_fit"
    plt.title("uid= %s:--->" % uid + title_qr, fontsize=20, y=1.02)

    r = result.best_values["radius"]
    delta_rho = result.best_values["delta_rho"]
    sigma = result.best_values["sigma"]

    ax.semilogy(q, iq, "ro", label="Form Factor")
    ax.semilogy(q_, result.best_fit / q_**fit_power, "-b", lw=3, label="Fit")

    txts = r"radius" + r" = %.2f " % (r / 10.0) + r"$ nm$"
    ax.text(x=0.02, y=0.35, s=txts, fontsize=14, transform=ax.transAxes)
    txts = r"sigma" + r" = %.3f" % (sigma)
    # txts = r'$\beta$' + r'$ = %.3f$'%(beta[i]) +  r'$ s^{-1}$'
    ax.text(x=0.02, y=0.25, s=txts, fontsize=14, transform=ax.transAxes)
    # txts = r'delta_rho' + r' = %.3e'%( delta_rho)
    # txts = r'$\beta$' + r'$ = %.3f$'%(beta[i]) +  r'$ s^{-1}$'
    # ax.text(x =0.02, y=.35, s=txts, fontsize=14, transform=ax.transAxes)
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

    fp = path + "uid=%s--form_factor--fit-" % (uid) + ".png"
    plt.savefig(fp, dpi=fig.dpi)
    # fig.tight_layout()
    plt.show()

    if return_fig:
        return fig


def fit_form_factor(
    q,
    iq,
    guess_values,
    fit_range=None,
    fit_variables=None,
    res_pargs=None,
    function="poly_sphere",
    fit_func="G",
    return_fig=False,
    *argv,
    **kwargs
):
    """
    Fit form factor

    The support fitting functions include
                poly_sphere (poly_sphere_form_factor_intensity),
                mono_sphere (mono_sphere_form_factor_intensity)
    Parameters
    ----------
    q: q vector
    iq: form factor
    res_pargs: a dict, contains keys, such path, uid...

    guess_values:a dict, contains keys
        radius: the initial guess of spherecentral radius
        sigma: the initial guess of sqrt root of variance in percent

    function:
        mono_sphere (mono_sphere_form_factor_intensity): fit by mono dispersed sphere model
        poly_sphere (poly_sphere_form_factor_intensity): fit by poly dispersed sphere model

    Returns
    -------
    fit resutls:
         radius
         sigma
    an example:
        result = fit_form_factor( q, iq,  res_pargs=None,function='poly_sphere'
    """

    result, q_ = get_form_factor_fit(
        q, iq, guess_values, fit_range=fit_range, fit_variables=fit_variables, function=function, fit_func=fit_func
    )
    plot_form_factor_with_fit(q, iq, q_, result, fit_power=0, res_pargs=res_pargs, return_fig=return_fig)

    return result


def fit_form_factor2(
    q,
    iq,
    guess_values,
    fit_range=None,
    fit_variables=None,
    res_pargs=None,
    function="poly_sphere",
    fit_func="G",
    *argv,
    **kwargs
):
    """
    Fit form factor

    The support fitting functions include
                poly_sphere (poly_sphere_form_factor_intensity),
                mono_sphere (mono_sphere_form_factor_intensity)
    Parameters
    ----------
    q: q vector
    iq: form factor
    res_pargs: a dict, contains keys, such path, uid...

    guess_values:a dict, contains keys
        radius: the initial guess of spherecentral radius
        sigma: the initial guess of sqrt root of variance in percent

    function:
        mono_sphere (mono_sphere_form_factor_intensity): fit by mono dispersed sphere model
        poly_sphere (poly_sphere_form_factor_intensity): fit by poly dispersed sphere model

    Returns
    -------
    fit resutls:
         radius
         sigma
    an example:
        result = fit_form_factor( q, iq,  res_pargs=None,function='poly_sphere'
    """

    if res_pargs is not None:
        uid = res_pargs["uid"]
        path = res_pargs["path"]
    else:
        if "uid" in kwargs.keys():
            uid = kwargs["uid"]
        else:
            uid = "uid"
        if "path" in kwargs.keys():
            path = kwargs["path"]
        else:
            path = ""

    if function == "poly_sphere":
        mod = Model(poly_sphere_form_factor_intensity)  # _q2 )
    elif function == "mono_sphere":
        mod = Model(mono_sphere_form_factor_intensity)
    else:
        print("The %s is not supported.The supported functions include poly_sphere and mono_sphere" % function)

    if fit_range is not None:
        x1, x2 = fit_range
        q1 = find_index(q, x1, tolerance=None)
        q2 = find_index(q, x2, tolerance=None)
    else:
        q1 = 0
        q2 = len(q)

    q_ = q[q1:q2]
    iq_ = iq[q1:q2]

    _r = guess_values["radius"]
    _sigma = guess_values["sigma"]
    _delta_rho = guess_values["delta_rho"]
    if "background" in list(guess_values.keys()):
        _bk = guess_values["background"]
    else:
        _bk = 0
    # _scale = guess_values['scale']
    # _baseline = guess_values['baseline']

    mod.set_param_hint("radius", min=_r / 10, max=_r * 10)
    mod.set_param_hint("sigma", min=_sigma / 10, max=_sigma * 10)
    # mod.set_param_hint( 'scale',   min= _scale/1E3, max= _scale*1E3  )
    mod.set_param_hint("background", min=0)
    # mod.set_param_hint( 'delta_rho',   min= 0  )
    mod.set_param_hint("delta_rho", min=_delta_rho / 1e6, max=_delta_rho * 1e6)
    pars = mod.make_params(
        radius=_r, sigma=_sigma, delta_rho=_delta_rho, background=_bk
    )  # scale= _scale, baseline =_baseline  )

    if fit_variables is not None:
        for var in list(fit_variables.keys()):
            pars[var].vary = fit_variables[var]
    # pars['delta_rho'].vary =False

    fig = plt.figure(figsize=(8, 6))
    title_qr = "form_factor_fit"
    plt.title("uid= %s:--->" % uid + title_qr, fontsize=20, y=1.02)

    fit_power = 0  # 2

    result = mod.fit(iq_ * q_**fit_power, pars, x=q_)  # ,fit_func= fit_func )

    if function == "poly_sphere":
        sigma = result.best_values["sigma"]
    elif function == "mono_sphere":
        sigma = 0

    r = result.best_values["radius"]
    # scale =  result.best_values['scale']
    # baseline = result.best_values['baseline']
    delta_rho = result.best_values["delta_rho"]

    # report_fit( result )

    ax = fig.add_subplot(1, 1, 1)

    ax.semilogy(q, iq, "ro", label="Form Factor")
    ax.semilogy(q_, result.best_fit / q_**fit_power, "-b", lw=3, label="Fit")

    txts = r"radius" + r" = %.2f " % (r / 10.0) + r"$ nm$"
    ax.text(x=0.02, y=0.35, s=txts, fontsize=14, transform=ax.transAxes)
    txts = r"sigma" + r" = %.3f" % (sigma)
    # txts = r'$\beta$' + r'$ = %.3f$'%(beta[i]) +  r'$ s^{-1}$'
    ax.text(x=0.02, y=0.25, s=txts, fontsize=14, transform=ax.transAxes)
    # txts = r'delta_rho' + r' = %.3e'%( delta_rho)
    # txts = r'$\beta$' + r'$ = %.3f$'%(beta[i]) +  r'$ s^{-1}$'
    # ax.text(x =0.02, y=.35, s=txts, fontsize=14, transform=ax.transAxes)
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

    fp = path + "%s_form_factor_fit" % (uid) + ".png"
    fig.savefig(fp, dpi=fig.dpi)
    fig.tight_layout()
    # plt.show()

    result = dict(radius=r, sigma=sigma, delta_rho=delta_rho)

    return result


def show_saxs_qmap(
    img,
    pargs,
    width=200,
    vmin=0.1,
    vmax=300,
    logs=True,
    image_name="",
    show_colorbar=True,
    file_name="",
    show_time=False,
    save=False,
    show_pixel=False,
    aspect=1,
    save_format="png",
    cmap="viridis",
):
    """
    Show a SAXS q-map by giving
    Parameter:
        image: the frame
        setup pargs, a dictionary, including
            dpix    #in mm, eiger 4m is 0.075 mm
            lambda_     # wavelegth of the X-rays in Angstroms
            Ldet     # detector to sample distance (mm)
            path  where to save data
            center: beam center in pixel, center[0] (x), should be image-y, and should be python-x
        width: the showed area centered at center
    Return:
        None
    """

    Ldet = pargs["Ldet"]
    dpix = pargs["dpix"]
    lambda_ = pargs["lambda_"]
    center = pargs["center"]
    cx, cy = center
    path = pargs["path"]
    lx, ly = img.shape
    # center = [ center[1], center[0] ] #due to python conventions
    w = width

    img_ = np.zeros([w, w])
    minW, maxW = min(center[0] - w, center[1] - w), max(center[0] - w, center[1] - w)
    if w < minW:
        img_ = img[cx - w // 2 : cx + w // 2, cy + w // 2 : cy + w // 2]
    # elif w > maxW:
    #    img_[ cx-w//2:cx+w//2, cy+w//2:cy+w//2 ] =

    ROI = [max(0, center[0] - w), min(center[0] + w, lx), max(0, center[1] - w), min(ly, center[1] + w)]
    # print( ROI )
    ax = plt.subplots()
    if not show_pixel:
        # print( 'here' )
        two_theta = utils.radius_to_twotheta(
            Ldet,
            np.array(
                [
                    (ROI[0] - cx) * dpix,
                    (ROI[1] - cx) * dpix,
                    (ROI[2] - cy) * dpix,
                    (ROI[3] - cy) * dpix,
                ]
            ),
        )
        qext = utils.twotheta_to_q(two_theta, lambda_)
        # print( two_theta, qext )

        show_img(
            1e-15 + img[ROI[0] : ROI[1], ROI[2] : ROI[3]],
            ax=ax,
            xlabel=r"$q_x$" + "(" + r"$\AA^{-1}$" + ")",
            ylabel=r"$q_y$" + "(" + r"$\AA^{-1}$" + ")",
            extent=[qext[3], qext[2], qext[0], qext[1]],
            vmin=vmin,
            vmax=vmax,
            logs=logs,
            image_name=image_name,
            file_name=file_name,
            show_time=show_time,
            save_format=save_format,
            cmap=cmap,
            show_colorbar=show_colorbar,
            save=save,
            path=path,
            aspect=aspect,
        )
    else:
        # qext = w
        show_img(
            1e-15 + img[ROI[0] : ROI[1], ROI[2] : ROI[3]],
            ax=ax,
            xlabel="pixel",
            ylabel="pixel",
            extent=[ROI[0], ROI[1], ROI[2], ROI[3]],
            vmin=vmin,
            vmax=vmax,
            logs=logs,
            image_name=image_name,
            save_format=save_format,
            cmap=cmap,
            show_colorbar=show_colorbar,
            file_name=file_name,
            show_time=show_time,
            save=save,
            path=path,
            aspect=aspect,
        )
    return ax


########################
##Fit sphere by scipy.leastsq fit


def fit_sphere_form_factor_func(parameters, ydata, xdata, yerror=None, nonvariables=None):
    """##Develop by YG at July 28, 2017 @CHX
    This function is for fitting form factor of polyderse spherical particles by using scipy.leastsq fit

    radius, sigma, delta_rho, background  = parameters
    """
    radius, sigma, delta_rho, background = parameters
    fit = poly_sphere_form_factor_intensity(
        xdata,
        radius=radius,
        sigma=sigma,
        delta_rho=delta_rho,
        background=background,
        num_points=10,
        spread=3,
        fit_func="G",
    )
    error = np.abs(ydata - fit)
    return np.sqrt(error)


def fit_sphere_form_factor_by_leastsq(
    p0,
    q,
    pq,
    fit_range=None,
):
    """##Develop by YG at July 28, 2017 @CHX
    Fitting form factor of polyderse spherical particles by using scipy.leastsq fit
    Input:
        radius, sigma, delta_rho, background  = p0
    Return
        fit  res, res[0] is the fitting parameters
    """
    if fit_range is not None:
        x1, x2 = fit_range
        q1, q2 = find_index(q, x1), find_index(q, x2)
    res = leastsq(
        fit_sphere_form_factor_func,
        [p0],
        args=(
            pq[q1:q2],
            q[q1:q2],
        ),
        ftol=1.49012e-38,
        xtol=1.49012e-38,
        factor=100,
        full_output=1,
    )
    return res


def plot_fit_sphere_form_factor(q, pq, res, p0=None, xlim=None, ylim=None):
    """##Develop by YG at July 28, 2017 @CHX"""

    if p0 is not None:
        radius, sigma, delta_rho, background = p0
        fit_init = poly_sphere_form_factor_intensity(
            q,
            radius=radius,
            sigma=sigma,
            delta_rho=delta_rho,
            background=background,
        )
    radius, sigma, delta_rho, background = res[0]
    fit = poly_sphere_form_factor_intensity(
        q,
        radius=radius,
        sigma=sigma,
        delta_rho=delta_rho,
        background=background,
    )

    fig, ax = plt.subplots()
    if p0 is not None:
        plot1D(x=q, y=fit_init, c="b", m="", ls="-", lw=3, ax=ax, logy=True, legend="Init_Fitting")
    plot1D(x=q, y=fit, c="r", m="", ls="-", lw=3, ax=ax, logy=True, legend="Fitting")
    plot1D(
        x=q,
        y=pq,
        c="k",
        m="X",
        ax=ax,
        markersize=3,
        ls="",
        legend="data",
        xlim=xlim,
        ylim=ylim,
        logx=True,
        xlabel="Q (A-1)",
        ylabel="P(Q)",
    )

    txts = r"radius" + r" = %.2f " % (res[0][0] / 10.0) + r"$ nm$"
    ax.text(x=0.02, y=0.25, s=txts, fontsize=14, transform=ax.transAxes)
    txts = r"sigma" + r" = %.3f" % (res[0][1])
    # txts = r'$\beta$' + r'$ = %.3f$'%(beta[i]) +  r'$ s^{-1}$'
    ax.text(x=0.02, y=0.15, s=txts, fontsize=14, transform=ax.transAxes)


def exm_plot():
    fig, ax = plt.subplots()

    ax.semilogy(q, iq, "ro", label="data")
    ax.semilogy(q, ff, "-b", label="fit")
    ax.set_xlim([0.0001, 0.01])
    ax.set_ylim([1e-2, 1e4])
    ax.legend(loc="best")
    # plot1D( iq, q, logy=True, xlim=[0.0001, .01], ylim=[1E-3,1E4], ax=ax, legend='data')
    # plot1D( ff, q, logy=True, xlim=[0.0001, .01], ax=ax, legend='cal')

    # %run /XF11ID/analysis/Analysis_Pipelines/Develop/pyCHX/pyCHX/XPCS_SAXS.py
    # %run /XF11ID/analysis/Analysis_Pipelines/Develop/pyCHX/pyCHX/chx_generic_functions.py
    # %run /XF11ID/analysis/Analysis_Pipelines/Develop/pyCHX/pyCHX/SAXS.py
