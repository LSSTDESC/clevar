"""@file clevar/match_metrics/scaling/array_funcs.py

Main scaling functions using arrays.
"""
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic

from ...utils import none_val, autobins, binmasks, deep_update
from .. import plot_helper as ph
from ..plot_helper import plt, NullFormatter


def _prep_fit_data(xvals, yvals, yerr=None, statistics='mean', bins_x=None, bins_y=None):
    """
    Prepare data for fit with binning.

    Parameters
    ----------
    xvals: array
        Input values for fit
    yvals: array
        Values to be fitted
    yerr: array, None
        Errors of y
    statistics: str
        Statistics to be used in fit (default=mean). Options are:

            * `individual` : Use each point
            * `mode` : Use mode of component 2 distribution in each comp 1 bin, requires bins2.
            * `mean` : Use mean of component 2 distribution in each comp 1 bin, requires bins2.

    bins_x: array, None
        Bins for component x
    bins_y: array, None
        Bins for component y

    Returns
    -------
    xdata, ydata, errdata: array
        Data for fit
    """
    if statistics=='individual':
        return xvals, yvals, yerr
    elif statistics=='mode':
        bins_hist = autobins(yvals, bins_y)
        bins_hist_m = 0.5*(bins_hist[1:]+bins_hist[:-1])
        stat_func = lambda vals: bins_hist_m[np.histogram(vals, bins=bins_hist)[0].argmax()]
    elif statistics=='mean':
        stat_func = lambda vals: np.mean(vals)
    else:
        raise ValueError(f'statistics ({statistics}) must be in (individual, mean, mode)')
    point_masks = [m for m in binmasks(xvals, autobins(xvals, bins_x)) if m[m].size>1]
    err = np.zeros(len(yvals)) if yerr is None else yerr
    err_func = lambda vals, err: np.mean(np.sqrt(np.std(vals)**2+err**2))
    return np.transpose([[np.mean(xvals[m]), stat_func(yvals[m]), err_func(yvals[m], err[m])]
                            for m in point_masks])


def _add_bindata_and_powlawfit(ax, values1, values2, err2, log=False, **kwargs):
    """
    Add binned data and powerlaw fit to plot

    Parameters
    ----------
    ax: matplotlib.axes
        Ax to add plot.
    values1, values2: array
        Components to be binned and fitted
    err2: array
        Error of component 2
    log: bool
        Bin and fit in log values (default=False).
    statistics: str
        Statistics to be used in fit (default=mean). Options are:

            * `individual` : Use each point
            * `mode` : Use mode of component 2 distribution in each comp 1 bin, requires bins2.
            * `mean` : Use mean of component 2 distribution in each comp 1 bin, requires bins2.

    bins1, bins2: array, None
        Bins for component x (default=10) and y (default=30).
    add_bindata: bool
        Plot binned data used for fit (default=False).
    add_fit: bool
        Fit and plot binned dat (default=False).
    bindata_kwargs: dict
        Additional arguments for pylab.errorbar.
    plt_kwargs: dict
        Additional arguments for plot of fit pylab.scatter.
    legend_kwargs: dict
        Additional arguments for plt.legend.
    label_components: tuple (of strings)
        Names of fitted components in fit line label, default=('x', 'y').

    Returns
    -------
    info: dict
        Information of data in the plots, it contains the sections:

            * binned_data (optional): input data for fitting, with values:

                * x: x values in fit (log of values1 if log=True).
                * y: y values in fit (log of values2 if log=True).
                * y_err: errorbar on y values (error_log if log=True).

            * fit (optional): fitting output dictionary, with values:

                * pars: fitted parameter.
                * cov: covariance of fitted parameters.
                * func: fitting function with fitted parameter.
                * func_plus: fitting function with fitted parameter plus 1x scatter.
                * func_minus: fitting function with fitted parameter minus 1x scatter.
                * func_scat: scatter of fited function.
                * func_chi: sqrt of chi_square(x, y) for the fitted function.

            * plots (optional): additional plots:

                * fit: fitted data
                * errorbar: binned data

    """
    info = {}
    # Default parameters
    mode = kwargs.get('statistics', 'mode')
    bins1 = kwargs.get('bins1', 10)
    bins2 = kwargs.get('bins2', 30)
    legend_kwargs = kwargs.get('legend_kwargs', {})
    add_bindata = kwargs.get('add_bindata', False)
    bindata_kwargs = kwargs.get('bindata_kwargs', {})
    add_fit = kwargs.get('add_fit', False)
    plot_kwargs = kwargs.get('plot_kwargs', {})
    xl, yl = kwargs.get('label_components', ('x', 'y'))
    xl = xl.replace('$', '')  if '$' in xl else xl.replace('_', r'\_')
    yl = yl.replace('$', '')  if '$' in yl else yl.replace('_', r'\_')
    if ((not add_bindata) and (not add_fit)) or len(values1)<=1:
        return info
    # set log/lin funcs
    tfunc, ifunc = (np.log10, lambda x: 10**x) if log else (lambda x:x, lambda x:x)
    # data
    data = _prep_fit_data(tfunc(values1), tfunc(values2),
                          bins_x=tfunc(bins1) if hasattr(bins1, '__len__') else bins1,
                          bins_y=tfunc(bins2) if hasattr(bins2, '__len__') else bins2,
                          yerr=None if (err2 is None or not log) else err2/(values2*np.log(10)),
                          statistics=mode)
    if len(data)==0:
        return info
    vbin_1, vbin_2, vbin_err2 = data
    info['binned_data'] = {'x': vbin_1, 'y': vbin_2, 'yerr': vbin_err2}
    # fit
    if add_fit:
        pw_func = lambda x, a, b: a*x+b
        fit, cov = curve_fit(pw_func, vbin_1, vbin_2,
            sigma=vbin_err2, absolute_sigma=True)
        # Functions with fit values
        fit_func = lambda x: pw_func(tfunc(x), *fit)
        scat_func = np.vectorize(
            lambda x: np.sqrt(np.dot([tfunc(x), 1], np.dot(cov, [tfunc(x), 1]))))
        info['fit'] = {
            'pars':fit, 'cov':cov,
            'func':lambda x: ifunc(fit_func(x)),
            'func_plus': lambda x: ifunc(fit_func(x)+scat_func(x)),
            'func_minus': lambda x: ifunc(fit_func(x)-scat_func(x)),
            'func_scat': scat_func,
            'func_chi': lambda x, y: (tfunc(y)-fit_func(x))/scat_func(x),
        }
        # labels
        sig = np.sqrt(np.diag(cov))
        fmt0 = lambda x: f'{x:.2f}' if 0.01<abs(fit[0])<100 else f'{x:.2e}'
        fmt1 = lambda x: f'{x:.2f}' if 0.01<abs(fit[1])<100 else f'{x:.2e}'
        fit0_lab = rf'({fmt0(fit[0])}\pm {fmt0(sig[0])})'
        fit1_lab = rf'{"-"*(fit[1]<0)}({fmt1(abs(fit[1]))}\pm {fmt1(sig[1])})'
        avg_label = rf'\left<{yl}\right|\left.{xl}\right>'
        fit_label = rf'${avg_label}=10^{{{fit1_lab}}}\;({xl})^{{{fit0_lab}}}$' if log\
            else rf"${avg_label}={fit0_lab}\;{xl}{'+'*(fit[1]>=0)}{fit1_lab}$"
        # plot fit
        plot_kwargs_ = {'color': 'r', 'label': fit_label}
        plot_kwargs_.update(plot_kwargs)
        sort = np.argsort(values1)
        xl = values1[sort]
        ax.plot(xl, info['fit']['func'](xl), **plot_kwargs_)
        deep_update(info,
            {'plots': {'fit': ax.fill_between(
                xl, info['fit']['func_plus'](xl), info['fit']['func_minus'](xl),
                color=plot_kwargs_['color'], alpha=.2, lw=0)}}
        )
    if add_bindata and not mode=='individual':
        eb_kwargs_ = {'elinewidth': 1, 'capsize': 2, 'fmt': '.',
                      'ms': 10, 'ls': '', 'color': 'm'}
        eb_kwargs_.update(bindata_kwargs)
        deep_update(info,
            {'plots': {'errorbar': ax.errorbar(
                ifunc(vbin_1), ifunc(vbin_2),
                yerr=(ifunc(vbin_2)*np.array([1-1/ifunc(vbin_err2), ifunc(vbin_err2)-1])
                    if log else vbin_err2),
                **eb_kwargs_)}}
        )
    # legend
    if any(c.get_label()[0]!='_' for c in ax.collections+ax.lines):
        legend_kwargs_ = {}
        legend_kwargs_.update(legend_kwargs)
        ax.legend(**legend_kwargs_)
    return info


def plot(values1, values2, err1=None, err2=None, ax=None, plt_kwargs={}, err_kwargs={},
         values_color=None, add_cb=True, cb_kwargs={}, **kwargs):
    """
    Scatter plot with errorbars and color based on input

    Parameters
    ----------
    values1, values2: array
        Components x and y for plot.
    err1, err2: array, None
        Errors of component x and y.
    ax: matplotlib.axes, None
        Ax to add plot. If equals `None`, one is created.
    plt_kwargs: dict
        Additional arguments for pylab.scatter
    err_kwargs: dict
        Additional arguments for pylab.errorbar
    values_color: array, None
        Values for color (cmap scale).
    add_cb: bool
        Plot colorbar when values_color is not `None`.
    cb_kwargs: dict
        Colorbar arguments

    Other Parameters
    -----------------
    add_bindata: bool
        Plot binned data used for fit (default=False).
    add_fit: bool
        Fit and plot binned dat (default=False).
    fit_err2: array, None
        Error of component 2 (set to err2 if not provided).
    fit_log: bool
        Bin and fit in log values (default=False).
    fit_statistics: str
        Statistics to be used in fit (default=mean). Options are:

            * `individual` : Use each point
            * `mode` : Use mode of component 2 distribution in each comp 1 bin, requires bins2.
            * `mean` : Use mean of component 2 distribution in each comp 1 bin, requires bins2.

    fit_bins1: array, None
        Bins for component 1 (default=10).
    fit_bins2: array, None
        Bins for component 2 (default=30).
    fit_legend_kwargs: dict
        Additional arguments for plt.legend.
    fit_bindata_kwargs: dict
        Additional arguments for pylab.errorbar.
    fit_plt_kwargs: dict
        Additional arguments for plot of fit pylab.scatter.
    fit_label_components: tuple (of strings)
        Names of fitted components in fit line label, default=('x', 'y').

    Returns
    -------
    info: dict
        Information of data in the plots, it contains the sections:

            * `ax`: ax used in the plot.
            * `ax_cb` (optional): ax of colorbar
            * `binned_data` (optional): input data for fitting, with values:

                * `x`: x values in fit (log of values if log=True).
                * `y`: y values in fit (log of values if log=True).
                * `y_err`: errorbar on y values (error_log if log=True).

            * `fit` (optional): fitting output dictionary, with values:

                * `pars`: fitted parameter.
                * `cov`: covariance of fitted parameters.
                * `func`: fitting function with fitted parameter.
                * `func_plus`: fitting function with fitted parameter plus 1x scatter.
                * `func_minus`: fitting function with fitted parameter minus 1x scatter.
                * `func_scat`: scatter of fited function.
                * `func_chi`: sqrt of chi_square(x, y) for the fitted function.

            * `plots` (optional): additional plots:

                * `fit`: fitted data
                * `errorbar`: binned data
    """
    info = {'ax': plt.axes() if ax is None else ax}
    ph.add_grid(info['ax'])
    if len(values1)==0:
        return info
    # Plot points
    plt_kwargs_ = {'s':1}
    if values_color is None:
        xplot, yplot = values1, values2
    else:
        isort = np.argsort(values_color)
        xplot, yplot, zplot = [v[isort] for v in (values1, values2, values_color)]
        plt_kwargs_['c'] = zplot
    plt_kwargs_.update(plt_kwargs)
    sc = info['ax'].scatter(xplot, yplot, **plt_kwargs_)
    # Plot errorbar
    err_kwargs_ = dict(elinewidth=.5, capsize=0, fmt='.', ms=0, ls='')
    if values_color is None:
        xerr, yerr = err1, err2
    else:
        cb = plt.colorbar(sc, ax=info['ax'], **cb_kwargs)
        xerr = err1[isort] if err1 is not None else None
        yerr = err2[isort] if err2 is not None else None
        err_kwargs_['ecolor'] = [cb.mappable.cmap(cb.mappable.norm(c)) for c in zplot]
    if err1 is not None or err2 is not None:
        err_kwargs_.update(err_kwargs)
        info['ax'].errorbar(xplot, yplot, xerr=xerr, yerr=yerr, **err_kwargs_)
    if values_color is not None:
        if add_cb:
            info['cb'] = cb
        else:
            cb.remove()
    # Bindata and fit
    kwargs['fit_err2'] = kwargs.get('fit_err2', err2)
    kwargs['fit_add_fit'] = kwargs.get('add_fit', False)
    kwargs['fit_add_bindata'] = kwargs.get('add_bindata', kwargs['fit_add_fit'])
    info.update(
        _add_bindata_and_powlawfit(
            info['ax'], values1, values2,
            **{k[4:]:v for k, v in kwargs.items() if k[:4]=='fit_'}))
    return info


def plot_density(values1, values2, bins1=30, bins2=30,
                 ax_rotation=0, rotation_resolution=30,
                 xscale='linear', yscale='linear',
                 err1=None, err2=None,
                 ax=None, plt_kwargs={},
                 add_cb=True, cb_kwargs={},
                 err_kwargs={}, **kwargs):

    """
    Scatter plot with errorbars and color based on point density

    Parameters
    ----------
    values1, values2: array
        Components x and y for plot.
    bins1, bins2: array, None
        Bins for component x and y (for density colors).
    ax_rotation: float
        Angle (in degrees) for rotation of axis of binning. Overwrites use of (bins1, bins2)
    rotation_resolution: int
        Number of bins to be used when ax_rotation!=0.
    xscale, yscale: str
        Scale for x/y axis.
    err1, err2: array, None
        Errors of component x and y.
    ax: matplotlib.axes, None
        Ax to add plot. If equals `None`, one is created.
    plt_kwargs: dict
        Additional arguments for pylab.scatter
    add_cb: bool
        Plot colorbar
    cb_kwargs: dict
        Colorbar arguments
    err_kwargs: dict
        Additional arguments for pylab.errorbar

    Other Parameters
    -----------------
    add_bindata: bool
        Plot binned data used for fit (default=False).
    add_fit: bool
        Fit and plot binned dat (default=False).
    **fit_kwargs:
        Other fit arguments (see `fit_*` paramters in `scaling.array_funcs.plot` for more info).

    Returns
    -------
    info: dict
        Information of data in the plots, it contains the sections:

            * `ax`: ax used in the plot.
            * `ax_cb` (optional): ax of colorbar
            * `binned_data` (optional): input data for fitting \
            (see `scaling.array_funcs.plot` for more info).
            * `fit` (optional): fitting output dictionary \
            (see `scaling.array_funcs.plot` for more info).
            * `plots` (optional): fit and binning plots \
            (see `scaling.array_funcs.plot` for more info).
    """
    values_color = ph.get_density_colors(values1, values2, bins1, bins2,
        ax_rotation=ax_rotation, rotation_resolution=rotation_resolution,
        xscale=xscale, yscale=yscale) if len(values1)>0 else []
    return plot(values1, values2, values_color=values_color,
            err1=err1, err2=err2, ax=ax, plt_kwargs=plt_kwargs,
            add_cb=add_cb, cb_kwargs=cb_kwargs, err_kwargs=err_kwargs, **kwargs)


def _plot_panel(plot_function, values_panel, bins_panel,
                panel_kwargs_list=None, panel_kwargs_errlist=None,
                fig_kwargs={}, add_label=True, label_format=lambda v: v,
                plt_kwargs={}, err_kwargs={}, **plt_func_kwargs):
    """
    Helper function to makes plots in panels

    Parameters
    ----------
    plot_function: function
        Plot function
    values_panel: array
        Values to bin data in panels
    bins_panel: array, int
        Bins defining panels
    panel_kwargs_list: list, None
        List of additional arguments for plotting each panel (using pylab.plot).
        Must have same size as len(bins2)-1
    panel_kwargs_errlist: list, None
        List of additional arguments for plotting each panel (using pylab.errorbar).
        Must have same size as len(bins2)-1
    fig_kwargs: dict
        Additional arguments for plt.subplots
    add_label: bool
        Add bin label to panel
    label_format: function
        Function to format the values of the bins
    **plt_func_kwargs
        All other parameters to be passed to plot_function

    Other Parameters
    -----------------
    add_bindata: bool
        Plot binned data used for fit (default=False).
    add_fit: bool
        Fit and plot binned dat (default=False).
    **fit_kwargs:
        Other fit arguments (see `fit_*` paramters in `scaling.array_funcs.plot` for more info).

    Returns
    -------
    Same as plot_function
    """
    edges = bins_panel if hasattr(bins_panel, '__len__') else\
        np.linspace(min(values_panel), max(values_panel), bins_panel)
    nj = int(np.ceil(np.sqrt(len(edges[:-1]))))
    ni = int(np.ceil(len(edges[:-1])/float(nj)))
    fig_kwargs_ = dict(sharex=True, sharey=True, figsize=(8, 6))
    fig_kwargs_.update(fig_kwargs)
    info = {key: value for key, value in zip(
        ('fig', 'axes'), plt.subplots(ni, nj, **fig_kwargs_))}
    panel_kwargs_list = none_val(panel_kwargs_list, [{} for m in edges[:-1]])
    panel_kwargs_errlist = none_val(panel_kwargs_errlist, [{} for m in edges[:-1]])
    masks = [(values_panel>=v0)*(values_panel<v1) for v0, v1 in zip(edges, edges[1:])]
    ax_conf = []
    for ax, mask, p_kwargs, p_e_kwargs in zip(
            info['axes'].flatten(), masks, panel_kwargs_list, panel_kwargs_errlist):
        ph.add_grid(ax)
        kwargs = {}
        kwargs.update(plt_kwargs)
        kwargs.update(p_kwargs)
        kwargs_e = {}
        kwargs_e.update(err_kwargs)
        kwargs_e.update(p_e_kwargs)
        ax_conf.append(
            plot_function(
                ax=ax, plt_kwargs=kwargs, err_kwargs=kwargs_e,
                **{k:v[mask] if (hasattr(v, '__len__') and len(v)==mask.size) and
                    (not isinstance(v, (str, dict))) else v
                    for k, v in plt_func_kwargs.items()}))
        ax_conf[-1].pop('ax')
    ax_conf += [{} for i in range(ni*nj-len(ax_conf))] # complete missing vals
    info['axes_conf'] = np.reshape(ax_conf, (ni, nj))
    for ax in info['axes'].flatten()[len(edges)-1:]:
        ax.axis('off')
    if add_label:
        ph.add_panel_bin_label(info['axes'],  edges[:-1], edges[1:],
                               format_func=label_format)
    return info


def plot_panel(
    values1, values2, values_panel, bins_panel, err1=None, err2=None, values_color=None,
    plt_kwargs={}, err_kwargs={}, add_cb=True, cb_kwargs={}, panel_kwargs_list=None,
    panel_kwargs_errlist=None, fig_kwargs={}, add_label=True, label_format=lambda v: v, **kwargs):
    """
    Scatter plot with errorbars and color based on input with panels

    Parameters
    ----------
    values1, values2: array
        Components x and y for plot.
    values_color: array
        Values for color (cmap scale)
    values_panel: array
        Values to bin data in panels
    bins_panel: array, int
        Bins defining panels
    err1, err2: array, None
        Errors of component x and y.
    values_color: array, None
        Values for color (cmap scale).
    plt_kwargs: dict
        Additional arguments for pylab.scatter
    err_kwargs: dict
        Additional arguments for pylab.errorbar
    add_cb: bool
        Plot colorbar when values_color is not `None`.
    cb_kwargs: dict
        Colorbar arguments
    panel_kwargs_list: list, None
        List of additional arguments for plotting each panel (using pylab.plot).
        Must have same size as len(bins2)-1
    panel_kwargs_errlist: list, None
        List of additional arguments for plotting each panel (using pylab.errorbar).
        Must have same size as len(bins2)-1
    fig_kwargs: dict
        Additional arguments for plt.subplots
    add_label: bool
        Add bin label to panel
    label_format: function
        Function to format the values of the bins

    Other Parameters
    -----------------
    add_bindata: bool
        Plot binned data used for fit (default=False).
    add_fit: bool
        Fit and plot binned dat (default=False).
    **fit_kwargs:
        Other fit arguments (see `fit_*` paramters in `scaling.array_funcs.plot` for more info).

    Returns
    -------
    info: dict
        Information of data in the plots, it contains the sections:

            * `fig`: `matplotlib.figure.Figure` object.
            * `axes`: `matplotlib.axes` used in the plot.
            * `ax_cb` (optional): ax of colorbar
            * `binned_data` (optional): input data for fitting \
            (see `scaling.array_funcs.plot` for more info).
            * `fit` (optional): fitting output dictionary \
            (see `scaling.array_funcs.plot` for more info).
            * `plots` (optional): fit and binning plots \
            (see `scaling.array_funcs.plot` for more info).
    """
    return _plot_panel(
        # _plot_panel arguments
        plot_function=plot,
        values_panel=values_panel, bins_panel=bins_panel,
        panel_kwargs_list=panel_kwargs_list, panel_kwargs_errlist=panel_kwargs_errlist,
        fig_kwargs=fig_kwargs, add_label=add_label, label_format=label_format,
        # plot_color arguments
        values1=values1, values2=values2, err1=err1, err2=err2,
        values_color=values_color, add_cb=add_cb, cb_kwargs=cb_kwargs,
        plt_kwargs=plt_kwargs, err_kwargs=err_kwargs,
        # for fit
        **kwargs,
        )


def plot_density_panel(values1, values2, values_panel, bins_panel,
    bins1=30, bins2=30, ax_rotation=0, rotation_resolution=30,
    xscale='linear', yscale='linear',
    err1=None, err2=None, plt_kwargs={}, add_cb=True, cb_kwargs={},
    err_kwargs={}, panel_kwargs_list=None, panel_kwargs_errlist=None,
    fig_kwargs={}, add_label=True, label_format=lambda v: v, **kwargs):

    """
    Scatter plot with errorbars and color based on point density with panels

    Parameters
    ----------
    values1, values2: array
        Components x and y for plot.
    values_panel: array
        Values to bin data in panels
    bins_panel: array, int
        Bins defining panels
    bins1, bins2: array, None
        Bins for component x and y (for density colors).
    ax_rotation: float
        Angle (in degrees) for rotation of axis of binning. Overwrites use of (bins1, bins2)
    rotation_resolution: int
        Number of bins to be used when ax_rotation!=0.
    xscale, yscale: str
        Scale for x/y axis.
    err1, err2: array, None
        Errors of component x and y.
    ax: matplotlib.axes, None
        Ax to add plot. If equals `None`, one is created.
    plt_kwargs: dict
        Additional arguments for pylab.scatter
    add_cb: bool
        Plot colorbar
    cb_kwargs: dict
        Colorbar arguments
    err_kwargs: dict
        Additional arguments for pylab.errorbar
    panel_kwargs_list: list, None
        List of additional arguments for plotting each panel (using pylab.plot).
        Must have same size as len(bins2)-1
    panel_kwargs_errlist: list, None
        List of additional arguments for plotting each panel (using pylab.errorbar).
        Must have same size as len(bins2)-1
    fig_kwargs: dict
        Additional arguments for plt.subplots
    add_label: bool
        Add bin label to panel
    label_format: function
        Function to format the values of the bins

    Other Parameters
    -----------------
    add_bindata: bool
        Plot binned data used for fit (default=False).
    add_fit: bool
        Fit and plot binned dat (default=False).
    **fit_kwargs:
        Other fit arguments (see `fit_*` paramters in `scaling.array_funcs.plot` for more info).

    Returns
    -------
    info: dict
        Information of data in the plots, it contains the sections:

            * `fig`: `matplotlib.figure.Figure` object.
            * `axes`: `matplotlib.axes` used in the plot.
            * `ax_cb` (optional): ax of colorbar
            * `binned_data` (optional): input data for fitting \
            (see `scaling.array_funcs.plot` for more info).
            * `fit` (optional): fitting output dictionary \
            (see `scaling.array_funcs.plot` for more info).
            * `plots` (optional): fit and binning plots \
            (see `scaling.array_funcs.plot` for more info).
    """
    return _plot_panel(
        # _plot_panel arguments
        plot_function=plot_density,
        values_panel=values_panel, bins_panel=bins_panel,
        xscale=xscale, yscale=yscale,
        panel_kwargs_list=panel_kwargs_list, panel_kwargs_errlist=panel_kwargs_errlist,
        fig_kwargs=fig_kwargs, add_label=add_label, label_format=label_format,
        # plot_density arguments
        values1=values1, values2=values2, err1=err1, err2=err2, bins1=bins1, bins2=bins2,
        ax_rotation=ax_rotation, rotation_resolution=rotation_resolution,
        plt_kwargs=plt_kwargs, err_kwargs=err_kwargs,
        add_cb=add_cb, cb_kwargs=cb_kwargs,
        # for fit
        **kwargs,
        )


def _plot_metrics(values1, values2, bins=30, mode='diff_z', ax=None,
                  metrics=['mean'], metrics_kwargs={}, rotated=False):
    """
    Plot metrics of 1 component.

    Parameters
    ----------
    values1, values2: array
        Components x and y for plot.
    bins: array, int
        Bins for component 1
    mode: str
        Mode to run metrics. Options are:

            * `simple` : metrics for `values2`.
            * `log` : metrics for `log10(values2)`.
            * `diff` : metrics for `values2-values1`.
            * `diff_log` : metrics for `log10(values2)-log10(values1)`.
            * `diff_z` : metrics for `(values2-values1)/(1+values1)`.

    ax: matplotlib.axes, None
        Ax to add plot. If equals `None`, one is created.
    metrics: list
        List of mettrics to be plotted. Possibilities are:

            * 'mean' : compute the mean of values for points within each bin.
            * 'std' : compute the standard deviation within each bin.
            * 'median' : compute the median of values for points within each bin.
            * 'count' : compute the count of points within each bin.
            * 'sum' : compute the sum of values for points within each bin.
            * 'min' : compute the minimum of values for points within each bin.
            * 'max' : compute the maximum of values for point within each bin.
            * 'p_#' : compute half the width where a percentile of data is found. Number must be
              between 0-100 (ex: 'p_68', 'p_95', 'p_99').

    metrics_kwargs: dict
        Dictionary of dictionary configs for each metric plots.
    rotated: bool
        Rotate ax of plot

    Returns
    -------
    info: dict
        Information of data in the plots, it contains the sections:

            * `ax`: ax used in the plot.
            * `plots` (optional): plot for each metric.
    """
    edges = autobins(values1, bins, log=mode=='log')
    bmask = np.array(binmasks(values1, edges))
    safe = [m[m].size>1 for m in bmask]
    values = {
        'diff': lambda v1, v2: v2-v1,
        'diff_z': lambda v1, v2: (v2-v1)/(1+v1),
        'diff_log': lambda v1, v2: np.log10(v2)-np.log10(v1),
        'simple': lambda v1, v2: v2,
        'log': lambda v1, v2: np.log10(v2),
        }[mode](values1, values2)
    values_mid = (10**(0.5*(np.log10(edges[1:])+np.log10(edges[:-1])))
        if mode=='log' else 0.5*(edges[1:]+edges[:-1]))
    # set for rotation
    info = {'ax': plt.axes() if ax is None else ax}
    ph.add_grid(info['ax'])
    # plot
    for metric in metrics:
        metric_name = metric.replace('.fill', '')
        kwargs = {'label':metric_name}
        if metric_name in ('mean', 'std', 'median', 'count', 'sum', 'min', 'max'):
            stat = binned_statistic(values1, values, bins=edges, statistic=metric_name)[0]
        elif metric[:2]=='p_':
            p = 0.01*float(metric_name[2:])
            q1 = binned_statistic(values1, values, bins=edges,
                statistic=lambda x: np.quantile(x, .5*(1-p)))[0]
            q2 = binned_statistic(values1, values, bins=edges,
                statistic=lambda x: np.quantile(x, .5*(1+p)))[0]
            stat = 0.5*(q2-q1)
        else:
            raise ValueError(f'Invalid value (={metric}) for metric.')
        if '.fill' in metric:
            kwargs.update({'alpha': .4})
            func = info['ax'].fill_betweenx if rotated else info['ax'].fill_between
            args = (values_mid, -stat, stat)
        else:
            func = info['ax'].plot
            args = (stat, values_mid) if rotated else (values_mid, stat)
        kwargs.update(metrics_kwargs.get(metric, {}))
        deep_update(info, {'plots': {metric: func(*(a[safe] for a in args), **kwargs)}})
    return info


def plot_metrics(values1, values2, bins1=30, bins2=30, mode='simple',
                 metrics=['mean', 'std'], metrics_kwargs={}, fig_kwargs={},
                 legend_kwargs={}):
    """
    Plot metrics of 1 component.

    Parameters
    ----------
    values1, values2: array
        Components x and y for plot.
    bins1, bins2: array, None
        Bins for component x and y.
    mode: str
        Mode to run metrics. Options are:

            * `simple` : metrics for `values2`.
            * `log` : metrics for `log10(values2)`.
            * `diff` : metrics for `values2-values1`.
            * `diff_log` : metrics for `log10(values2)-log10(values1)`.
            * `diff_z` : metrics for `(values2-values1)/(1+values1)`.

    metrics: list
        List of mettrics to be plotted. Possibilities are:

            * 'mean' : compute the mean of values for points within each bin.
            * 'std' : compute the standard deviation within each bin.
            * 'median' : compute the median of values for points within each bin.
            * 'count' : compute the count of points within each bin.
            * 'sum' : compute the sum of values for points within each bin.
            * 'min' : compute the minimum of values for points within each bin.
            * 'max' : compute the maximum of values for point within each bin.
            * 'p_#' : compute half the width where a percentile of data is found. Number must be
              between 0-100 (ex: 'p_68', 'p_95', 'p_99').

        If `'.fill'` is added to each metric, it will produce a filled region between (-metric,
        metric).

    metrics_kwargs: dict
        Dictionary of dictionary configs for each metric plots.
    fig_kwargs: dict
        Additional arguments for plt.subplots
    legend_kwargs: dict
        Additional arguments for plt.legend

    Returns
    -------
    info: dict
        Information of data in the plots, it contains the sections:

            * `fig`: `matplotlib.figure.Figure` object.
            * `axes`: `matplotlib.axes` used in the plot.
    """
    fig_kwargs_ = dict(figsize=(8, 6))
    fig_kwargs_.update(fig_kwargs)
    info = {key: value for key, value in zip(
        ('fig', 'axes'), plt.subplots(2, **fig_kwargs_))}
    # default args
    info['top'] = _plot_metrics(values1, values2, bins=bins1, mode=mode, ax=info['axes'][0],
                                metrics=metrics, metrics_kwargs=metrics_kwargs)
    info['bottom'] = _plot_metrics(values2, values1, bins=bins2, mode=mode, ax=info['axes'][1],
                                   metrics=metrics, metrics_kwargs=metrics_kwargs)
    info['axes'][0].legend(**legend_kwargs)
    info['axes'][0].xaxis.tick_top()
    info['axes'][0].xaxis.set_label_position('top')
    return info


def plot_density_metrics(values1, values2, bins1=30, bins2=30,
    ax_rotation=0, rotation_resolution=30, xscale='linear', yscale='linear',
    err1=None, err2=None, metrics_mode='simple', metrics=['std'],
    plt_kwargs={}, add_cb=True, cb_kwargs={},
    err_kwargs={}, metrics_kwargs={}, fig_kwargs={},
    fig_pos=(0.1, 0.1, 0.95, 0.95), fig_frac=(0.8, 0.01, 0.02), **kwargs):
    """
    Scatter plot with errorbars and color based on point density with scatter and bias panels

    Parameters
    ----------
    values1, values2: array
        Components x and y for plot.
    bins1, bins2: array, None
        Bins for component x and y.
    ax_rotation: float
        Angle (in degrees) for rotation of axis of binning. Overwrites use of (bins1, bins2) on main
        plot.
    rotation_resolution: int
        Number of bins to be used when ax_rotation!=0.
    xscale, yscale: str
        Scale for x/y axis.
    err1, err2: array, None
        Errors of component x and y.
    metrics_mode: str
        Mode to run metrics. Options are:

            * `simple` : metrics for `values2`.
            * `log` : metrics for `log10(values2)`.
            * `diff` : metrics for `values2-values1`.
            * `diff_log` : metrics for `log10(values2)-log10(values1)`.
            * `diff_z` : metrics for `(values2-values1)/(1+values1)`.

    metrics: list
        List of mettrics to be plotted. Possibilities are:

            * 'mean' : compute the mean of values for points within each bin.
            * 'std' : compute the standard deviation within each bin.
            * 'median' : compute the median of values for points within each bin.
            * 'count' : compute the count of points within each bin.
            * 'sum' : compute the sum of values for points within each bin.
            * 'min' : compute the minimum of values for points within each bin.
            * 'max' : compute the maximum of values for point within each bin.
            * 'p_#' : compute half the width where a percentile of data is found. Number must be
              between 0-100 (ex: 'p_68', 'p_95', 'p_99').

    plt_kwargs: dict
        Additional arguments for pylab.scatter
    add_cb: bool
        Plot colorbar
    cb_kwargs: dict
        Colorbar arguments
    err_kwargs: dict
        Additional arguments for pylab.errorbar
    metrics_kwargs: dict
        Dictionary of dictionary configs for each metric plots.
    fig_kwargs: dict
        Additional arguments for plt.subplots
    fig_pos: tuple
        List with edges for the figure. Must be in format (left, bottom, right, top)
    fig_frac: tuple
        Sizes of each panel in the figure. Must be in the format (main_panel, gap, colorbar)
        and have values: [0, 1]. Colorbar is only used with add_cb key.

    Other Parameters
    -----------------
    add_bindata: bool
        Plot binned data used for fit (default=False).
    add_fit: bool
        Fit and plot binned dat (default=False).
    **fit_kwargs:
        Other fit arguments (see `fit_*` paramters in `scaling.array_funcs.plot` for more info).

    Returns
    -------
    info: dict
        Information of data in the plots, it contains the sections:

            * `fig`: `matplotlib.figure.Figure` object.
            * `axes`: dictionary with each ax of the plot.
            * `metrics`: dictionary with the plots for each metric.
            * `binned_data` (optional): input data for fitting \
            (see `scaling.array_funcs.plot` for more info).
            * `fit` (optional): fitting output dictionary \
            (see `scaling.array_funcs.plot` for more info).
            * `plots` (optional): fit and binning plots \
            (see `scaling.array_funcs.plot` for more info).
    """
    fig_kwargs_ = dict(figsize=(8, 6))
    fig_kwargs_.update(fig_kwargs)
    info = {'fig': plt.figure(**fig_kwargs_), 'axes':{}, 'metrics':{}}
    left, bottom, right, top = fig_pos
    frac, gap, cb = fig_frac
    cb = cb if add_cb else 0
    xmain, xgap, xpanel = (right-left)*np.array([frac, gap, 1-frac-gap-cb])
    ymain, ygap, ypanel, ycb = (top-bottom)*np.array([frac, gap, 1-frac-gap-cb, cb-gap])
    info['axes']['main'] = info['fig'].add_axes([left, bottom, xmain, ymain])
    info['axes']['right'] = info['fig'].add_axes([left+xmain+xgap, bottom, xpanel, ymain])
    info['axes']['top'] = info['fig'].add_axes([left, bottom+ymain+ygap, xmain, ypanel])
    info['axes']['label'] = info['fig'].add_axes(
        [left+xmain+xgap, bottom+ymain+ygap, xpanel, ypanel])
    info['axes']['colorbar'] = info['fig'].add_axes(
        [left, bottom+ymain+2*ygap+ypanel, xmain+xgap+xpanel, ycb]) if add_cb else None
    # Main plot
    cb_kwargs_ = {'cax': info['axes']['colorbar'], 'orientation': 'horizontal'}
    cb_kwargs_.update(cb_kwargs)
    main_info = plot_density(
        values1, values2, bins1=bins1, bins2=bins2,
        ax_rotation=ax_rotation, rotation_resolution=rotation_resolution,
        xscale=xscale, yscale=yscale, err1=err1, err2=err2, ax=info['axes']['main'],
        plt_kwargs=plt_kwargs, add_cb=add_cb, cb_kwargs=cb_kwargs_,
        err_kwargs=err_kwargs, **kwargs)
    main_info.pop('ax')
    main_info.pop('ax_cb', None)
    info.update(main_info)
    if add_cb:
        info['axes']['colorbar'].xaxis.tick_top()
        info['axes']['colorbar'].xaxis.set_label_position('top')
    # Metrics plot
    info['metrics']['top'] = _plot_metrics(
        values1, values2, bins=bins1, mode=metrics_mode, ax=info['axes']['top'], metrics=metrics,
        metrics_kwargs=metrics_kwargs)['plots']
    info['metrics']['right'] = _plot_metrics(
        values2, values1, bins=bins2, mode=metrics_mode, ax=info['axes']['right'], rotated=True,
        metrics=metrics, metrics_kwargs=metrics_kwargs)['plots']
    # Adjust plots
    labels = [c.get_label() for c in info['axes']['right'].collections+info['axes']['right'].lines]
    labels = [rf"$\sigma_{{{l.replace('p_', '')}}}$" if l[:2]=='p_' else l for l in labels]
    info['axes']['label'].legend(
        info['axes']['right'].collections+info['axes']['right'].lines, labels)
    info['axes']['main'].set_xscale(xscale)
    info['axes']['main'].set_yscale(yscale)
    # Horizontal
    info['axes']['top'].set_xscale(xscale)
    info['axes']['top'].xaxis.set_minor_formatter(NullFormatter())
    info['axes']['top'].xaxis.set_major_formatter(NullFormatter())
    info['axes']['top'].set_xlim(info['axes']['main'].get_xlim())
    # Vertical
    info['axes']['right'].set_yscale(yscale)
    info['axes']['right'].yaxis.set_minor_formatter(NullFormatter())
    info['axes']['right'].yaxis.set_major_formatter(NullFormatter())
    info['axes']['right'].set_ylim(info['axes']['main'].get_ylim())
    # Label
    info['axes']['label'].axis('off')
    return info


def plot_dist(values1, values2, bins1_dist, bins2, values_aux=None, bins_aux=5,
              log_vals=False, log_aux=False, transpose=False,
              shape='steps', plt_kwargs={}, line_kwargs_list=None,
              fig_kwargs={}, legend_kwargs={}, panel_label_prefix='',
              add_panel_label=True, panel_label_format=lambda v: v,
              add_line_label=True, line_label_format=lambda v: v):
    """
    Plot distribution of a parameter, binned by other component in panels,
    and an optional secondary component in lines.

    Parameters
    ----------
    values1, values2: array
        Components x and y for plot.
    bins1_dist: array, int
        Bins for distribution of component 1.
    bins2: array, int
        Bins for component 2 (for panels/lines).
    values_aux: array
        Auxiliary component (to bin data in lines/panels).
    bins_aux: array, int
        Bins for component aux (for lines/panels).
    log_vals: bool
        Log scale for values (and int bins)
    log_aux: bool
        Log scale for aux values (and int bins)
    transpose: bool
        Invert lines and panels
    shape: str
        Shape of the lines. Can be steps or line.
    plt_kwargs: dict
        Additional arguments for pylab.plot
    line_kwargs_list: list, None
        List of additional arguments for plotting each line (using pylab.plot).
        Must have same size as len(bins_aux)-1
    fig_kwargs: dict
        Additional arguments for plt.subplots
    legend_kwargs: dict
        Additional arguments for plt.legend
    add_panel_label: bool
        Add bin label to panel
    panel_label_format: function
        Function to format the values of the bins
    panel_label_prefix: str
        Prefix to add to panel label
    add_line_label: bool
        Add bin label to line
    line_label_format: function
        Function to format the values of the bins

    Returns
    -------
    info: dict
        Information of data in the plots, it contains the sections:

            * `fig`: `matplotlib.figure.Figure` object.
            * `axes`: `matplotlib.axes` used in the plot.
    """
    if transpose and (values_aux is None):
        raise ValueError('transpose=true can only be used with values_aux!=None')
    edges1_dist = autobins(values1, bins1_dist, log=log_vals)
    edges2 = autobins(values2, bins2)
    edges_aux = None if values_aux is None else autobins(values_aux, bins_aux, log_aux)
    masks2 = binmasks(values2, edges2)
    masks_aux = [np.ones(len(values1), dtype=bool)] if values_aux is None\
                else binmasks(values_aux, edges_aux)
    steps1 = np.log(edges1_dist[1:])-np.log(edges1_dist[:-1]) if log_vals\
            else edges1_dist[1:]-edges1_dist[:-1]
    # Use quantities relative to panel and lines:
    panel_masks, line_masks = (masks_aux, masks2) if transpose else (masks2, masks_aux)
    panel_edges, line_edges = (edges_aux, edges2) if transpose else (edges2, edges_aux)
    nj = int(np.ceil(np.sqrt(panel_edges[:-1].size)))
    ni = int(np.ceil(panel_edges[:-1].size/float(nj)))
    fig_kwargs_ = dict(sharex=True, figsize=(8, 6))
    fig_kwargs_.update(fig_kwargs)
    info = {key: value for key, value in zip(
        ('fig', 'axes'), plt.subplots(ni, nj, **fig_kwargs_))}
    line_kwargs_list = none_val(line_kwargs_list, [{}] if values_aux is None else
        [{'label': ph.get_bin_label(vb, vt, line_label_format)}
            for vb, vt in zip(line_edges, line_edges[1:])])
    for ax, maskp in zip(info['axes'].flatten(), panel_masks):
        ph.add_grid(ax)
        kwargs = {}
        kwargs.update(plt_kwargs)
        for maskl, p_kwargs in zip(line_masks, line_kwargs_list):
            kwargs.update(p_kwargs)
            hist = np.histogram(values1[maskp*maskl], bins=edges1_dist)[0]
            norm = (hist*steps1).sum()
            norm = norm if norm>0 else 1
            ph.plot_hist_line(hist/norm, edges1_dist,
                              ax=ax, shape=shape, **kwargs)
        ax.set_xscale('log' if log_vals else 'linear')
        ax.set_yticklabels([])
    for ax in info['axes'].flatten()[len(panel_edges)-1:]:
        ax.axis('off')
    if add_panel_label:
        ph.add_panel_bin_label(info['axes'],  panel_edges[:-1], panel_edges[1:],
                               format_func=panel_label_format,
                               prefix=panel_label_prefix)
    if values_aux is not None and add_line_label:
        info['axes'].flatten()[0].legend(**legend_kwargs)
    return info


def plot_density_dist(values1, values2, bins1=30, bins2=30,
    ax_rotation=0, rotation_resolution=30, xscale='linear', yscale='linear',
    err1=None, err2=None, plt_kwargs={}, add_cb=True, cb_kwargs={},
    err_kwargs={}, fig_kwargs={}, fig_pos=(0.1, 0.1, 0.95, 0.95), fig_frac=(0.8, 0.01, 0.02),
    vline_kwargs={}, **kwargs):
    """
    Scatter plot with errorbars and color based on point density with distribution panels.

    Parameters
    ----------
    values1, values2: array
        Components x and y for plot.
    bins1, bins2: array, None
        Bins for component x and y (for density colors).
    ax_rotation: float
        Angle (in degrees) for rotation of axis of binning. Overwrites use of (bins1, bins2) on main
        plot.
    rotation_resolution: int
        Number of bins to be used when ax_rotation!=0.
    xscale, yscale: str
        Scale for x/y axis.
    err1, err2: array, None
        Errors of component x and y.
    plt_kwargs: dict
        Additional arguments for pylab.scatter
    add_cb: bool
        Plot colorbar
    cb_kwargs: dict
        Colorbar arguments
    err_kwargs: dict
        Additional arguments for pylab.errorbar
    fig_kwargs: dict
        Additional arguments for plt.subplots
    fig_pos: tuple
        List with edges for the figure. Must be in format (left, bottom, right, top)
    fig_frac: tuple
        Sizes of each panel in the figure. Must be in the format (main_panel, gap, colorbar)
        and have values: [0, 1]. Colorbar is only used with add_cb key.

    Other Parameters
    -----------------
    add_bindata: bool
        Plot binned data used for fit (default=False).
    add_fit: bool
        Fit and plot binned dat (default=False).
    **fit_kwargs:
        Other fit arguments (see `fit_*` paramters in `scaling.array_funcs.plot` for more info).
    vline_kwargs: dict
        Arguments for vlines marking bins in main plot, used in plt.axvline.

    Returns
    -------
    info: dict
        Information of data in the plots, it contains the sections:

            * `fig`: `matplotlib.figure.Figure` object.
            * `axes`: dictionary with each ax of the plot.
            * `binned_data` (optional): input data for fitting \
            (see `scaling.array_funcs.plot` for more info).
            * `fit` (optional): fitting output dictionary \
            (see `scaling.array_funcs.plot` for more info).
            * `plots` (optional): fit and binning plots \
            (see `scaling.array_funcs.plot` for more info).
    """
    # Fig
    fig_kwargs_ = dict(figsize=(8, 6))
    fig_kwargs_.update(fig_kwargs)
    info = {'fig': plt.figure(**fig_kwargs_), 'axes':{}}
    left, bottom, right, top = fig_pos
    frac, gap, cb = fig_frac
    cb = cb if add_cb else 0
    xmain, xgap = (right-left)*np.array([frac, gap])
    ymain, ygap, ypanel, ycb = (top-bottom)*np.array([frac, gap, 1-frac-gap-cb, cb-gap])
    info['axes']['main'] = info['fig'].add_axes([left, bottom, xmain, ymain])
    info['axes']['colorbar'] = info['fig'].add_axes(
        [left+xmain+xgap, bottom, ycb, ymain]) if add_cb else None
    # Main plot
    cb_kwargs_ = {'cax': info['axes']['colorbar'], 'orientation': 'vertical'}
    cb_kwargs_.update(cb_kwargs)
    plot_density(values1, values2, bins1=bins1, bins2=bins2,
        ax_rotation=ax_rotation, rotation_resolution=rotation_resolution,
        xscale=xscale, yscale=yscale, err1=err1, err2=err2, ax=info['axes']['main'],
        plt_kwargs=plt_kwargs, add_cb=add_cb, cb_kwargs=cb_kwargs_,
        err_kwargs=err_kwargs)
    if add_cb:
        info['axes']['colorbar'].xaxis.tick_top()
        info['axes']['colorbar'].xaxis.set_label_position('top')
    info['axes']['main'].set_xscale(xscale)
    info['axes']['main'].set_yscale(yscale)
    # Add v lines
    info['axes']['main'].xaxis.grid(False, which='both')
    fit_bins1 = autobins(values1, kwargs.get('fit_bins1', 10), xscale=='log')
    vline_kwargs_ = {'lw':.5, 'color':'0'}
    vline_kwargs_.update(vline_kwargs)
    for v in fit_bins1:
        info['axes']['main'].axvline(v, **vline_kwargs_)
    # Dist plot
    fit_bins2 = autobins(values2, kwargs.get('fit_bins2', 30), yscale=='log')
    masks1 = binmasks(values1, fit_bins1)
    xlims = info['axes']['main'].get_xlim()
    if xscale=='log':
        xlims, fit_bins1 = np.log(xlims), np.log(fit_bins1)
    xpos = [xmain*(x-xlims[0])/(xlims[1]-xlims[0]) for x in fit_bins1]
    info['axes']['top'] = [info['fig'].add_axes([left+xl, bottom+ymain+ygap, xr-xl, ypanel]) # top
                for xl, xr in zip(xpos, xpos[1:])]
    fit_line_kwargs_list = kwargs.get('fit_line_kwargs_list', [{} for m in masks1])
    dlims = (np.inf, -np.inf)
    for ax, mask, lkwarg in zip(info['axes']['top'], masks1, fit_line_kwargs_list):
        ph.add_grid(ax)
        kwargs_ = {}
        kwargs_.update(kwargs.get('fit_plt_kwargs', {}))
        kwargs_.update(lkwarg)
        ph.plot_hist_line(*np.histogram(values2[mask], bins=fit_bins2),
                          ax=ax, shape='line', rotate=True, **kwargs_)
        ax.set_xlim(ax.get_xlim()[::-1])
        ax.set_xticklabels([])
        ax.set_yscale(xscale)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
        dlims = min(dlims[0], ax.get_ylim()[0]), max(dlims[1], ax.get_ylim()[1])
    for ax in info['axes']['top']:
        ax.set_ylim(dlims)
    for ax in info['axes']['top'][:-1]:
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
    # Bindata and fit
    kwargs['fit_err2'] = kwargs.get('fit_err2', err2)
    kwargs['fit_add_fit'] = kwargs.get('add_fit', False)
    kwargs['fit_add_bindata'] = kwargs.get('add_bindata', kwargs['fit_add_fit'])
    info.update(_add_bindata_and_powlawfit(
        info['axes']['main'], values1, values2,
        **{k[4:]:v for k, v in kwargs.items() if k[:4]=='fit_'}))
    if kwargs['fit_add_bindata'] and info.get('plots', {}).get('errorbar', False):
        color = info['plots']['errorbar'].lines[0]._color
        for ax, m, b, t in zip(
                info['axes']['top'], info['plots']['errorbar'].lines[0]._y,
                info['plots']['errorbar'].lines[1][0]._y, info['plots']['errorbar'].lines[1][1]._y):
            xlim = ax.get_xlim()
            ax.axhline(m, color=color)
            ax.fill_between(xlim, b, t, alpha=.3, lw=0, color=color)
            ax.set_xlim(xlim)
    return info
