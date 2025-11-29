"""@file clevar/match_metrics/scaling/aux_funcs.py
Auxiliary functions for scaling array functions.
"""

import warnings

import numpy as np
from scipy.interpolate import UnivariateSpline as spline
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic

from ...utils import autobins, binmasks, deep_update, gaussian, none_val, updated_dict
from .. import plot_helper as ph
from ..plot_helper import NullFormatter, plt


def _prep_fit_data(xvals, yvals, yerr=None, statistics="mean", bins_x=None, bins_y=None):
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
    if statistics == "individual":
        return xvals, yvals, yerr
    if statistics == "mode":
        bins_hist = autobins(yvals, bins_y)
        bins_hist_m = 0.5 * (bins_hist[1:] + bins_hist[:-1])
        # pylint: disable=unnecessary-lambda-assignment
        statistic = lambda vals: bins_hist_m[np.histogram(vals, bins=bins_hist)[0].argmax()]
    elif statistics == "mean":
        statistic = "mean"
    else:
        raise ValueError(f"statistics ({statistics}) must be in (individual, mean, mode)")
    xbins = autobins(xvals, bins_x)
    xdata = binned_statistic(xvals, xvals, bins=xbins, statistic="mean")[0]
    ydata = binned_statistic(xvals, yvals, bins=xbins, statistic=statistic)[0]
    std = binned_statistic(xvals, yvals, bins=xbins, statistic="std")[0]
    err = binned_statistic(
        xvals, none_val(yerr, np.zeros(len(yvals))), bins=xbins, statistic="mean"
    )[0]
    valid = ~np.isnan(xdata)
    return xdata[valid], ydata[valid], np.sqrt(std**2 + err**2)[valid]


def _pw_func(xvals, coeff_ang, coeff_lin):
    """Linear function for powerlaw fit"""
    return coeff_ang * xvals + coeff_lin


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
    bindata_kwargs: dict, None
        Additional arguments for pylab.errorbar.
    plt_kwargs: dict, None
        Additional arguments for plot of fit pylab.scatter.
    legend_kwargs: dict, None
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
                * func_scat: scatter of fitted function.
                * func_dist: P(y|x) - Probability of having y given a value for x, assumes \
                normal distribution and uses scatter of the fitted function.
                * func_scat_interp: interpolated scatter from data.
                * func_dist_interp: P(y|x) using interpolated scatter.

            * plots (optional): additional plots:

                * fit: fitted data
                * errorbar: binned data

    """
    # pylint: disable=too-many-locals
    info = {}
    # Default parameters
    add_bindata = kwargs.get("add_bindata", False)
    add_fit = kwargs.get("add_fit", False)
    if ((not add_bindata) and (not add_fit)) or len(values1) <= 1:
        warnings.warn("Too few data points for fit!")
        return info
    mode = kwargs.get("statistics", "mode")
    bins1 = kwargs.get("bins1", 10)
    bins2 = kwargs.get("bins2", 30)
    # set log/lin funcs
    tfunc, ifunc = (np.log10, lambda x: 10**x) if log else (lambda x: x, lambda x: x)
    # data
    vbin_1, vbin_2, vbin_err2 = _prep_fit_data(
        tfunc(values1),
        tfunc(values2),
        bins_x=tfunc(bins1) if hasattr(bins1, "__len__") else bins1,
        bins_y=tfunc(bins2) if hasattr(bins2, "__len__") else bins2,
        yerr=err2 if (err2 is None or not log) else err2 / (values2 * np.log(10)),
        statistics=mode,
    )
    if len(vbin_1) == 0:
        return info
    info["binned_data"] = {"x": vbin_1, "y": vbin_2, "yerr": vbin_err2}
    # fit
    if add_fit:
        fit, cov = curve_fit(_pw_func, vbin_1, vbin_2, sigma=vbin_err2, absolute_sigma=True)[:2]

        # Functions with fit values
        def __fit_func__(xvals):
            return _pw_func(tfunc(xvals), *fit)

        __scat_func__ = np.vectorize(
            lambda xvals: np.sqrt(np.dot([tfunc(xvals), 1], np.dot(cov, [tfunc(xvals), 1])))
        )
        scat_spline = spline(vbin_1, vbin_err2, k=1)

        def __scat_interp__(xvals):
            return scat_spline(tfunc(xvals))

        info["fit"] = {
            "pars": fit,
            "cov": cov,
            "func": lambda x: ifunc(__fit_func__(x)),
            "func_plus": lambda x: ifunc(__fit_func__(x) + __scat_func__(x)),
            "func_minus": lambda x: ifunc(__fit_func__(x) - __scat_func__(x)),
            "func_scat": __scat_func__,
            "func_dist": lambda y, x: gaussian(tfunc(y), __fit_func__(x), __scat_func__(x)),
            "func_scat_interp": __scat_interp__,
            "func_dist_interp": lambda y, x: gaussian(
                tfunc(y), __fit_func__(x), __scat_interp__(x)
            ),
        }
        # labels
        xlabel, ylabel = kwargs.get("label_components", ("x", "y"))
        xlabel = xlabel.replace("$", "") if "$" in xlabel else xlabel.replace("_", r"\_")
        ylabel = ylabel.replace("$", "") if "$" in ylabel else ylabel.replace("_", r"\_")
        sig = np.sqrt(np.diag(cov))

        def _fmt0(xval):
            return f"{xval:.2f}" if 0.01 < abs(fit[0]) < 100 else f"{xval:.2e}"

        def _fmt1(xval):
            return f"{xval:.2f}" if 0.01 < abs(fit[1]) < 100 else f"{xval:.2e}"

        fit0_lab = rf"({_fmt0(fit[0])}\pm {_fmt0(sig[0])})"
        fit1_lab = rf'{"-"*int(fit[1]<0)}({_fmt1(abs(fit[1]))}\pm {_fmt1(sig[1])})'
        avg_label = rf"\left<{ylabel}\right|\left.{xlabel}\right>"
        fit_label = (
            rf"${avg_label}=10^{{{fit1_lab}}}\;({xlabel})^{{{fit0_lab}}}$"
            if log
            else rf"${avg_label}={fit0_lab}\;{xlabel}{'+'*(fit[1]>=0)}{fit1_lab}$"
        )
        # plot fit
        plot_kwargs_ = updated_dict(
            {"color": "r", "label": fit_label},
            kwargs.get("plot_kwargs", {}),
        )
        sort = np.argsort(values1)
        xvals = values1[sort]
        ax.plot(xvals, info["fit"]["func"](xvals), **plot_kwargs_)
        deep_update(
            info,
            {
                "plots": {
                    "fit": ax.fill_between(
                        xvals,
                        info["fit"]["func_plus"](xvals),
                        info["fit"]["func_minus"](xvals),
                        color=plot_kwargs_["color"],
                        alpha=0.2,
                        lw=0,
                    )
                }
            },
        )
    if add_bindata and not mode == "individual":
        deep_update(
            info,
            {
                "plots": {
                    "errorbar": ax.errorbar(
                        ifunc(vbin_1),
                        ifunc(vbin_2),
                        yerr=(
                            ifunc(vbin_2)
                            * np.array([1 - 1 / ifunc(vbin_err2), ifunc(vbin_err2) - 1])
                            if log
                            else vbin_err2
                        ),
                        **updated_dict(
                            {
                                "elinewidth": 1,
                                "capsize": 2,
                                "fmt": ".",
                                "ms": 10,
                                "ls": "",
                                "color": "m",
                            },
                            kwargs.get("bindata_kwargs", {}),
                        ),
                    )
                }
            },
        )
    # legend
    if any(c.get_label()[0] != "_" for c in ax.collections + ax.lines):
        ax.legend(**updated_dict(kwargs.get("legend_kwargs", {})))
    return info


def _add_bindata_and_powlawfit_array(ax, values1, values2, err2=None, **kwargs):
    """
    Add binned data and powerlaw fit to plot. To be used by array_funcs,
    same as _add_bindata_and_powlawfit, but takes fit_* aux arguments.

    Parameters
    ----------
    ax: matplotlib.axes
        Ax to add plot.
    values1, values2: array
        Components to be binned and fitted
    err2: array
        Error of component 2. Used if fit_err2 not provided.
    fit_err2: array
        Error of component 2.
    fit_log: bool
        Bin and fit in log values (default=False).
    fit_statistics: str
        Statistics to be used in fit (default=mean). Options are:

            * `individual` : Use each point
            * `mode` : Use mode of component 2 distribution in each comp 1 bin, requires bins2.
            * `mean` : Use mean of component 2 distribution in each comp 1 bin, requires bins2.

    fit_bins1, fit_bins2: array, None
        Bins for component x (default=10) and y (default=30).
    fit_add_bindata: bool
        Plot binned data used for fit (default=False).
    add_fit: bool
        Fit and plot binned dat (default=False).
    fit_bindata_kwargs: dict, None
        Additional arguments for pylab.errorbar.
    fit_plt_kwargs: dict, None
        Additional arguments for plot of fit pylab.scatter.
    fit_legend_kwargs: dict, None
        Additional arguments for plt.legend.
    fit_label_components: tuple (of strings)
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
                * func_scat: scatter of fitted function.
                * func_dist: P(y|x) - Probability of having y given a value for x, assumes \
                normal distribution and uses scatter of the fitted function.
                * func_scat_interp: interpolated scatter from data.
                * func_dist_interp: P(y|x) using interpolated scatter.

            * plots (optional): additional plots:

                * fit: fitted data
                * errorbar: binned data

    """
    use_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == "fit_"}
    use_kwargs.update(
        {
            "err2": use_kwargs.get("err2", err2),
            "add_bindata": kwargs.get("add_bindata", kwargs.get("add_fit", False)),
            "add_fit": kwargs.get("add_fit", False),
        }
    )
    return _add_bindata_and_powlawfit(ax, values1, values2, **use_kwargs)


def _plot_panel(
    plot_function,
    values_panel,
    bins_panel,
    panel_kwargs_list=None,
    panel_kwargs_errlist=None,
    fig_kwargs=None,
    add_label=True,
    label_format=lambda v: v,
    plt_kwargs=None,
    err_kwargs=None,
    **plt_func_kwargs,
):
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
    fig_kwargs: dict, None
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
    # pylint: disable=too-many-locals
    edges = (
        bins_panel
        if hasattr(bins_panel, "__len__")
        else np.linspace(min(values_panel), max(values_panel), bins_panel)
    )
    ncol = int(np.ceil(np.sqrt(len(edges[:-1]))))
    nrow = int(np.ceil(len(edges[:-1]) / float(ncol)))
    fig, axes = plt.subplots(
        nrow, ncol, **updated_dict({"sharex": True, "sharey": True, "figsize": (8, 6)}, fig_kwargs)
    )
    info = {"fig": fig, "axes": axes}
    ax_conf = []
    for ax, mask, p_kwargs, p_e_kwargs in zip(
        info["axes"].flatten(),
        [(values_panel >= v0) * (values_panel < v1) for v0, v1 in zip(edges, edges[1:])],
        none_val(panel_kwargs_list, iter(lambda: {}, 1)),
        none_val(panel_kwargs_errlist, iter(lambda: {}, 1)),
    ):
        ph.add_grid(ax)
        ax_conf.append(
            plot_function(
                ax=ax,
                plt_kwargs=updated_dict(plt_kwargs, p_kwargs),
                err_kwargs=updated_dict(err_kwargs, p_e_kwargs),
                **{
                    k: (
                        v[mask]
                        if (hasattr(v, "__len__") and len(v) == mask.size)
                        and (not isinstance(v, (str, dict)))
                        else v
                    )
                    for k, v in plt_func_kwargs.items()
                },
            )
        )
        ax_conf[-1].pop("ax")
    ax_conf += [{} for i in range(nrow * ncol - len(ax_conf))]  # complete missing vals
    info["axes_conf"] = np.reshape(ax_conf, (nrow, ncol))
    for ax in info["axes"].flatten()[len(edges) - 1 :]:
        ax.axis("off")
    if add_label:
        ph.add_panel_bin_label(info["axes"], edges[:-1], edges[1:], format_func=label_format)
    return info


def _plot_metrics(
    values1,
    values2,
    bins=30,
    mode="diff_z",
    ax=None,
    metrics=("mean"),
    metrics_kwargs=None,
    rotated=False,
):
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

    metrics_kwargs: dict, None
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
    # pylint: disable=too-many-locals
    edges = autobins(values1, bins, log=mode == "log")
    bmask = np.array(binmasks(values1, edges))
    safe = [m[m].size > 1 for m in bmask]
    values = {
        "diff": lambda v1, v2: v2 - v1,
        "diff_z": lambda v1, v2: (v2 - v1) / (1 + v1),
        "diff_log": lambda v1, v2: np.log10(v2) - np.log10(v1),
        "simple": lambda v1, v2: v2,
        "log": lambda v1, v2: np.log10(v2),
    }[mode](values1, values2)
    values_mid = (
        10 ** (0.5 * (np.log10(edges[1:]) + np.log10(edges[:-1])))
        if mode == "log"
        else 0.5 * (edges[1:] + edges[:-1])
    )
    # set for rotation
    info = {"ax": plt.axes() if ax is None else ax}
    ph.add_grid(info["ax"])
    # plot
    for metric in metrics:
        metric_name = metric.replace(".fill", "")
        kwargs = {"label": metric_name}
        if metric_name in ("mean", "std", "median", "count", "sum", "min", "max"):
            stat = binned_statistic(values1, values, bins=edges, statistic=metric_name)[0]
        elif metric[:2] == "p_":
            # pylint: disable=cell-var-from-loop
            _perc = 0.01 * float(metric_name[2:])
            quant1 = binned_statistic(
                values1, values, bins=edges, statistic=lambda x: np.quantile(x, 0.5 * (1 - _perc))
            )[0]
            quant2 = binned_statistic(
                values1, values, bins=edges, statistic=lambda x: np.quantile(x, 0.5 * (1 + _perc))
            )[0]
            stat = 0.5 * (quant2 - quant1)
        else:
            raise ValueError(f"Invalid value (={metric}) for metric.")
        if ".fill" in metric:
            kwargs.update({"alpha": 0.4})
            func = info["ax"].fill_betweenx if rotated else info["ax"].fill_between
            args = (values_mid, -stat, stat)
        else:
            func = info["ax"].plot
            args = (stat, values_mid) if rotated else (values_mid, stat)
        kwargs.update(updated_dict(metrics_kwargs).get(metric, {}))
        deep_update(info, {"plots": {metric: func(*(a[safe] for a in args), **kwargs)}})
    return info


def _plot_dist_vertical(axes, values, masks, fit_line_kwargs_list, fit_bins, kwargs, xscale):
    dlims = (np.inf, -np.inf)
    for ax, mask, lkwarg in zip(axes, masks, fit_line_kwargs_list):
        ph.add_grid(ax)
        ph.plot_hist_line(
            *np.histogram(values[mask], bins=fit_bins),
            ax=ax,
            shape="line",
            rotate=True,
            **updated_dict(kwargs.get("fit_plt_kwargs"), lkwarg),
        )
        ax.set_xlim(ax.get_xlim()[::-1])
        ax.set_xticklabels([])
        ax.set_yscale(xscale)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        dlims = min(dlims[0], ax.get_ylim()[0]), max(dlims[1], ax.get_ylim()[1])
    for ax in axes:
        ax.set_ylim(dlims)
    for ax in axes[:-1]:
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
