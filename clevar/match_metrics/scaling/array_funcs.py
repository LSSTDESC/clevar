"""@file clevar/match_metrics/scaling/array_funcs.py

Main scaling functions using arrays.
"""
# pylint: disable=too-many-lines

import numpy as np

from ...utils import (
    autobins,
    binmasks,
    none_val,
    subdict,
    subdict_exclude,
    updated_dict,
)
from .. import plot_helper as ph
from ..plot_helper import plt
from .aux_funcs import (
    _add_bindata_and_powlawfit_array,
    _plot_dist_vertical,
    _plot_metrics,
    _plot_panel,
)


def plot(
    values1,
    values2,
    err1=None,
    err2=None,
    ax=None,
    plt_kwargs=None,
    err_kwargs=None,
    values_color=None,
    add_cb=True,
    cb_kwargs=None,
    **kwargs,
):
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
    plt_kwargs: dict, None
        Additional arguments for pylab.scatter
    err_kwargs: dict, None
        Additional arguments for pylab.errorbar
    values_color: array, None
        Values for color (cmap scale).
    add_cb: bool
        Plot colorbar when values_color is not `None`.
    cb_kwargs: dict, None
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
    fit_legend_kwargs: dict, None
        Additional arguments for plt.legend.
    fit_bindata_kwargs: dict, None
        Additional arguments for pylab.errorbar.
    fit_plt_kwargs: dict, None
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
                * `func_dist`: `P(y|x)` - Probability of having y given a value for x, assumes \
                normal distribution and uses scatter of the fitted function.
                * `func_scat_interp`: interpolated scatter from data.
                * `func_dist_interp`: `P(y|x)` using interpolated scatter.

            * `plots` (optional): additional plots:

                * `fit`: fitted data
                * `errorbar`: binned data
    """
    # pylint: disable=too-many-locals
    info = {"ax": plt.axes() if ax is None else ax}
    ph.add_grid(info["ax"])
    if len(values1) == 0:
        return info
    # Plot points
    plt_kwargs_ = {"s": 1}
    if values_color is None:
        xplot, yplot = values1, values2
    else:
        isort = np.argsort(values_color)
        xplot, yplot, zplot = [v[isort] for v in (values1, values2, values_color)]
        plt_kwargs_["c"] = zplot
    scat = info["ax"].scatter(xplot, yplot, **updated_dict(plt_kwargs_, plt_kwargs))
    # Plot errorbar
    err_kwargs_ = {"elinewidth": 0.5, "capsize": 0, "fmt": ".", "ms": 0, "ls": ""}
    if values_color is None:
        xerr, yerr = err1, err2
    else:
        cbar = plt.colorbar(scat, ax=info["ax"], **updated_dict(cb_kwargs))
        xerr = err1[isort] if err1 is not None else None
        yerr = err2[isort] if err2 is not None else None
        err_kwargs_["ecolor"] = [cbar.mappable.cmap(cbar.mappable.norm(c)) for c in zplot]
    if err1 is not None or err2 is not None:
        info["ax"].errorbar(
            xplot, yplot, xerr=xerr, yerr=yerr, **updated_dict(err_kwargs_, err_kwargs)
        )
    if values_color is not None:
        if add_cb:
            info["cb"] = cbar
        else:
            cbar.remove()
    # Bindata and fit
    kwargs["fit_err2"] = kwargs.get("fit_err2", err2)
    kwargs["fit_add_fit"] = kwargs.get("add_fit", False)
    kwargs["fit_add_bindata"] = kwargs.get("add_bindata", kwargs["fit_add_fit"])
    info.update(_add_bindata_and_powlawfit_array(info["ax"], values1, values2, **kwargs))
    return info


def plot_density(
    values1,
    values2,
    bins1=30,
    bins2=30,
    ax_rotation=0,
    rotation_resolution=30,
    xscale="linear",
    yscale="linear",
    err1=None,
    err2=None,
    ax=None,
    plt_kwargs=None,
    add_cb=True,
    cb_kwargs=None,
    err_kwargs=None,
    **kwargs,
):
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
    plt_kwargs: dict, None
        Additional arguments for pylab.scatter
    add_cb: bool
        Plot colorbar
    cb_kwargs: dict, None
        Colorbar arguments
    err_kwargs: dict, None
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
    # pylint: disable=unused-argument
    # pylint: disable=possibly-unused-variable
    _kwargs = locals()
    _kwargs["values_color"] = (
        ph.get_density_colors(
            values1,
            values2,
            bins1,
            bins2,
            **subdict(
                locals(),
                ["ax_rotation", "rotation_resolution", "xscale", "yscale"],
            ),
        )
        if len(values1) > 0
        else None
    )
    _kwargs.update(_kwargs.pop("kwargs"))
    return plot(**_kwargs)


def plot_panel(
    values1,
    values2,
    values_panel,
    bins_panel,
    err1=None,
    err2=None,
    values_color=None,
    plt_kwargs=None,
    err_kwargs=None,
    add_cb=True,
    cb_kwargs=None,
    panel_kwargs_list=None,
    panel_kwargs_errlist=None,
    fig_kwargs=None,
    add_label=True,
    label_format=lambda v: v,
    **kwargs,
):
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
    plt_kwargs: dict, None
        Additional arguments for pylab.scatter
    err_kwargs: dict, None
        Additional arguments for pylab.errorbar
    add_cb: bool
        Plot colorbar when values_color is not `None`.
    cb_kwargs: dict, None
        Colorbar arguments
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
    # pylint: disable=unused-argument
    return _plot_panel(
        # _plot_panel arguments
        plot_function=plot,
        **subdict_exclude(locals(), ["kwargs"]),
    )


def plot_density_panel(
    values1,
    values2,
    values_panel,
    bins_panel,
    bins1=30,
    bins2=30,
    ax_rotation=0,
    rotation_resolution=30,
    xscale="linear",
    yscale="linear",
    err1=None,
    err2=None,
    plt_kwargs=None,
    add_cb=True,
    cb_kwargs=None,
    err_kwargs=None,
    panel_kwargs_list=None,
    panel_kwargs_errlist=None,
    fig_kwargs=None,
    add_label=True,
    label_format=lambda v: v,
    **kwargs,
):
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
    plt_kwargs: dict, None
        Additional arguments for pylab.scatter
    add_cb: bool
        Plot colorbar
    cb_kwargs: dict, None
        Colorbar arguments
    err_kwargs: dict, None
        Additional arguments for pylab.errorbar
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
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-arguments
    # pylint: disable=unused-argument
    _kwargs = locals()
    _kwargs.update(_kwargs.pop("kwargs"))
    return _plot_panel(
        plot_function=plot_density,
        **_kwargs,
    )


def plot_metrics(
    values1,
    values2,
    bins1=30,
    bins2=30,
    mode="simple",
    metrics=("mean", "std"),
    metrics_kwargs=None,
    fig_kwargs=None,
    legend_kwargs=None,
):
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

    metrics_kwargs: dict, None
        Dictionary of dictionary configs for each metric plots.
    fig_kwargs: dict, None
        Additional arguments for plt.subplots
    legend_kwargs: dict, None
        Additional arguments for plt.legend

    Returns
    -------
    info: dict
        Information of data in the plots, it contains the sections:

            * `fig`: `matplotlib.figure.Figure` object.
            * `axes`: `matplotlib.axes` used in the plot.
    """
    fig, axes = plt.subplots(2, **updated_dict({"figsize": (8, 6)}, fig_kwargs))
    info = {"fig": fig, "axes": axes}
    # default args
    _common = {"mode": mode, "metrics": metrics, "metrics_kwargs": metrics_kwargs}
    info["top"] = _plot_metrics(values1, values2, bins=bins1, ax=info["axes"][0], **_common)
    info["bottom"] = _plot_metrics(values2, values1, bins=bins2, ax=info["axes"][1], **_common)
    info["axes"][0].legend(**updated_dict(legend_kwargs))
    info["axes"][0].xaxis.tick_top()
    info["axes"][0].xaxis.set_label_position("top")
    return info


# pylint: disable=unused-argument
def plot_density_metrics(
    values1,
    values2,
    bins1=30,
    bins2=30,
    ax_rotation=0,
    rotation_resolution=30,
    xscale="linear",
    yscale="linear",
    err1=None,
    err2=None,
    metrics_mode="simple",
    metrics=("std"),
    plt_kwargs=None,
    add_cb=True,
    cb_kwargs=None,
    err_kwargs=None,
    metrics_kwargs=None,
    fig_kwargs=None,
    fig_pos=(0.1, 0.1, 0.95, 0.95),
    fig_frac=(0.8, 0.01, 0.02),
    **kwargs,
):
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

    plt_kwargs: dict, None
        Additional arguments for pylab.scatter
    add_cb: bool
        Plot colorbar
    cb_kwargs: dict, None
        Colorbar arguments
    err_kwargs: dict, None
        Additional arguments for pylab.errorbar
    metrics_kwargs: dict, None
        Dictionary of dictionary configs for each metric plots.
    fig_kwargs: dict, None
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
    # pylint: disable=too-many-locals
    fig = plt.figure(**updated_dict({"figsize": (8, 6)}, fig_kwargs))
    axes = {}
    info = {
        "metrics": {},
    }
    left, bottom, right, top = fig_pos
    fmain, fgap, fcb = fig_frac
    fcb = fcb if add_cb else 0
    xmain, xgap, xpanel = (right - left) * np.array([fmain, fgap, 1 - fmain - fgap - fcb])
    ymain, ygap, ypanel, ycb = (top - bottom) * np.array(
        [fmain, fgap, 1 - fmain - fgap - fcb, fcb - fgap]
    )
    axes["main"] = fig.add_axes([left, bottom, xmain, ymain])
    axes["right"] = fig.add_axes([left + xmain + xgap, bottom, xpanel, ymain])
    axes["top"] = fig.add_axes([left, bottom + ymain + ygap, xmain, ypanel])
    axes["label"] = fig.add_axes([left + xmain + xgap, bottom + ymain + ygap, xpanel, ypanel])
    axes["colorbar"] = (
        fig.add_axes([left, bottom + ymain + 2 * ygap + ypanel, xmain + xgap + xpanel, ycb])
        if add_cb
        else None
    )
    # Main plot
    info.update(
        subdict_exclude(
            plot_density(
                values1,
                values2,
                **subdict(
                    locals(),
                    [
                        "bins1",
                        "bins2",
                        "ax_rotation",
                        "rotation_resolution",
                        "xscale",
                        "yscale",
                        "err1",
                        "err2",
                        "plt_kwargs",
                        "add_cb",
                        "err_kwargs",
                    ],
                ),
                ax=axes["main"],
                cb_kwargs=updated_dict(
                    {"cax": axes["colorbar"], "orientation": "horizontal"}, cb_kwargs
                ),
                **kwargs,
            ),
            ["ax", "ax_cb"],
        )
    )
    if add_cb:
        axes["colorbar"].xaxis.tick_top()
        axes["colorbar"].xaxis.set_label_position("top")
    # Metrics plot
    _common = {"mode": metrics_mode, "metrics": metrics, "metrics_kwargs": metrics_kwargs}
    info["metrics"]["top"] = _plot_metrics(values1, values2, bins=bins1, ax=axes["top"], **_common)[
        "plots"
    ]
    info["metrics"]["right"] = _plot_metrics(
        values2, values1, bins=bins2, ax=axes["right"], rotated=True, **_common
    )["plots"]
    # Adjust plots
    labels = [
        rf"$\sigma_{{{l.replace('p_', '')}}}$" if l[:2] == "p_" else l
        for l in (c.get_label() for c in axes["right"].collections + axes["right"].lines)
    ]
    axes["label"].legend(axes["right"].collections + axes["right"].lines, labels)
    axes["main"].set_xscale(xscale)
    axes["main"].set_yscale(yscale)
    # Horizontal
    axes["top"].set_xscale(xscale)
    ph.rm_axis_ticklabels(axes["top"].xaxis)
    axes["top"].set_xlim(axes["main"].get_xlim())
    # Vertical
    axes["right"].set_yscale(yscale)
    ph.rm_axis_ticklabels(axes["right"].yaxis)
    axes["right"].set_ylim(axes["main"].get_ylim())
    # Label
    axes["label"].axis("off")
    return updated_dict({"fig": fig, "axes": axes}, info)


def plot_dist(
    values1,
    values2,
    bins1_dist,
    bins2,
    values_aux=None,
    bins_aux=5,
    log_vals=False,
    log_aux=False,
    transpose=False,
    shape="steps",
    plt_kwargs=None,
    line_kwargs_list=None,
    fig_kwargs=None,
    legend_kwargs=None,
    panel_label_prefix="",
    add_panel_label=True,
    panel_label_format=lambda v: v,
    add_line_label=True,
    line_label_format=lambda v: v,
):
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
    plt_kwargs: dict, None
        Additional arguments for pylab.plot
    line_kwargs_list: list, None
        List of additional arguments for plotting each line (using pylab.plot).
        Must have same size as len(bins_aux)-1
    fig_kwargs: dict, None
        Additional arguments for plt.subplots
    legend_kwargs: dict, None
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
    # pylint: disable=too-many-locals
    if transpose and (values_aux is None):
        raise ValueError("transpose=true can only be used with values_aux!=None")
    edges1_dist = autobins(values1, bins1_dist, log=log_vals)
    edges2 = autobins(values2, bins2)
    edges_aux = None if values_aux is None else autobins(values_aux, bins_aux, log_aux)
    masks2 = binmasks(values2, edges2)
    masks_aux = (
        [np.ones(len(values1), dtype=bool)]
        if values_aux is None
        else binmasks(values_aux, edges_aux)
    )
    steps1 = (
        (np.log10(edges1_dist[1:]) - np.log10(edges1_dist[:-1]))
        if log_vals
        else edges1_dist[1:] - edges1_dist[:-1]
    )
    # Use quantities relative to panel and lines:
    panel_masks, line_masks = (masks_aux, masks2) if transpose else (masks2, masks_aux)
    panel_edges, line_edges = (edges_aux, edges2) if transpose else (edges2, edges_aux)
    ncol = int(np.ceil(np.sqrt(panel_edges[:-1].size)))
    nrow = int(np.ceil(panel_edges[:-1].size / float(ncol)))
    fig, axes = plt.subplots(
        nrow, ncol, **updated_dict({"sharex": True, "figsize": (8, 6)}, fig_kwargs)
    )
    info = {"fig": fig, "axes": axes}
    line_kwargs_list = none_val(
        line_kwargs_list,
        (
            [{}]
            if values_aux is None
            else [
                {"label": ph.get_bin_label(vb, vt, line_label_format)}
                for vb, vt in zip(line_edges, line_edges[1:])
            ]
        ),
    )
    for ax, maskp in zip(info["axes"].flatten(), panel_masks):
        ph.add_grid(ax)
        for maskl, p_kwargs in zip(line_masks, line_kwargs_list):
            hist = np.histogram(values1[maskp * maskl], bins=edges1_dist)[0]
            norm = (hist * steps1).sum() if hist.sum() > 0 else 1
            ph.plot_hist_line(
                hist / norm, edges1_dist, ax=ax, shape=shape, **updated_dict(plt_kwargs, p_kwargs)
            )
        ax.set_xscale("log" if log_vals else "linear")
        ax.set_yticklabels([])
    for ax in info["axes"].flatten()[len(panel_edges) - 1 :]:
        ax.axis("off")
    if add_panel_label:
        ph.add_panel_bin_label(
            info["axes"],
            panel_edges[:-1],
            panel_edges[1:],
            format_func=panel_label_format,
            prefix=panel_label_prefix,
        )
    if values_aux is not None and add_line_label:
        info["axes"].flatten()[0].legend(**updated_dict(legend_kwargs))
    return info


# pylint: disable=unused-argument
def plot_density_dist(
    values1,
    values2,
    bins1=30,
    bins2=30,
    ax_rotation=0,
    rotation_resolution=30,
    xscale="linear",
    yscale="linear",
    err1=None,
    err2=None,
    plt_kwargs=None,
    add_cb=True,
    cb_kwargs=None,
    err_kwargs=None,
    fig_kwargs=None,
    fig_pos=(0.1, 0.1, 0.95, 0.95),
    fig_frac=(0.8, 0.01, 0.02),
    vline_kwargs=None,
    **kwargs,
):
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
    plt_kwargs: dict, None
        Additional arguments for pylab.scatter
    add_cb: bool
        Plot colorbar
    cb_kwargs: dict, None
        Colorbar arguments
    err_kwargs: dict, None
        Additional arguments for pylab.errorbar
    fig_kwargs: dict, None
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
    vline_kwargs: dict, None
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
    # pylint: disable=too-many-locals
    info = {"fig": plt.figure(**updated_dict({"figsize": (8, 6)}, fig_kwargs)), "axes": {}}
    left, bottom, right, top = fig_pos
    fmain, fgap, fcb = fig_frac
    fcb = fcb if add_cb else 0
    xmain, xgap = (right - left) * np.array([fmain, fgap])
    ymain, ygap, ypanel, ycb = (top - bottom) * np.array(
        [fmain, fgap, 1 - fmain - fgap - fcb, fcb - fgap]
    )
    info["axes"]["main"] = info["fig"].add_axes([left, bottom, xmain, ymain])
    info["axes"]["colorbar"] = (
        info["fig"].add_axes([left + xmain + xgap, bottom, ycb, ymain]) if add_cb else None
    )
    # Main plot
    plot_density(
        values1,
        values2,
        **subdict(
            locals(),
            [
                "bins1",
                "bins2",
                "ax_rotation",
                "rotation_resolution",
                "xscale",
                "yscale",
                "err1",
                "err2",
                "plt_kwargs",
                "add_cb",
                "err_kwargs",
            ],
        ),
        ax=info["axes"]["main"],
        cb_kwargs=updated_dict(
            {"cax": info["axes"]["colorbar"], "orientation": "vertical"}, cb_kwargs
        ),
    )
    if add_cb:
        info["axes"]["colorbar"].xaxis.tick_top()
        info["axes"]["colorbar"].xaxis.set_label_position("top")
    info["axes"]["main"].set_xscale(xscale)
    info["axes"]["main"].set_yscale(yscale)
    # Add v lines
    info["axes"]["main"].xaxis.grid(False, which="both")
    fit_bins1 = autobins(values1, kwargs.get("fit_bins1", 10), xscale == "log")
    for value in fit_bins1:
        info["axes"]["main"].axvline(value, **updated_dict({"lw": 0.5, "color": "0"}, vline_kwargs))
    # Dist plot
    fit_bins2 = autobins(values2, kwargs.get("fit_bins2", 30), yscale == "log")
    masks1 = binmasks(values1, fit_bins1)
    xlims = info["axes"]["main"].get_xlim()
    if xscale == "log":
        xlims, fit_bins1 = np.log(xlims), np.log(fit_bins1)
    xpos = [xmain * (x - xlims[0]) / (xlims[1] - xlims[0]) for x in fit_bins1]
    info["axes"]["top"] = [
        info["fig"].add_axes([left + xl, bottom + ymain + ygap, xr - xl, ypanel])  # top
        for xl, xr in zip(xpos, xpos[1:])
    ]
    fit_line_kwargs_list = kwargs.get("fit_line_kwargs_list", [{} for m in masks1])
    _plot_dist_vertical(
        info["axes"]["top"], values2, masks1, fit_line_kwargs_list, fit_bins2, kwargs, xscale
    )
    # Bindata and fit
    info.update(_add_bindata_and_powlawfit_array(info["axes"]["main"], values1, values2, **kwargs))
    if kwargs.get("add_bindata", kwargs.get("add_fit", False)) and info.get("plots", {}).get(
        "errorbar", False
    ):
        # pylint: disable=protected-access
        color = info["plots"]["errorbar"].lines[0]._color
        lines = info["plots"]["errorbar"].lines
        for axis, mid, bottom, top in zip(
            info["axes"]["top"], lines[0]._y, lines[1][0]._y, lines[1][1]._y
        ):
            xlim = axis.get_xlim()
            axis.axhline(mid, color=color)
            axis.fill_between(xlim, bottom, top, alpha=0.3, lw=0, color=color)
            axis.set_xlim(xlim)
    return info
