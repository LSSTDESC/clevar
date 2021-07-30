"""@file clevar/match_metrics/scaling/catalog_funcs.py

Main scaling functions using catalogs, wrapper of array_funcs functions
"""
import numpy as np

from ...utils import autobins
from ...match import get_matched_pairs
from .. import plot_helper as ph
from . import array_funcs

_local_args = ('xlabel', 'ylabel', 'xscale', 'yscale', 'add_err', 'add_fit_err',
              'label1', 'label2', 'scale1', 'scale2', 'mask1', 'mask2')


def _prep_kwargs(cat1, cat2, matching_type, col, kwargs={}):
    """
    Prepare kwargs into args for this class and args for function

    Parameters
    ----------
    cat1, cat2: clevar.ClCatalog
        ClCatalogs with matching information.
    matching_type: str
        Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
    col: str
        Name of column to be plotted
    kwargs: dict
        Input arguments

    Returns
    -------
    class_kwargs: dict
        Arguments for class
    func_kwargs: dict
        Arguments for function
    mp: clevar.match.MatchedPairs
        Matched catalogs
    """
    func_kwargs = {k:v for k, v in kwargs.items() if k not in _local_args}
    mt1, mt2 = get_matched_pairs(cat1, cat2, matching_type,
                      mask1=kwargs.get('mask1', None),
                      mask2=kwargs.get('mask2', None))
    func_kwargs['values1'] = mt1[col]
    func_kwargs['values2'] = mt2[col]
    func_kwargs['err1'] = mt1.get(f'{col}_err') if kwargs.get('add_err', True) else None
    func_kwargs['err2'] = mt2.get(f'{col}_err') if kwargs.get('add_err', True) else None
    func_kwargs['fit_err2'] = mt2.get(f'{col}_err') if kwargs.get('add_fit_err', True) else None
    class_kwargs = {
        'xlabel': kwargs.get('xlabel', f'${cat1.labels[col]}$'),
        'ylabel': kwargs.get('ylabel', f'${cat2.labels[col]}$'),
        'xscale': kwargs.get('xscale', 'linear'),
        'yscale': kwargs.get('yscale', 'linear'),
    }
    if kwargs.get('add_fit', False):
        xlabel = kwargs.get('label1', class_kwargs['xlabel'])
        ylabel = kwargs.get('label2', class_kwargs['ylabel'])
        func_kwargs['fit_label_components'] = kwargs.get('fit_label_components', (xlabel, ylabel))
    return class_kwargs, func_kwargs, mt1, mt2


def _fmt_plot(ax, **kwargs):
    """
    Format plot (scale and label of ax)

    Parameters
    ----------
    ax: matplotlib.axes
        Ax to add plot.
    **kwargs
        Other arguments
    """
    ax.set_xlabel(kwargs['xlabel'])
    ax.set_ylabel(kwargs['ylabel'])
    ax.set_xscale(kwargs['xscale'])
    ax.set_yscale(kwargs['yscale'])


def plot(cat1, cat2, matching_type, col, col_color=None,
         color1=True, color_log=False, **kwargs):
    """
    Scatter plot with errorbars. Color can be based on input column.

    Parameters
    ----------
    cat1, cat2: clevar.ClCatalog
        ClCatalogs with matching information.
    matching_type: str
        Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
    col: str
        Name of column to be plotted
    col_color: str, None
        Name of column for color
    color1: bool
        Use catalog 1 for color. If false uses catalog 2
    color_log: bool
        Use log of col_color
    add_err: bool
        Add errorbars
    mask1, mask2: array, None
        Masks for clusters 1(2), must have size=cat1(2).size

    Other Parameters
    ----------------
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
    xlabel, ylabel: str
        Label of x/y axis (default=cat1.labels[col]/cat2.labels[col]).
    xscale, yscale: str
        Scale for x/y axis.
    add_bindata: bool
        Plot binned data used for fit (default=False).
    add_fit: bool
        Fit and plot binned dat (default=False).
    **fit_kwargs:
        Other fit arguments (see `fit_*` paramters in `scaling.catalog_funcs.plot` for more info).

    Returns
    -------
    info: dict
        Information of data in the plots, it contains the sections:

            * `ax`: ax used in the plot.
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
    cl_kwargs, f_kwargs, mt1, mt2 = _prep_kwargs(cat1, cat2, matching_type, col, kwargs)
    if col_color is not None:
        f_kwargs['values_color'] = mt1[col_color] if color1 else mt2[col_color]
        f_kwargs['values_color'] = np.log10(f_kwargs['values_color']
                                            ) if color_log else f_kwargs['values_color']
    info = array_funcs.plot(**f_kwargs)
    _fmt_plot(info['ax'], **cl_kwargs)
    return info


def plot_density(cat1, cat2, matching_type, col, **kwargs):
    """
    Scatter plot with errorbars and color based on point density

    Parameters
    ----------
    cat1, cat2: clevar.ClCatalog
        ClCatalogs with matching information.
    matching_type: str
        Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
    col: str
        Name of column to be plotted
    bins1, bins2: array, None
        Bins for component x and y (for density colors).
    add_err: bool
        Add errorbars
    mask1, mask2: array, None
        Masks for clusters 1(2), must have size=cat1(2).size

    Other Parameters
    ----------------
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
    ax_rotation: float
        Angle (in degrees) for rotation of axis of binning. Overwrites use of (bins1, bins2)
    rotation_resolution: int
        Number of bins to be used when ax_rotation!=0.
    xlabel, ylabel: str
        Label of x/y axis (default=cat1.labels[col]/cat2.labels[col]).
    xscale, yscale: str
        Scale for x/y axis.
    add_bindata: bool
        Plot binned data used for fit (default=False).
    add_fit: bool
        Fit and plot binned dat (default=False).
    **fit_kwargs:
        Other fit arguments (see `fit_*` paramters in `scaling.catalog_funcs.plot` for more info).

    Returns
    -------
    info: dict
        Information of data in the plots, it contains the sections:

            * `ax`: ax used in the plot.
            * `ax_cb` (optional): ax of colorbar
            * `fit` (optional): fitting output dictionary \
            (see `scaling.catalog_funcs.plot` for more info).
            * `plots` (optional): fit and binning plots \
            (see `scaling.catalog_funcs.plot` for more info).
    """
    cl_kwargs, f_kwargs = _prep_kwargs(cat1, cat2, matching_type, col, kwargs)[:2]
    f_kwargs['xscale'] = kwargs.get('xscale', 'linear')
    f_kwargs['yscale'] = kwargs.get('yscale', 'linear')
    info = array_funcs.plot_density(**f_kwargs)
    _fmt_plot(info['ax'], **cl_kwargs)
    return info


def _get_panel_args(cat1, cat2, matching_type, col,
    col_panel, bins_panel, panel_cat1=True, log_panel=False,
    **kwargs):
    """
    Prepare args for panel

    Parameters
    ----------
    panel_plot_function: function
        Plot function
    cat1, cat2: clevar.ClCatalog
        ClCatalogs with matching information.
    matching_type: str
        Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
    col: str
        Name of column to be plotted
    col_panel: str
        Name of column to make panels
    bins_panel: array, int
        Bins to make panels
    panel_cat1: bool
        Used catalog 1 for col_panel. If false uses catalog 2
    log_panel: bool
        Scale of the panel values
    add_err: bool
        Add errorbars
    **kwargs
        Other arguments

    Returns
    -------
    class_kwargs: dict
        Arguments for class
    func_kwargs: dict
        Arguments for function
    mp: clevar.match.MatchedPairs
        Matched catalogs
    """
    cl_kwargs, f_kwargs, mt1, mt2 = _prep_kwargs(cat1, cat2, matching_type, col, kwargs)
    f_kwargs['values_panel'] = mt1[col_panel] if panel_cat1 else mt2[col_panel]
    f_kwargs['bins_panel'] = autobins(f_kwargs['values_panel'], bins_panel, log_panel)
    ph._set_label_format(f_kwargs, 'label_format', 'label_fmt', log_panel)
    return cl_kwargs, f_kwargs, mt1, mt2


def plot_panel(
    cat1, cat2, matching_type, col, col_panel, bins_panel, panel_cat1=True,
    col_color=None, color1=True, log_panel=False, **kwargs):
    """
    Scatter plot with errorbars and color based on input with panels

    Parameters
    ----------
    cat1, cat2: clevar.ClCatalog
        ClCatalogs with matching information.
    matching_type: str
        Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
    col: str
        Name of column to be plotted
    col_panel: str
        Name of column to make panels
    bins_panel: array, int
        Bins to make panels
    panel_cat1: bool
        Used catalog 1 for col_panel. If false uses catalog 2
    col_color: str, None
        Name of column for color.
    color1: bool
        Use catalog 1 for color. If false uses catalog 2
    log_panel: bool
        Scale of the panel values
    add_err: bool
        Add errorbars
    mask1, mask2: array, None
        Masks for clusters 1(2), must have size=cat1(2).size

    Other Parameters
    ----------------
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
        Function
    xlabel, ylabel: str
        Label of x/y axis (default=cat1.labels[col]/cat2.labels[col]).
    xscale, yscale: str
        Scale for x/y axis.
    add_bindata: bool
        Plot binned data used for fit (default=False).
    add_fit: bool
        Fit and plot binned dat (default=False).
    **fit_kwargs:
        Other fit arguments (see `fit_*` paramters in `scaling.catalog_funcs.plot` for more info).

    Returns
    -------
    info: dict
        Information of data in the plots, it contains the sections:

            * `fig`: `matplotlib.figure.Figure` object.
            * `axes`: `matplotlib.axes` used in the plot.
            * `ax_cb` (optional): ax of colorbar
            * `fit` (optional): fitting output dictionary \
            (see `scaling.catalog_funcs.plot` for more info).
            * `plots` (optional): fit and binning plots \
            (see `scaling.catalog_funcs.plot` for more info).
    """
    cl_kwargs, f_kwargs, mt1, mt2 = _get_panel_args(cat1, cat2, matching_type, col,
        col_panel, bins_panel, panel_cat1=panel_cat1, log_panel=log_panel, **kwargs)
    if col_color is not None:
        f_kwargs['values_color'] = mt1[col_color] if color1 else mt2[col_color]
    info = array_funcs.plot_panel(**f_kwargs)
    ph.nice_panel(info['axes'], **cl_kwargs)
    return info


def plot_density_panel(cat1, cat2, matching_type, col,
    col_panel, bins_panel, panel_cat1=True, log_panel=False, **kwargs):
    """
    Scatter plot with errorbars and color based on point density with panels

    Parameters
    ----------
    cat1, cat2: clevar.ClCatalog
        ClCatalogs with matching information.
    matching_type: str
        Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
    col: str
        Name of column to be plotted
    col_panel: str
        Name of column to make panels
    bins_panel: array, int
        Bins to make panels
    panel_cat1: bool
        Used catalog 1 for col_panel. If false uses catalog 2
    bins1, bins2: array, None
        Bins for component x and y (for density colors).
    log_panel: bool
        Scale of the panel values
    add_err: bool
        Add errorbars
    mask1, mask2: array, None
        Masks for clusters 1(2), must have size=cat1(2).size

    Other Parameters
    ----------------
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
    ax_rotation: float
        Angle (in degrees) for rotation of axis of binning. Overwrites use of (bins1, bins2)
    rotation_resolution: int
        Number of bins to be used when ax_rotation!=0.
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
    xlabel, ylabel: str
        Label of x/y axis (default=cat1.labels[col]/cat2.labels[col]).
    xscale, yscale: str
        Scale for x/y axis.
    add_bindata: bool
        Plot binned data used for fit (default=False).
    add_fit: bool
        Fit and plot binned dat (default=False).
    **fit_kwargs:
        Other fit arguments (see `fit_*` paramters in `scaling.catalog_funcs.plot` for more info).

    Returns
    -------
    info: dict
        Information of data in the plots, it contains the sections:

            * `fig`: `matplotlib.figure.Figure` object.
            * `axes`: `matplotlib.axes` used in the plot.
            * `ax_cb` (optional): ax of colorbar
            * `fit` (optional): fitting output dictionary \
            (see `scaling.catalog_funcs.plot` for more info).
            * `plots` (optional): fit and binning plots \
            (see `scaling.catalog_funcs.plot` for more info).
    """
    cl_kwargs, f_kwargs = _get_panel_args(
        cat1, cat2, matching_type, col, col_panel, bins_panel, panel_cat1=panel_cat1,
        log_panel=log_panel, **kwargs)[:2]
    f_kwargs['xscale'] = kwargs.get('xscale', 'linear')
    f_kwargs['yscale'] = kwargs.get('yscale', 'linear')
    info = array_funcs.plot_density_panel(**f_kwargs)
    ph.nice_panel(info['axes'], **cl_kwargs)
    return info


def plot_metrics(cat1, cat2, matching_type, col, bins1=30, bins2=30, **kwargs):
    """
    Plot metrics.

    Parameters
    ----------
    cat1, cat2: clevar.ClCatalog
        ClCatalogs with matching information.
    matching_type: str
        Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
    col: str
        Name of column to be plotted
    bins1, bins2: array, None
        Bins for component of catalog 1 and 2.
    mode: str
        Mode to run (default=`simple`). Options are:

            * `simple` : metrics for `values2`.
            * `log` : metrics for `log10(values2)`.
            * `diff` : metrics for `values2-values1`.
            * `diff_log` : metrics for `log10(values2)-log10(values1)`.
            * `diff_z` : metrics for `(values2-values1)/(1+values1)`.

    metrics: list
        List of mettrics to be plotted (default=[`mean`, `std`]). Possibilities are:

            * `mean` : compute the mean of values for points within each bin.
            * `std` : compute the standard deviation within each bin.
            * `median` : compute the median of values for points within each bin.
            * `count` : compute the count of points within each bin.
            * `sum` : compute the sum of values for points within each bin.
            * `min` : compute the minimum of values for points within each bin.
            * `max` : compute the maximum of values for point within each bin.
            * `p_#` : compute half the width where a percentile of data is found. Number must be
              between 0-100 (ex: `p_68`, `p_95`, `p_99`).

    mask1, mask2: array, None
        Masks for clusters 1(2), must have size=cat1(2).size

    Other Parameters
    ----------------
    metrics_kwargs: dict
        Dictionary of dictionary configs for each metric plots.
    fig_kwargs: dict
        Additional arguments for plt.subplots
    legend_kwargs: dict
        Additional arguments for plt.legend
    label1, label2: str
        Label for catalog 1/2 components.
    scale1, scale2: str
        Scale of catalog 1/2 components.

    Returns
    -------
    info: dict
        Information of data in the plots, it contains the sections:

            * `fig`: `matplotlib.figure.Figure` object.
            * `axes`: `matplotlib.axes` used in the plot.
    """
    cl_kwargs, f_kwargs = _prep_kwargs(cat1, cat2, matching_type, col, kwargs)[:2]
    f_kwargs.pop('fit_err2', None)
    f_kwargs.pop('err1', None)
    f_kwargs.pop('err2', None)
    f_kwargs['bins1'] = bins1
    f_kwargs['bins2'] = bins2
    info = array_funcs.plot_metrics(**f_kwargs)
    info['axes'][0].set_ylabel(cat1.name)
    info['axes'][1].set_ylabel(cat2.name)
    info['axes'][0].set_xlabel(kwargs.get('label1', cl_kwargs['xlabel']))
    info['axes'][1].set_xlabel(kwargs.get('label2', cl_kwargs['ylabel']))
    info['axes'][0].set_xscale(kwargs.get('scale1', cl_kwargs['xscale']))
    info['axes'][1].set_xscale(kwargs.get('scale2', cl_kwargs['yscale']))
    return info


def plot_density_metrics(cat1, cat2, matching_type, col, bins1=30, bins2=30, **kwargs):
    """
    Scatter plot with errorbars and color based on point density with scatter and bias panels

    Parameters
    ----------
    cat1, cat2: clevar.ClCatalog
        ClCatalogs with matching information.
    matching_type: str
        Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
    col: str
        Name of column to be plotted
    bins1, bins2: array, None
        Bins for component x and y.
    metrics_mode: str
        Mode to run metrics (default=`simple`). Options are:

            * `simple` : metrics for `values2`.
            * `log` : metrics for `log10(values2)`.
            * `diff` : metrics for `values2-values1`.
            * `diff_log` : metrics for `log10(values2)-log10(values1)`.
            * `diff_z` : metrics for `(values2-values1)/(1+values1)`.

    metrics: list
        List of mettrics to be plotted (default=[`mean`, `std`]). Possibilities are:

            * `mean` : compute the mean of values for points within each bin.
            * `std` : compute the standard deviation within each bin.
            * `median` : compute the median of values for points within each bin.
            * `count` : compute the count of points within each bin.
            * `sum` : compute the sum of values for points within each bin.
            * `min` : compute the minimum of values for points within each bin.
            * `max` : compute the maximum of values for point within each bin.
            * `p_#` : compute half the width where a percentile of data is found. Number must be
              between 0-100 (ex: `p_68`, `p_95`, `p_99`).

    add_err: bool
        Add errorbars
    mask1, mask2: array, None
        Masks for clusters 1(2), must have size=cat1(2).size

    Other Parameters
    ----------------
    fig_kwargs: dict
        Additional arguments for plt.subplots
    ax_rotation: float
        Angle (in degrees) for rotation of axis of binning. Overwrites use of (bins1, bins2) on main plot.
    rotation_resolution: int
        Number of bins to be used when ax_rotation!=0.
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
    xscale, yscale: str
        Scale for x/y axis.
    fig_pos: tuple
        List with edges for the figure. Must be in format (left, bottom, right, top)
    fig_frac: tuple
        Sizes of each panel in the figure. Must be in the format (main_panel, gap, colorbar)
        and have values: [0, 1]. Colorbar is only used with add_cb key.
    add_bindata: bool
        Plot binned data used for fit (default=False).
    add_fit: bool
        Fit and plot binned dat (default=False).
    **fit_kwargs:
        Other fit arguments (see `fit_*` paramters in `scaling.catalog_funcs.plot` for more info).

    Returns
    -------
    info: dict
        Information of data in the plots, it contains the sections:

            * `fig`: `matplotlib.figure.Figure` object.
            * `axes`: dictionary with each ax of the plot.
            * `metrics`: dictionary with the plots for each metric.
    """
    cl_kwargs, f_kwargs = _prep_kwargs(cat1, cat2, matching_type, col, kwargs)[:2]
    f_kwargs['xscale'] = kwargs.get('scale1', cl_kwargs['xscale'])
    f_kwargs['yscale'] = kwargs.get('scale2', cl_kwargs['yscale'])
    f_kwargs['bins1'] = bins1
    f_kwargs['bins2'] = bins2
    info = array_funcs.plot_density_metrics(**f_kwargs)
    xlabel = kwargs.get('label1', cl_kwargs['xlabel'])
    ylabel = kwargs.get('label2', cl_kwargs['ylabel'])
    info['axes']['main'].set_xlabel(xlabel)
    info['axes']['main'].set_ylabel(ylabel)
    return info


def plot_dist(cat1, cat2, matching_type, col, bins1=30, bins2=5, col_aux=None, bins_aux=5,
              log_vals=False, log_aux=False, transpose=False, **kwargs):
    """
    Plot distribution of a cat1 column, binned by the cat2 column in panels,
    with option for a second cat2 column in lines.

    Parameters
    ----------
    cat1, cat2: clevar.ClCatalog
        ClCatalogs with matching information.
    matching_type: str
        Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
    col: str
        Name of column to be plotted
    bins1: array, int
        Bins for distribution of the cat1 column.
    bins2: array, int
        Bins for cat2 column (for panels/lines).
    col_aux: array
        Auxiliary colum of cat2 (to bin data in lines/panels).
    bins_aux: array, int
        Bins for component aux
    log_vals: bool
        Log scale for values (and int bins)
    log_aux: bool
        Log scale for aux values (and int bins)
    transpose: bool
        Invert lines and panels

    Other Parameters
    ----------------
    fig_kwargs: dict
        Additional arguments for plt.subplots
    shape: str
        Shape of the lines. Can be steps or line.
    plt_kwargs: dict
        Additional arguments for pylab.plot
    line_kwargs_list: list, None
        List of additional arguments for plotting each line (using pylab.plot).
        Must have same size as len(bins_aux)-1
    legend_kwargs: dict
        Additional arguments for plt.legend
    add_panel_label: bool
        Add bin label to panel
    panel_label_format: function
        Function to format the values of the bins
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
    f_kwargs, mt2 = _prep_kwargs(cat1, cat2, matching_type, col, kwargs)[1::2]
    f_kwargs.pop('err1', None)
    f_kwargs.pop('err2', None)
    f_kwargs.pop('fit_err2', None)
    f_kwargs['values_aux'] = None if col_aux is None else mt2[col_aux]
    f_kwargs['bins1_dist'] = bins1
    f_kwargs['bins2'] = bins2
    f_kwargs['bins_aux'] = bins_aux
    f_kwargs['log_vals'] = log_vals
    f_kwargs['log_aux'] = log_aux
    f_kwargs['transpose'] = transpose
    f_kwargs['panel_label_prefix'] = f'{cat2.labels[col]}\,-\,'
    log_panel, log_line = (log_aux, log_vals) if transpose else (log_vals, log_aux)
    ph._set_label_format(f_kwargs, 'panel_label_format', 'panel_label_fmt', log_panel)
    ph._set_label_format(f_kwargs, 'line_label_format', 'line_label_fmt', log_line)
    info = array_funcs.plot_dist(**f_kwargs)
    xlabel = kwargs.get('label', f'${cat1.labels[col]}$')
    for ax in (info['axes'][-1,:] if len(info['axes'].shape)>1 else info['axes']):
        ax.set_xlabel(xlabel)
    return info


def plot_dist_self(cat, col, bins1=30, bins2=5, col_aux=None, bins_aux=5,
                   log_vals=False, log_aux=False, transpose=False, mask=None, **kwargs):
    """
    Plot distribution of a cat1 column, binned by the same column in panels,
    with option for a second column in lines. Is is useful to compare with plot_dist results.

    Parameters
    ----------
    cat: clevar.ClCatalog
        Input catalog
    col: str
        Name of column to be plotted
    bins1: array, int
        Bins for distribution of the cat1 column.
    bins2: array, int
        Bins for cat2 column (for panels/lines).
    col_aux: array
        Auxiliary colum of cat2 (to bin data in lines/panels).
    bins_aux: array, int
        Bins for component aux
    log_vals: bool
        Log scale for values (and int bins)
    log_aux: bool
        Log scale for aux values (and int bins)
    transpose: bool
        Invert lines and panels
    mask: ndarray
        Mask for catalog

    Other Parameters
    ----------------
    fig_kwargs: dict
        Additional arguments for plt.subplots
    shape: str
        Shape of the lines. Can be steps or line.
    plt_kwargs: dict
        Additional arguments for pylab.plot
    line_kwargs_list: list, None
        List of additional arguments for plotting each line (using pylab.plot).
        Must have same size as len(bins_aux)-1
    legend_kwargs: dict
        Additional arguments for plt.legend
    add_panel_label: bool
        Add bin label to panel
    panel_label_format: function
        Function to format the values of the bins
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
    f_kwargs = {k:v for k, v in kwargs.items() if k not in _local_args}
    mask = np.ones(cat.size, dtype=bool) if mask is None else mask
    f_kwargs['values1'] = cat[col][mask]
    f_kwargs['values2'] = cat[col][mask]
    f_kwargs['values_aux'] = None if col_aux is None else cat[col_aux][mask]
    f_kwargs['bins1_dist'] = bins1
    f_kwargs['bins2'] = bins2
    f_kwargs['bins_aux'] = bins_aux
    f_kwargs['log_vals'] = log_vals
    f_kwargs['log_aux'] = log_aux
    f_kwargs['transpose'] = transpose
    f_kwargs['panel_label_prefix'] = f'{cat.labels[col]}\,-\,'
    log_panel, log_line = (log_aux, log_vals) if transpose else (log_vals, log_aux)
    ph._set_label_format(f_kwargs, 'panel_label_format', 'panel_label_fmt', log_panel)
    ph._set_label_format(f_kwargs, 'line_label_format', 'line_label_fmt', log_line)
    info = array_funcs.plot_dist(**f_kwargs)
    xlabel = kwargs.get('label', f'${cat.labels[col]}$')
    for ax in (info['axes'][-1,:] if len(info['axes'].shape)>1 else info['axes']):
        ax.set_xlabel(xlabel)
    return info


def plot_density_dist(cat1, cat2, matching_type, col, **kwargs):
    """
    Scatter plot with errorbars and color based on point density with distribution panels.

    Parameters
    ----------
    cat1, cat2: clevar.ClCatalog
        ClCatalogs with matching information.
    matching_type: str
        Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
    col: str
        Name of column to be plotted
    bins1, bins2: array, None
        Bins for component x and y (for density colors).
    add_err: bool
        Add errorbars
    mask1, mask2: array, None
        Masks for clusters 1(2), must have size=cat1(2).size

    Other Parameters
    ----------------
    fig_kwargs: dict
        Additional arguments for plt.subplots
    ax_rotation: float
        Angle (in degrees) for rotation of axis of binning. Overwrites use of (bins1, bins2) on main plot.
    rotation_resolution: int
        Number of bins to be used when ax_rotation!=0.
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
    xscale, yscale: str
        Scale for x/y axis.
    fig_pos: tuple
        List with edges for the figure. Must be in format (left, bottom, right, top)
    fig_frac: tuple
        Sizes of each panel in the figure. Must be in the format (main_panel, gap, colorbar)
        and have values: [0, 1]. Colorbar is only used with add_cb key.
    add_bindata: bool
        Plot binned data used for fit (default=False).
    add_fit: bool
        Fit and plot binned dat (default=False).
    **fit_kwargs:
        Other fit arguments (see `fit_*` paramters in `scaling.catalog_funcs.plot` for more info).

    Returns
    -------
    info: dict
        Information of data in the plots, it contains the sections:

            * `fig`: `matplotlib.figure.Figure` object.
            * `axes`: dictionary with each ax of the plot.
            * `fit` (optional): fitting output dictionary \
            (see `scaling.catalog_funcs.plot` for more info).
            * `plots` (optional): fit and binning plots \
            (see `scaling.catalog_funcs.plot` for more info).
    """
    cl_kwargs, f_kwargs = _prep_kwargs(cat1, cat2, matching_type, col, kwargs)[:2]
    f_kwargs['xscale'] = kwargs.get('scale1', cl_kwargs['xscale'])
    f_kwargs['yscale'] = kwargs.get('scale2', cl_kwargs['yscale'])
    info = array_funcs.plot_density_dist(**f_kwargs)
    info['axes']['main'].set_xlabel(kwargs.get('label1', cl_kwargs['xlabel']))
    info['axes']['main'].set_ylabel(kwargs.get('label2', cl_kwargs['ylabel']))
    info['axes']['top'][-1].set_ylabel(kwargs.get('label2', cl_kwargs['ylabel']))
    return info
