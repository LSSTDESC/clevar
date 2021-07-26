"""@file clevar/match_metrics/scaling/catalog_funcs.py

Main scaling functions using catalogs, wrapper of array_funcs functions
"""
import pylab as plt
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
    cat1: clevar.ClCatalog
        ClCatalog with matching information
    cat2: clevar.ClCatalog
        ClCatalog matched to
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
        Ax to add plot
    **kwargs
        Other arguments
    """
    ax.set_xlabel(kwargs['xlabel'])
    ax.set_ylabel(kwargs['ylabel'])
    ax.set_xscale(kwargs['xscale'])
    ax.set_yscale(kwargs['yscale'])
def plot(cat1, cat2, matching_type, col, **kwargs):
    """
    Scatter plot with errorbars and color based on input

    Parameters
    ----------
    cat1: clevar.ClCatalog
        ClCatalog with matching information
    cat2: clevar.ClCatalog
        ClCatalog matched to
    matching_type: str
        Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
    col: str
        Name of column to be plotted
    add_err: bool
        Add errorbars
    mask1: array, None
        Mask for clusters 1 properties, must have size=cat1.size
    mask2: array, None
        Mask for clusters 2 properties, must have size=cat2.size

    Fit Parameters
    --------------
    add_bindata: bool
        Plot binned data used for fit (default=False).
    add_fit: bool
        Fit and plot binned dat (default=False).
    add_fit_err: bool
        Use error of component 2 in fit (default=True).
    fit_statistics: str
        Statistics to be used in fit (default=mean). Options are:
            `individual` - Use each point
            `mode` - Use mode of component 2 distribution in each comp 1 bin, requires fit_bins2.
            `mean` - Use mean of component 2 distribution in each comp 1 bin, requires fit_bins2.
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
        Names of fitted components in fit line label, default=(xlabel, ylabel).

    Other parameters
    ----------------
    ax: matplotlib.axes
        Ax to add plot
    plt_kwargs: dict
        Additional arguments for pylab.scatter
    err_kwargs: dict
        Additional arguments for pylab.errorbar
    xlabel: str
        Label of x axis (default=cat1.labels[col]).
    ylabel: str
        Label of y axis (default=cat2.labels[col]).
    xscale: str
        Scale xaxis.
    yscale: str
        Scale yaxis.

    Returns
    -------
    ax: matplotlib.axes
        Axis of the plot
    """
    cl_kwargs, f_kwargs, mt1, mt2 = _prep_kwargs(cat1, cat2, matching_type, col, kwargs)
    conf = array_funcs.plot(**f_kwargs)
    _fmt_plot(conf['ax'], **cl_kwargs)
    return conf
def plot_color(cat1, cat2, matching_type, col, col_color,
               color1=True, color_log=False, **kwargs):
    """
    Scatter plot with errorbars and color based on input

    Parameters
    ----------
    cat1: clevar.ClCatalog
        ClCatalog with matching information
    cat2: clevar.ClCatalog
        ClCatalog matched to
    matching_type: str
        Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
    col: str
        Name of column to be plotted
    col_color: str
        Name of column for color
    color1: bool
        Use catalog 1 for color. If false uses catalog 2
    color_log: bool
        Use log of col_color
    add_err: bool
        Add errorbars
    mask1: array, None
        Mask for clusters 1 properties, must have size=cat1.size
    mask2: array, None
        Mask for clusters 2 properties, must have size=cat2.size

    Fit Parameters
    --------------
    add_bindata: bool
        Plot binned data used for fit (default=False).
    add_fit: bool
        Fit and plot binned dat (default=False).
    add_fit_err: bool
        Use error of component 2 in fit (default=True).
    fit_statistics: str
        Statistics to be used in fit (default=mean). Options are:
            `individual` - Use each point
            `mode` - Use mode of component 2 distribution in each comp 1 bin, requires fit_bins2.
            `mean` - Use mean of component 2 distribution in each comp 1 bin, requires fit_bins2.
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
        Names of fitted components in fit line label, default=(xlabel, ylabel).

    Other parameters
    ----------------
    ax: matplotlib.axes
        Ax to add plot
    plt_kwargs: dict
        Additional arguments for pylab.scatter
    add_cb: bool
        Plot colorbar
    cb_kwargs: dict
        Colorbar arguments
    err_kwargs: dict
        Additional arguments for pylab.errorbar
    xlabel: str
        Label of x axis (default=cat1.labels[col]).
    ylabel: str
        Label of y axis (default=cat2.labels[col]).
    xscale: str
        Scale xaxis.
    yscale: str
        Scale yaxis.

    Returns
    -------
    ax: matplotlib.axes
        Axis of the plot
    matplotlib.colorbar.Colorbar (optional)
        Colorbar of the recovey rates. Only returned if add_cb=True.
    """
    cl_kwargs, f_kwargs, mt1, mt2 = _prep_kwargs(cat1, cat2, matching_type, col, kwargs)
    f_kwargs['values_color'] = mt1[col_color] if color1 else mt2[col_color]
    f_kwargs['values_color'] = np.log10(f_kwargs['values_color']
                                        ) if color_log else f_kwargs['values_color']
    conf = array_funcs.plot_color(**f_kwargs)
    _fmt_plot(conf['ax'], **cl_kwargs)
    return conf
def plot_density(cat1, cat2, matching_type, col, **kwargs):
    """
    Scatter plot with errorbars and color based on point density

    Parameters
    ----------
    cat1: clevar.ClCatalog
        ClCatalog with matching information
    cat2: clevar.ClCatalog
        ClCatalog matched to
    matching_type: str
        Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
    col: str
        Name of column to be plotted
    bins1: array, int
        Bins of component 1 for density
    bins2: array, int
        Bins of component 2 for density
    add_err: bool
        Add errorbars
    mask1: array, None
        Mask for clusters 1 properties, must have size=cat1.size
    mask2: array, None
        Mask for clusters 2 properties, must have size=cat2.size

    Fit Parameters
    --------------
    add_bindata: bool
        Plot binned data used for fit (default=False).
    add_fit: bool
        Fit and plot binned dat (default=False).
    add_fit_err: bool
        Use error of component 2 in fit (default=True).
    fit_statistics: str
        Statistics to be used in fit (default=mean). Options are:
            `individual` - Use each point
            `mode` - Use mode of component 2 distribution in each comp 1 bin, requires fit_bins2.
            `mean` - Use mean of component 2 distribution in each comp 1 bin, requires fit_bins2.
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
        Names of fitted components in fit line label, default=(xlabel, ylabel).

    Other parameters
    ----------------
    ax: matplotlib.axes
        Ax to add plot
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
    xlabel: str
        Label of x axis (default=cat1.labels[col]).
    ylabel: str
        Label of y axis (default=cat2.labels[col]).
    xscale: str
        Scale xaxis.
    yscale: str
        Scale yaxis.

    Returns
    -------
    ax: matplotlib.axes
        Axis of the plot
    matplotlib.colorbar.Colorbar (optional)
        Colorbar of the recovey rates. Only returned if add_cb=True.
    """
    cl_kwargs, f_kwargs, mt1, mt2 = _prep_kwargs(cat1, cat2, matching_type, col, kwargs)
    f_kwargs['xscale'] = kwargs.get('xscale', 'linear')
    f_kwargs['yscale'] = kwargs.get('yscale', 'linear')
    conf = array_funcs.plot_density(**f_kwargs)
    _fmt_plot(conf['ax'], **cl_kwargs)
    return conf
def _get_panel_args(cat1, cat2, matching_type, col,
    col_panel, bins_panel, panel_cat1=True, log_panel=False,
    **kwargs):
    """
    Prepare args for panel

    Parameters
    ----------
    panel_plot_function: function
        Plot function
    cat1: clevar.ClCatalog
        ClCatalog with matching information
    cat2: clevar.ClCatalog
        ClCatalog matched to
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
def plot_panel(cat1, cat2, matching_type, col,
    col_panel, bins_panel, panel_cat1=True, log_panel=False,
    **kwargs):
    """
    Scatter plot with errorbars and color based on input with panels

    Parameters
    ----------
    cat1: clevar.ClCatalog
        ClCatalog with matching information
    cat2: clevar.ClCatalog
        ClCatalog matched to
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
    mask1: array, None
        Mask for clusters 1 properties, must have size=cat1.size
    mask2: array, None
        Mask for clusters 2 properties, must have size=cat2.size

    Fit Parameters
    --------------
    add_bindata: bool
        Plot binned data used for fit (default=False).
    add_fit: bool
        Fit and plot binned dat (default=False).
    add_fit_err: bool
        Use error of component 2 in fit (default=True).
    fit_statistics: str
        Statistics to be used in fit (default=mean). Options are:
            `individual` - Use each point
            `mode` - Use mode of component 2 distribution in each comp 1 bin, requires fit_bins2.
            `mean` - Use mean of component 2 distribution in each comp 1 bin, requires fit_bins2.
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
        Names of fitted components in fit line label, default=(xlabel, ylabel).

    Other parameters
    ----------------
    plt_kwargs: dict
        Additional arguments for pylab.scatter
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
    xlabel: str
        Label of x axis (default=cat1.labels[col]).
    ylabel: str
        Label of y axis (default=cat2.labels[col]).
    xscale: str
        Scale xaxis.
    yscale: str
        Scale yaxis.

    Returns
    -------
    fig: matplotlib.figure.Figure
        `matplotlib.figure.Figure` object
    ax: matplotlib.axes
        Axes with the panels
    """
    cl_kwargs, f_kwargs, mt1, mt2 = _get_panel_args(cat1, cat2, matching_type, col,
        col_panel, bins_panel, panel_cat1=panel_cat1, log_panel=log_panel, **kwargs)
    conf = array_funcs.plot_panel(**f_kwargs)
    ph.nice_panel(conf['axes'], **cl_kwargs)
    return conf
def plot_color_panel(cat1, cat2, matching_type, col, col_color,
    col_panel, bins_panel, panel_cat1=True, color1=True, log_panel=False, **kwargs):
    """
    Scatter plot with errorbars and color based on input with panels

    Parameters
    ----------
    pltfunc: function
        array_funcs function
    cat1: clevar.ClCatalog
        ClCatalog with matching information
    cat2: clevar.ClCatalog
        ClCatalog matched to
    matching_type: str
        Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
    col: str
        Name of column to be plotted
    col_color: str
        Name of column for color
    col_panel: str
        Name of column to make panels
    bins_panel: array, int
        Bins to make panels
    panel_cat1: bool
        Used catalog 1 for col_panel. If false uses catalog 2
    color1: bool
        Use catalog 1 for color. If false uses catalog 2
    log_panel: bool
        Scale of the panel values
    add_err: bool
        Add errorbars
    mask1: array, None
        Mask for clusters 1 properties, must have size=cat1.size
    mask2: array, None
        Mask for clusters 2 properties, must have size=cat2.size

    Fit Parameters
    --------------
    add_bindata: bool
        Plot binned data used for fit (default=False).
    add_fit: bool
        Fit and plot binned dat (default=False).
    add_fit_err: bool
        Use error of component 2 in fit (default=True).
    fit_statistics: str
        Statistics to be used in fit (default=mean). Options are:
            `individual` - Use each point
            `mode` - Use mode of component 2 distribution in each comp 1 bin, requires fit_bins2.
            `mean` - Use mean of component 2 distribution in each comp 1 bin, requires fit_bins2.
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
        Names of fitted components in fit line label, default=(xlabel, ylabel).

    Other parameters
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
    xlabel: str
        Label of x axis (default=cat1.labels[col]).
    ylabel: str
        Label of y axis (default=cat2.labels[col]).
    xscale: str
        Scale xaxis.
    yscale: str
        Scale yaxis.

    Returns
    -------
    fig: matplotlib.figure.Figure
        `matplotlib.figure.Figure` object
    ax: matplotlib.axes
        Axis of the plot
    """
    cl_kwargs, f_kwargs, mt1, mt2 = _get_panel_args(cat1, cat2, matching_type, col,
        col_panel, bins_panel, panel_cat1=panel_cat1, log_panel=log_panel, **kwargs)
    f_kwargs['values_color'] = mt1[col_color] if color1 else mt2[col_color]
    conf = array_funcs.plot_color_panel(**f_kwargs)
    ph.nice_panel(conf['axes'], **cl_kwargs)
    return conf
def plot_density_panel(cat1, cat2, matching_type, col,
    col_panel, bins_panel, panel_cat1=True, log_panel=False, **kwargs):
    """
    Scatter plot with errorbars and color based on point density with panels

    Parameters
    ----------
    cat1: clevar.ClCatalog
        ClCatalog with matching information
    cat2: clevar.ClCatalog
        ClCatalog matched to
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
    bins1: array, int
        Bins of component 1 for density
    bins2: array, int
        Bins of component 2 for density
    log_panel: bool
        Scale of the panel values
    add_err: bool
        Add errorbars
    mask1: array, None
        Mask for clusters 1 properties, must have size=cat1.size
    mask2: array, None
        Mask for clusters 2 properties, must have size=cat2.size

    Fit Parameters
    --------------
    add_bindata: bool
        Plot binned data used for fit (default=False).
    add_fit: bool
        Fit and plot binned dat (default=False).
    add_fit_err: bool
        Use error of component 2 in fit (default=True).
    fit_statistics: str
        Statistics to be used in fit (default=mean). Options are:
            `individual` - Use each point
            `mode` - Use mode of component 2 distribution in each comp 1 bin, requires fit_bins2.
            `mean` - Use mean of component 2 distribution in each comp 1 bin, requires fit_bins2.
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
        Names of fitted components in fit line label, default=(xlabel, ylabel).

    Other parameters
    ----------------
    ax: matplotlib.axes
        Ax to add plot
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
    xlabel: str
        Label of x axis (default=cat1.labels[col]).
    ylabel: str
        Label of y axis (default=cat2.labels[col]).
    xscale: str
        Scale xaxis.
    yscale: str
        Scale yaxis.

    Returns
    -------
    fig: matplotlib.figure.Figure
        `matplotlib.figure.Figure` object
    ax: matplotlib.axes
        Axis of the plot
    """
    cl_kwargs, f_kwargs, mt1, mt2 = _get_panel_args(cat1, cat2, matching_type, col,
        col_panel, bins_panel, panel_cat1=panel_cat1, log_panel=log_panel, **kwargs)
    f_kwargs['xscale'] = kwargs.get('xscale', 'linear')
    f_kwargs['yscale'] = kwargs.get('yscale', 'linear')
    conf = array_funcs.plot_density_panel(**f_kwargs)
    ph.nice_panel(conf['axes'], **cl_kwargs)
    return conf
def plot_metrics(cat1, cat2, matching_type, col, bins1=30, bins2=30, **kwargs):
    """
    Plot metrics.

    Parameters
    ----------
    cat1: clevar.ClCatalog
        ClCatalog with matching information
    cat2: clevar.ClCatalog
        ClCatalog matched to
    matching_type: str
        Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
    col: str
        Name of column to be plotted
    bins1: array, int
        Bins for catalog 1
    bins2: array, int
        Bins for catalog 2
    mode: str
        Mode to run. Options are:
        simple - used simple difference
        redshift - metrics for (values2-values1)/(1+values1)
        log - metrics for log of values
    mask1: array, None
        Mask for clusters 1 properties, must have size=cat1.size
    mask2: array, None
        Mask for clusters 2 properties, must have size=cat2.size

    Other parameters
    ----------------
    metrics_kwargs: dict
        Dictionary of dictionary configs for each metric plots.
    fig_kwargs: dict
        Additional arguments for plt.subplots
    legend_kwargs: dict
        Additional arguments for plt.legend
    label1: str
        Label of component from catalog 1.
    label2: str
        Label of component from catalog 2.
    scale1: str
        Scale of component from catalog 1.
    scale2: str
        Scale of component from catalog 2.

    Returns
    -------
    fig: matplotlib.figure.Figure
        `matplotlib.figure.Figure` object
    ax: matplotlib.axes
        Axis of the plot
    """
    cl_kwargs, f_kwargs, mt1, mt2 = _prep_kwargs(cat1, cat2, matching_type, col, kwargs)
    f_kwargs.pop('fit_err2', None)
    f_kwargs.pop('err1', None)
    f_kwargs.pop('err2', None)
    f_kwargs['bins1'] = bins1
    f_kwargs['bins2'] = bins2
    conf = array_funcs.plot_metrics(**f_kwargs)
    conf['axes'][0].set_ylabel(cat1.name)
    conf['axes'][1].set_ylabel(cat2.name)
    conf['axes'][0].set_xlabel(kwargs.get('label1', cl_kwargs['xlabel']))
    conf['axes'][1].set_xlabel(kwargs.get('label2', cl_kwargs['ylabel']))
    conf['axes'][0].set_xscale(kwargs.get('scale1', cl_kwargs['xscale']))
    conf['axes'][1].set_xscale(kwargs.get('scale2', cl_kwargs['yscale']))
    return conf
def plot_density_metrics(cat1, cat2, matching_type, col, bins1=30, bins2=30, **kwargs):
    """
    Scatter plot with errorbars and color based on point density with scatter and bias panels

    Parameters
    ----------
    cat1: clevar.ClCatalog
        ClCatalog with matching information
    cat2: clevar.ClCatalog
        ClCatalog matched to
    matching_type: str
        Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
    col: str
        Name of column to be plotted
    bins1: array, int
        Bins for component 1
    bins2: array, int
        Bins for component 2
    metrics_mode: str
        Mode to run. Options are:
        simple - used simple difference
        redshift - metrics for (values2-values1)/(1+values1)
        log - metrics for log of values
    add_err: bool
        Add errorbars
    mask1: array, None
        Mask for clusters 1 properties, must have size=cat1.size
    mask2: array, None
        Mask for clusters 2 properties, must have size=cat2.size

    Fit Parameters
    --------------
    add_bindata: bool
        Plot binned data used for fit (default=False).
    add_fit: bool
        Fit and plot binned dat (default=False).
    add_fit_err: bool
        Use error of component 2 in fit (default=True).
    fit_statistics: str
        Statistics to be used in fit (default=mean). Options are:
            `individual` - Use each point
            `mode` - Use mode of component 2 distribution in each comp 1 bin, requires fit_bins2.
            `mean` - Use mean of component 2 distribution in each comp 1 bin, requires fit_bins2.
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
        Names of fitted components in fit line label, default=(xlabel, ylabel).

    Other parameters
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
    xscale: str
        Scale xaxis.
    yscale: str
        Scale yaxis.
    fig_pos: tuple
        List with edges for the figure. Must be in format (left, bottom, right, top)
    fig_frac: tuple
        Sizes of each panel in the figure. Must be in the format (main_panel, gap, colorbar)
        and have values: [0, 1]. Colorbar is only used with add_cb key.

    Returns
    -------
    fig: matplotlib.figure.Figure
        `matplotlib.figure.Figure` object
    list
        Axes with the panels (main, right, top, label)
    """
    cl_kwargs, f_kwargs, mt1, mt2 = _prep_kwargs(cat1, cat2, matching_type, col, kwargs)
    f_kwargs['xscale'] = kwargs.get('scale1', cl_kwargs['xscale'])
    f_kwargs['yscale'] = kwargs.get('scale2', cl_kwargs['yscale'])
    f_kwargs['bins1'] = bins1
    f_kwargs['bins2'] = bins2
    conf = array_funcs.plot_density_metrics(**f_kwargs)
    xlabel = kwargs.get('label1', cl_kwargs['xlabel'])
    ylabel = kwargs.get('label2', cl_kwargs['ylabel'])
    conf['axes']['main'].set_xlabel(xlabel)
    conf['axes']['main'].set_ylabel(ylabel)
    return conf

def plot_dist(cat1, cat2, matching_type, col, bins1=30, bins2=5, col_aux=None, bins_aux=5,
              log_vals=False, log_aux=False, transpose=False, **kwargs):
    """
    Plot distribution of a cat1 column, binned by the cat2 column in panels,
    with option for a second cat2 column in lines.

    Parameters
    ----------
    cat1: clevar.ClCatalog
        ClCatalog with matching information
    cat2: clevar.ClCatalog
        ClCatalog matched to
    matching_type: str
        Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
    col: str
        Name of column to be plotted
    bins1: array, int
        Bins for distribution of the cat1 column
    bins2: array, int
        Bins for cat2 column
    col_aux: array
        Auxiliary colum of cat2 (to bin data in lines)
    bins_aux: array, int
        Bins for component aux
    log_vals: bool
        Log scale for values (and int bins)
    log_aux: bool
        Log scale for aux values (and int bins)
    transpose: bool
        Invert lines and panels

    Other parameters
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
    fig: matplotlib.figure.Figure
        `matplotlib.figure.Figure` object
    ax: matplotlib.axes
        Axes with the panels
    """
    cl_kwargs, f_kwargs, mt1, mt2 = _prep_kwargs(cat1, cat2, matching_type, col, kwargs)
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
    conf = array_funcs.plot_dist(**f_kwargs)
    xlabel = kwargs.get('label', f'${cat1.labels[col]}$')
    for ax in (conf['axes'][-1,:] if len(conf['axes'].shape)>1 else conf['axes']):
        ax.set_xlabel(xlabel)
    return conf
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
        Bins for distribution of the column
    bins2: array, int
        Bins for panels
    col_aux: array
        Auxiliary colum (to bin data in lines)
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

    Other parameters
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
    fig: matplotlib.figure.Figure
        `matplotlib.figure.Figure` object
    ax: matplotlib.axes
        Axes with the panels
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
    conf = array_funcs.plot_dist(**f_kwargs)
    xlabel = kwargs.get('label', f'${cat.labels[col]}$')
    for ax in (conf['axes'][-1,:] if len(conf['axes'].shape)>1 else conf['axes']):
        ax.set_xlabel(xlabel)
    return conf
def plot_density_dist(cat1, cat2, matching_type, col, **kwargs):
    """
    Scatter plot with errorbars and color based on point density with distribution panels.

    Parameters
    ----------
    cat1: clevar.ClCatalog
        ClCatalog with matching information
    cat2: clevar.ClCatalog
        ClCatalog matched to
    matching_type: str
        Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
    col: str
        Name of column to be plotted
    bins1: array, int
        Bins for component 1 (for density colors).
    bins2: array, int
        Bins for component 2 (for density colors).
    add_err: bool
        Add errorbars
    mask1: array, None
        Mask for clusters 1 properties, must have size=cat1.size
    mask2: array, None
        Mask for clusters 2 properties, must have size=cat2.size

    Fit Parameters
    --------------
    add_bindata: bool
        Plot binned data used for fit (default=False).
    add_fit: bool
        Fit and plot binned dat (default=False).
    add_fit_err: bool
        Use error of component 2 in fit (default=True).
    fit_statistics: str
        Statistics to be used in fit (default=mean). Options are:
            `individual` - Use each point
            `mode` - Use mode of component 2 distribution in each comp 1 bin, requires fit_bins2.
            `mean` - Use mean of component 2 distribution in each comp 1 bin, requires fit_bins2.
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
        Names of fitted components in fit line label, default=(xlabel, ylabel).
    vline_kwargs: dict
        Arguments for vlines marking bins in main plot, used in plt.axvline.

    Other parameters
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
    xscale: str
        Scale xaxis.
    yscale: str
        Scale yaxis.
    fig_pos: tuple
        List with edges for the figure. Must be in format (left, bottom, right, top)
    fig_frac: tuple
        Sizes of each panel in the figure. Must be in the format (main_panel, gap, colorbar)
        and have values: [0, 1]. Colorbar is only used with add_cb key.

    Returns
    -------
    fig: matplotlib.figure.Figure
        `matplotlib.figure.Figure` object
    list
        Axes with the panels (main, right, top, label)
    """
    cl_kwargs, f_kwargs, mt1, mt2 = _prep_kwargs(cat1, cat2, matching_type, col, kwargs)
    f_kwargs['xscale'] = kwargs.get('scale1', cl_kwargs['xscale'])
    f_kwargs['yscale'] = kwargs.get('scale2', cl_kwargs['yscale'])
    conf = array_funcs.plot_density_dist(**f_kwargs)
    conf['axes']['main'].set_xlabel(kwargs.get('label1', cl_kwargs['xlabel']))
    conf['axes']['main'].set_ylabel(kwargs.get('label2', cl_kwargs['ylabel']))
    conf['axes']['top'][-1].set_ylabel(kwargs.get('label2', cl_kwargs['ylabel']))
    return conf
