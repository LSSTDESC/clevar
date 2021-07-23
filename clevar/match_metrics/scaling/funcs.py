"""@file clevar/match_metrics/scaling/funcs.py

Main scaling functions for mass and redshift plots,
wrapper of catalog_funcs functions
"""
from . import catalog_funcs

def redshift(cat1, cat2, matching_type, **kwargs):
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
        Use redshift errors of catalog 2 in fit (default=True).
    fit_statistics: str
        Statistics to be used in fit (default=mean). Options are:
            `individual` - Use each point
            `mode` - Use mode of component 2 distribution in each comp 1 bin, requires fit_bins2.
            `mean` - Use mean of component 2 distribution in each comp 1 bin, requires fit_bins2.
    fit_bins1: array, None
        Bins for redshift of catalog 1 (default=10).
    fit_bins2: array, None
        Bins for redshift of catalog 2 (default=30).
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
        Label of x axis (default=cat1.labels['z']).
    ylabel: str
        Label of y axis (default=cat2.labels['z']).

    Returns
    -------
    ax: matplotlib.axes
        Axis of the plot
    """
    return catalog_funcs.plot(cat1, cat2, matching_type, col='z', **kwargs)
def redshift_density(cat1, cat2, matching_type, **kwargs):
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
        Bins of redshift 1 for density
    bins2: array, int
        Bins of redshift 2 for density
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
        Use redshift errors of catalog 2 in fit (default=True).
    fit_statistics: str
        Statistics to be used in fit (default=mean). Options are:
            `individual` - Use each point
            `mode` - Use mode of component 2 distribution in each comp 1 bin, requires fit_bins2.
            `mean` - Use mean of component 2 distribution in each comp 1 bin, requires fit_bins2.
    fit_bins1: array, None
        Bins for redshift of catalog 1 (default=10).
    fit_bins2: array, None
        Bins for redshift of catalog 2 (default=30).
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
        Label of x axis (default=cat1.labels['z']).
    ylabel: str
        Label of y axis (default=cat2.labels['z']).
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
    return catalog_funcs.plot_density(cat1, cat2, matching_type, col='z', **kwargs)
def redshift_masscolor(cat1, cat2, matching_type, log_mass=True, color1=True, **kwargs):
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
    log_mass: bool
        Log scale for mass
    color1: bool
        Use catalog 1 for color. If false uses catalog 2
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
        Use redshift errors of catalog 2 in fit (default=True).
    fit_statistics: str
        Statistics to be used in fit (default=mean). Options are:
            `individual` - Use each point
            `mode` - Use mode of component 2 distribution in each comp 1 bin, requires fit_bins2.
            `mean` - Use mean of component 2 distribution in each comp 1 bin, requires fit_bins2.
    fit_bins1: array, None
        Bins for redshift of catalog 1 (default=10).
    fit_bins2: array, None
        Bins for redshift of catalog 2 (default=30).
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
        Label of x axis (default=cat1.labels['z']).
    ylabel: str
        Label of y axis (default=cat2.labels['z']).

    Returns
    -------
    ax: matplotlib.axes
        Axis of the plot
    matplotlib.colorbar.Colorbar (optional)
        Colorbar of the recovey rates. Only returned if add_cb=True.
    """
    return catalog_funcs.plot_color(cat1, cat2, matching_type, col='z', col_color='mass',
            color1=color1, color_log=log_mass, **kwargs)
def redshift_masspanel(cat1, cat2, matching_type, mass_bins=5, log_mass=True, **kwargs):
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
    mass_bins: int, array, int
        Mass bins to make panels
    log_mass: bool
        Log scale for mass
    panel_cat1: bool
        Used catalog 1 for col_panel. If false uses catalog 2
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
        Use redshift errors of catalog 2 in fit (default=True).
    fit_statistics: str
        Statistics to be used in fit (default=mean). Options are:
            `individual` - Use each point
            `mode` - Use mode of component 2 distribution in each comp 1 bin, requires fit_bins2.
            `mean` - Use mean of component 2 distribution in each comp 1 bin, requires fit_bins2.
    fit_bins1: array, None
        Bins for redshift of catalog 1 (default=10).
    fit_bins2: array, None
        Bins for redshift of catalog 2 (default=30).
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
    label_fmt: str
        Format the values of binedges (ex: '.2f')
    xlabel: str
        Label of x axis (default=cat1.labels['z']).
    ylabel: str
        Label of y axis (default=cat2.labels['z']).

    Returns
    -------
    fig: matplotlib.figure.Figure
        `matplotlib.figure.Figure` object
    ax: matplotlib.axes
        Axes with the panels
    """
    kwargs['label_fmt'] = kwargs.get('label_fmt', '.1f')
    return catalog_funcs.plot_panel(cat1, cat2, matching_type, col='z',
            col_panel='mass', bins_panel=mass_bins, log_panel=log_mass,
            **kwargs)
def redshift_density_masspanel(cat1, cat2, matching_type, mass_bins=5, log_mass=True, **kwargs):
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
    mass_bins: int, array, int
        Mass bins to make panels
    log_mass: bool
        Log scale for mass
    panel_cat1: bool
        Used catalog 1 for col_panel. If false uses catalog 2
    bins1: array, int
        Bins of redshift 1 for density
    bins2: array, int
        Bins of redshift 2 for density
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
        Use redshift errors of catalog 2 in fit (default=True).
    fit_statistics: str
        Statistics to be used in fit (default=mean). Options are:
            `individual` - Use each point
            `mode` - Use mode of component 2 distribution in each comp 1 bin, requires fit_bins2.
            `mean` - Use mean of component 2 distribution in each comp 1 bin, requires fit_bins2.
    fit_bins1: array, None
        Bins for redshift of catalog 1 (default=10).
    fit_bins2: array, None
        Bins for redshift of catalog 2 (default=30).
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
    label_fmt: str
        Format the values of binedges (ex: '.2f')
    xlabel: str
        Label of x axis (default=cat1.labels['z']).
    ylabel: str
        Label of y axis (default=cat2.labels['z']).

    Returns
    -------
    ax: matplotlib.axes
        Axis of the plot
    matplotlib.colorbar.Colorbar (optional)
        Colorbar of the recovey rates. Only returned if add_cb=True.
    """
    kwargs['label_fmt'] = kwargs.get('label_fmt', '.1f')
    return catalog_funcs.plot_density_panel(cat1, cat2, matching_type, col='z',
            col_panel='mass', bins_panel=mass_bins, log_panel=log_mass,
            **kwargs)
def redshift_metrics(cat1, cat2, matching_type, **kwargs):
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
    bins1: array, int
        Bins for catalog 1
    bins2: array, int
        Bins for catalog 2
    mode: str
        Mode to run (default=redshit). Options are:
        simple - used simple difference
        redshift - metrics for (values2-values1)/(1+values1)
        log - metrics for log of values
    mask1: array, None
        Mask for clusters 1 properties, must have size=cat1.size
    mask2: array, None
        Mask for clusters 2 properties, must have size=cat2.size

    Other parameters
    ----------------
    fig_kwargs: dict
        Additional arguments for plt.subplots
    metrics_kwargs: dict
        Dictionary of dictionary configs for each metric plots.
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
    mode = kwargs.pop('mode', 'diff_z')
    return catalog_funcs.plot_metrics(cat1, cat2, matching_type, col='z', mode=mode,
                                       **kwargs)
def redshift_density_metrics(cat1, cat2, matching_type, **kwargs):
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
        Mode to run (default=redshit). Options are:
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
        Use redshift errors of catalog 2 in fit (default=True).
    fit_statistics: str
        Statistics to be used in fit (default=mean). Options are:
            `individual` - Use each point
            `mode` - Use mode of component 2 distribution in each comp 1 bin, requires fit_bins2.
            `mean` - Use mean of component 2 distribution in each comp 1 bin, requires fit_bins2.
    fit_bins1: array, None
        Bins for redshift of catalog 1 (default=10).
    fit_bins2: array, None
        Bins for redshift of catalog 2 (default=30).
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
    kwargs['metrics_mode'] = kwargs.get('metrics_mode', 'diff_z')
    kwargs['metrics'] = kwargs.get('metrics', ['std.fill', 'mean'])
    return catalog_funcs.plot_density_metrics(cat1, cat2, matching_type, col='z', **kwargs)
def redshift_dist(cat1, cat2, matching_type, redshift_bins_dist=30, redshift_bins=5, mass_bins=5,
              log_mass=True, transpose=False, **kwargs):
    """
    Plot distribution of a cat1 redshift, binned by the cat2 redshift (in panels),
    with option for cat2 mass bins (in lines).

    Parameters
    ----------
    cat1: clevar.ClCatalog
        ClCatalog with matching information
    cat2: clevar.ClCatalog
        ClCatalog matched to
    matching_type: str
        Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
    redshift_bins_dist: array, int
        Bins for distribution of the cat1 redshift
    redshift_bins: array, int
        Bins for cat2 redshift
    mass_bins: array, int
        Bins for cat2 mass
    log_mass: bool
        Log scale for mass
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
    kwargs_ = {}
    kwargs_.update(kwargs)
    kwargs_.update({
        'col': 'z',
        'col_aux': 'mass',
        'bins1': redshift_bins_dist,
        'bins2': redshift_bins,
        'bins_aux': mass_bins,
        'log_vals': False,
        'log_aux': log_mass,
        'transpose': transpose,
    })
    return catalog_funcs.plot_dist(cat1, cat2, matching_type, **kwargs_)
def redshift_dist_self(cat, bins1=30, redshift_bins_dist=30, redshift_bins=5, mass_bins=5,
                   log_mass=True, transpose=False, mask=None, **kwargs):
    """
    Plot distribution of a cat redshift, binned by redshift (in panels),
    with option for mass bins (in lines).
    Is is useful to compare with redshift_dist results.

    Parameters
    ----------
    cat: clevar.ClCatalog
        Input Catalog
    redshift_bins_dist: array, int
        Bins for distribution of redshift
    redshift_bins: array, int
        Bins for redshift panels
    mass_bins: array, int
        Bins for mass
    log_mass: bool
        Log scale for mass
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
    kwargs_ = {}
    kwargs_.update(kwargs)
    kwargs_.update({
        'col': 'z',
        'col_aux': 'mass',
        'bins1': redshift_bins_dist,
        'bins2': redshift_bins,
        'bins_aux': mass_bins,
        'log_vals': False,
        'log_aux': log_mass,
        'transpose': transpose,
        'mask': mask,
    })
    return catalog_funcs.plot_dist_self(cat, **kwargs_)
def redshift_density_dist(cat1, cat2, matching_type, **kwargs):
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
        Use redshift errors of catalog 2 in fit (default=True).
    fit_statistics: str
        Statistics to be used in fit (default=mean). Options are:
            `individual` - Use each point
            `mode` - Use mode of component 2 distribution in each comp 1 bin, requires fit_bins2.
            `mean` - Use mean of component 2 distribution in each comp 1 bin, requires fit_bins2.
    fit_bins1: array, None
        Bins for redshift of catalog 1 (default=10).
    fit_bins2: array, None
        Bins for redshift of catalog 2 (default=30).
    fit_legend_kwargs: dict
        Additional arguments for plt.legend.
    fit_bindata_kwargs: dict
        Additional arguments for pylab.errorbar.
    fit_plt_kwargs: dict
        Additional arguments for plot of fit pylab.scatter.
    fit_label_components: tuple (of strings)
        Names of fitted components in fit line label, default=(xlabel, ylabel).

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
    return catalog_funcs.plot_density_dist(cat1, cat2, matching_type, col='z', **kwargs)

############################################################################################
### Mass Plots #############################################################################
############################################################################################
def mass(cat1, cat2, matching_type, log_mass=True, **kwargs):
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
    log_mass: bool
        Log scale for mass
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
        Use mass errors of catalog 2 in fit (default=True).
    fit_log: bool
        Bin and fit in log values (default=log_mass).
    fit_statistics: str
        Statistics to be used in fit (default=mean). Options are:
            `individual` - Use each point
            `mode` - Use mode of component 2 distribution in each comp 1 bin, requires fit_bins2.
            `mean` - Use mean of component 2 distribution in each comp 1 bin, requires fit_bins2.
    fit_bins1: array, None
        Bins for mass of catalog 1 (default=10).
    fit_bins2: array, None
        Bins for mass of catalog 2 (default=30).
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
        Label of x axis (default=cat1.labels['mass']).
    ylabel: str
        Label of y axis (default=cat2.labels['mass']).

    Returns
    -------
    ax: matplotlib.axes
        Axis of the plot
    """
    kwargs['fit_log'] = kwargs.get('fit_log', log_mass)
    return catalog_funcs.plot(cat1, cat2, matching_type, col='mass',
            xscale='log' if log_mass else 'linear',
            yscale='log' if log_mass else 'linear',
            **kwargs)
def mass_zcolor(cat1, cat2, matching_type, log_mass=True, color1=True, **kwargs):
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
    log_mass: bool
        Log scale for mass
    color1: bool
        Use catalog 1 for color. If false uses catalog 2
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
        Use mass errors of catalog 2 in fit (default=True).
    fit_log: bool
        Bin and fit in log values (default=log_mass).
    fit_statistics: str
        Statistics to be used in fit (default=mean). Options are:
            `individual` - Use each point
            `mode` - Use mode of component 2 distribution in each comp 1 bin, requires fit_bins2.
            `mean` - Use mean of component 2 distribution in each comp 1 bin, requires fit_bins2.
    fit_bins1: array, None
        Bins for mass of catalog 1 (default=10).
    fit_bins2: array, None
        Bins for mass of catalog 2 (default=30).
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
        Label of x axis (default=cat1.labels['mass']).
    ylabel: str
        Label of y axis (default=cat2.labels['mass']).

    Returns
    -------
    ax: matplotlib.axes
        Axis of the plot
    matplotlib.colorbar.Colorbar (optional)
        Colorbar of the recovey rates. Only returned if add_cb=True.
    """
    kwargs['fit_log'] = kwargs.get('fit_log', log_mass)
    return catalog_funcs.plot_color(cat1, cat2, matching_type, col='mass', col_color='z',
            xscale='log' if log_mass else 'linear',
            yscale='log' if log_mass else 'linear',
            color1=color1, **kwargs)
def mass_density(cat1, cat2, matching_type, log_mass=True, **kwargs):
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
    log_mass: bool
        Log scale for mass
    bins1: array, int
        Bins of mass 1 for density
    bins2: array, int
        Bins of mass 2 for density
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
        Use mass errors of catalog 2 in fit (default=True).
    fit_log: bool
        Bin and fit in log values (default=log_mass).
    fit_statistics: str
        Statistics to be used in fit (default=mean). Options are:
            `individual` - Use each point
            `mode` - Use mode of component 2 distribution in each comp 1 bin, requires fit_bins2.
            `mean` - Use mean of component 2 distribution in each comp 1 bin, requires fit_bins2.
    fit_bins1: array, None
        Bins for mass of catalog 1 (default=10).
    fit_bins2: array, None
        Bins for mass of catalog 2 (default=30).
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
        Label of x axis (default=cat1.labels['mass']).
    ylabel: str
        Label of y axis (default=cat2.labels['mass']).

    Returns
    -------
    ax: matplotlib.axes
        Axis of the plot
    matplotlib.colorbar.Colorbar (optional)
        Colorbar of the recovey rates. Only returned if add_cb=True.
    """
    kwargs['fit_log'] = kwargs.get('fit_log', log_mass)
    return catalog_funcs.plot_density(cat1, cat2, matching_type, col='mass',
            xscale='log' if log_mass else 'linear',
            yscale='log' if log_mass else 'linear',
            **kwargs)
def mass_zpanel(cat1, cat2, matching_type, redshift_bins=5, log_mass=True, **kwargs):
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
    redshift_bins: int, array, int
        Redshift bins to make panels
    log_mass: bool
        Log scale for mass
    panel_cat1: bool
        Used catalog 1 for col_panel. If false uses catalog 2
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
        Use mass errors of catalog 2 in fit (default=True).
    fit_log: bool
        Bin and fit in log values (default=log_mass).
    fit_statistics: str
        Statistics to be used in fit (default=mean). Options are:
            `individual` - Use each point
            `mode` - Use mode of component 2 distribution in each comp 1 bin, requires fit_bins2.
            `mean` - Use mean of component 2 distribution in each comp 1 bin, requires fit_bins2.
    fit_bins1: array, None
        Bins for mass of catalog 1 (default=10).
    fit_bins2: array, None
        Bins for mass of catalog 2 (default=30).
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
    label_fmt: str
        Format the values of binedges (ex: '.2f')
    xlabel: str
        Label of x axis (default=cat1.labels['mass']).
    ylabel: str
        Label of y axis (default=cat2.labels['mass']).

    Returns
    -------
    fig: matplotlib.figure.Figure
        `matplotlib.figure.Figure` object
    ax: matplotlib.axes
        Axes with the panels
    """
    kwargs['label_format'] = kwargs.get('label_format',
        lambda v: f'%{kwargs.pop("label_fmt", ".2f")}'%v)
    kwargs['fit_log'] = kwargs.get('fit_log', log_mass)
    return catalog_funcs.plot_panel(cat1, cat2, matching_type, col='mass',
            col_panel='z', bins_panel=redshift_bins,
            xscale='log' if log_mass else 'linear',
            yscale='log' if log_mass else 'linear',
            **kwargs)
def mass_density_zpanel(cat1, cat2, matching_type, redshift_bins=5, log_mass=True, **kwargs):
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
    redshift_bins: int, array, int
        Redshift bins to make panels
    log_mass: bool
        Log scale for mass
    panel_cat1: bool
        Used catalog 1 for col_panel. If false uses catalog 2
    bins1: array, int
        Bins of mass 1 for density
    bins2: array, int
        Bins of mass 2 for density
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
        Use mass errors of catalog 2 in fit (default=True).
    fit_log: bool
        Bin and fit in log values (default=log_mass).
    fit_statistics: str
        Statistics to be used in fit (default=mean). Options are:
            `individual` - Use each point
            `mode` - Use mode of component 2 distribution in each comp 1 bin, requires fit_bins2.
            `mean` - Use mean of component 2 distribution in each comp 1 bin, requires fit_bins2.
    fit_bins1: array, None
        Bins for mass of catalog 1 (default=10).
    fit_bins2: array, None
        Bins for mass of catalog 2 (default=30).
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
    label_fmt: str
        Format the values of binedges (ex: '.2f')
    xlabel: str
        Label of x axis (default=cat1.labels['mass']).
    ylabel: str
        Label of y axis (default=cat2.labels['mass']).

    Returns
    -------
    ax: matplotlib.axes
        Axis of the plot
    matplotlib.colorbar.Colorbar (optional)
        Colorbar of the recovey rates. Only returned if add_cb=True.
    """
    kwargs['fit_log'] = kwargs.get('fit_log', log_mass)
    return catalog_funcs.plot_density_panel(cat1, cat2, matching_type, col='mass',
            col_panel='z', bins_panel=redshift_bins,
            xscale='log' if log_mass else 'linear',
            yscale='log' if log_mass else 'linear',
            **kwargs)
def mass_metrics(cat1, cat2, matching_type, log_mass=True, **kwargs):
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
    log_mass: bool
        Log scale for mass
    bins1: array, int
        Bins for catalog 1
    bins2: array, int
        Bins for catalog 2
    mode: str
        Mode to run (default=log). Options are:
        simple - used simple difference
        redshift - metrics for (values2-values1)/(1+values1)
        log - metrics for log of values
    mask1: array, None
        Mask for clusters 1 properties, must have size=cat1.size
    mask2: array, None
        Mask for clusters 2 properties, must have size=cat2.size

    Other parameters
    ----------------
    fig_kwargs: dict
        Additional arguments for plt.subplots
    metrics_kwargs: dict
        Dictionary of dictionary configs for each metric plots.
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
    mode = kwargs.pop('mode', 'log')
    return catalog_funcs.plot_metrics(cat1, cat2, matching_type, col='mass', mode=mode,
            xscale='log' if log_mass else 'linear',
            yscale='log' if log_mass else 'linear',
            **kwargs)
def mass_density_metrics(cat1, cat2, matching_type, log_mass=True, **kwargs):
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
    log_mass: bool
        Log scale for mass
    bins1: array, int
        Bins for component 1
    bins2: array, int
        Bins for component 2
    metrics_mode: str
        Mode to run (default=log). Options are:
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
        Use mass errors of catalog 2 in fit (default=True).
    fit_log: bool
        Bin and fit in log values (default=log_mass).
    fit_statistics: str
        Statistics to be used in fit (default=mean). Options are:
            `individual` - Use each point
            `mode` - Use mode of component 2 distribution in each comp 1 bin, requires fit_bins2.
            `mean` - Use mean of component 2 distribution in each comp 1 bin, requires fit_bins2.
    fit_bins1: array, None
        Bins for mass of catalog 1 (default=10).
    fit_bins2: array, None
        Bins for mass of catalog 2 (default=30).
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
    metrics_mode = kwargs.pop('metrics_mode', 'log')
    kwargs['fit_log'] = kwargs.get('fit_log', log_mass)
    return catalog_funcs.plot_density_metrics(cat1, cat2, matching_type, col='mass',
            xscale='log' if log_mass else 'linear',
            yscale='log' if log_mass else 'linear',
            metrics_mode=metrics_mode, **kwargs)
def mass_dist(cat1, cat2, matching_type, mass_bins_dist=30, mass_bins=5, redshift_bins=5,
              log_mass=True, transpose=False, **kwargs):
    """
    Plot distribution of a cat1 mass, binned by the cat2 mass (in panels),
    with option for cat2 redshift bins (in lines).

    Parameters
    ----------
    cat1: clevar.ClCatalog
        ClCatalog with matching information
    cat2: clevar.ClCatalog
        ClCatalog matched to
    matching_type: str
        Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
    mass_bins_dist: array, int
        Bins for distribution of the cat1 mass
    mass_bins: array, int
        Bins for cat2 mass
    redshift_bins: array, int
        Bins for cat2 redshift
    log_mass: bool
        Log scale for mass
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
    kwargs_ = {}
    kwargs_.update(kwargs)
    kwargs_.update({
        'col': 'mass',
        'col_aux': 'z',
        'bins1': mass_bins_dist,
        'bins2': mass_bins,
        'bins_aux': redshift_bins,
        'log_vals': log_mass,
        'log_aux': False,
        'transpose': transpose,
    })
    return catalog_funcs.plot_dist(cat1, cat2, matching_type, **kwargs_)
def mass_dist_self(cat, bins1=30, mass_bins_dist=30, mass_bins=5, redshift_bins=5,
                   log_mass=True, transpose=False, mask=None, **kwargs):
    """
    Plot distribution of a cat mass, binned by mass (in panels),
    with option for redshift bins (in lines).
    Is is useful to compare with mass_dist results.

    Parameters
    ----------
    cat: clevar.ClCatalog
        Input Catalog
    mass_bins_dist: array, int
        Bins for distribution of mass
    mass_bins: array, int
        Bins for mass panels
    redshift_bins: array, int
        Bins for redshift
    log_mass: bool
        Log scale for mass
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
    kwargs_ = {}
    kwargs_.update(kwargs)
    kwargs_.update({
        'col': 'mass',
        'col_aux': 'z',
        'bins1': mass_bins_dist,
        'bins2': mass_bins,
        'bins_aux': redshift_bins,
        'log_vals': log_mass,
        'log_aux': False,
        'transpose': transpose,
        'mask': mask,
    })
    return catalog_funcs.plot_dist_self(cat, **kwargs_)

def mass_density_dist(cat1, cat2, matching_type, log_mass=True, **kwargs):
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
        Bins for component 1 (for density colors).
    bins2: array, int
        Bins for component 2 (for density colors).
    add_err: bool
        Add errorbars
    mask1: array, None
        Mask for clusters 1 properties, must have size=cat1.size
    mask2: array, None
        Mask for clusters 2 properties, must have size=cat2.size
    log_mass: bool
        Log scale for mass

    Fit Parameters
    --------------
    add_bindata: bool
        Plot binned data used for fit (default=False).
    add_fit: bool
        Fit and plot binned dat (default=False).
    add_fit_err: bool
        Use mass errors of catalog 2 in fit (default=True).
    fit_log: bool
        Bin and fit in log values (default=log_mass).
    fit_statistics: str
        Statistics to be used in fit (default=mean). Options are:
            `individual` - Use each point
            `mode` - Use mode of component 2 distribution in each comp 1 bin, requires fit_bins2.
            `mean` - Use mean of component 2 distribution in each comp 1 bin, requires fit_bins2.
    fit_bins1: array, None
        Bins for mass of catalog 1 (default=10).
    fit_bins2: array, None
        Bins for mass of catalog 2 (default=30).
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
    kwargs['fit_log'] = kwargs.get('fit_log', True)
    kwargs['xscale'] = 'log' if log_mass else 'linear'
    kwargs['yscale'] = 'log' if log_mass else 'linear'
    kwargs['fit_log'] = kwargs.get('fit_log', log_mass)
    return catalog_funcs.plot_density_dist(cat1, cat2, matching_type, col='mass', **kwargs)
