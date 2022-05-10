"""@file clevar/match_metrics/scaling/funcs.py

Main scaling functions for mass and redshift plots,
wrapper of catalog_funcs functions
"""
from ...utils import deep_update
from . import catalog_funcs

############################################################################################
### Redshift Plots #########################################################################
############################################################################################


def redshift(cat1, cat2, matching_type, **kwargs):
    """
    Scatter plot with errorbars and color based on input

    Parameters
    ----------
    cat1, cat2: clevar.ClCatalog
        ClCatalogs with matching information.
    matching_type: str
        Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
    add_err: bool
        Add errorbars
    mask1, mask2: array, None
        Masks for clusters 1(2), must have size=cat1(2).size

    Other Parameters
    ----------------
    ax: matplotlib.axes, None
        Ax to add plot. If equals `None`, one is created.
    plt_kwargs: dict, None
        Additional arguments for pylab.scatter
    err_kwargs: dict, None
        Additional arguments for pylab.errorbar
    xlabel, ylabel: str
        Label of x/y axis (default=cat1.labels['z']/cat2.labels['z']).

    Other Parameters
    ----------------
    add_bindata: bool
        Plot binned data used for fit (default=False).
    add_fit: bool
        Fit and plot binned dat (default=False).
    add_fit_err: bool
        Use redshift errors of catalog 2 in fit (default=True).
    fit_statistics: str
        Statistics to be used in fit (default=mean). Options are:

            * `individual` - Use each point.
            * `mode` - Use mode of catalog 2 redshift distribution in each catalog 1 redshift bin,\
            requires fit_bins2.
            * `mean` - Use mean of catalog 2 redshift distribution in each catalog 1 redshift bin,\
            requires fit_bins2.

    fit_bins1: array, None
        Bins for redshift of catalog 1 (default=10).
    fit_bins2: array, None
        Bins for redshift of catalog 2 (default=30).
    fit_legend_kwargs: dict, None
        Additional arguments for plt.legend.
    fit_bindata_kwargs: dict, None
        Additional arguments for pylab.errorbar.
    fit_plt_kwargs: dict, None
        Additional arguments for plot of fit pylab.scatter.
    fit_label_components: tuple (of strings)
        Names of fitted components in fit line label, default=(xlabel, ylabel).

    Returns
    -------
    info: dict
        Information of data in the plots, it contains the sections:

            * `ax`: ax used in the plot.
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
    return catalog_funcs.plot(cat1, cat2, matching_type, col='z', **kwargs)


def redshift_density(cat1, cat2, matching_type, **kwargs):
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
        Bins for redshift of catalogs 1 and 2 (for density colors).
    add_err: bool
        Add errorbars
    mask1, mask2: array, None
        Masks for clusters 1(2), must have size=cat1(2).size

    Other Parameters
    ----------------
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
    ax_rotation: float
        Angle (in degrees) for rotation of axis of binning. Overwrites use of (bins1, bins2)
    rotation_resolution: int
        Number of bins to be used when ax_rotation!=0.
    xlabel, ylabel: str
        Label of x/y axis (default=cat1.labels['z']/cat2.labels['z']).
    xscale, yscale: str
        Scale for x/y axis.
    **fit_kwargs:
        Other fit arguments (see `fit_*` paramters in `scaling.redshift` for more info).

    Returns
    -------
    info: dict
        Information of data in the plots, it contains the sections:

            * `ax`: ax used in the plot.
            * `ax_cb` (optional): ax of colorbar
            * `binned_data` (optional): input data for fitting \
            (see `scaling.redshift` for more info).
            * `fit` (optional): fitting output dictionary \
            (see `scaling.redshift` for more info).
            * `plots` (optional): fit and binning plots \
            (see `scaling.redshift` for more info).
    """
    return catalog_funcs.plot_density(cat1, cat2, matching_type, col='z', **kwargs)


def redshift_masscolor(cat1, cat2, matching_type, log_mass=True, color1=True, **kwargs):
    """
    Scatter plot with errorbars and color based on input

    Parameters
    ----------
    cat1, cat2: clevar.ClCatalog
        ClCatalogs with matching information.
    matching_type: str
        Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
    log_mass: bool
        Log scale for mass
    color1: bool
        Use catalog 1 for color. If false uses catalog 2
    add_err: bool
        Add errorbars
    mask1, mask2: array, None
        Masks for clusters 1(2), must have size=cat1(2).size

    Other Parameters
    ----------------
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
    xlabel, ylabel: str
        Label of x/y axis (default=cat1.labels['z']/cat2.labels['z']).
    **fit_kwargs:
        Other fit arguments (see `fit_*` paramters in `scaling.redshift` for more info).

    Returns
    -------
    info: dict
        Information of data in the plots, it contains the sections:

            * `ax`: ax used in the plot.
            * `ax_cb` (optional): ax of colorbar
            * `binned_data` (optional): input data for fitting \
            (see `scaling.redshift` for more info).
            * `fit` (optional): fitting output dictionary \
            (see `scaling.redshift` for more info).
            * `plots` (optional): fit and binning plots \
            (see `scaling.redshift` for more info).
    """
    return catalog_funcs.plot(cat1, cat2, matching_type, col='z', col_color='mass',
            color1=color1, color_log=log_mass, **kwargs)


def redshift_masspanel(cat1, cat2, matching_type, mass_bins=5, log_mass=True, **kwargs):
    """
    Scatter plot with errorbars and color based on input with panels

    Parameters
    ----------
    cat1, cat2: clevar.ClCatalog
        ClCatalogs with matching information.
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
    mask1, mask2: array, None
        Masks for clusters 1(2), must have size=cat1(2).size

    Other Parameters
    ----------------
    plt_kwargs: dict, None
        Additional arguments for pylab.scatter
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
    label_fmt: str
        Format the values of binedges (ex: '.2f')
    xlabel, ylabel: str
        Label of x/y axis (default=cat1.labels['z']/cat2.labels['z']).
    **fit_kwargs:
        Other fit arguments (see `fit_*` paramters in `scaling.redshift` for more info).

    Returns
    -------
    info: dict
        Information of data in the plots, it contains the sections:

            * `fig`: `matplotlib.figure.Figure` object.
            * `axes`: `matplotlib.axes` used in the plot.
            * `binned_data` (optional): input data for fitting \
            (see `scaling.redshift` for more info).
            * `fit` (optional): fitting output dictionary \
            (see `scaling.redshift` for more info).
            * `plots` (optional): fit and binning plots \
            (see `scaling.redshift` for more info).
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
    cat1, cat2: clevar.ClCatalog
        ClCatalogs with matching information.
    matching_type: str
        Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
    mass_bins: int, array, int
        Mass bins to make panels
    log_mass: bool
        Log scale for mass
    panel_cat1: bool
        Used catalog 1 for col_panel. If false uses catalog 2
    bins1, bins2: array, None
        Bins for redshift of catalogs 1 and 2 (for density colors).
    add_err: bool
        Add errorbars
    mask1, mask2: array, None
        Masks for clusters 1(2), must have size=cat1(2).size

    Other Parameters
    ----------------
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
    fig_kwargs: dict, None
        Additional arguments for plt.subplots
    add_label: bool
        Add bin label to panel
    label_format: function
        Function to format the values of the bins
    label_fmt: str
        Format the values of binedges (ex: '.2f')
    xlabel, ylabel: str
        Label of x/y axis (default=cat1.labels['z']/cat2.labels['z']).
    **fit_kwargs:
        Other fit arguments (see `fit_*` paramters in `scaling.redshift` for more info).

    Returns
    -------
    info: dict
        Information of data in the plots, it contains the sections:

            * `fig`: `matplotlib.figure.Figure` object.
            * `axes`: `matplotlib.axes` used in the plot.
            * `ax_cb` (optional): ax of colorbar
            * `binned_data` (optional): input data for fitting \
            (see `scaling.redshift` for more info).
            * `fit` (optional): fitting output dictionary \
            (see `scaling.redshift` for more info).
            * `plots` (optional): fit and binning plots \
            (see `scaling.redshift` for more info).
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
    cat1, cat2: clevar.ClCatalog
        ClCatalogs with matching information.
    matching_type: str
        Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
    bins1, bins2: array, None
        Bins for redshift of catalog 1 and 2.
    metrics_mode: str
        Mode to run metrics (default=`diff_z`). Options are:

            * `simple` : metrics for `values2`.
            * `log` : metrics for `log10(values2)`.
            * `diff` : metrics for `values2-values1`.
            * `diff_log` : metrics for `log10(values2)-log10(values1)`.
            * `diff_z` : metrics for `(values2-values1)/(1+values1)`.

    metrics: list
        List of mettrics to be plotted (default=[`std.fill`, `mean`]). Possibilities are:

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
    fig_kwargs: dict, None
        Additional arguments for plt.subplots
    metrics_kwargs: dict, None
        Dictionary of dictionary configs for each metric plots.
    legend_kwargs: dict, None
        Additional arguments for plt.legend
    label1, label2: str
        Label for catalog 1/2 redshifts.
    scale1, scale2: str
        Scale of catalog 1/2 redshifts.

    Returns
    -------
    info: dict
        Information of data in the plots, it contains the sections:

            * `fig`: `matplotlib.figure.Figure` object.
            * `axes`: `matplotlib.axes` used in the plot.
    """
    kwargs['mode'] = kwargs.get('mode', 'diff_z')
    kwargs['metrics_kwargs'] = deep_update(
        {'std':{'label': '$\sigma_z$'}, 'mean':{'label': '$bias_z$'}},
        kwargs.get('metrics_kwargs', {}))
    return catalog_funcs.plot_metrics(cat1, cat2, matching_type, col='z', **kwargs)


def redshift_density_metrics(cat1, cat2, matching_type, **kwargs):
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
        Bins for redshift of catalog 1 and 2.
    metrics_mode: str
        Mode to run metrics (default=`diff_z`). Options are:

            * `simple` : metrics for `values2`.
            * `log` : metrics for `log10(values2)`.
            * `diff` : metrics for `values2-values1`.
            * `diff_log` : metrics for `log10(values2)-log10(values1)`.
            * `diff_z` : metrics for `(values2-values1)/(1+values1)`.

    metrics: list
        List of mettrics to be plotted (default=[`std.fill`, `mean`]). Possibilities are:

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
    fig_kwargs: dict, None
        Additional arguments for plt.subplots
    ax_rotation: float
        Angle (in degrees) for rotation of axis of binning. Overwrites use of (bins1, bins2) on main plot.
    rotation_resolution: int
        Number of bins to be used when ax_rotation!=0.
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
    xscale, yscale: str
        Scale for x/y axis.
    fig_pos: tuple
        List with edges for the figure. Must be in format (left, bottom, right, top)
    fig_frac: tuple
        Sizes of each panel in the figure. Must be in the format (main_panel, gap, colorbar)
        and have values: [0, 1]. Colorbar is only used with add_cb key.
    **fit_kwargs:
        Other fit arguments (see `fit_*` paramters in `scaling.redshift` for more info).

    Returns
    -------
    info: dict
        Information of data in the plots, it contains the sections:

            * `fig`: `matplotlib.figure.Figure` object.
            * `axes`: dictionary with each ax of the plot.
            * `metrics`: dictionary with the plots for each metric.
    """
    kwargs['metrics'] = kwargs.get('metrics', ['std.fill', 'mean'])
    kwargs['metrics_mode'] = kwargs.get('metrics_mode', 'diff_z')
    kwargs['metrics_kwargs'] = deep_update(
        {
            'std.fill': {'label':'$\sigma_z$'},
            'std': {'label':'$\sigma_z$'},
            'mean': {'label':'$bias_z$'},
        },
        kwargs.get('metrics_kwargs', {}))
    return catalog_funcs.plot_density_metrics(cat1, cat2, matching_type, col='z', **kwargs)


def redshift_dist(cat1, cat2, matching_type, redshift_bins_dist=30, redshift_bins=5, mass_bins=5,
              log_mass=True, transpose=False, **kwargs):
    """
    Plot distribution of a cat1 redshift, binned by the cat2 redshift (in panels),
    with option for cat2 mass bins (in lines).

    Parameters
    ----------
    cat1, cat2: clevar.ClCatalog
        ClCatalogs with matching information.
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

    Other Parameters
    ----------------
    fig_kwargs: dict, None
        Additional arguments for plt.subplots
    shape: str
        Shape of the lines. Can be steps or line.
    plt_kwargs: dict, None
        Additional arguments for pylab.plot
    line_kwargs_list: list, None
        List of additional arguments for plotting each line (using pylab.plot).
        Must have same size as len(bins_aux)-1
    legend_kwargs: dict, None
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


def redshift_dist_self(cat, redshift_bins_dist=30, redshift_bins=5, mass_bins=5, log_mass=True,
                       transpose=False, mask=None, **kwargs):
    """
    Plot distribution of a cat redshift, binned by redshift (in panels),
    with option for mass bins (in lines).
    Is is useful to compare with redshift_dist results.

    Parameters
    ----------
    cat1, cat2: clevar.ClCatalog
        ClCatalogs with matching information.
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

    Other Parameters
    ----------------
    fig_kwargs: dict, None
        Additional arguments for plt.subplots
    shape: str
        Shape of the lines. Can be steps or line.
    plt_kwargs: dict, None
        Additional arguments for pylab.plot
    line_kwargs_list: list, None
        List of additional arguments for plotting each line (using pylab.plot).
        Must have same size as len(bins_aux)-1
    legend_kwargs: dict, None
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
    cat1, cat2: clevar.ClCatalog
        ClCatalogs with matching information.
    matching_type: str
        Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
    col: str
        Name of column to be plotted
    bins1, bins2: array, None
        Bins for redshift of catalogs 1 and 2 (for density colors).
    add_err: bool
        Add errorbars
    mask1, mask2: array, None
        Masks for clusters 1(2), must have size=cat1(2).size

    Other Parameters
    ----------------
    fig_kwargs: dict, None
        Additional arguments for plt.subplots
    ax_rotation: float
        Angle (in degrees) for rotation of axis of binning. Overwrites use of (bins1, bins2) on main plot.
    rotation_resolution: int
        Number of bins to be used when ax_rotation!=0.
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
    xscale, yscale: str
        Scale for x/y axis.
    fig_pos: tuple
        List with edges for the figure. Must be in format (left, bottom, right, top)
    fig_frac: tuple
        Sizes of each panel in the figure. Must be in the format (main_panel, gap, colorbar)
        and have values: [0, 1]. Colorbar is only used with add_cb key.
    **fit_kwargs:
        Other fit arguments (see `fit_*` paramters in `scaling.redshift` for more info).
    vline_kwargs: dict, None
        Arguments for vlines marking bins in main plot, used in plt.axvline.


    Returns
    -------
    info: dict
        Information of data in the plots, it contains the sections:

            * `fig`: `matplotlib.figure.Figure` object.
            * `axes`: dictionary with each ax of the plot.
            * `binned_data` (optional): input data for fitting \
            (see `scaling.redshift` for more info).
            * `fit` (optional): fitting output dictionary \
            (see `scaling.redshift` for more info).
            * `plots` (optional): fit and binning plots \
            (see `scaling.redshift` for more info).
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
    cat1, cat2: clevar.ClCatalog
        ClCatalogs with matching information.
    matching_type: str
        Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
    log_mass: bool
        Log scale for mass
    add_err: bool
        Add errorbars
    mask1, mask2: array, None
        Masks for clusters 1(2), must have size=cat1(2).size

    Other Parameters
    ----------------
    ax: matplotlib.axes, None
        Ax to add plot. If equals `None`, one is created.
    plt_kwargs: dict, None
        Additional arguments for pylab.scatter
    err_kwargs: dict, None
        Additional arguments for pylab.errorbar
    xlabel, ylabel: str
        Label of x/y axis (default=cat1.labels['mass']/cat2.labels['mass']).

    Other Parameters
    ----------------
    add_bindata: bool
        Plot binned data used for fit (default=False).
    add_fit: bool
        Fit and plot binned dat (default=False).
    add_fit_err: bool
        Use mass errors of catalog 2 in fit (default=True).
    fit_log: bool
        Bin and fit in log values (default=True).
    fit_statistics: str
        Statistics to be used in fit (default=mean). Options are:

            * `individual` - Use each point.
            * `mode` - Use mode of catalog 2 mass distribution in each catalog 1 mass bin,\
            requires fit_bins2.
            * `mean` - Use mean of catalog 2 mass distribution in each catalog 1 mass bin,\
            requires fit_bins2.

    fit_bins1: array, None
        Bins for mass of catalog 1 (default=10).
    fit_bins2: array, None
        Bins for mass of catalog 2 (default=30).
    fit_legend_kwargs: dict, None
        Additional arguments for plt.legend.
    fit_bindata_kwargs: dict, None
        Additional arguments for pylab.errorbar.
    fit_plt_kwargs: dict, None
        Additional arguments for plot of fit pylab.scatter.
    fit_label_components: tuple (of strings)
        Names of fitted components in fit line label, default=(xlabel, ylabel).

    Returns
    -------
    info: dict
        Information of data in the plots, it contains the sections:

            * `ax`: ax used in the plot.
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
    cat1, cat2: clevar.ClCatalog
        ClCatalogs with matching information.
    matching_type: str
        Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
    log_mass: bool
        Log scale for mass
    color1: bool
        Use catalog 1 for color. If false uses catalog 2
    add_err: bool
        Add errorbars
    mask1, mask2: array, None
        Masks for clusters 1(2), must have size=cat1(2).size

    Other Parameters
    ----------------
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
    xlabel, ylabel: str
        Label of x/y axis (default=cat1.labels['mass']/cat2.labels['mass']).
    **fit_kwargs:
        Other fit arguments (see `fit_*` paramters in `scaling.mass` for more info).

    Returns
    -------
    info: dict
        Information of data in the plots, it contains the sections:

            * `fig`: `matplotlib.figure.Figure` object.
            * `axes`: `matplotlib.axes` used in the plot.
            * `ax_cb` (optional): ax of colorbar
            * `binned_data` (optional): input data for fitting \
            (see `scaling.mass` for more info).
            * `fit` (optional): fitting output dictionary \
            (see `scaling.mass` for more info).
            * `plots` (optional): fit and binning plots \
            (see `scaling.mass` for more info).
    """
    kwargs['fit_log'] = kwargs.get('fit_log', log_mass)
    return catalog_funcs.plot(cat1, cat2, matching_type, col='mass', col_color='z',
                              xscale='log' if log_mass else 'linear',
                              yscale='log' if log_mass else 'linear',
                              color1=color1, **kwargs)


def mass_density(cat1, cat2, matching_type, log_mass=True, **kwargs):
    """
    Scatter plot with errorbars and color based on point density

    Parameters
    ----------
    cat1, cat2: clevar.ClCatalog
        ClCatalogs with matching information.
    matching_type: str
        Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
    log_mass: bool
        Log scale for mass
    bins1, bins2: array, None
        Bins for mass of catalogs 1 and 2 (for density colors).
    add_err: bool
        Add errorbars
    mask1, mask2: array, None
        Masks for clusters 1(2), must have size=cat1(2).size

    Other Parameters
    ----------------
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
    ax_rotation: float
        Angle (in degrees) for rotation of axis of binning. Overwrites use of (bins1, bins2)
    rotation_resolution: int
        Number of bins to be used when ax_rotation!=0.
    xlabel, ylabel: str
        Label of x/y axis (default=cat1.labels['mass']/cat2.labels['mass']).
    **fit_kwargs:
        Other fit arguments (see `fit_*` paramters in `scaling.mass` for more info).

    Returns
    -------
    info: dict
        Information of data in the plots, it contains the sections:

            * `ax`: ax used in the plot.
            * `ax_cb` (optional): ax of colorbar
            * `binned_data` (optional): input data for fitting \
            (see `scaling.mass` for more info).
            * `fit` (optional): fitting output dictionary \
            (see `scaling.mass` for more info).
            * `plots` (optional): fit and binning plots \
            (see `scaling.mass` for more info).
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
    cat1, cat2: clevar.ClCatalog
        ClCatalogs with matching information.
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
    mask1, mask2: array, None
        Masks for clusters 1(2), must have size=cat1(2).size

    Other Parameters
    ----------------
    plt_kwargs: dict, None
        Additional arguments for pylab.scatter
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
    label_fmt: str
        Format the values of binedges (ex: '.2f')
    xlabel, ylabel: str
        Label of x/y axis (default=cat1.labels['mass']/cat2.labels['mass']).
    **fit_kwargs:
        Other fit arguments (see `fit_*` paramters in `scaling.mass` for more info).

    Returns
    -------
    info: dict
        Information of data in the plots, it contains the sections:

            * `fig`: `matplotlib.figure.Figure` object.
            * `axes`: `matplotlib.axes` used in the plot.
            * `binned_data` (optional): input data for fitting \
            (see `scaling.mass` for more info).
            * `fit` (optional): fitting output dictionary \
            (see `scaling.mass` for more info).
            * `plots` (optional): fit and binning plots \
            (see `scaling.mass` for more info).
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
    cat1, cat2: clevar.ClCatalog
        ClCatalogs with matching information.
    matching_type: str
        Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
    redshift_bins: int, array, int
        Redshift bins to make panels
    log_mass: bool
        Log scale for mass
    panel_cat1: bool
        Used catalog 1 for col_panel. If false uses catalog 2
    bins1, bins2: array, None
        Bins for mass of catalogs 1 and 2 (for density colors).
    add_err: bool
        Add errorbars
    mask1, mask2: array, None
        Masks for clusters 1(2), must have size=cat1(2).size

    Other Parameters
    ----------------
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
    fig_kwargs: dict, None
        Additional arguments for plt.subplots
    add_label: bool
        Add bin label to panel
    label_format: function
        Function to format the values of the bins
    label_fmt: str
        Format the values of binedges (ex: '.2f')
    xlabel, ylabel: str
        Label of x/y axis (default=cat1.labels['mass']/cat2.labels['mass']).
    **fit_kwargs:
        Other fit arguments (see `fit_*` paramters in `scaling.mass` for more info).

    Returns
    -------
    info: dict
        Information of data in the plots, it contains the sections:

            * `fig`: `matplotlib.figure.Figure` object.
            * `axes`: `matplotlib.axes` used in the plot.
            * `ax_cb` (optional): ax of colorbar
            * `binned_data` (optional): input data for fitting \
            (see `scaling.mass` for more info).
            * `fit` (optional): fitting output dictionary \
            (see `scaling.mass` for more info).
            * `plots` (optional): fit and binning plots \
            (see `scaling.mass` for more info).
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
    cat1, cat2: clevar.ClCatalog
        ClCatalogs with matching information.
    matching_type: str
        Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
    log_mass: bool
        Log scale for mass
    bins1, bins2: array, None
        Bins for mass of catalogs 1 and 2.
    metrics_mode: str
        Mode to run metrics (default=`log`). Options are:

            * `simple` : metrics for `values2`.
            * `log` : metrics for `log10(values2)`.
            * `diff` : metrics for `values2-values1`.
            * `diff_log` : metrics for `log10(values2)-log10(values1)`.
            * `diff_z` : metrics for `(values2-values1)/(1+values1)`.

    metrics: list
        List of mettrics to be plotted (default=[`std`]). Possibilities are:

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
    fig_kwargs: dict, None
        Additional arguments for plt.subplots
    metrics_kwargs: dict, None
        Dictionary of dictionary configs for each metric plots.
    legend_kwargs: dict, None
        Additional arguments for plt.legend
    label1, label2: str
        Label for catalog 1/2 masses.
    scale1, scale2: str
        Scale of catalog 1/2 masses.

    Returns
    -------
    info: dict
        Information of data in the plots, it contains the sections:

            * `fig`: `matplotlib.figure.Figure` object.
            * `axes`: `matplotlib.axes` used in the plot.
    """
    kwargs['mode'] = kwargs.get('mode', 'log')
    kwargs['metrics'] = kwargs.get('metrics', ['std'])
    kwargs['metrics_kwargs'] = deep_update(
        {'std':{'label': r'$\sigma_{\log(M)}$'}},
        kwargs.get('metrics_kwargs', {}))
    return catalog_funcs.plot_metrics(cat1, cat2, matching_type, col='mass',
                                      xscale='log' if log_mass else 'linear',
                                      yscale='log' if log_mass else 'linear',
                                      **kwargs)


def mass_density_metrics(cat1, cat2, matching_type, log_mass=True, **kwargs):
    """
    Scatter plot with errorbars and color based on point density with scatter and bias panels

    Parameters
    ----------
    cat1, cat2: clevar.ClCatalog
        ClCatalogs with matching information.
    matching_type: str
        Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
    log_mass: bool
        Log scale for mass
    bins1, bins2: array, None
        Bins for mass of catalogs 1 and 2.
    metrics_mode: str
        Mode to run metrics (default=`log`). Options are:

            * `simple` : metrics for `values2`.
            * `log` : metrics for `log10(values2)`.
            * `diff` : metrics for `values2-values1`.
            * `diff_log` : metrics for `log10(values2)-log10(values1)`.
            * `diff_z` : metrics for `(values2-values1)/(1+values1)`.

    metrics: list
        List of mettrics to be plotted (default=[`std`]). Possibilities are:

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
    fig_kwargs: dict, None
        Additional arguments for plt.subplots
    ax_rotation: float
        Angle (in degrees) for rotation of axis of binning. Overwrites use of (bins1, bins2) on main plot.
    rotation_resolution: int
        Number of bins to be used when ax_rotation!=0.
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
    fig_pos: tuple
        List with edges for the figure. Must be in format (left, bottom, right, top)
    fig_frac: tuple
        Sizes of each panel in the figure. Must be in the format (main_panel, gap, colorbar)
        and have values: [0, 1]. Colorbar is only used with add_cb key.
    **fit_kwargs:
        Other fit arguments (see `fit_*` paramters in `scaling.mass` for more info).

    Returns
    -------
    info: dict
        Information of data in the plots, it contains the sections:

            * `fig`: `matplotlib.figure.Figure` object.
            * `axes`: dictionary with each ax of the plot.
            * `metrics`: dictionary with the plots for each metric.
    """
    kwargs['metrics_mode'] = kwargs.get('mode', 'log')
    kwargs['metrics'] = kwargs.get('metrics', ['std'])
    kwargs['metrics_kwargs'] = deep_update(
        {'std':{'label': r'$\sigma_{\log(M)}$'}},
        kwargs.get('metrics_kwargs', {}))
    kwargs['fit_log'] = kwargs.get('fit_log', log_mass)
    return catalog_funcs.plot_density_metrics(cat1, cat2, matching_type, col='mass',
                                              xscale='log' if log_mass else 'linear',
                                              yscale='log' if log_mass else 'linear',
                                              **kwargs)


def mass_dist(cat1, cat2, matching_type, mass_bins_dist=30, mass_bins=5, redshift_bins=5,
              log_mass=True, transpose=False, **kwargs):
    """
    Plot distribution of a cat1 mass, binned by the cat2 mass (in panels),
    with option for cat2 redshift bins (in lines).

    Parameters
    ----------
    cat1, cat2: clevar.ClCatalog
        ClCatalogs with matching information.
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

    Other Parameters
    ----------------
    fig_kwargs: dict, None
        Additional arguments for plt.subplots
    shape: str
        Shape of the lines. Can be steps or line.
    plt_kwargs: dict, None
        Additional arguments for pylab.plot
    line_kwargs_list: list, None
        List of additional arguments for plotting each line (using pylab.plot).
        Must have same size as len(bins_aux)-1
    legend_kwargs: dict, None
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


def mass_dist_self(cat, mass_bins_dist=30, mass_bins=5, redshift_bins=5,
                   log_mass=True, transpose=False, mask=None, **kwargs):
    """
    Plot distribution of a cat mass, binned by mass (in panels),
    with option for redshift bins (in lines).
    Is is useful to compare with mass_dist results.

    Parameters
    ----------
    cat1, cat2: clevar.ClCatalog
        ClCatalogs with matching information.
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

    Other Parameters
    ----------------
    fig_kwargs: dict, None
        Additional arguments for plt.subplots
    shape: str
        Shape of the lines. Can be steps or line.
    plt_kwargs: dict, None
        Additional arguments for pylab.plot
    line_kwargs_list: list, None
        List of additional arguments for plotting each line (using pylab.plot).
        Must have same size as len(bins_aux)-1
    legend_kwargs: dict, None
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
    cat1, cat2: clevar.ClCatalog
        ClCatalogs with matching information.
    matching_type: str
        Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
    col: str
        Name of column to be plotted
    bins1, bins2: array, None
        Bins for mass of catalogs 1 and 2 (for density colors).
    add_err: bool
        Add errorbars
    mask1, mask2: array, None
        Masks for clusters 1(2), must have size=cat1(2).size
    log_mass: bool
        Log scale for mass

    Other Parameters
    ----------------
    fig_kwargs: dict, None
        Additional arguments for plt.subplots
    ax_rotation: float
        Angle (in degrees) for rotation of axis of binning. Overwrites use of (bins1, bins2) on main plot.
    rotation_resolution: int
        Number of bins to be used when ax_rotation!=0.
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
    fig_pos: tuple
        List with edges for the figure. Must be in format (left, bottom, right, top)
    fig_frac: tuple
        Sizes of each panel in the figure. Must be in the format (main_panel, gap, colorbar)
        and have values: [0, 1]. Colorbar is only used with add_cb key.
    **fit_kwargs:
        Other fit arguments (see `fit_*` paramters in `scaling.mass` for more info).

    Returns
    -------
    info: dict
        Information of data in the plots, it contains the sections:

            * `fig`: `matplotlib.figure.Figure` object.
            * `axes`: dictionary with each ax of the plot.
            * `binned_data` (optional): input data for fitting \
            (see `scaling.mass` for more info).
            * `fit` (optional): fitting output dictionary \
            (see `scaling.mass` for more info).
            * `plots` (optional): fit and binning plots \
            (see `scaling.mass` for more info).
    """
    kwargs['fit_log'] = kwargs.get('fit_log', True)
    kwargs['xscale'] = 'log' if log_mass else 'linear'
    kwargs['yscale'] = 'log' if log_mass else 'linear'
    kwargs['fit_log'] = kwargs.get('fit_log', log_mass)
    return catalog_funcs.plot_density_dist(cat1, cat2, matching_type, col='mass', **kwargs)
