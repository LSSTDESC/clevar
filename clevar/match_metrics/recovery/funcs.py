"""@file clevar/match_metrics/recovery/funcs.py

Main recovery functions for mass and redshift plots,
wrapper of catalog_funcs functions
"""
import numpy as np
from .. import plot_helper as ph
from . import catalog_funcs

def _plot_base(pltfunc, cat, matching_type, redshift_bins, mass_bins,
               transpose=False, **kwargs):
    """
    Adapts a ClCatalogFuncs function for main functions using mass and redshift.

    Parameters
    ----------
    pltfunc: function
        ClCatalogFuncs function
    cat: clevar.ClCatalog
        ClCatalog with matching information
    matching_type: str
        Type of matching to be considered. Must be in:
        'cross', 'cat1', 'cat2', 'multi_cat1', 'multi_cat2', 'multi_join'
    redshift_bins: array, int
        Bins for redshift
    mass_bins: array, int
        Bins for mass
    transpose: bool
        Transpose mass and redshift in plots
    **kwargs:
        Additional arguments to be passed to pltfunc

    Returns
    -------
    Same as pltfunc
    """
    args = ('mass', 'z', mass_bins, redshift_bins) if transpose\
        else ('z', 'mass', redshift_bins, mass_bins)
    return pltfunc(cat, *args, matching_type, **kwargs)


def plot(cat, matching_type, redshift_bins, mass_bins, transpose=False, log_mass=True,
         redshift_label=None, mass_label=None, recovery_label=None, **kwargs):
    """
    Plot recovery rate as lines, with each line binned by redshift inside a mass bin.

    Parameters
    ----------
    cat: clevar.ClCatalog
        ClCatalog with matching information
    matching_type: str
        Type of matching to be considered. Must be in:
        'cross', 'cat1', 'cat2', 'multi_cat1', 'multi_cat2', 'multi_join'
    redshift_bins: array, int
        Bins for redshift
    mass_bins: array, int
        Bins for mass
    transpose: bool
        Transpose mass and redshift in plots
    log_mass: bool
        Plot mass in log scale
    mask: array
        Mask of unwanted clusters
    mask_unmatched: array
        Mask of unwanted unmatched clusters (ex: out of footprint)


    Other parameters
    ----------------
    shape: str
        Shape of the lines. Can be steps or line.
    mass_label: str
        Label for mass.
    redshift_label: str
        Label for redshift.
    recovery_label: str
        Label for recovery rate.
    ax: matplotlib.axes
        Ax to add plot
    plt_kwargs: dict
        Additional arguments for pylab.plot
    lines_kwargs_list: list, None
        List of additional arguments for plotting each line (using pylab.plot).
        Must have same size as len(bins2)-1
    add_legend: bool
        Add legend of bins
    legend_format: function
        Function to format the values of the bins in legend
    legend_fmt: str
        Format the values of binedges (ex: '.2f')
    legend_kwargs: dict
        Additional arguments for pylab.legend

    Returns
    -------
    info: dict
        Information of data in the plots, it contains the sections:

            * `ax`: ax used in the plot.
            * `data`: Binned data used in the plot. It has the sections:

                * `recovery`: Recovery rate binned with (bin1, bin2).\
                bins where no cluster was found have nan value.
                * `edges1`: The bin edges along the first dimension.
                * `edges2`: The bin edges along the second dimension.
                * `counts`: Counts of all clusters in bins.
                * `matched`: Counts of matched clusters in bins.
    """
    legend_fmt = kwargs.pop("legend_fmt", ".1f" if log_mass*(not transpose) else ".2f")
    kwargs['legend_format'] = kwargs.get('legend_format',
        lambda v: f'10^{{%{legend_fmt}}}'%np.log10(v) if log_mass*(not transpose)\
             else f'%{legend_fmt}'%v)
    return _plot_base(catalog_funcs.plot, cat, matching_type,
                      redshift_bins, mass_bins, transpose,
                      scale1='log' if log_mass*transpose else 'linear',
                      xlabel=mass_label if transpose else redshift_label,
                      ylabel=recovery_label,
                      **kwargs)


def plot_panel(cat, matching_type, redshift_bins, mass_bins, transpose=False, log_mass=True,
               redshift_label=None, mass_label=None, recovery_label=None, **kwargs):
    """
    Plot recovery rate as lines in panels, with each line binned by redshift
    and each panel is based on the data inside a mass bin.

    Parameters
    ----------
    cat: clevar.ClCatalog
        ClCatalog with matching information
    matching_type: str
        Type of matching to be considered. Must be in:
        'cross', 'cat1', 'cat2', 'multi_cat1', 'multi_cat2', 'multi_join'
    redshift_bins: array, int
        Bins for redshift
    mass_bins: array, int
        Bins for mass
    transpose: bool
        Transpose mass and redshift in plots
    log_mass: bool
        Plot mass in log scale
    mask: array
        Mask of unwanted clusters
    mask_unmatched: array
        Mask of unwanted unmatched clusters (ex: out of footprint)

    Other parameters
    ----------------
    shape: str
        Shape of the lines. Can be steps or line.
    mass_label: str
        Label for mass.
    redshift_label: str
        Label for redshift.
    recovery_label: str
        Label for recovery rate.
    ax: matplotlib.axes
        Ax to add plot
    plt_kwargs: dict
        Additional arguments for pylab.plot
    panel_kwargs_list: list, None
        List of additional arguments for plotting each panel (using pylab.plot).
        Must have same size as len(bins2)-1
    fig_kwargs: dict
        Additional arguments for plt.subplots
    add_label: bool
        Add bin label to panel
    label_format: function
        Function to format the values of the bins
    label_fmt: str
        Format the values of binedges (ex: '.2f')

    Returns
    -------
    info: dict
        Information of data in the plots, it contains the sections:

            * `fig`: `matplotlib.figure.Figure` object.
            * `axes`: `matplotlib.axes` used in the plot.
            * `data`: Binned data used in the plot. It has the sections:

                * `recovery`: Recovery rate binned with (bin1, bin2).\
                bins where no cluster was found have nan value.
                * `edges1`: The bin edges along the first dimension.
                * `edges2`: The bin edges along the second dimension.
                * `counts`: Counts of all clusters in bins.
                * `matched`: Counts of matched clusters in bins.
    """
    log = log_mass*(not transpose)
    ph._set_label_format(kwargs, 'label_format', 'label_fmt',
                         log=log, default_fmt=".1f" if log else ".2f")
    return _plot_base(catalog_funcs.plot_panel, cat, matching_type,
                      redshift_bins, mass_bins, transpose,
                      scale1='log' if log_mass*transpose else 'linear',
                      xlabel=mass_label if transpose else redshift_label,
                      ylabel=recovery_label,
                      **kwargs)


def plot2D(cat, matching_type, redshift_bins, mass_bins, transpose=False, log_mass=True,
           redshift_label=None, mass_label=None, recovery_label=None, **kwargs):
    """
    Plot recovery rate as in 2D (redshift, mass) bins.

    Parameters
    ----------
    cat: clevar.ClCatalog
        ClCatalog with matching information
    matching_type: str
        Type of matching to be considered. Must be in:
        'cross', 'cat1', 'cat2', 'multi_cat1', 'multi_cat2', 'multi_join'
    redshift_bins: array, int
        Bins for redshift
    mass_bins: array, int
        Bins for mass
    transpose: bool
        Transpose mass and redshift in plots
    log_mass: bool
        Plot mass in log scale
    mask: array
        Mask of unwanted clusters
    mask_unmatched: array
        Mask of unwanted unmatched clusters (ex: out of footprint)

    Other parameters
    ----------------
    mass_label: str
        Label for mass.
    redshift_label: str
        Label for redshift.
    recovery_label: str
        Label for recovery rate.
    ax: matplotlib.axes
        Ax to add plot
    plt_kwargs: dict
        Additional arguments for pylab.plot
    add_cb: bool
        Plot colorbar
    cb_kwargs: dict
        Colorbar arguments
    add_num: int
        Add numbers in each bin
    num_kwargs: dict
        Arguments for number plot (used in plt.text)

    Returns
    -------
    info: dict
        Information of data in the plots, it contains the sections:

            * `ax`: ax used in the plot.
            * `cb` (optional): colorbar.
            * `data`: Binned data used in the plot. It has the sections:

                * `recovery`: Recovery rate binned with (bin1, bin2).\
                bins where no cluster was found have nan value.
                * `edges1`: The bin edges along the first dimension.
                * `edges2`: The bin edges along the second dimension.
                * `counts`: Counts of all clusters in bins.
                * `matched`: Counts of matched clusters in bins.
    """
    return _plot_base(catalog_funcs.plot2D, cat, matching_type,
                      redshift_bins, mass_bins, transpose,
                      scale1='log' if log_mass*transpose else 'linear',
                      scale2='log' if log_mass*(not transpose) else 'linear',
                      xlabel=mass_label if transpose else redshift_label,
                      ylabel=redshift_label if transpose else mass_label,
                      **kwargs)
