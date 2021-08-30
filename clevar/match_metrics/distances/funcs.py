"""@file clevar/match_metrics/distances/funcs.py

Main distances functions, wrapper of catalog_funcs functions
"""
import numpy as np
from . import catalog_funcs

def central_position(cat1, cat2, matching_type, radial_bins=20, radial_bin_units='degrees', cosmo=None,
                     quantity_bins=None, bins=None, log_quantity=False, ax=None, **kwargs):
    """
    Plot recovery rate as lines, with each line binned by redshift inside a mass bin.

    Parameters
    ----------
    cat1: clevar.ClCatalog
        ClCatalog with matching information
    cat2: clevar.ClCatalog
        ClCatalog matched to
    matching_type: str
        Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
    radial_bins: array, int
        Bins for radial distances
    radial_bin_units: str
        Units of radial bins
    cosmo: clevar.Cosmology
        Cosmology (used if physical units required)
    quantity_bins: str
        Column to bin the data
    bins: array, int
        Bins for quantity
    log_quantity: bool
        Display label in log fmt
    mask1: array, None
        Mask for clusters 1 properties, must have size=cat1.size
    mask2: array, None
        Mask for clusters 2 properties, must have size=cat2.size

    Other parameters
    ----------------
    shape: str
        Shape of the lines. Can be steps or line.
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
            * `distances`: values of distances.
            * `data`: Binned data used in the plot. It has the sections:

                * `hist`: Binned angular distances with (distance_bins, bin2).\
                bins where no cluster was found have nan value.
                * `distance_bins`: The bin edges for distances.
                * `bins2` (optional): The bin edges along the second dimension.
    """
    legend_fmt = kwargs.pop("legend_fmt", ".1f" if log_quantity else ".2f")
    kwargs['legend_format'] = kwargs.get('legend_format',
        lambda v: f'10^{{%{legend_fmt}}}'%np.log10(v) if log_quantity else f'%{legend_fmt}'%v)
    kwargs['add_legend'] = kwargs.get('add_legend', True)*(bins is not None)
    return catalog_funcs.central_position(
        cat1, cat2, matching_type, radial_bins=radial_bins,
        radial_bin_units=radial_bin_units, cosmo=cosmo, col2=quantity_bins,
        bins2=bins, ax=ax, **kwargs)

def redshift(cat1, cat2, matching_type, redshift_bins=20, normalize=None,
             quantity_bins=None, bins=None, log_quantity=False,
             ax=None, **kwargs):
    """
    Plot recovery rate as lines, with each line binned by redshift inside a mass bin.

    Parameters
    ----------
    cat1: clevar.ClCatalog
        ClCatalog with matching information
    cat2: clevar.ClCatalog
        ClCatalog matched to
    matching_type: str
        Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
    redshift_bins: array, int
        Bins for redshift distances
    normalize: str, None
        Normalize difference by (1+z). Can be 'cat1' for (1+z1), 'cat2' for (1+z2)
        or 'mean' for (1+(z1+z2)/2).
    quantity_bins: str
        Column to bin the data
    bins: array, int
        Bins for quantity
    log_quantity: bool
        Display label in log fmt
    mask1: array, None
        Mask for clusters 1 properties, must have size=cat1.size
    mask2: array, None
        Mask for clusters 2 properties, must have size=cat2.size

    Other parameters
    ----------------
    shape: str
        Shape of the lines. Can be steps or line.
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
            * `distances`: values of distances.
            * `data`: Binned data used in the plot. It has the sections:

                * `hist`: Binned redshift distances with (distance_bins, bin2).\
                bins where no cluster was found have nan value.
                * `distance_bins`: The bin edges for distances.
                * `bins2` (optional): The bin edges along the second dimension.
    """
    legend_fmt = kwargs.pop("legend_fmt", ".1f" if log_quantity else ".2f")
    kwargs['legend_format'] = kwargs.get('legend_format',
        lambda v: f'10^{{%{legend_fmt}}}'%np.log10(v) if log_quantity else f'%{legend_fmt}'%v)
    kwargs['add_legend'] = kwargs.get('add_legend', True)*(bins is not None)
    return catalog_funcs.redshift(
        cat1, cat2, matching_type, redshift_bins=redshift_bins,
        normalize=normalize, col2=quantity_bins, bins2=bins, ax=ax, **kwargs)
