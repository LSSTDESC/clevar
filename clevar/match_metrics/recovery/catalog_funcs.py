"""@file clevar/match_metrics/recovery/catalog_funcs.py

Main recovery functions using catalogs, wrapper of array_funcs functions
"""
import numpy as np
from ...utils import none_val
from .. import plot_helper as ph
from . import array_funcs

def _plot_base(pltfunc, cat, col1, col2, bins1, bins2, matching_type,
               mask=None, mask_unmatched=None, **kwargs):
    """
    Adapts local function to use a ArrayFuncs function.

    Parameters
    ----------
    pltfunc: function
        ArrayFuncs function
    cat: clevar.ClCatalog
        ClCatalog with matching information
    col1: str
        Name of column 1
    col2: str
        Name of column 2
    bins1: array, int
        Bins for component 1
    bins2: array, int
        Bins for component 2
    matching_type: str
        Type of matching to be considered. Must be in:
        'cross', 'cat1', 'cat2', 'multi_cat1', 'multi_cat2', 'multi_join'
    mask: array
        Mask of unwanted clusters
    mask_unmatched: array
        Mask of unwanted unmatched clusters (ex: out of footprint)
    **kwargs:
        Additional arguments to be passed to pltfunc
    """
    # convert matching type to the values expected by get_matching_mask
    matching_type_conv = matching_type.replace('cat1', 'self').replace('cat2', 'other')
    is_matched = cat.get_matching_mask(matching_type_conv)
    # mask_ to apply mask and mask_unmatched
    mask_ = none_val(mask, True)*(~(~is_matched*none_val(mask_unmatched, False)))
    # make sure bins stay consistent regardless of mask
    edges1, edges2 = np.histogram2d(cat[col1], cat[col2], bins=(bins1, bins2))[1:]
    return pltfunc(cat[mask_][col1], cat[mask_][col2], edges1, edges2,
                   is_matched=is_matched[mask_], **kwargs)
def plot(cat, col1, col2, bins1, bins2, matching_type,
         xlabel=None, ylabel=None, scale1='linear', **kwargs):
    """
    Plot recovery rate as lines, with each line binned by bins1 inside a bin of bins2.

    Parameters
    ----------
    cat: clevar.ClCatalog
        ClCatalog with matching information
    col1: str
        Name of column 1
    col2: str
        Name of column 2
    bins1: array, int
        Bins for component 1
    bins2: array, int
        Bins for component 2
    matching_type: str
        Type of matching to be considered. Must be in:
        'cross', 'cat1', 'cat2', 'multi_cat1', 'multi_cat2', 'multi_join'
    mask: array
        Mask of unwanted clusters
    mask_unmatched: array
        Mask of unwanted unmatched clusters (ex: out of footprint)

    Other parameters
    ----------------
    shape: str
        Shape of the lines. Can be steps or line.
    ax: matplotlib.axes
        Ax to add plot
    xlabel: str
        Label of component 1. Default is col1.
    ylabel: str
        Label of recovery rate.
    scale1: str
        Scale of col 1 component
    plt_kwargs: dict
        Additional arguments for pylab.plot
    lines_kwargs_list: list, None
        List of additional arguments for plotting each line (using pylab.plot).
        Must have same size as len(bins2)-1
    add_legend: bool
        Add legend of bins
    legend_format: function
        Function to format the values of the bins in legend
    legend_kwargs: dict
        Additional arguments for pylab.legend

    Returns
    -------
    ax: matplotlib.axes
        Axis of the plot
    """
    ax = _plot_base(array_funcs.plot,
            cat, col1, col2, bins1, bins2, matching_type, **kwargs)
    ax.set_xlabel(xlabel if xlabel else f'${cat.labels[col1]}$')
    ax.set_ylabel(ylabel if ylabel else 'recovery rate')
    ax.set_xscale(scale1)
    ax.set_ylim(-.01, 1.05)
    return ax
def plot_panel(cat, col1, col2, bins1, bins2, matching_type,
               xlabel=None, ylabel=None, scale1='linear', **kwargs):
    """
    Plot recovery rate as lines in panels, with each line binned by bins1
    and each panel is based on the data inside a bins2 bin.

    Parameters
    ----------
    cat: clevar.ClCatalog
        ClCatalog with matching information
    col1: str
        Name of column 1
    col2: str
        Name of column 2
    bins1: array, int
        Bins for component 1
    bins2: array, int
        Bins for component 2
    matching_type: str
        Type of matching to be considered. Must be in:
        'cross', 'cat1', 'cat2', 'multi_cat1', 'multi_cat2', 'multi_join'
    mask: array
        Mask of unwanted clusters
    mask_unmatched: array
        Mask of unwanted unmatched clusters (ex: out of footprint)

    Other parameters
    ----------------
    shape: str
        Shape of the lines. Can be steps or line.
    ax: matplotlib.axes
        Ax to add plot
    xlabel: str
        Label of component 1. Default is col1.
    ylabel: str
        Label of recovery rate.
    scale1: str
        Scale of col 1 component
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

    Returns
    -------
    fig: matplotlib.figure.Figure
        `matplotlib.figure.Figure` object
    axes: array
        Axes with the panels
    """
    fig, axes = _plot_base(array_funcs.plot_panel,
            cat, col1, col2, bins1, bins2, matching_type, **kwargs)
    ph.nice_panel(axes, xlabel=none_val(xlabel, f'${cat.labels[col1]}$'),
                  ylabel=none_val(ylabel, 'recovery rate'),
                  xscale=scale1, yscale='linear')
    axes.flatten()[0].set_ylim(-.01, 1.05)
    return fig, axes
def plot2D(cat, col1, col2, bins1, bins2, matching_type,
           xlabel=None, ylabel=None, scale1='linear', scale2='linear',
           **kwargs):
    """
    Plot recovery rate as in 2D bins.

    Parameters
    ----------
    cat: clevar.ClCatalog
        ClCatalog with matching information
    col1: str
        Name of column 1
    col2: str
        Name of column 2
    bins1: array, int
        Bins for component 1
    bins2: array, int
        Bins for component 2
    matching_type: str
        Type of matching to be considered. Must be in:
        'cross', 'cat1', 'cat2', 'multi_cat1', 'multi_cat2', 'multi_join'
    mask: array
        Mask of unwanted clusters
    mask_unmatched: array
        Mask of unwanted unmatched clusters (ex: out of footprint)

    Other parameters
    ----------------
    ax: matplotlib.axes
        Ax to add plot
    xlabel: str
        Label of component 1. Default is col1.
    ylabel: str
        Label of component 2. Default is col2.
    scale1: str
        Scale of col 1 component
    scale2: str
        Scale of col 2 component
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
    ax: matplotlib.axes
        Axis of the plot
    matplotlib.colorbar.Colorbar
        Colorbar of the recovey rates
    """
    ax, cb = _plot_base(array_funcs.plot2D,
            cat, col1, col2, bins1, bins2, matching_type, **kwargs)
    ax.set_xlabel(xlabel if xlabel else f'${cat.labels[col1]}$')
    ax.set_ylabel(ylabel if ylabel else f'${cat.labels[col2]}$')
    ax.set_xscale(scale1)
    ax.set_yscale(scale2)
    return ax, cb
