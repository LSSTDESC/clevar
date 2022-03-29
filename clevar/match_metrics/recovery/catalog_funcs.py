"""@file clevar/match_metrics/recovery/catalog_funcs.py

Main recovery functions using catalogs, wrapper of array_funcs functions
"""
import numpy as np
from ...utils import none_val
from .. import plot_helper as ph
from . import array_funcs


def _rec_masks(cat, matching_type, mask=None, mask_unmatched=None):
    """
    Get masks to be used in recovery rate.

    Parameters
    ----------
    cat: clevar.ClCatalog
        ClCatalog with matching information
    matching_type: str
        Type of matching to be considered. Must be in:
        'cross', 'cat1', 'cat2', 'multi_cat1', 'multi_cat2', 'multi_join'
    mask: array
        Mask of unwanted clusters
    mask_unmatched: array
        Mask of unwanted unmatched clusters (ex: out of footprint)

    Returns
    -------
    use_mask: array
        Mask of clusters to be used.
    is_matched: array
        Mask for matched clusters (use_mask has to be applied to it).
    """
    # convert matching type to the values expected by get_matching_mask
    matching_type_conv = matching_type.replace('cat1', 'self').replace('cat2', 'other')
    is_matched = cat.get_matching_mask(matching_type_conv)
    # mask_ to apply mask and mask_unmatched
    use_mask = none_val(mask, True)*(~(~is_matched*none_val(mask_unmatched, False)))
    return use_mask, is_matched


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
    col1, col2: str
        Names of columns 1 and 2.
    bins1, bins2: array, int
        Bins for components 1 and 2.
    matching_type: str
        Type of matching to be considered. Must be in:
        'cross', 'cat1', 'cat2', 'multi_cat1', 'multi_cat2', 'multi_join'
    mask: array
        Mask of unwanted clusters
    mask_unmatched: array
        Mask of unwanted unmatched clusters (ex: out of footprint)
    **kwargs:
        Additional arguments to be passed to pltfunc

    Returns
    -------
    Same as pltfunc
    """
    mask_, is_matched = _rec_masks(cat, matching_type, mask, mask_unmatched)
    # make sure bins stay consistent regardless of mask
    edges1, edges2 = np.histogram2d(cat[col1], cat[col2], bins=(bins1, bins2))[1:]
    return pltfunc(cat[col1][mask_], cat[col2][mask_], edges1, edges2,
                   is_matched=is_matched[mask_], **kwargs)


def plot(cat, col1, col2, bins1, bins2, matching_type,
         xlabel=None, ylabel=None, scale1='linear', **kwargs):
    """
    Plot recovery rate as lines, with each line binned by bins1 inside a bin of bins2.

    Parameters
    ----------
    cat: clevar.ClCatalog
        ClCatalog with matching information
    col1, col2: str
        Names of columns 1 and 2.
    bins1, bins2: array, int
        Bins for components 1 and 2.
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
    info = _plot_base(
        array_funcs.plot, cat, col1, col2, bins1, bins2, matching_type,
        **kwargs)
    info['ax'].set_xlabel(xlabel if xlabel else f'${cat.labels[col1]}$')
    info['ax'].set_ylabel(ylabel if ylabel else 'recovery rate')
    info['ax'].set_xscale(scale1)
    info['ax'].set_ylim(-.01, 1.05)
    return info


def plot_panel(cat, col1, col2, bins1, bins2, matching_type,
               xlabel=None, ylabel=None, scale1='linear', **kwargs):
    """
    Plot recovery rate as lines in panels, with each line binned by bins1
    and each panel is based on the data inside a bins2 bin.

    Parameters
    ----------
    cat: clevar.ClCatalog
        ClCatalog with matching information
    col1, col2: str
        Names of columns 1 and 2.
    bins1, bins2: array, int
        Bins for components 1 and 2.
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
    info = _plot_base(array_funcs.plot_panel,
            cat, col1, col2, bins1, bins2, matching_type, **kwargs)
    ph.nice_panel(info['axes'], xlabel=none_val(xlabel, f'${cat.labels[col1]}$'),
                  ylabel=none_val(ylabel, 'recovery rate'),
                  xscale=scale1, yscale='linear')
    info['axes'].flatten()[0].set_ylim(-.01, 1.05)
    return info


def plot2D(cat, col1, col2, bins1, bins2, matching_type,
           xlabel=None, ylabel=None, scale1='linear', scale2='linear',
           **kwargs):
    """
    Plot recovery rate as in 2D bins.

    Parameters
    ----------
    cat: clevar.ClCatalog
        ClCatalog with matching information
    col1, col2: str
        Names of columns 1 and 2.
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
    xlabel, ylabel: str
        Labels of components 1 and 2. Default is col1, col2.
    scale1, scale2: str
        Scales of col 1, 2 components.
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
    info = _plot_base(array_funcs.plot2D,
            cat, col1, col2, bins1, bins2, matching_type, **kwargs)
    info['ax'].set_xlabel(xlabel if xlabel else f'${cat.labels[col1]}$')
    info['ax'].set_ylabel(ylabel if ylabel else f'${cat.labels[col2]}$')
    info['ax'].set_xscale(scale1)
    info['ax'].set_yscale(scale2)
    return info


def skyplot(cat, matching_type, nside=256, nest=True, mask=None, mask_unmatched=None,
            auto_lim=False, ra_lim=None, dec_lim=None, recovery_label='Recovery Rate',
            fig=None, figsize=None, **kwargs):
    """
    Plot recovery rate in healpix pixels.

    Parameters
    ----------
    cat: clevar.ClCatalog
        ClCatalog with matching information
    matching_type: str
        Type of matching to be considered. Must be in:
        'cross', 'cat1', 'cat2', 'multi_cat1', 'multi_cat2', 'multi_join'
    nside: int
        Healpix nside
    nest: bool
        If ordering is nested
    mask: array
        Mask of unwanted clusters
    mask_unmatched: array
        Mask of unwanted unmatched clusters (ex: out of footprint)
    auto_lim: bool
        Set automatic limits for ra/dec.
    ra_lim: None, list
        Min/max RA for plot.
    dec_lim: None, list
        Min/max DEC for plot.
    recovery_label: str
        Lable for colorbar. Default: 'recovery rate'
    fig: matplotlib.figure.Figure, None
        Matplotlib figure object. If not provided a new one is created.
    figsize: tuple
        Width, height in inches (float, float). Default value from hp.cartview.
    **kwargs:
        Extra arguments for hp.cartview:

            * xsize (int) : The size of the image. Default: 800
            * title (str) : The title of the plot. Default: None
            * min (float) : The minimum range value
            * max (float) : The maximum range value
            * remove_dip (bool) : If :const:`True`, remove the dipole+monopole
            * remove_mono (bool) : If :const:`True`, remove the monopole
            * gal_cut (float, scalar) : Symmetric galactic cut for \
            the dipole/monopole fit. Removes points in latitude range \
            [-gal_cut, +gal_cut]
            * format (str) : The format of the scale label. Default: '%g'
            * cbar (bool) : Display the colorbar. Default: True
            * notext (bool) : If True, no text is printed around the map
            * norm ({'hist', 'log', None}) : Color normalization, \
            hist= histogram equalized color mapping, log= logarithmic color \
            mapping, default: None (linear color mapping)
            * cmap (a color map) :  The colormap to use (see matplotlib.cm)
            * badcolor (str) : Color to use to plot bad values
            * bgcolor (str) : Color to use for background
            * margins (None or sequence) : Either None, or a \
            sequence (left,bottom,right,top) giving the margins on \
            left,bottom,right and top of the axes. Values are relative to \
            figure (0-1). Default: None

    Returns
    -------
    info: dict
        Information of data in the plots, it contains the sections:

            * `fig` (matplotlib.pyplot.figure): Figure of the plot. The main can be accessed at\
            `info['fig'].axes[0]`, and the colorbar at `info['fig'].axes[1]`.
            * `nc_pix`: Dictionary with the number of clusters in each pixel.
            * `nc_mt_pix`: Dictionary with the number of matched clusters in each pixel.
    """
    mask_, is_matched = _rec_masks(cat, matching_type, mask, mask_unmatched)
    return array_funcs.skyplot(
        cat['ra'][mask_], cat['dec'][mask_], is_matched[mask_],
        nside=nside, nest=nest, auto_lim=auto_lim, ra_lim=ra_lim, dec_lim=dec_lim,
        recovery_label=recovery_label, fig=fig, figsize=figsize, **kwargs)


def _plot_fscore_base(pltfunc, cat1, cat1_col1, cat1_col2, cat1_bins1, cat1_bins2,
                      cat2, cat2_col1, cat2_col2, cat2_bins1, cat2_bins2, matching_type,
                      mask1=None, mask2=None, mask_unmatched1=None, mask_unmatched2=None,
                      **kwargs):
    """
    Adapts local function to use a ArrayFuncs function.

    Parameters
    ----------
    pltfunc: function
        ArrayFuncs function
    cat: clevar.ClCatalog
        ClCatalog with matching information
    col1, col2: str
        Names of columns 1 and 2.
    bins1, bins2: array, int
        Bins for components 1 and 2.
    matching_type: str
        Type of matching to be considered. Must be in:
        'cross', 'cat1', 'cat2', 'multi_cat1', 'multi_cat2', 'multi_join'
    mask: array
        Mask of unwanted clusters
    mask_unmatched: array
        Mask of unwanted unmatched clusters (ex: out of footprint)
    **kwargs:
        Additional arguments to be passed to pltfunc

    Returns
    -------
    Same as pltfunc
    """
    c1_mask, c1_is_matched = _rec_masks(cat1, matching_type, mask1, mask_unmatched1)
    c2_mask, c2_is_matched = _rec_masks(cat2, matching_type, mask2, mask_unmatched2)
    # make sure bins stay consistent regardless of mask
    c1_edges1, c1_edges2 = np.histogram2d(cat1[cat1_col1], cat1[cat1_col2],
                                          bins=(cat1_bins1, cat1_bins2))[1:]
    c2_edges1, c2_edges2 = np.histogram2d(cat2[cat2_col1], cat2[cat2_col2],
                                          bins=(cat2_bins1, cat2_bins2))[1:]
    return pltfunc(
        cat1[cat1_col1][c1_mask], cat1[cat1_col2][c1_mask], c1_edges1, c1_edges2, c1_is_matched[c1_mask],
        cat2[cat2_col1][c2_mask], cat2[cat2_col2][c2_mask], c2_edges1, c2_edges2, c2_is_matched[c2_mask],
        **kwargs)


def plot_fscore(cat1, cat1_col1, cat1_col2, cat1_bins1, cat1_bins2,
                cat2, cat2_col1, cat2_col2, cat2_bins1, cat2_bins2,
                matching_type, beta=1, pref='cat1', xlabel=None, ylabel=None,
                scale1='linear', **kwargs):
    """
    Plot recovery rate as lines, with each line binned by bins1 inside a bin of bins2.

    Parameters
    ----------
    cat: clevar.ClCatalog
        ClCatalog with matching information
    col1, col2: str
        Names of columns 1 and 2.
    bins1, bins2: array, int
        Bins for components 1 and 2.
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
    info = _plot_fscore_base(array_funcs.plot_fscore,
                             cat1, cat1_col1, cat1_col2, cat1_bins1, cat1_bins2,
                             cat2, cat2_col1, cat2_col2, cat2_bins1, cat2_bins2,
                             matching_type, beta=beta, pref=pref, **kwargs)
    for ax in info['axes'][-1]:
        ax.set_xlabel(xlabel if xlabel else f'${cat1.labels[cat1_col1]}$')
    for ax in info['axes'][:, 0]:
        ax.set_ylabel(ylabel if ylabel else f'$F_{beta}$ score')
    for ax in info['axes'].flatten():
        ax.set_xscale(scale1)
        ax.set_ylim(-.01, 1.05)
    return info
