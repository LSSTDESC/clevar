"""@file clevar/match_metrics/recovery/catalog_funcs.py

Main recovery functions using catalogs, wrapper of array_funcs functions
"""

import numpy as np
import pylab as plt

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
    matching_type_conv = matching_type.replace("cat1", "self").replace("cat2", "other")
    is_matched = cat.get_matching_mask(matching_type_conv)
    # mask_ to apply mask and mask_unmatched
    use_mask = none_val(mask, True) * (~(~is_matched * none_val(mask_unmatched, False)))
    return use_mask, is_matched


def _plot_base(
    pltfunc, cat, col1, col2, bins1, bins2, matching_type, mask=None, mask_unmatched=None, **kwargs
):
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
    return pltfunc(
        cat[col1][mask_], cat[col2][mask_], edges1, edges2, is_matched=is_matched[mask_], **kwargs
    )


def plot(
    cat,
    col1,
    col2,
    bins1,
    bins2,
    matching_type,
    xlabel=None,
    ylabel=None,
    scale1="linear",
    **kwargs,
):
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
    plt_kwargs: dict, None
        Additional arguments for pylab.plot.
        It also includes the possibility of smoothening the line with `n_increase, scheme`
        arguments. See `clevar.utils.smooth_line` for details.
    lines_kwargs_list: list, None
        List of additional arguments for plotting each line (using pylab.plot).
        Must have same size as len(bins2)-1
    add_legend: bool
        Add legend of bins
    legend_format: function
        Function to format the values of the bins in legend
    legend_kwargs: dict, None
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
    info = _plot_base(array_funcs.plot, cat, col1, col2, bins1, bins2, matching_type, **kwargs)
    info["ax"].set_xlabel(xlabel if xlabel else f"${cat.labels[col1]}$")
    info["ax"].set_ylabel(ylabel if ylabel else "recovery rate")
    info["ax"].set_xscale(scale1)
    info["ax"].set_ylim(-0.01, 1.05)
    return info


def plot_panel(
    cat,
    col1,
    col2,
    bins1,
    bins2,
    matching_type,
    xlabel=None,
    ylabel=None,
    scale1="linear",
    **kwargs,
):
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
    xlabel: str
        Label of component 1. Default is col1.
    ylabel: str
        Label of recovery rate.
    scale1: str
        Scale of col 1 component
    plt_kwargs: dict, None
        Additional arguments for pylab.plot.
        It also includes the possibility of smoothening the line with `n_increase, scheme`
        arguments. See `clevar.utils.smooth_line` for details.
    panel_kwargs_list: list, None
        List of additional arguments for plotting each panel (using pylab.plot).
        Must have same size as len(bins2)-1
    fig_kwargs: dict, None
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
    info = _plot_base(
        array_funcs.plot_panel, cat, col1, col2, bins1, bins2, matching_type, **kwargs
    )
    ph.nice_panel(
        info["axes"],
        xlabel=none_val(xlabel, f"${cat.labels[col1]}$"),
        ylabel=none_val(ylabel, "recovery rate"),
        xscale=scale1,
        yscale="linear",
    )
    info["axes"].flatten()[0].set_ylim(-0.01, 1.05)
    return info


def plot2D(
    cat,
    col1,
    col2,
    bins1,
    bins2,
    matching_type,
    xlabel=None,
    ylabel=None,
    scale1="linear",
    scale2="linear",
    **kwargs,
):
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
    plt_kwargs: dict, None
        Additional arguments for pylab.pcolor.
    add_cb: bool
        Plot colorbar
    cb_kwargs: dict, None
        Colorbar arguments
    add_num: int
        Add numbers in each bin
    num_kwargs: dict, None
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
    # pylint: disable=invalid-name
    info = _plot_base(array_funcs.plot2D, cat, col1, col2, bins1, bins2, matching_type, **kwargs)
    info["ax"].set_xlabel(xlabel if xlabel else f"${cat.labels[col1]}$")
    info["ax"].set_ylabel(ylabel if ylabel else f"${cat.labels[col2]}$")
    info["ax"].set_xscale(scale1)
    info["ax"].set_yscale(scale2)
    return info


def skyplot(
    cat,
    matching_type,
    nside=256,
    nest=True,
    mask=None,
    mask_unmatched=None,
    auto_lim=False,
    ra_lim=None,
    dec_lim=None,
    recovery_label="Recovery Rate",
    fig=None,
    figsize=None,
    **kwargs,
):
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
        cat["ra"][mask_],
        cat["dec"][mask_],
        is_matched[mask_],
        nside=nside,
        nest=nest,
        auto_lim=auto_lim,
        ra_lim=ra_lim,
        dec_lim=dec_lim,
        recovery_label=recovery_label,
        fig=fig,
        figsize=figsize,
        **kwargs,
    )


#######
# ROC #
#######


def threshold_counts(values, thresholds):
    r"""Compute vectorized counts above given thresholds.

    Parameters
    ----------
    values: np.ndarray
        Values for counts
    thresholds: np.ndarray
        Thresholds for counts

    Returns
    -------
    np.ndarray
        Vectorized counts above thresholds
    """
    return np.cumsum(np.histogram(values, bins=np.append(thresholds, np.inf))[0][::-1])[::-1]


def get_rates_snr(is_matched, snr, snr_cat2, snr_thresholds):
    """Compute vectorized recovery rates above snr_thresholds.

    Parameters
    ----------
    is_matched: np.ndarray
        Mask pointing to matched objects
    snr: np.ndarray
        Signal-to-noise of the base catalog
    snr_cat2: np.ndarray
        Signal-to-noise of base catalog object matched to the target catalog
    snr_thresholds: np.ndarray
        Thresholds for snr computation

    Returns
    -------
    np.ndarray, np.ndarray
        Recovery rates for cat1 and cat2 above snr thresholds provided
    """
    num_cat_th = threshold_counts(snr, snr_thresholds)
    num_cat_mt_th = threshold_counts(snr[is_matched], snr_thresholds)

    _msk = num_cat_th > 0

    recovery1 = np.full(len(snr_thresholds), np.nan)
    recovery1[_msk] = num_cat_mt_th[_msk] / num_cat_th[_msk]

    recovery2 = threshold_counts(snr_cat2, snr_thresholds) / len(snr_cat2)

    return recovery1, recovery2


def plot_roc(cat1, cat2, matching_type, col_th, thresholds, **kwargs):
    """
    Plot redshift distance between matched clusters, binned by a second quantity.

    Parameters
    ----------
    cat1, cat2: clevar.ClCatalog
        ClCatalogs with matching information.
    matching_type: str
        Type of matching to be considered. Must be in
        'cross', 'self', 'other', 'multi_self', 'multi_other', 'multi_join'
        Method to assign a corresponding snr to catalog2. Options are: 'matching', 'max'
    col_th: str
        Column in catalog1 to be use for threshold values
    thresholds: np.ndarray
        Thresholds for computation
    mask1, mask2: array, None
        Masks for clusters 1(2), must have size=cat1(2).size

    Other parameters
    ----------------
    ax: matplotlib.axes
        Ax to add plot
    plt_kwargs: dict, None
        Additional arguments for pylab.plot.

    Returns
    -------
    info: dict
        Information of data in the plots, it contains the sections:

            * `ax`: ax used in the plot.
            * `data`: Recovery rates and thresholds used in the plot (rec1, rec2, ths).
    """
    mt_msk = cat1.get_matching_mask(matching_type)

    vals_th_cat2 = np.zeros(cat2.size)
    if "multi" in matching_type:
        if matching_type in ("multi_self", "multi_join"):
            vals_th_cat2 = np.maximum(
                vals_th_cat2,
                [
                    cat1[col_th][cat1.ids2inds(ids)].max() if len(ids) > 0 else 0
                    for ids in cat2["mt_multi_self"]
                ],
            )
        if matching_type in ("multi_other", "multi_join"):
            vals_th_cat2 = np.maximum(
                vals_th_cat2,
                [
                    cat1[col_th][cat1.ids2inds(ids)].max() if len(ids) > 0 else 0
                    for ids in cat2["mt_multi_other"]
                ],
            )
    else:
        mt_msk2 = cat2.ids2inds(cat1[f"mt_{matching_type}"][mt_msk])
        vals_th_cat2[mt_msk2] = cat1[col_th][mt_msk]

    mask1 = kwargs.get("mask1", None)
    if mask1 is None:
        mask1 = np.ones(cat1.size, dtype=bool)
    mask2 = kwargs.get("mask2", None)
    if mask2 is None:
        mask2 = np.ones(cat2.size, dtype=bool)

    # if rm_msk_from_mt:
    #    cl_mt_msk *= clean_mt(_cat, halos, matching_type_cat, mask_halo)
    #    print(f"     + msk h : {cl_mt_msk.sum():,}")

    ax = kwargs.get("ax", None)
    info = {
        "data": [
            *get_rates_snr(mt_msk[mask1], cat1[col_th][mask1], vals_th_cat2[mask2], thresholds),
            thresholds,
        ],
        "ax": plt.axes() if ax is None else ax,
    }

    info["ax"].plot(*info["data"][:2], **kwargs.get("plt_kwargs", {}))
    info["ax"].set_xlabel("Recovery rate cat1")
    info["ax"].set_ylabel("Recovery rate cat2")
    return info
