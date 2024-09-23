"""@file clevar/match_metrics/recovery/array_funcs.py

Main recovery functions using arrays.
"""
import numpy as np
import healpy as hp

from ...utils import none_val, updated_dict, index_list
from .. import plot_helper as ph
from ..plot_helper import plt


def get_recovery_rate(values1, values2, bins1, bins2, is_matched):
    """
    Get recovery rate binned in 2 components

    Parameters
    ----------
    values1, values2: array
        Component 1 and 2.
    bins1, bins2: array, int
        Bins for components 1 and 2.
    is_matched: array (boolean)
        Boolean array indicating matching status

    Returns
    -------
    dict
        Binned recovery rate. Its sections are:

            * `recovery`: Recovery rate binned with (bin1, bin2).\
            bins where no cluster was found have nan value.
            * `edges1`: The bin edges along the first dimension.
            * `edges2`: The bin edges along the second dimension.
            * `counts`: Counts of all clusters in bins.
            * `matched`: Counts of matched clusters in bins.
    """
    hist_counts, edges1, edges2 = np.histogram2d(values1, values2, bins=(bins1, bins2))
    hist_matched = np.histogram2d(values1[is_matched], values2[is_matched], bins=(edges1, edges2))[
        0
    ]
    recovery = np.zeros(hist_counts.shape)
    recovery[:] = np.nan
    recovery[hist_counts > 0] = hist_matched[hist_counts > 0] / hist_counts[hist_counts > 0]
    return {
        "recovery": recovery,
        "edges1": edges1,
        "edges2": edges2,
        "matched": np.array(hist_matched, dtype=int),
        "counts": np.array(hist_counts, dtype=int),
    }


def plot(
    values1,
    values2,
    bins1,
    bins2,
    is_matched,
    shape="steps",
    ax=None,
    plt_kwargs=None,
    lines_kwargs_list=None,
    add_legend=True,
    legend_format=lambda v: v,
    legend_kwargs=None,
):
    """
    Plot recovery rate as lines, with each line binned by bins1 inside a bin of bins2.

    Parameters
    ----------
    values1, values2: array
        Component 1 and 2.
    bins1, bins2: array, int
        Bins for components 1 and 2.
    is_matched: array (boolean)
        Boolean array indicating matching status
    shape: str
        Shape of the lines. Can be steps or line.
    ax: matplotlib.axes
        Ax to add plot
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
    info = {
        "data": get_recovery_rate(values1, values2, bins1, bins2, is_matched),
        "ax": plt.axes() if ax is None else ax,
    }
    ph.plot_histograms(
        info["data"]["recovery"].T,
        info["data"]["edges1"],
        info["data"]["edges2"],
        ax=info["ax"],
        shape=shape,
        plt_kwargs=plt_kwargs,
        lines_kwargs_list=lines_kwargs_list,
        add_legend=add_legend,
        legend_kwargs=legend_kwargs,
        legend_format=legend_format,
    )
    return info


def plot_panel(
    values1,
    values2,
    bins1,
    bins2,
    is_matched,
    shape="steps",
    plt_kwargs=None,
    panel_kwargs_list=None,
    fig_kwargs=None,
    add_label=True,
    label_format=lambda v: v,
):
    """
    Plot recovery rate as lines in panels, with each line binned by bins1
    and each panel is based on the data inside a bins2 bin.

    Parameters
    ----------
    values1, values2: array
        Component 1 and 2.
    bins1, bins2: array, int
        Bins for components 1 and 2.
    is_matched: array (boolean)
        Boolean array indicating matching status
    shape: str
        Shape of the lines. Can be steps or line.
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
    info = {"data": get_recovery_rate(values1, values2, bins1, bins2, is_matched)}
    ncol = int(np.ceil(np.sqrt(info["data"]["edges2"][:-1].size)))
    nrow = int(np.ceil(info["data"]["edges2"][:-1].size / float(ncol)))
    fig, axes = plt.subplots(
        nrow, ncol, **updated_dict({"sharex": True, "sharey": True, "figsize": (8, 6)}, fig_kwargs)
    )
    info.update({"fig": fig, "axes": axes})
    for ax, rec_line, p_kwargs in zip(
        info["axes"].flatten(),
        info["data"]["recovery"].T,
        none_val(panel_kwargs_list, iter(lambda: {}, 1)),
    ):
        ph.add_grid(ax)
        ph.plot_hist_line(
            rec_line, info["data"]["edges1"], ax, shape, **updated_dict(plt_kwargs, p_kwargs)
        )
    for ax in info["axes"].flatten()[len(info["data"]["edges2"]) - 1 :]:
        ax.axis("off")
    if add_label:
        ph.add_panel_bin_label(
            info["axes"],
            info["data"]["edges2"][:-1],
            info["data"]["edges2"][1:],
            format_func=label_format,
        )
    return info


def plot2D(
    values1,
    values2,
    bins1,
    bins2,
    is_matched,
    ax=None,
    plt_kwargs=None,
    add_cb=True,
    cb_kwargs=None,
    add_num=False,
    num_kwargs=None,
):
    """
    Plot recovery rate as in 2D bins.

    Parameters
    ----------
    values1, values2: array
        Component 1 and 2.
    bins1, bins2: array, int
        Bins for components 1 and 2.
    is_matched: array (boolean)
        Boolean array indicating matching status
    ax: matplotlib.axes
        Ax to add plot
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
    # pylint: disable=too-many-locals
    # pylint: disable=invalid-name
    info = {
        "data": get_recovery_rate(values1, values2, bins1, bins2, is_matched),
        "ax": plt.axes() if ax is None else ax,
    }
    pcolor = info["ax"].pcolor(
        info["data"]["edges1"],
        info["data"]["edges2"],
        info["data"]["recovery"].T,
        **updated_dict({"vmin": 0, "vmax": 1}, plt_kwargs),
    )
    if add_num:
        xpositions = 0.5 * (info["data"]["edges1"][:-1] + info["data"]["edges1"][1:])
        ypositions = 0.5 * (info["data"]["edges2"][:-1] + info["data"]["edges2"][1:])
        num_kwargs_ = updated_dict({"va": "center", "ha": "center"}, num_kwargs)
        for xpos, line_matched, line_total in zip(
            xpositions, info["data"]["matched"], info["data"]["counts"]
        ):
            for ypos, matched, total in zip(ypositions, line_matched, line_total):
                if total > 0:
                    info["ax"].text(
                        xpos, ypos, rf"$\frac{{{matched:.0f}}}{{{total:.0f}}}$", **num_kwargs_
                    )
    if add_cb:
        info["cb"] = plt.colorbar(pcolor, **updated_dict({"ax": info["ax"]}, cb_kwargs))
    return info


def skyplot(
    ra,
    dec,
    is_matched,
    nside=256,
    nest=True,
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
    ra: numpy array
        Ra array in degrees
    dec: numpy array
        Dec array in degrees
    is_matched: array (boolean)
        Boolean array indicating matching status
    nside: int
        Healpix nside
    nest: bool
        If ordering is nested
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
    # pylint: disable=too-many-locals
    all_pix, mt_pix = {}, {}
    for pix, is_mt in zip(hp.ang2pix(nside, ra, dec, nest=nest, lonlat=True), is_matched):
        all_pix[pix] = all_pix.get(pix, 0) + 1
        mt_pix[pix] = mt_pix.get(pix, 0) + is_mt
    map_ = np.full(hp.nside2npix(nside), np.nan)
    map_[list(all_pix.keys())] = np.divide(list(mt_pix.values()), list(all_pix.values()))

    kwargs_ = {}
    vmin, vmax = min(map_[list(all_pix.keys())]), max(map_[list(all_pix.keys())])
    if vmin == vmax:
        kwargs_["min"] = vmin - 1e-10
        kwargs_["max"] = vmax + 1e-10
    fig, _, cbar = ph.plot_healpix_map(
        map_,
        nest=nest,
        auto_lim=auto_lim,
        bad_val=np.nan,
        ra_lim=ra_lim,
        dec_lim=dec_lim,
        fig=fig,
        figsize=figsize,
        **updated_dict(kwargs_, kwargs),
    )

    if cbar:
        cbar.set_xlabel(recovery_label)

    info = {"fig": fig, "nc_pix": all_pix, "nc_mt_pix": mt_pix}
    return info

def get_fscore(cat1_values1, cat1_values2, cat1_bins1, cat1_bins2, cat1_is_matched,
               cat2_values1, cat2_values2, cat2_bins1, cat2_bins2, cat2_is_matched,
               beta=1, pref='cat1'):
    """
    Computes fscore

    Parameters
    ----------
    cat1_values1, cat2_values2: array
        Component 1 and 2 of catalog 1.
    cat1_bins1, cat1_bins2: array, int
        Bins for components 1 and 2 of catalog 1.
    cat1_is_matched: array (boolean)
        Boolean array indicating matching status of catalog 1.
    cat2_values1, cat2_values2: array
        Component 1 and 2 of catalog 2.
    cat2_bins1, cat2_bins2: array, int
        Bins for components 1 and 2 of catalog 2.
    cat2_is_matched: array (boolean)
        Boolean array indicating matching status of catalog 2.
    beta: float
        Additional recall weight in f-score
    pref: str
        Peference to which recovery rate beta is applied, must be cat1 or cat2.


    Returns
    -------
    dict
        Binned fscore and recovery rate. Its sections are:

            * `fscore`: F-n score binned with (cat2_bins1, cat2_bins2, cat1_bins1, cat1_bins2).
            * `cat1`: Dictionary with recovery rate of catalog 1, see get_recovery_rate for info.
            * `cat2`: Dictionary with recovery rate of catalog 2, see get_recovery_rate for info.
    """
    rec1 = get_recovery_rate(
        cat1_values1, cat1_values2, cat1_bins1, cat1_bins2, cat1_is_matched)
    rec2 = get_recovery_rate(
        cat2_values1, cat2_values2, cat2_bins1, cat2_bins2, cat2_is_matched)

    r1_grid = np.outer(np.ones(rec2['recovery'].size), rec1['recovery'].flatten())
    r2_grid = np.outer(rec2['recovery'].flatten(), np.ones(rec1['recovery'].size))

    beta2 = beta**2
    fscore = (1+beta2)*r1_grid*r2_grid
    if pref=='cat1':
        fscore /= (beta2*r1_grid+r2_grid)
    elif pref=='cat2':
        fscore /= (r1_grid+beta2*r2_grid)
    else:
        raise ValueError(f'pref (={pref}) must be cat1 or cat2')
    return {'fscore': fscore.reshape(*rec2['recovery'].shape, *rec1['recovery'].shape),
            'cat1':rec1, 'cat2':rec2}

def plot_fscore(cat1_val1, cat1_val2, cat1_bins1, cat1_bins2, cat1_is_matched,
                cat2_val1, cat2_val2, cat2_bins1, cat2_bins2, cat2_is_matched,
                beta=1, pref='cat1', par_order=(0, 1, 2, 3), shape='steps', plt_kwargs={},
                lines_kwargs_list=None, fig_kwargs={}, legend_kwargs={},
                cat1_val1_label=None, cat1_val2_label=None,
                cat2_val1_label=None, cat2_val2_label=None,
                cat1_datalabel1_format=lambda v: v, cat1_datalabel2_format=lambda v: v,
                cat2_datalabel1_format=lambda v: v, cat2_datalabel2_format=lambda v: v,
                ):
    """
    Plot recovery rate as lines in panels, with each line binned by bins1
    and each panel is based on the data inside a bins2 bin.

    Parameters
    ----------
    cat1_values1, cat2_values2: array
        Component 1 and 2 of catalog 1.
    cat1_bins1, cat1_bins2: array, int
        Bins for components 1 and 2 of catalog 1.
    cat1_is_matched: array (boolean)
        Boolean array indicating matching status of catalog 1.
    cat2_values1, cat2_values2: array
        Component 1 and 2 of catalog 2.
    cat2_bins1, cat2_bins2: array, int
        Bins for components 1 and 2 of catalog 2.
    cat2_is_matched: array (boolean)
        Boolean array indicating matching status of catalog 2.
    beta: float
        Additional recall weight in f-score
    pref: str
        Peference to which recovery rate beta is applied, must be cat1 or cat2.
    par_order: list, bool
        It transposes quantities used, must be a percolation of (0, 1, 2, 3).
    shape: str
        Shape of the lines. Can be steps or line.
    plt_kwargs: dict
        Additional arguments for pylab.plot
    lines_kwargs_list: list, None
        List of additional arguments for plotting each line (using pylab.plot).
        Must have same size as len(bins2)-1
    fig_kwargs: dict
        Additional arguments for plt.subplots
    legend_kwargs: dict
        Additional arguments for pylab.legend
    cat1_val1_label, cat1_val2_label: str
        Labels for quantities 1 and 2 of catalog 1.
    cat2_val1_label, cat2_val2_label: str
        Labels for quantities 1 and 2 of catalog 2.
    cat1_datalabel1_format, cat1_datalabel2_format: function
        Function to format the values of labels for catalog 1.
    cat2_datalabel1_format, cat2_datalabel2_format: function
        Function to format the values of labels for catalog 2.


    Returns
    -------
    info: dict
        Information of data in the plots, it contains the sections:

            * `fig`: `matplotlib.figure.Figure` object.
            * `axes`: `matplotlib.axes` used in the plot.
            * `data`: Binned data used in the plot. It has the sections:

                * `fscore`: F-n score binned with (cat2_bins1, cat2_bins2, cat1_bins1, cat1_bins2).
                * `cat1`: Dictionary with recovery rate of catalog 1, see get_recovery_rate.
                * `cat2`: Dictionary with recovery rate of catalog 2, see get_recovery_rate.
    """
    info = {'data': get_fscore(
        cat1_val1, cat1_val2, cat1_bins1, cat1_bins2, cat1_is_matched,
        cat2_val1, cat2_val2, cat2_bins1, cat2_bins2, cat2_is_matched,
        beta=beta, pref=pref)}

    # Order of parameters
    edges = index_list([info['data']['cat1']['edges1'], info['data']['cat1']['edges2'],
                        info['data']['cat2']['edges1'], info['data']['cat2']['edges2']],
                         par_order)
    labels = index_list([cat1_val1_label, cat1_val2_label,
                         cat2_val1_label, cat2_val2_label], par_order)
    label_formats = index_list(
        [cat1_datalabel1_format, cat1_datalabel2_format,
         cat2_datalabel1_format, cat2_datalabel2_format], par_order)
    tr_order = index_list([2, 3, 0, 1], index_list(par_order, [2, 3, 0, 1]))

    ni = edges[2].size-1
    nj = edges[3].size-1
    fig_kwargs_ = dict(sharex=True, sharey=True, figsize=(8, 6))
    fig_kwargs_.update(fig_kwargs)
    info.update({key: value for key, value in zip(
        ('fig', 'axes'), plt.subplots(ni, nj, **fig_kwargs_))})
    add_legend = True
    for axl, fscl in zip(info['axes'], info['data']['fscore'].transpose(*tr_order)):
        for ax, fsc in zip(axl, fscl):
            ph.add_grid(ax)
            ph.plot_histograms(fsc.T, edges[0], edges[1],
                               ax=ax, shape=shape, plt_kwargs=plt_kwargs,
                               lines_kwargs_list=lines_kwargs_list,
                               add_legend=add_legend, legend_format=label_formats[1],
                               legend_kwargs=legend_kwargs)
            add_legend = False
    for ax in info['axes'][:,0]:
        ax.set_ylabel(f'$F_{{{beta}}}$ score')
    for ax in info['axes'][-1,:]:
        ax.set_xlabel(f'${labels[0]}$')
    ph.add_panel_bin_label(
        info['axes'], edges[3][:-1], edges[3][1:],
        prefix='' if labels[3] is None else f'{labels[3]}: ',
        format_func=label_formats[3])
    ph.add_panel_bin_label(
        info['axes'][:,-1], edges[2][:-1], edges[2][1:],
        prefix='' if labels[2] is None else f'{labels[2]}: ',
        format_func=label_formats[2], position='right')

    return info
