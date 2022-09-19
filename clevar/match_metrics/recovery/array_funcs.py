"""@file clevar/match_metrics/recovery/array_funcs.py

Main recovery functions using arrays.
"""
import numpy as np
import healpy as hp

from ...utils import none_val, updated_dict
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
    hist_matched = np.histogram2d(values1[is_matched], values2[is_matched],
                                  bins=(edges1, edges2))[0]
    recovery = np.zeros(hist_counts.shape)
    recovery[:] = np.nan
    recovery[hist_counts>0] = hist_matched[hist_counts>0]/hist_counts[hist_counts>0]
    return {'recovery':recovery, 'edges1':edges1, 'edges2':edges2,
            'matched':np.array(hist_matched, dtype=int),
            'counts': np.array(hist_counts, dtype=int)}


def plot(values1, values2, bins1, bins2, is_matched, shape='steps',
         ax=None, plt_kwargs=None, lines_kwargs_list=None,
         add_legend=True, legend_format=lambda v: v, legend_kwargs=None):
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
        'data': get_recovery_rate(values1, values2, bins1, bins2, is_matched),
        'ax': plt.axes() if ax is None else ax,}
    ph.plot_histograms(
        info['data']['recovery'].T, info['data']['edges1'],
        info['data']['edges2'], ax=info['ax'], shape=shape,
        plt_kwargs=plt_kwargs, lines_kwargs_list=lines_kwargs_list,
        add_legend=add_legend, legend_format=legend_format,
        legend_kwargs=legend_kwargs)
    return info


def plot_panel(values1, values2, bins1, bins2, is_matched, shape='steps',
               plt_kwargs=None, panel_kwargs_list=None,
               fig_kwargs=None, add_label=True, label_format=lambda v: v):
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
    info = {'data': get_recovery_rate(values1, values2, bins1, bins2, is_matched)}
    nj = int(np.ceil(np.sqrt(info['data']['edges2'][:-1].size)))
    ni = int(np.ceil(info['data']['edges2'][:-1].size/float(nj)))
    fig, axes = plt.subplots(
        ni, nj, **updated_dict({'sharex':True, 'sharey':True, 'figsize':(8, 6)}, fig_kwargs))
    info.update({'fig': fig, 'axes':axes})
    for ax, rec_line, p_kwargs in zip(
            info['axes'].flatten(),
            info['data']['recovery'].T,
            none_val(panel_kwargs_list, iter(lambda: {}, 1))
        ):
        ph.add_grid(ax)
        ph.plot_hist_line(rec_line, info['data']['edges1'], ax, shape,
                          **updated_dict(plt_kwargs, p_kwargs))
    for ax in info['axes'].flatten()[len(info['data']['edges2'])-1:]:
        ax.axis('off')
    if add_label:
        ph.add_panel_bin_label(info['axes'],  info['data']['edges2'][:-1],
                               info['data']['edges2'][1:], format_func=label_format)
    return info


def plot2D(values1, values2, bins1, bins2, is_matched,
           ax=None, plt_kwargs=None, add_cb=True, cb_kwargs=None,
           add_num=False, num_kwargs=None):
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
    info = {
        'data': get_recovery_rate(values1, values2, bins1, bins2, is_matched),
        'ax': plt.axes() if ax is None else ax,}
    c = info['ax'].pcolor(info['data']['edges1'], info['data']['edges2'],
                          info['data']['recovery'].T,
                          **updated_dict({'vmin':0, 'vmax':1}, plt_kwargs))
    if add_num:
        xpositions = .5*(info['data']['edges1'][:-1]+info['data']['edges1'][1:])
        ypositions = .5*(info['data']['edges2'][:-1]+info['data']['edges2'][1:])
        num_kwargs_ = updated_dict({'va':'center', 'ha':'center'}, num_kwargs)
        for xpos, line_matched, line_total in zip(
                xpositions, info['data']['matched'], info['data']['counts']):
            for ypos, matched, total in zip(ypositions, line_matched, line_total):
                if total>0:
                    info['ax'].text(xpos, ypos,
                                    rf'$\frac{{{matched:.0f}}}{{{total:.0f}}}$', **num_kwargs_)
    if add_cb:
        info['cb'] = plt.colorbar(c, **updated_dict({'ax': info['ax']}, cb_kwargs))
    return info


def skyplot(ra, dec, is_matched, nside=256, nest=True, auto_lim=False, ra_lim=None, dec_lim=None,
            recovery_label='Recovery Rate', fig=None, figsize=None, **kwargs):
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
    all_pix, mt_pix = {}, {}
    for pix, mt in zip(hp.ang2pix(nside, ra, dec, nest=nest, lonlat=True), is_matched):
        all_pix[pix] = all_pix.get(pix, 0)+1
        mt_pix[pix] = mt_pix.get(pix, 0)+mt
    map_ = np.full(hp.nside2npix(nside), np.nan)
    map_[list(all_pix.keys())] = np.divide(list(mt_pix.values()), list(all_pix.values()))

    kwargs_ = {}
    vmin, vmax = min(map_[list(all_pix.keys())]), max(map_[list(all_pix.keys())])
    if vmin==vmax:
        kwargs_['min'] = vmin-1e-10
        kwargs_['max'] = vmax+1e-10
    fig, ax, cb = ph.plot_healpix_map(
        map_, nest=nest, auto_lim=auto_lim, bad_val=np.nan,
        ra_lim=ra_lim, dec_lim=dec_lim, fig=fig, figsize=figsize,
        **updated_dict(kwargs_, kwargs))

    if cb:
        cb.set_xlabel(recovery_label)

    info = {'fig':fig,  'nc_pix':all_pix, 'nc_mt_pix':mt_pix}
    return info
