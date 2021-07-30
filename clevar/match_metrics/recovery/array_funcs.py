"""@file clevar/match_metrics/recovery/array_funcs.py

Main recovery functions using arrays.
"""
import numpy as np

from ...utils import none_val
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
         ax=None, plt_kwargs={}, lines_kwargs_list=None,
         add_legend=True, legend_format=lambda v: v, legend_kwargs={}):
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
               plt_kwargs={}, panel_kwargs_list=None,
               fig_kwargs={}, add_label=True, label_format=lambda v: v):
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
    fig_kwargs_ = dict(sharex=True, sharey=True, figsize=(8, 6))
    fig_kwargs_.update(fig_kwargs)
    info.update({key: value for key, value in zip(
        ('fig', 'axes'), plt.subplots(ni, nj, **fig_kwargs_))})
    for ax, rec_line, p_kwargs in zip(
            info['axes'].flatten(),
            info['data']['recovery'].T,
            none_val(panel_kwargs_list, iter(lambda: {}, 1))
        ):
        ph.add_grid(ax)
        kwargs = {}
        kwargs.update(plt_kwargs)
        kwargs.update(p_kwargs)
        ph.plot_hist_line(rec_line, info['data']['edges1'], ax, shape, **kwargs)
    for ax in info['axes'].flatten()[len(info['data']['edges2'])-1:]:
        ax.axis('off')
    if add_label:
        ph.add_panel_bin_label(info['axes'],  info['data']['edges2'][:-1],
                               info['data']['edges2'][1:], format_func=label_format)
    return info


def plot2D(values1, values2, bins1, bins2, is_matched,
           ax=None, plt_kwargs={}, add_cb=True, cb_kwargs={},
           add_num=False, num_kwargs={}):
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
    info = {
        'data': get_recovery_rate(values1, values2, bins1, bins2, is_matched),
        'ax': plt.axes() if ax is None else ax,}
    plt_kwargs_ = {'vmin':0, 'vmax':1}
    plt_kwargs_.update(plt_kwargs)
    c = info['ax'].pcolor(info['data']['edges1'], info['data']['edges2'],
                          info['data']['recovery'].T, **plt_kwargs_)
    if add_num:
        xpositions = .5*(info['data']['edges1'][:-1]+info['data']['edges1'][1:])
        ypositions = .5*(info['data']['edges2'][:-1]+info['data']['edges2'][1:])
        num_kwargs_ = {'va':'center', 'ha':'center'}
        num_kwargs_.update(num_kwargs)
        for xpos, line_matched, line_total in zip(
                xpositions, info['data']['matched'], info['data']['counts']):
            for ypos, matched, total in zip(ypositions, line_matched, line_total):
                if total>0:
                    info['ax'].text(xpos, ypos,
                                    rf'$\frac{{{matched:.0f}}}{{{total:.0f}}}$', **num_kwargs_)
    if add_cb:
        cb_kwargs_ = {'ax': info['ax']}
        cb_kwargs_.update(cb_kwargs)
        info['cb'] = plt.colorbar(c, **cb_kwargs_)
    return info
