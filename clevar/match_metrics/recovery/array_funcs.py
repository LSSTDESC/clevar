"""@file clevar/match_metrics/recovery/array_funcs.py

Main recovery functions using arrays.
"""
# Set mpl backend run plots on github actions
import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == 'test':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import pylab as plt
import numpy as np

from ...utils import none_val
from .. import plot_helper as ph

def get_recovery_rate(values1, values2, bins1, bins2, is_matched):
    """
    Get recovery rate binned in 2 components

    Parameters
    ----------
    values1: array
        Component 1
    values2: array
        Component 2
    bins1: array, int
        Bins for component 1
    bins2: array, int
        Bins for component 2
    is_matched: array (boolean)
        Boolean array indicating matching status

    Returns
    -------
    recovery: ndarray (2D)
        Recovery rate binned with (bin1, bin2). bins where no cluster was found have nan value.
    edges1: ndarray
        The bin edges along the first dimension.
    edges2: ndarray
        The bin edges along the second dimension.
    """
    hist_all, edges1, edges2 = np.histogram2d(values1, values2, bins=(bins1, bins2))
    hist_matched = np.histogram2d(values1[is_matched], values2[is_matched],
                                  bins=(edges1, edges2))[0]
    recovery = np.zeros(hist_all.shape)
    recovery[:] = np.nan
    recovery[hist_all>0] = hist_matched[hist_all>0]/hist_all[hist_all>0]
    return recovery, edges1, edges2
def plot(values1, values2, bins1, bins2, is_matched, shape='steps',
         ax=None, plt_kwargs={}, lines_kwargs_list=None,
         add_legend=True, legend_format=lambda v: v, legend_kwargs={}):
    """
    Plot recovery rate as lines, with each line binned by bins1 inside a bin of bins2.

    Parameters
    ----------
    values1: array
        Component 1
    values2: array
        Component 2
    bins1: array, int
        Bins for component 1
    bins2: array, int
        Bins for component 2
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
    ax: matplotlib.axes
        Axis of the plot
    """
    ax = plt.axes() if ax is None else ax
    ph.add_grid(ax)
    recovery, edges1, edges2 = get_recovery_rate(values1, values2, bins1, bins2, is_matched)
    lines_kwargs_list = none_val(lines_kwargs_list, [{} for m in edges2[:-1]])
    for rec_line, l_kwargs, e0, e1 in zip(recovery.T, lines_kwargs_list, edges2, edges2[1:]):
        kwargs = {}
        kwargs['label'] = ph.get_bin_label(e0, e1, legend_format) if add_legend else None
        kwargs.update(plt_kwargs)
        kwargs.update(l_kwargs)
        ph.plot_hist_line(rec_line, edges1, ax, shape, **kwargs)
    if add_legend:
        ax.legend(**legend_kwargs)
    return ax
def plot_panel(values1, values2, bins1, bins2, is_matched, shape='steps',
               plt_kwargs={}, panel_kwargs_list=None,
               fig_kwargs={}, add_label=True, label_format=lambda v: v):
    """
    Plot recovery rate as lines in panels, with each line binned by bins1
    and each panel is based on the data inside a bins2 bin.

    Parameters
    ----------
    values1: array
        Component 1
    values2: array
        Component 2
    bins1: array, int
        Bins for component 1
    bins2: array, int
        Bins for component 2
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
    fig: matplotlib.figure.Figure
        `matplotlib.figure.Figure` object
    ax: matplotlib.axes
        Axes with the panels
    """
    recovery, edges1, edges2 = get_recovery_rate(values1, values2, bins1, bins2, is_matched)
    nj = int(np.ceil(np.sqrt(edges2[:-1].size)))
    ni = int(np.ceil(edges2[:-1].size/float(nj)))
    fig_kwargs_ = dict(sharex=True, sharey=True, figsize=(8, 6))
    fig_kwargs_.update(fig_kwargs)
    f, axes = plt.subplots(ni, nj, **fig_kwargs_)
    panel_kwargs_list = none_val(panel_kwargs_list, [{} for m in edges2[:-1]])
    for ax, rec_line, p_kwargs in zip(axes.flatten(), recovery.T, panel_kwargs_list):
        ph.add_grid(ax)
        kwargs = {}
        kwargs.update(plt_kwargs)
        kwargs.update(p_kwargs)
        ph.plot_hist_line(rec_line, edges1, ax, shape, **kwargs)
    for ax in axes.flatten()[len(edges2)-1:]:
        ax.axis('off')
    if add_label:
        ph.add_panel_bin_label(axes,  edges2[:-1], edges2[1:],
                               format_func=label_format)
    return f, axes
def plot2D(values1, values2, bins1, bins2, is_matched,
           ax=None, plt_kwargs={}, add_cb=True, cb_kwargs={},
           add_num=False, num_kwargs={}):
    """
    Plot recovery rate as in 2D bins.

    Parameters
    ----------
    values1: array
        Component 1
    values2: array
        Component 2
    bins1: array, int
        Bins for component 1
    bins2: array, int
        Bins for component 2
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
    ax: matplotlib.axes
        Axis of the plot
    matplotlib.colorbar.Colorbar
        Colorbar of the recovey rates
    """
    recovery, edges1, edges2 = get_recovery_rate(values1, values2, bins1, bins2, is_matched)
    ax = plt.axes() if ax is None else ax
    plt_kwargs_ = {'vmin':0, 'vmax':1}
    plt_kwargs_.update(plt_kwargs)
    c = ax.pcolor(edges1, edges2, recovery.T, **plt_kwargs_)
    if add_num:
        hist_all = np.histogram2d(values1, values2, bins=(bins1, bins2))[0]
        hist_matched = np.histogram2d(values1[is_matched], values2[is_matched],
                              bins=(bins1, bins2))[0]
        xp, yp = .5*(edges1[:-1]+edges1[1:]), .5*(edges2[:-1]+edges2[1:])
        num_kwargs_ = {'va':'center', 'ha':'center'}
        num_kwargs_.update(num_kwargs)
        for x, ht_, hb_ in zip(xp, hist_matched, hist_all):
            for y, ht, hb in zip(yp, ht_, hb_):
                if hb>0:
                    ax.text(x, y, f'$\\frac{{{ht:.0f}}}{{{hb:.0f}}}$', **num_kwargs_)
    cb_kwargs_ = {'ax':ax}
    cb_kwargs_.update(cb_kwargs)
    return ax, plt.colorbar(c, **cb_kwargs_)
