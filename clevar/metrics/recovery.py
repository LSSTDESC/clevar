# Set mpl backend run plots on github actions
import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == 'test':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import pylab as plt
import numpy as np

from ..utils import none_val
from . import plot_helper as ph

def get_recovery_rate(values1, values2, bins1, bins2, is_matched):
    """
    Get recovery rate binned in 2 components

    Parameters
    ----------
    values1: array
        Component 1
    values2: array
        Component 2
    bins1: array
        Bins for component 1
    bins2: array
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
                                  bins=(bins1, bins2))[0]
    recovery = np.zeros(hist_all.shape)
    recovery[:] = np.nan
    recovery[hist_all>0] = hist_matched[hist_all>0]/hist_all[hist_all>0]
    return recovery, edges1, edges2
class ArrayFuncs():
    """
    Class of plot functions with arrays as inputs
    """
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
        bins1: array
            Bins for component 1
        bins2: array
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
                   fig_kwargs={}, add_label=False, label_format=lambda v: v):
        """
        Plot recovery rate as lines in panels, with each line binned by bins1
        and each panel is based on the data inside a bins2 bin.

        Parameters
        ----------
        values1: array
            Component 1
        values2: array
            Component 2
        bins1: array
            Bins for component 1
        bins2: array
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
        bins1: array
            Bins for component 1
        bins2: array
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
        c = ax.pcolor(edges1, edges2, recovery.T, **plt_kwargs)
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
class CatalogFuncs():
    """
    Class of plot functions with clevar.Catalog as inputs
    """
    def _plot_base(pltfunc, cat, col1, col2, bins1, bins2, matching_type,
                   mask=None, mask_unmatched=None, **kwargs):
        """
        Adapts a CatalogFuncs function to use a ArrayFuncs function.

        Parameters
        ----------
        pltfunc: function
            ArrayFuncs function
        cat: clevar.Catalog
            Catalog with matching information
        col1: str
            Name of column 1
        col2: str
            Name of column 2
        bins1: array
            Bins for component 1
        bins2: array
            Bins for component 2
        matching_type: str
            Type of matching to be considered. Must be in:
            'mt_cross', 'mt_self', 'mt_other', 'mt_multi_self', 'mt_multi_other', 'mt_multi_join'
        mask: array
            Mask unwanted clusters
        mask_unmatched: array
            Mask unwanted unmatched clusters (ex: out of footprint)
        **kwargs:
            Additional arguments to be passed to pltfunc
        """
        # convert matching type to the values expected by get_matching_mask
        matching_type_conv = matching_type.replace('cat1', 'self').replace('cat2', 'other')
        is_matched = cat.get_matching_mask(matching_type_conv)
        # mask_ to apply mask and mask_unmatched
        mask_ = none_val(mask, True)*(~(~is_matched*none_val(mask_unmatched, False)))
        return pltfunc(cat[mask_][col1], cat[mask_][col2], bins1, bins2,
                       is_matched=is_matched[mask_], **kwargs)
    def plot(cat, col1, col2, bins1, bins2, matching_type,
             xlabel=None, ylabel=None, scale1='linear', **kwargs):
        """
        Plot recovery rate as lines, with each line binned by bins1 inside a bin of bins2.

        Parameters
        ----------
        cat: clevar.Catalog
            Catalog with matching information
        col1: str
            Name of column 1
        col2: str
            Name of column 2
        bins1: array
            Bins for component 1
        bins2: array
            Bins for component 2
        matching_type: str
            Type of matching to be considered. Must be in:
            'mt_cross', 'mt_self', 'mt_other', 'mt_multi_self', 'mt_multi_other', 'mt_multi_join'
        mask: array
            Mask unwanted clusters
        mask_unmatched: array
            Mask unwanted unmatched clusters (ex: out of footprint)

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
        ax = CatalogFuncs._plot_base(ArrayFuncs.plot,
                cat, col1, col2, bins1, bins2, matching_type, **kwargs)
        ax.set_xlabel(xlabel if xlabel else f'${col1}_{{{cat.name}}}$')
        ax.set_ylabel(ylabel if ylabel else 'recovery rate')
        ax.set_xscale(scale1)
        return ax
    def plot_panel(cat, col1, col2, bins1, bins2, matching_type,
                   xlabel=None, ylabel=None, scale1='linear', **kwargs):
        """
        Plot recovery rate as lines in panels, with each line binned by bins1
        and each panel is based on the data inside a bins2 bin.

        Parameters
        ----------
        cat: clevar.Catalog
            Catalog with matching information
        col1: str
            Name of column 1
        col2: str
            Name of column 2
        bins1: array
            Bins for component 1
        bins2: array
            Bins for component 2
        matching_type: str
            Type of matching to be considered. Must be in:
            'mt_cross', 'mt_self', 'mt_other', 'mt_multi_self', 'mt_multi_other', 'mt_multi_join'
        mask: array
            Mask unwanted clusters
        mask_unmatched: array
            Mask unwanted unmatched clusters (ex: out of footprint)

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
        fig, axes = CatalogFuncs._plot_base(ArrayFuncs.plot_panel,
                cat, col1, col2, bins1, bins2, matching_type, **kwargs)
        ph.nice_panel(axes, xlabel=none_val(xlabel, col1), ylabel=none_val(xlabel, col2),
                      xscale=scale1, yscale='linear')
        return fig, axes
    def plot2D(cat, col1, col2, bins1, bins2, matching_type,
               xlabel=None, ylabel=None, scale1='linear', scale2='linear',
               **kwargs):
        """
        Plot recovery rate as in 2D bins.

        Parameters
        ----------
        cat: clevar.Catalog
            Catalog with matching information
        col1: str
            Name of column 1
        col2: str
            Name of column 2
        bins1: array
            Bins for component 1
        bins2: array
            Bins for component 2
        matching_type: str
            Type of matching to be considered. Must be in:
            'mt_cross', 'mt_self', 'mt_other', 'mt_multi_self', 'mt_multi_other', 'mt_multi_join'
        mask: array
            Mask unwanted clusters
        mask_unmatched: array
            Mask unwanted unmatched clusters (ex: out of footprint)

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
        ax, cb = CatalogFuncs._plot_base(ArrayFuncs.plot2D,
                cat, col1, col2, bins1, bins2, matching_type, **kwargs)
        ax.set_xlabel(xlabel if xlabel else f'${col1}_{{{cat.name}}}$')
        ax.set_ylabel(ylabel if ylabel else f'${col2}_{{{cat.name}}}$')
        ax.set_xscale(scale1)
        ax.set_yscale(scale2)
        return ax, cb
def _plot_base(pltfunc, cat, matching_type, redshift_bins, mass_bins,
               transpose=False, **kwargs):
    """
    Adapts a CatalogFuncs function for main functions using mass and redshift.

    Parameters
    ----------
    pltfunc: function
        CatalogFuncs function
    cat: clevar.Catalog
        Catalog with matching information
    matching_type: str
        Type of matching to be considered. Must be in:
        'mt_cross', 'mt_self', 'mt_other', 'mt_multi_self', 'mt_multi_other', 'mt_multi_join'
    redshift_bins: array
        Bins for redshift
    mass_bins: array
        Bins for mass
    transpose: bool
        Transpose mass and redshift in plots
    **kwargs:
        Additional arguments to be passed to pltfunc
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
    cat: clevar.Catalog
        Catalog with matching information
    matching_type: str
        Type of matching to be considered. Must be in:
        'mt_cross', 'mt_self', 'mt_other', 'mt_multi_self', 'mt_multi_other', 'mt_multi_join'
    redshift_bins: array
        Bins for redshift
    mass_bins: array
        Bins for mass
    transpose: bool
        Transpose mass and redshift in plots
    log_mass: bool
        Plot mass in log scale

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
    """
    legend_fmt = kwargs.pop("legend_fmt", ".1f" if log_mass*(not transpose) else ".2f")
    kwargs['legend_format'] = kwargs.get('legend_format',
        lambda v: f'10^{{%{legend_fmt}}}'%np.log10(v) if log_mass*(not transpose)\
             else f'%{legend_fmt}'%v)
    return _plot_base(CatalogFuncs.plot, cat, matching_type,
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
    cat: clevar.Catalog
        Catalog with matching information
    matching_type: str
        Type of matching to be considered. Must be in:
        'mt_cross', 'mt_self', 'mt_other', 'mt_multi_self', 'mt_multi_other', 'mt_multi_join'
    redshift_bins: array
        Bins for redshift
    mass_bins: array
        Bins for mass
    transpose: bool
        Transpose mass and redshift in plots
    log_mass: bool
        Plot mass in log scale

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
    fig: matplotlib.figure.Figure
        `matplotlib.figure.Figure` object
    ax: matplotlib.axes
        Axes with the panels
    """
    label_fmt = kwargs.pop("label_fmt", ".1f" if log_mass*(not transpose) else ".2f")
    kwargs['label_format'] = kwargs.get('label_format',
        lambda v: f'10^{{%{label_fmt}}}'%np.log10(v) if log_mass*(not transpose)\
             else f'%{label_fmt}'%v)
    return _plot_base(CatalogFuncs.plot_panel, cat, matching_type,
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
    cat: clevar.Catalog
        Catalog with matching information
    matching_type: str
        Type of matching to be considered. Must be in:
        'mt_cross', 'mt_self', 'mt_other', 'mt_multi_self', 'mt_multi_other', 'mt_multi_join'
    redshift_bins: array
        Bins for redshift
    mass_bins: array
        Bins for mass
    transpose: bool
        Transpose mass and redshift in plots
    log_mass: bool
        Plot mass in log scale

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
    matplotlib.colorbar.Colorbar
        Colorbar of the recovey rates
    """
    return _plot_base(CatalogFuncs.plot2D, cat, matching_type,
                      redshift_bins, mass_bins, transpose,
                      scale1='log' if log_mass*transpose else 'linear',
                      scale2='log' if log_mass*(not transpose) else 'linear',
                      xlabel=mass_label if transpose else redshift_label,
                      ylabel=redshift_label if transpose else mass_label,
                      **kwargs)
