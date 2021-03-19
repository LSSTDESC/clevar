import numpy as np
import pylab as plt

from ..utils import none_val, bin_masks
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
    array (2D)
        Recovery rate binned with (bin1, bin2). bins where no cluster was found have nan value.
    """
    hist_all = np.histogram2d(values1, values2, bins=(bins1, bins2))[0]
    hist_matched = np.histogram2d(values1[is_matched], values2[is_matched],
                                  bins=(bins1, bins2))[0]
    recovery = np.zeros(hist_all.shape)
    recovery[:] = np.nan
    recovery[hist_all>0] = hist_matched[hist_all>0]/hist_all[hist_all>0]
    return recovery
def plot_recovery_line(hist_values, bins, ax, shape='steps', kwargs={}):
    """
    Plot recovey rate as lines. Can be in steps or continuous

    Parameters
    ----------
    hist_values: array
        Values of each bin in the histogram
    bins: array
        Bins of histogram
    ax: matplotlib.axes
        Ax to add plot
    shape: str
        Shape of the line. Can be steps or line.
    """
    if shape=='steps':
        data = (np.transpose([bins[:-1], bins[1:]]).flatten(),
                np.transpose([hist_values, hist_values]).flatten())
    elif shape=='line':
        data = (0.5*(bins[:-1]+bins[1:]), hist_values)
    else:
        raise ValueError(f"shape ({shape}) must be 'steps' or 'line'")
    ax.plot(*data, **kwargs)
class ArrayFuncs():
    """
    Class of plot functions with arrays as inputs
    """
    def plot(values1, values2, bins1, bins2, is_matched, shape='steps',
             ax=None, plt_kwargs={}, lines_kwargs_list=None):
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
        """
        ax = plt.axes() if ax is None else ax
        ph.add_grid(ax)
        recovery = get_recovery_rate(values1, values2, bins1, bins2, is_matched).T
        lines_kwargs_list = none_val(lines_kwargs_list, [{} for m in bins2[:-1]])
        for rec_line, l_kwargs in zip(recovery, lines_kwargs_list):
            kwargs = {}
            kwargs.update(plt_kwargs)
            kwargs.update(l_kwargs)
            plot_recovery_line(rec_line, bins1, ax, shape, kwargs)
    def plot_panel(values1, values2, bins1, bins2, is_matched, shape='steps',
                   plt_kwargs={}, panel_kwargs_list=None,
                   fig_kwargs={}):
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

        Returns
        -------
        fig: matplotlib.figure.Figure
            `matplotlib.figure.Figure` object
        ax: matplotlib.axes
            Axes with the panels
        """
        ni = int(np.floor(np.sqrt(len(bins2))))
        nj = ni if ni**2>=len(bins2) else ni+1
        f, axes = plt.subplots(ni, nj, sharex=True, sharey=True, **fig_kwargs)
        recovery = get_recovery_rate(values1, values2, bins1, bins2, is_matched).T
        panel_kwargs_list = none_val(panel_kwargs_list, [{} for m in bins2[:-1]])
        for ax, rec_line, p_kwargs in zip(axes, recovery, panel_kwargs_list):
            ph.add_grid(ax)
            kwargs = {}
            kwargs.update(plt_kwargs)
            kwargs.update(p_kwargs)
            plot_recovery_line(rec_line, bins1, ax, shape, kwargs)
        return f, axes
    def plot2D(values1, values2, bins1, bins2, is_matched,
               ax=None, plt_kwargs={}, add_cb=True, cb_kwargs={}):
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

        Returns
        -------
        matplotlib.colorbar.Colorbar
            Colorbar of the recovey rates
        """
        recovery = get_recovery_rate(values1, values2, bins1, bins2, is_matched).T
        ax = plt.axes() if ax is None else ax
        c = ax.pcolor(bins1, bins2, recovery, **plt_kwargs)
        return plt.colorbar(c, **cb_kwargs)
class CatalogFuncs():
    """
    Class of plot functions with clevar.Catalog as inputs
    """
    def _plot_base(pltfunc, cat, col1, col2, bins1, bins2, matching_type, **kwargs):
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
            'cross', 'self', 'other', 'multi_self', 'multi_other', 'multi_join'
        **kwargs:
            Additional arguments to be passed to pltfunc
        """
        return pltfunc(cat.data[col1], cat.data[col2], bins1, bins2,
                       is_matched=cat.get_matching_mask(matching_type), **kwargs)
    def plot(cat, col1, col2, bins1, bins2, matching_type, **kwargs):
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
            'cross', 'self', 'other', 'multi_self', 'multi_other', 'multi_join'
        shape: str
            Shape of the lines. Can be steps or line.
        ax: matplotlib.axes
            Ax to add plot
        plt_kwargs: dict
            Additional arguments for pylab.plot
        lines_kwargs_list: list, None
            List of additional arguments for plotting each line (using pylab.plot).
            Must have same size as len(bins2)-1
        """
        return CatalogFuncs._plot_base(ArrayFuncs.plot,
                cat, col1, col2, bins1, bins2, matching_type, **kwargs)
    def plot_panel(cat, col1, col2, bins1, bins2, matching_type, **kwargs):
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
            'cross', 'self', 'other', 'multi_self', 'multi_other', 'multi_join'
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

        Returns
        -------
        fig: matplotlib.figure.Figure
            `matplotlib.figure.Figure` object
        ax: matplotlib.axes
            Axes with the panels
        """
        return CatalogFuncs._plot_base(ArrayFuncs.plot_panel,
                cat, col1, col2, bins1, bins2, matching_type, **kwargs)
    def plot2D(cat, col1, col2, bins1, bins2, matching_type, **kwargs):
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
            'cross', 'self', 'other', 'multi_self', 'multi_other', 'multi_join'
        ax: matplotlib.axes
            Ax to add plot
        plt_kwargs: dict
            Additional arguments for pylab.plot
        add_cb: bool
            Plot colorbar
        cb_kwargs: dict
            Colorbar arguments

        Returns
        -------
        matplotlib.colorbar.Colorbar
            Colorbar of the recovey rates
        """
        return CatalogFuncs._plot_base(ArrayFuncs.plot2D,
                cat, col1, col2, bins1, bins2, matching_type, **kwargs)
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
        'cross', 'self', 'other', 'multi_self', 'multi_other', 'multi_join'
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
def plot(cat, matching_type, redshift_bins, mass_bins, transpose=False, **kwargs):
    """
    Plot recovery rate as lines, with each line binned by redshift inside a mass bin.

    Parameters
    ----------
    cat: clevar.Catalog
        Catalog with matching information
    matching_type: str
        Type of matching to be considered. Must be in:
        'cross', 'self', 'other', 'multi_self', 'multi_other', 'multi_join'
    redshift_bins: array
        Bins for redshift
    mass_bins: array
        Bins for mass
    transpose: bool
        Transpose mass and redshift in plots
    shape: str
        Shape of the lines. Can be steps or line.
    ax: matplotlib.axes
        Ax to add plot
    plt_kwargs: dict
        Additional arguments for pylab.plot
    lines_kwargs_list: list, None
        List of additional arguments for plotting each line (using pylab.plot).
        Must have same size as len(bins2)-1
    """
    return _plot_base(CatalogFuncs.plot, cat, matching_type,
                      redshift_bins, mass_bins, transpose, **kwargs)
def plot_panel(cat, matching_type, redshift_bins, mass_bins, transpose=False, **kwargs):
    """
    Plot recovery rate as lines in panels, with each line binned by redshift
    and each panel is based on the data inside a mass bin.

    Parameters
    ----------
    cat: clevar.Catalog
        Catalog with matching information
    matching_type: str
        Type of matching to be considered. Must be in:
        'cross', 'self', 'other', 'multi_self', 'multi_other', 'multi_join'
    redshift_bins: array
        Bins for redshift
    mass_bins: array
        Bins for mass
    transpose: bool
        Transpose mass and redshift in plots
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

    Returns
    -------
    fig: matplotlib.figure.Figure
        `matplotlib.figure.Figure` object
    ax: matplotlib.axes
        Axes with the panels
    """
    return _plot_base(CatalogFuncs.plot_panel, cat, matching_type,
                      redshift_bins, mass_bins, transpose, **kwargs)
def plot2D(cat, matching_type, redshift_bins, mass_bins, transpose=False, **kwargs):
    """
    Plot recovery rate as in 2D (redshift, mass) bins.

    Parameters
    ----------
    cat: clevar.Catalog
        Catalog with matching information
    matching_type: str
        Type of matching to be considered. Must be in:
        'cross', 'self', 'other', 'multi_self', 'multi_other', 'multi_join'
    redshift_bins: array
        Bins for redshift
    mass_bins: array
        Bins for mass
    transpose: bool
        Transpose mass and redshift in plots
    ax: matplotlib.axes
        Ax to add plot
    plt_kwargs: dict
        Additional arguments for pylab.plot
    add_cb: bool
        Plot colorbar
    cb_kwargs: dict
        Colorbar arguments

    Returns
    -------
    matplotlib.colorbar.Colorbar
        Colorbar of the recovey rates
    """
    return _plot_base(CatalogFuncs.plot2D, cat, matching_type,
                      redshift_bins, mass_bins, transpose, **kwargs)
