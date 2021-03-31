# Set mpl backend run plots on github actions
import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == 'test':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
from matplotlib.ticker import ScalarFormatter, NullFormatter
import pylab as plt
import numpy as np

from ..utils import none_val
from ..match import MatchedPairs
from . import plot_helper as ph

class ArrayFuncs():
    """
    Class of plot functions with arrays as inputs
    """
    def plot(values1, values2, err1=None, err2=None,
                   ax=None, plt_kwargs={}, err_kwargs={}):
        """
        Scatter plot with errorbars and color based on input

        Parameters
        ----------
        values1: array
            Component 1
        values2: array
            Component 2
        err1: array
            Error of component 1
        err2: array
            Error of component 2
        ax: matplotlib.axes
            Ax to add plot
        plt_kwargs: dict
            Additional arguments for pylab.scatter
        err_kwargs: dict
            Additional arguments for pylab.errorbar

        Returns
        -------
        ax: matplotlib.axes
            Axis of the plot
        """
        ax = plt.axes() if ax is None else ax
        ax.scatter(values1, values2, **plt_kwargs)
        if err1 is not None or err2 is not None:
            err_kwargs_ = dict(elinewidth=.5, capsize=0, fmt='.', ms=0, ls='')
            err_kwargs_.update(err_kwargs)
            ax.errorbar(values1, values2, xerr=err1, yerr=err2, **err_kwargs_)
        return ax
    def plot_color(values1, values2, values_color, err1=None, err2=None,
                   ax=None, plt_kwargs={}, add_cb=True, cb_kwargs={},
                   err_kwargs={}):
        """
        Scatter plot with errorbars and color based on input

        Parameters
        ----------
        values1: array
            Component 1
        values2: array
            Component 2
        values_color: array
            Values for color (cmap scale)
        err1: array
            Error of component 1
        err2: array
            Error of component 2
        ax: matplotlib.axes
            Ax to add plot
        plt_kwargs: dict
            Additional arguments for pylab.scatter
        add_cb: bool
            Plot colorbar
        cb_kwargs: dict
            Colorbar arguments
        err_kwargs: dict
            Additional arguments for pylab.errorbar

        Returns
        -------
        ax: matplotlib.axes
            Axis of the plot
        matplotlib.colorbar.Colorbar (optional)
            Colorbar of the recovey rates. Only returned if add_cb=True.
        """
        ax = plt.axes() if ax is None else ax
        isort = np.argsort(values_color)
        xp, yp, zp = [v[isort] for v in (values1, values2, values_color)]
        sc = ax.scatter(xp, yp, c=zp, **plt_kwargs)
        cb_kwargs_ = {'ax':ax}
        cb_kwargs_.update(cb_kwargs)
        cb = plt.colorbar(sc, **cb_kwargs_)
        if err1 is not None or err2 is not None:
            xerr = err1[isort] if err1 is not None else [None for i in isort]
            yerr = err2[isort] if err2 is not None else [None for i in isort]
            #err_kwargs_ = dict(elinewidth=.5, capsize=0, fmt='.', ms=0, ls='')
            err_kwargs_ = {}
            err_kwargs_.update(err_kwargs)
            cols = [cb.mappable.cmap(cb.mappable.norm(c)) for c in zp]
            for i in range(xp.size):
                ax.errorbar(xp[i], yp[i], xerr=xerr[i], yerr=yerr[i],
                    c=cols[i], **err_kwargs_)
        if add_cb:
            return ax, cb
        cb.remove()
        return ax
    def plot_density(values1, values2, bins1=30, bins2=30,
                     ax_rotation=0, rotation_resolution=30,
                     err1=None, err2=None,
                     ax=None, plt_kwargs={},
                     add_cb=True, cb_kwargs={},
                     err_kwargs={}):

        """
        Scatter plot with errorbars and color based on point density

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
        ax_rotation: float
            Angle (in degrees) for rotation of axis of binning. Overwrites use of (bins1, bins2)
        rotation_resolution: int
            Number of bins to be used when ax_rotation!=0.
        err1: array
            Error of component 1
        err2: array
            Error of component 2
        ax: matplotlib.axes
            Ax to add plot
        plt_kwargs: dict
            Additional arguments for pylab.scatter
        add_cb: bool
            Plot colorbar
        cb_kwargs: dict
            Colorbar arguments
        err_kwargs: dict
            Additional arguments for pylab.errorbar

        Returns
        -------
        ax: matplotlib.axes
            Axis of the plot
        matplotlib.colorbar.Colorbar (optional)
            Colorbar of the recovey rates. Only returned if add_cb=True.
        """
        values_color = ph.get_density_colors(values1, values2, bins1, bins2,
            ax_rotation=ax_rotation, rotation_resolution=rotation_resolution)
        return ArrayFuncs.plot_color(values1, values2, values_color=values_color,
                err1=err1, err2=err2, ax=ax, plt_kwargs=plt_kwargs,
                add_cb=add_cb, cb_kwargs=cb_kwargs, err_kwargs=err_kwargs)
    def _plot_panel(plot_function, values_panel, bins_panel,
                    panel_kwargs_list=None, panel_kwargs_errlist=None,
                    fig_kwargs={}, add_label=False, label_format=lambda v: v,
                    plt_kwargs={}, err_kwargs={}, **plt_func_kwargs):
        """
        Helper function to makes plots in panels

        Parameters
        ----------
        plot_function: function
            Plot function
        values_panel: array
            Values to bin data in panels
        bins_panel: array
            Bins defining panels
        panel_kwargs_list: list, None
            List of additional arguments for plotting each panel (using pylab.plot).
            Must have same size as len(bins2)-1
        panel_kwargs_errlist: list, None
            List of additional arguments for plotting each panel (using pylab.errorbar).
            Must have same size as len(bins2)-1
        fig_kwargs: dict
            Additional arguments for plt.subplots
        add_label: bool
            Add bin label to panel
        label_format: function
            Function to format the values of the bins
        **plt_func_kwargs
            All other parameters to be passed to plot_function


        Returns
        -------
        Same as plot_function
        """
        nj = int(np.ceil(np.sqrt(len(bins_panel[:-1]))))
        ni = int(np.ceil(len(bins_panel[:-1])/float(nj)))
        fig_kwargs_ = dict(sharex=True, sharey=True, figsize=(8, 6))
        fig_kwargs_.update(fig_kwargs)
        f, axes = plt.subplots(ni, nj, **fig_kwargs_)
        panel_kwargs_list = none_val(panel_kwargs_list, [{} for m in bins_panel[:-1]])
        panel_kwargs_errlist = none_val(panel_kwargs_errlist, [{} for m in bins_panel[:-1]])
        masks = [(values_panel>=v0)*(values_panel<v1) for v0, v1 in zip(bins_panel, bins_panel[1:])]
        for ax, mask, p_kwargs, p_e_kwargs in zip(axes.flatten(), masks,
                                    panel_kwargs_list, panel_kwargs_errlist):
            ph.add_grid(ax)
            kwargs = {}
            kwargs.update(plt_kwargs)
            kwargs.update(p_kwargs)
            kwargs_e = {}
            kwargs_e.update(err_kwargs)
            kwargs_e.update(p_e_kwargs)
            plot_function(ax=ax, plt_kwargs=kwargs, err_kwargs=kwargs_e,
                **{k:v[mask] if (hasattr(v, '__len__') and len(v)==mask.size) else v
                for k, v in plt_func_kwargs.items()})
        for ax in axes.flatten()[len(bins_panel)-1:]:
            ax.axis('off')
        if add_label:
            ph.add_panel_bin_label(axes,  edges2[:-1], edges2[1:],
                                   format_func=label_format)
        return f, axes
    def plot_panel(values1, values2, values_panel, bins_panel,
                   err1=None, err2=None, plt_kwargs={}, err_kwargs={},
                   panel_kwargs_list=None, panel_kwargs_errlist=None,
                   fig_kwargs={}, add_label=False, label_format=lambda v: v):
        """
        Scatter plot with errorbars and color based on input with panels

        Parameters
        ----------
        values1: array
            Component 1
        values2: array
            Component 2
        values_panel: array
            Values to bin data in panels
        bins_panel: array
            Bins defining panels
        err1: array
            Error of component 1
        err2: array
            Error of component 2
        ax: matplotlib.axes
            Ax to add plot
        plt_kwargs: dict
            Additional arguments for pylab.scatter
        err_kwargs: dict
            Additional arguments for pylab.errorbar
        panel_kwargs_list: list, None
            List of additional arguments for plotting each panel (using pylab.plot).
            Must have same size as len(bins2)-1
        panel_kwargs_errlist: list, None
            List of additional arguments for plotting each panel (using pylab.errorbar).
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
        return ArrayFuncs._plot_panel(
            # _plot_panel arguments
            plot_function=ArrayFuncs.plot,
            values_panel=values_panel, bins_panel=bins_panel,
            panel_kwargs_list=panel_kwargs_list, panel_kwargs_errlist=panel_kwargs_errlist,
            fig_kwargs=fig_kwargs, add_label=add_label, label_format=label_format,
            # plot arguments
            values1=values1, values2=values2, err1=err1, err2=err2,
            plt_kwargs=plt_kwargs, err_kwargs=err_kwargs,
            )
    def plot_color_panel(values1, values2, values_color, values_panel, bins_panel,
                   err1=None, err2=None, plt_kwargs={}, err_kwargs={},
                   panel_kwargs_list=None, panel_kwargs_errlist=None,
                   fig_kwargs={}, add_label=False, label_format=lambda v: v):
        """
        Scatter plot with errorbars and color based on input with panels

        Parameters
        ----------
        values1: array
            Component 1
        values2: array
            Component 2
        values_color: array
            Values for color (cmap scale)
        values_panel: array
            Values to bin data in panels
        bins_panel: array
            Bins defining panels
        err1: array
            Error of component 1
        err2: array
            Error of component 2
        ax: matplotlib.axes
            Ax to add plot
        plt_kwargs: dict
            Additional arguments for pylab.scatter
        err_kwargs: dict
            Additional arguments for pylab.errorbar
        panel_kwargs_list: list, None
            List of additional arguments for plotting each panel (using pylab.plot).
            Must have same size as len(bins2)-1
        panel_kwargs_errlist: list, None
            List of additional arguments for plotting each panel (using pylab.errorbar).
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
        return ArrayFuncs._plot_panel(
            # _plot_panel arguments
            plot_function=ArrayFuncs.plot_color,
            values_panel=values_panel, bins_panel=bins_panel,
            panel_kwargs_list=panel_kwargs_list, panel_kwargs_errlist=panel_kwargs_errlist,
            fig_kwargs=fig_kwargs, add_label=add_label, label_format=label_format,
            # plot_color arguments
            values1=values1, values2=values2, err1=err1, err2=err2,
            values_color=values_color,
            plt_kwargs=plt_kwargs, err_kwargs=err_kwargs,
            )
    def plot_density_panel(values1, values2, values_panel, bins_panel,
        bins1=30, bins2=30, ax_rotation=0, rotation_resolution=30,
        err1=None, err2=None, plt_kwargs={},add_cb=True, cb_kwargs={},
        err_kwargs={}, panel_kwargs_list=None, panel_kwargs_errlist=None,
        fig_kwargs={}, add_label=False, label_format=lambda v: v):

        """
        Scatter plot with errorbars and color based on point density

        Parameters
        ----------
        values1: array
            Component 1
        values2: array
            Component 2
        values_panel: array
            Values to bin data in panels
        bins_panel: array
            Bins defining panels
        bins1: array, int
            Bins for component 1
        bins2: array, int
            Bins for component 2
        ax_rotation: float
            Angle (in degrees) for rotation of axis of binning. Overwrites use of (bins1, bins2)
        rotation_resolution: int
            Number of bins to be used when ax_rotation!=0.
        err1: array
            Error of component 1
        err2: array
            Error of component 2
        ax: matplotlib.axes
            Ax to add plot
        plt_kwargs: dict
            Additional arguments for pylab.scatter
        add_cb: bool
            Plot colorbar
        cb_kwargs: dict
            Colorbar arguments
        err_kwargs: dict
            Additional arguments for pylab.errorbar
        panel_kwargs_list: list, None
            List of additional arguments for plotting each panel (using pylab.plot).
            Must have same size as len(bins2)-1
        panel_kwargs_errlist: list, None
            List of additional arguments for plotting each panel (using pylab.errorbar).
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
        return ArrayFuncs._plot_panel(
            # _plot_panel arguments
            plot_function=ArrayFuncs.plot_density,
            values_panel=values_panel, bins_panel=bins_panel,
            panel_kwargs_list=panel_kwargs_list, panel_kwargs_errlist=panel_kwargs_errlist,
            fig_kwargs=fig_kwargs, add_label=add_label, label_format=label_format,
            # plot_density arguments
            values1=values1, values2=values2, err1=err1, err2=err2, bins1=bins1, bins2=bins2,
            ax_rotation=ax_rotation, rotation_resolution=rotation_resolution,
            plt_kwargs=plt_kwargs, err_kwargs=err_kwargs,
            )
class CatalogFuncs():
    """
    Class of plot functions with clevar.Catalog as inputs
    """
    def _get_err(mp, col, add_err):
        """
        Get err values for plotting

        Parameters
        ----------
        mp: clevar.match.MatchedPairs
            Matched catalogs
        col: str
            Name of column to be plotted
        add_err: bool
            Add errorbars

        Returns
        -------
        err1, err2: ndarray, None
            Value of errors for plotting
        """
        return (mp.data1[f'{col}_err'] if add_err and f'{col}_err' in  mp.data1.colnames\
                else None,
                mp.data2[f'{col}_err'] if add_err and f'{col}_err' in  mp.data2.colnames\
                else None)
        err1, err2 = None, None
        if add_err and f'{col}_err' in  mp.data1.colnames:
            err1 = mp.data1[f'{col}_err']
        if add_err and f'{col}_err' in  mp.data2.colnames:
            err2 = mp.data2[f'{col}_err']
        return err1, err2
    def plot(cat1, cat2, matching_type, col, add_err=False, **kwargs):
        """
        Adapts a CatalogFuncs function to use a ArrayFuncs function.

        Parameters
        ----------
        pltfunc: function
            ArrayFuncs function
        cat1: clevar.Catalog
            Catalog with matching information
        cat2: clevar.Catalog
            Catalog matched to
        matching_type: str
            Type of matching to be considered. Must be in:
            'cross', 'self' (catalog 1), 'other'(catalog 2)
        col: str
            Name of column to be plotted
        add_err: bool
            Add errorbars

        Other parameters
        ----------------
        ax: matplotlib.axes
            Ax to add plot
        plt_kwargs: dict
            Additional arguments for pylab.scatter
        err_kwargs: dict
            Additional arguments for pylab.errorbar

        Returns
        -------
        ax: matplotlib.axes
            Axis of the plot
        """
        mp = MatchedPairs(cat1, cat2, matching_type)
        err1, err2 = CatalogFuncs._get_err(mp, col, add_err)
        ax = ArrayFuncs.plot(mp.data1[col], mp.data2[col], err1=err1, err2=err2, **kwargs)
        return ax
    def plot_color(cat1, cat2, matching_type, col, col_color,
                   color1=True, add_err=False, **kwargs):
        """
        Adapts a CatalogFuncs function to use a ArrayFuncs function.

        Parameters
        ----------
        pltfunc: function
            ArrayFuncs function
        cat1: clevar.Catalog
            Catalog with matching information
        cat2: clevar.Catalog
            Catalog matched to
        matching_type: str
            Type of matching to be considered. Must be in:
            'cross', 'self' (catalog 1), 'other'(catalog 2)
        col: str
            Name of column to be plotted
        col_color: str
            Name of column for color
        color1: bool
            Use catalog 1 for color. If false uses catalog 2
        add_err: bool
            Add errorbars

        Other parameters
        ----------------
        ax: matplotlib.axes
            Ax to add plot
        plt_kwargs: dict
            Additional arguments for pylab.scatter
        add_cb: bool
            Plot colorbar
        cb_kwargs: dict
            Colorbar arguments
        err_kwargs: dict
            Additional arguments for pylab.errorbar

        Returns
        -------
        ax: matplotlib.axes
            Axis of the plot
        matplotlib.colorbar.Colorbar (optional)
            Colorbar of the recovey rates. Only returned if add_cb=True.
        """
        mp = MatchedPairs(cat1, cat2, matching_type)
        err1, err2 = CatalogFuncs._get_err(mp, col, add_err)
        values_color = mp.data1[col_color] if color1 else mp.data2[col_color]
        return ArrayFuncs.plot_color(mp.data1[col], mp.data2[col],
                values_color=values_color, err1=err1, err2=err2, **kwargs)
    def plot_density(cat1, cat2, matching_type, col, bins=30, add_err=False,
                     ax_rotation=0, rotation_resolution=30, **kwargs):
        """
        Adapts a CatalogFuncs function to use a ArrayFuncs function.

        Parameters
        ----------
        pltfunc: function
            ArrayFuncs function
        cat1: clevar.Catalog
            Catalog with matching information
        cat2: clevar.Catalog
            Catalog matched to
        matching_type: str
            Type of matching to be considered. Must be in:
            'cross', 'self', 'other', 'multi_self', 'multi_other', 'multi_join'
        col: str
            Name of column to be plotted
        bins: array, int
            Bins for density
        add_err: bool
            Add errorbars

        Other parameters
        ----------------
        ax: matplotlib.axes
            Ax to add plot
        plt_kwargs: dict
            Additional arguments for pylab.scatter
        add_cb: bool
            Plot colorbar
        cb_kwargs: dict
            Colorbar arguments
        err_kwargs: dict
            Additional arguments for pylab.errorbar
        ax_rotation: float
            Angle (in degrees) for rotation of axis of binning. Overwrites use of (bins1, bins2)
        rotation_resolution: int
            Number of bins to be used when ax_rotation!=0.

        Returns
        -------
        ax: matplotlib.axes
            Axis of the plot
        matplotlib.colorbar.Colorbar (optional)
            Colorbar of the recovey rates. Only returned if add_cb=True.
        """
        mp = MatchedPairs(cat1, cat2, matching_type)
        err1, err2 = CatalogFuncs._get_err(mp, col, add_err)
        return ArrayFuncs.plot_density(mp.data1[col], mp.data2[col],
                bins1=bins, bins2=bins, ax_rotation=ax_rotation,
                rotation_resolution=rotation_resolution,
                err1=err1, err2=err2, **kwargs)
    def plot_panel(cat1, cat2, matching_type, col,
        col_panel, panel_bins, panel_cat1=True,
        add_err=False, **kwargs):
        """
        Adapts a CatalogFuncs function to use a ArrayFuncs function.

        Parameters
        ----------
        pltfunc: function
            ArrayFuncs function
        cat1: clevar.Catalog
            Catalog with matching information
        cat2: clevar.Catalog
            Catalog matched to
        matching_type: str
            Type of matching to be considered. Must be in:
            'cross', 'self' (catalog 1), 'other'(catalog 2)
        col: str
            Name of column to be plotted
        col_panel: str
            Name of column to make panels
        panel_bins: array
            Bins to make panels
        panel_cat1: bool
            Used catalog 1 for col_panel. If false uses catalog 2
        add_err: bool
            Add errorbars

        Other parameters
        ----------------
        plt_kwargs: dict
            Additional arguments for pylab.scatter
        err_kwargs: dict
            Additional arguments for pylab.errorbar
        panel_kwargs_list: list, None
            List of additional arguments for plotting each panel (using pylab.plot).
            Must have same size as len(bins2)-1
        panel_kwargs_errlist: list, None
            List of additional arguments for plotting each panel (using pylab.errorbar).
            Must have same size as len(bins2)-1
        fig_kwargs: dict
            Additional arguments for plt.subplots
        add_label: bool
            Add bin label to panel
        label_format: function
            Function to format the values of the bins

        Returns
        -------
        ax: matplotlib.axes
            Axis of the plot
        matplotlib.colorbar.Colorbar (optional)
            Colorbar of the recovey rates. Only returned if add_cb=True.
        """
        mp = MatchedPairs(cat1, cat2, matching_type)
        err1, err2 = CatalogFuncs._get_err(mp, col, add_err)
        return ArrayFuncs.plot_panel(mp.data1[col], mp.data2[col], err1=err1, err2=err2,
                values_panel=mp.data1[col_panel] if panel_cat1 else mp.data2[col_panel],
                bins_panel=panel_bins, **kwargs)
    def plot_color_panel(cat1, cat2, matching_type, col, col_color,
        col_panel, panel_bins, panel_cat1=True, color1=True, add_err=False, **kwargs):
        """
        Adapts a CatalogFuncs function to use a ArrayFuncs function.

        Parameters
        ----------
        pltfunc: function
            ArrayFuncs function
        cat1: clevar.Catalog
            Catalog with matching information
        cat2: clevar.Catalog
            Catalog matched to
        matching_type: str
            Type of matching to be considered. Must be in:
            'cross', 'self' (catalog 1), 'other'(catalog 2)
        col: str
            Name of column to be plotted
        col_color: str
            Name of column for color
        col_panel: str
            Name of column to make panels
        panel_bins: array
            Bins to make panels
        panel_cat1: bool
            Used catalog 1 for col_panel. If false uses catalog 2
        color1: bool
            Use catalog 1 for color. If false uses catalog 2
        add_err: bool
            Add errorbars

        Other parameters
        ----------------
        plt_kwargs: dict
            Additional arguments for pylab.scatter
        add_cb: bool
            Plot colorbar
        cb_kwargs: dict
            Colorbar arguments
        err_kwargs: dict
            Additional arguments for pylab.errorbar
        panel_kwargs_list: list, None
            List of additional arguments for plotting each panel (using pylab.plot).
            Must have same size as len(bins2)-1
        panel_kwargs_errlist: list, None
            List of additional arguments for plotting each panel (using pylab.errorbar).
            Must have same size as len(bins2)-1
        fig_kwargs: dict
            Additional arguments for plt.subplots
        add_label: bool
            Add bin label to panel
        label_format: function
            Function to format the values of the bins

        Returns
        -------
        ax: matplotlib.axes
            Axis of the plot
        matplotlib.colorbar.Colorbar (optional)
            Colorbar of the recovey rates. Only returned if add_cb=True.
        """
        mp = MatchedPairs(cat1, cat2, matching_type)
        err1, err2 = CatalogFuncs._get_err(mp, col, add_err)
        values_color = mp.data1[col_color] if color1 else mp.data2[col_color]
        return ArrayFuncs.plot_color_panel(mp.data1[col], mp.data2[col],
                values_panel=mp.data1[col_panel] if panel_cat1 else mp.data2[col_panel],
                bins_panel=panel_bins,
                values_color=values_color, err1=err1, err2=err2,
                **kwargs)
    def plot_density_panel(cat1, cat2, matching_type, col,
        col_panel, panel_bins, panel_cat1=True,
        bins=30, add_err=False, ax_rotation=0, rotation_resolution=30, **kwargs):
        """
        Adapts a CatalogFuncs function to use a ArrayFuncs function.

        Parameters
        ----------
        pltfunc: function
            ArrayFuncs function
        cat1: clevar.Catalog
            Catalog with matching information
        cat2: clevar.Catalog
            Catalog matched to
        matching_type: str
            Type of matching to be considered. Must be in:
            'cross', 'self', 'other', 'multi_self', 'multi_other', 'multi_join'
        col: str
            Name of column to be plotted
        col_panel: str
            Name of column to make panels
        panel_bins: array
            Bins to make panels
        panel_cat1: bool
            Used catalog 1 for col_panel. If false uses catalog 2
        bins: array, int
            Bins for density
        add_err: bool
            Add errorbars

        Other parameters
        ----------------
        ax: matplotlib.axes
            Ax to add plot
        plt_kwargs: dict
            Additional arguments for pylab.scatter
        add_cb: bool
            Plot colorbar
        cb_kwargs: dict
            Colorbar arguments
        err_kwargs: dict
            Additional arguments for pylab.errorbar
        ax_rotation: float
            Angle (in degrees) for rotation of axis of binning. Overwrites use of (bins1, bins2)
        rotation_resolution: int
            Number of bins to be used when ax_rotation!=0.
        panel_kwargs_list: list, None
            List of additional arguments for plotting each panel (using pylab.plot).
            Must have same size as len(bins2)-1
        panel_kwargs_errlist: list, None
            List of additional arguments for plotting each panel (using pylab.errorbar).
            Must have same size as len(bins2)-1
        fig_kwargs: dict
            Additional arguments for plt.subplots
        add_label: bool
            Add bin label to panel
        label_format: function
            Function to format the values of the bins

        Returns
        -------
        ax: matplotlib.axes
            Axis of the plot
        matplotlib.colorbar.Colorbar (optional)
            Colorbar of the recovey rates. Only returned if add_cb=True.
        """
        mp = MatchedPairs(cat1, cat2, matching_type)
        err1, err2 = CatalogFuncs._get_err(mp, col, add_err)
        return ArrayFuncs.plot_density_panel(mp.data1[col], mp.data2[col],
                values_panel=mp.data1[col_panel] if panel_cat1 else mp.data2[col_panel],
                bins_panel=panel_bins,
                bins1=bins, bins2=bins, ax_rotation=ax_rotation,
                rotation_resolution=rotation_resolution,
                err1=err1, err2=err2, **kwargs)
#def plot_z():
