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
        ax = none_val(ax, plt.axes())
        isort = np.argsort(values_color)
        xp, yp, zp = [v[isort] for v in (values1, values2, values_color)]
        sc = ax.scatter(xp, yp, c=zp, **plt_kwargs)
        cb = plt.colorbar(sc, **cb_kwargs)
        if err1 is not None or err2 is not None:
            xerr = err1[isort] if err1 is not None else [None for i in isort]
            yerr = err2[isort] if err2 is not None else [None for i in isort]
            err_kwargs_ = dict(elinewidth=.5, capsize=0, fmt='.', ms=0, ls='')
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
    def plot_color(cat1, cat2, matching_type, col, col_color, color1=True, add_err=False,
               ax=None, plt_kwargs={}, add_cb=True, cb_kwargs={},
               err_kwargs={}):
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
                values_color=values_color, err1=err1, err2=err2,
                ax=ax, plt_kwargs=plt_kwargs, add_cb=add_cb,
                cb_kwargs=cb_kwargs, err_kwargs=err_kwargs)
    def plot_density(cat1, cat2, matching_type, col, bins=30, add_err=False,
                     ax_rotation=0, rotation_resolution=30,
                     ax=None, plt_kwargs={}, add_cb=True, cb_kwargs={},
                     err_kwargs={}):
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
                err1=err1, err2=err2,
                ax=ax, plt_kwargs=plt_kwargs, add_cb=add_cb,
                cb_kwargs=cb_kwargs, err_kwargs=err_kwargs)
#def plot_z():
