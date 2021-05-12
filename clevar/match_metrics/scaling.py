# Set mpl backend run plots on github actions
import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == 'test':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
from matplotlib.ticker import ScalarFormatter, NullFormatter
import pylab as plt
import numpy as np
from scipy.optimize import curve_fit

from ..utils import none_val, autobins, binmasks
from ..match import MatchedPairs
from . import plot_helper as ph

def _prep_fit_data(x, y, yerr=None, statistics='mean', bins_x=None, bins_y=None):
    """
    Prepare data for fit with binning.

    Parameters
    ----------
    x: array
        Input values for fit
    y: array
        Values to be fitted
    yerr: array, None
        Errors of y
    statistics: str
        Statistics to be used. Options are:
            `individual` - Use each point
            `mode` - Use mode of y distribution in each x bin, requires bins_y.
            `mean` - Use mean of y distribution in each x bin, requires bins_y.
    bins_x: array, None
        Bins for component x
    bins_y: array, None
        Bins for component y

    Returns
    -------
    xdata, ydata, errdata: array
        Data for fit
    """
    if statistics=='individual':
        return x, y, yerr
    elif statistics=='mode':
        bins_hist = autobins(y, bins_y)
        bins_hist_m = 0.5*(bins_hist[1:]+bins_hist[:-1])
        stat_func = lambda vals: bins_hist_m[np.histogram(vals, bins=bins_hist)[0].argmax()]
    elif statistics=='mean':
        stat_func = lambda vals: np.mean(vals)
    else:
        raise ValueError(f'statistics ({statistics}) must be in (individual, mean, mode)')
    point_masks = [m for m in binmasks(x, autobins(x, bins_x)) if m[m].size>1]
    err = np.zeros(len(y)) if yerr is None else yerr
    err_func = lambda vals, err: np.mean(np.sqrt(np.std(vals)**2+err**2))
    return np.transpose([[np.mean(x[m]), stat_func(y[m]), err_func(y[m], err[m])]
                            for m in point_masks])
class ArrayFuncs():
    """
    Class of plot functions with arrays as inputs
    """
    def _add_bindata_and_powlawfit(ax, values1, values2, err2, log=False, **kwargs):
        """
        Add binned data and powerlaw fit to plot

        Parameters
        ----------
        ax: matplotlib.axes
            Ax to add plot
        values1: array
            Component 1
        values2: array
            Component 2
        err2: array
            Error of component 2
        log: bool
            Bin and fit in log values
        mode: str
            Statistics to be used in fit. Options are:
                `individual` - Use each point
                `mode` - Use mode of component 2 distribution in each comp 1 bin, requires bins2.
                `mean` - Use mean of component 2 distribution in each comp 1 bin, requires bins2.
        bins1: array, None
            Bins for component 1 (default=10).
        bins2: array, None
            Bins for component 2 (default=30).
        add_bindata: bool
            Plot binned data used for fit.
        add_fit: bool
            Fit and plot binned dat.
        bindata_kwargs: dict
            Additional arguments for pylab.errorbar.
        plt_kwargs: dict
            Additional arguments for plot of fit pylab.scatter.
        legend_kwargs: dict
            Additional arguments for plt.legend.
        """
        # Default parameters
        mode = kwargs.get('mode', 'mode')
        bins1 = kwargs.get('bins1', 10)
        bins2 = kwargs.get('bins2', 30)
        legend_kwargs = kwargs.get('legend_kwargs', {})
        add_bindata = kwargs.get('add_bindata', False)
        bindata_kwargs = kwargs.get('bindata_kwargs', {})
        add_fit = kwargs.get('add_fit', False)
        plot_kwargs = kwargs.get('plot_kwargs', {})
        if (not add_bindata) and (not add_fit):
            return
        # set log/lin funcs
        tfunc, ifunc = (np.log, np.exp) if log else (lambda x:x, lambda x:x)
        # data
        vbin_1, vbin_2, vbin_err2 = _prep_fit_data(tfunc(values1), tfunc(values2),
                    bins_x=tfunc(bins1) if hasattr(bins1, '__len__') else bins1,
                    bins_y=tfunc(bins2) if hasattr(bins2, '__len__') else bins2,
                    yerr=None if (err2 is None or not log) else err2/values2,
                    statistics=mode)
        if add_bindata:
            eb_kwargs_ = {'elinewidth': 1, 'capsize': 2, 'fmt': '.',
                          'ms': 10, 'ls': '', 'color': 'm'}
            eb_kwargs_.update(bindata_kwargs)
            ax.errorbar(ifunc(vbin_1), ifunc(vbin_2),
                        yerr=vbin_err2*ifunc(vbin_2) if log else vbin_err2,
                        **eb_kwargs_)
        # fit
        if add_fit:
            pw_func = lambda x, a, b: a*x+b
            fit, cov = curve_fit(pw_func, vbin_1, vbin_2, sigma=vbin_err2)
            fit1_lab = f'{ifunc(fit[1]):.2f}' if ifunc(fit[1])<100\
                  else f'10^{{{np.log10(ifunc(fit[1])):.2f}}}'
            fit_label = f'$f(x)={fit1_lab}\;x^{{{fit[0]:.2f}}}$' if log\
                else f'$f(x)={fit[0]:.2f}\;x%s$'%(fit1_lab if fit[1]<0 else '+'+fit1_lab)
            plot_kwargs_ = {'color': 'r', 'label': fit_label}
            plot_kwargs_.update(plot_kwargs)
            sort = np.argsort(values1)
            ax.plot(values1[sort], ifunc(pw_func(tfunc(values1)[sort], *fit)), **plot_kwargs_)
        # legend
        if any(c.get_label()[0]!='_' for c in ax.collections+ax.lines):
            legend_kwargs_ = {}
            legend_kwargs_.update(legend_kwargs)
            ax.legend(**legend_kwargs_)
    def plot(values1, values2, err1=None, err2=None,
             ax=None, plt_kwargs={}, err_kwargs={}, **kwargs):
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

        Fit Parameters
        --------------
        add_bindata: bool
            Plot binned data used for fit.
        add_fit: bool
            Fit and plot binned dat.
        fit_err2: array
            Error of component 2
        fit_mode: str
            Statistics to be used in fit. Options are:
                `individual` - Use each point
                `mode` - Use mode of component 2 distribution in each comp 1 bin, requires fit_bins2.
                `mean` - Use mean of component 2 distribution in each comp 1 bin, requires fit_bins2.
        fit_bins1: array, None
            Bins for component 1 (default=10).
        fit_bins2: array, None
            Bins for component 2 (default=30).
        fit_legend_kwargs: dict
            Additional arguments for plt.legend.
        fit_bindata_kwargs: dict
            Additional arguments for pylab.errorbar.
        fit_plt_kwargs: dict
            Additional arguments for plot of fit pylab.scatter.

        Returns
        -------
        ax: matplotlib.axes
            Axis of the plot
        """
        ax = plt.axes() if ax is None else ax
        ph.add_grid(ax)
        plt_kwargs_ = {'s':1}
        plt_kwargs_.update(plt_kwargs)
        ax.scatter(values1, values2, **plt_kwargs_)
        if err1 is not None or err2 is not None:
            err_kwargs_ = dict(elinewidth=.5, capsize=0, fmt='.', ms=0, ls='')
            err_kwargs_.update(err_kwargs)
            ax.errorbar(values1, values2, xerr=err1, yerr=err2, **err_kwargs_)
        # Bindata and fit
        kwargs['fit_err2'] = kwargs.get('fit_err2', err2)
        kwargs['fit_add_fit'] = kwargs.get('add_fit', False)
        kwargs['fit_add_bindata'] = kwargs.get('add_bindata', kwargs['fit_add_fit'])
        ArrayFuncs._add_bindata_and_powlawfit(ax, values1, values2,
                                         **{k[4:]:v for k, v in kwargs.items()
                                             if k[:4]=='fit_'})
        return ax
    def plot_color(values1, values2, values_color, err1=None, err2=None,
                   ax=None, plt_kwargs={}, add_cb=True, cb_kwargs={},
                   err_kwargs={}, **kwargs):
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

        Fit Parameters
        --------------
        add_bindata: bool
            Plot binned data used for fit.
        add_fit: bool
            Fit and plot binned dat.
        fit_err2: array
            Error of component 2
        fit_mode: str
            Statistics to be used in fit. Options are:
                `individual` - Use each point
                `mode` - Use mode of component 2 distribution in each comp 1 bin, requires fit_bins2.
                `mean` - Use mean of component 2 distribution in each comp 1 bin, requires fit_bins2.
        fit_bins1: array, None
            Bins for component 1 (default=10).
        fit_bins2: array, None
            Bins for component 2 (default=30).
        fit_legend_kwargs: dict
            Additional arguments for plt.legend.
        fit_bindata_kwargs: dict
            Additional arguments for pylab.errorbar.
        add_fit: bool
            Fit and plot binned dat.
        fit_plt_kwargs: dict
            Additional arguments for plot of fit pylab.scatter.

        Returns
        -------
        ax: matplotlib.axes
            Axis of the plot
        matplotlib.colorbar.Colorbar (optional)
            Colorbar of the recovey rates. Only returned if add_cb=True.
        """
        ax = plt.axes() if ax is None else ax
        ph.add_grid(ax)
        if len(values1)==0:
            return (ax, None) if add_cb else ax
        isort = np.argsort(values_color)
        xp, yp, zp = [v[isort] for v in (values1, values2, values_color)]
        plt_kwargs_ = {'s':1}
        plt_kwargs_.update(plt_kwargs)
        sc = ax.scatter(xp, yp, c=zp, **plt_kwargs_)
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
        # Bindata and fit
        kwargs['fit_err2'] = kwargs.get('fit_err2', err2)
        kwargs['fit_add_fit'] = kwargs.get('add_fit', False)
        kwargs['fit_add_bindata'] = kwargs.get('add_bindata', kwargs['fit_add_fit'])
        ArrayFuncs._add_bindata_and_powlawfit(ax, values1, values2,
                                         **{k[4:]:v for k, v in kwargs.items()
                                             if k[:4]=='fit_'})
        if add_cb:
            return ax, cb
        cb.remove()
        return ax
    def plot_density(values1, values2, bins1=30, bins2=30,
                     ax_rotation=0, rotation_resolution=30,
                     xscale='linear', yscale='linear',
                     err1=None, err2=None,
                     ax=None, plt_kwargs={},
                     add_cb=True, cb_kwargs={},
                     err_kwargs={}, **kwargs):

        """
        Scatter plot with errorbars and color based on point density

        Parameters
        ----------
        values1: array
            Component 1
        values2: array
            Component 2
        bins1: array, int
            Bins for component 1 (for density colors).
        bins2: array, int
            Bins for component 2 (for density colors).
        ax_rotation: float
            Angle (in degrees) for rotation of axis of binning. Overwrites use of (bins1, bins2)
        rotation_resolution: int
            Number of bins to be used when ax_rotation!=0.
        xscale: str
            Scale xaxis.
        yscale: str
            Scale yaxis.
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

        Fit Parameters
        --------------
        add_bindata: bool
            Plot binned data used for fit.
        add_fit: bool
            Fit and plot binned dat.
        fit_err2: array
            Error of component 2
        fit_mode: str
            Statistics to be used in fit. Options are:
                `individual` - Use each point
                `mode` - Use mode of component 2 distribution in each comp 1 bin, requires fit_bins2.
                `mean` - Use mean of component 2 distribution in each comp 1 bin, requires fit_bins2.
        fit_bins1: array, None
            Bins for component 1 (default=10).
        fit_bins2: array, None
            Bins for component 2 (default=30).
        fit_legend_kwargs: dict
            Additional arguments for plt.legend.
        fit_bindata_kwargs: dict
            Additional arguments for pylab.errorbar.
        add_fit: bool
            Fit and plot binned dat.
        fit_plt_kwargs: dict
            Additional arguments for plot of fit pylab.scatter.

        Returns
        -------
        ax: matplotlib.axes
            Axis of the plot
        matplotlib.colorbar.Colorbar (optional)
            Colorbar of the recovey rates. Only returned if add_cb=True.
        """
        values_color = ph.get_density_colors(values1, values2, bins1, bins2,
            ax_rotation=ax_rotation, rotation_resolution=rotation_resolution,
            xscale=xscale, yscale=yscale) if len(values1)>0 else []
        return ArrayFuncs.plot_color(values1, values2, values_color=values_color,
                err1=err1, err2=err2, ax=ax, plt_kwargs=plt_kwargs,
                add_cb=add_cb, cb_kwargs=cb_kwargs, err_kwargs=err_kwargs, **kwargs)
    def _plot_panel(plot_function, values_panel, bins_panel,
                    panel_kwargs_list=None, panel_kwargs_errlist=None,
                    fig_kwargs={}, add_label=True, label_format=lambda v: v,
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

        Fit Parameters
        --------------
        add_bindata: bool
            Plot binned data used for fit.
        add_fit: bool
            Fit and plot binned dat.
        fit_err2: array
            Error of component 2
        fit_mode: str
            Statistics to be used in fit. Options are:
                `individual` - Use each point
                `mode` - Use mode of component 2 distribution in each comp 1 bin, requires fit_bins2.
                `mean` - Use mean of component 2 distribution in each comp 1 bin, requires fit_bins2.
        fit_bins1: array, None
            Bins for component 1 (default=10).
        fit_bins2: array, None
            Bins for component 2 (default=30).
        fit_legend_kwargs: dict
            Additional arguments for plt.legend.
        fit_bindata_kwargs: dict
            Additional arguments for pylab.errorbar.
        add_fit: bool
            Fit and plot binned dat.
        fit_plt_kwargs: dict
            Additional arguments for plot of fit pylab.scatter.

        Returns
        -------
        Same as plot_function
        """
        edges = bins_panel if hasattr(bins_panel, '__len__') else\
            np.linspace(min(values_panel), max(values_panel), bins_panel)
        nj = int(np.ceil(np.sqrt(len(edges[:-1]))))
        ni = int(np.ceil(len(edges[:-1])/float(nj)))
        fig_kwargs_ = dict(sharex=True, sharey=True, figsize=(8, 6))
        fig_kwargs_.update(fig_kwargs)
        f, axes = plt.subplots(ni, nj, **fig_kwargs_)
        panel_kwargs_list = none_val(panel_kwargs_list, [{} for m in edges[:-1]])
        panel_kwargs_errlist = none_val(panel_kwargs_errlist, [{} for m in edges[:-1]])
        masks = [(values_panel>=v0)*(values_panel<v1) for v0, v1 in zip(edges, edges[1:])]
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
        for ax in axes.flatten()[len(edges)-1:]:
            ax.axis('off')
        if add_label:
            ph.add_panel_bin_label(axes,  edges[:-1], edges[1:],
                                   format_func=label_format)
        return f, axes
    def plot_panel(values1, values2, values_panel, bins_panel,
                   err1=None, err2=None, plt_kwargs={}, err_kwargs={},
                   panel_kwargs_list=None, panel_kwargs_errlist=None,
                   fig_kwargs={}, add_label=True, label_format=lambda v: v, **kwargs):
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

        Fit Parameters
        --------------
        add_bindata: bool
            Plot binned data used for fit.
        add_fit: bool
            Fit and plot binned dat.
        fit_err2: array
            Error of component 2
        fit_mode: str
            Statistics to be used in fit. Options are:
                `individual` - Use each point
                `mode` - Use mode of component 2 distribution in each comp 1 bin, requires fit_bins2.
                `mean` - Use mean of component 2 distribution in each comp 1 bin, requires fit_bins2.
        fit_bins1: array, None
            Bins for component 1 (default=10).
        fit_bins2: array, None
            Bins for component 2 (default=30).
        fit_legend_kwargs: dict
            Additional arguments for plt.legend.
        fit_bindata_kwargs: dict
            Additional arguments for pylab.errorbar.
        fit_plt_kwargs: dict
            Additional arguments for plot of fit pylab.scatter.

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
            # for fit
            **kwargs,
            )
    def plot_color_panel(values1, values2, values_color, values_panel, bins_panel,
                   err1=None, err2=None, plt_kwargs={}, err_kwargs={},
                   panel_kwargs_list=None, panel_kwargs_errlist=None,
                   fig_kwargs={}, add_label=True, label_format=lambda v: v, **kwargs):
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

        Fit Parameters
        --------------
        add_bindata: bool
            Plot binned data used for fit.
        add_fit: bool
            Fit and plot binned dat.
        fit_err2: array
            Error of component 2
        fit_mode: str
            Statistics to be used in fit. Options are:
                `individual` - Use each point
                `mode` - Use mode of component 2 distribution in each comp 1 bin, requires fit_bins2.
                `mean` - Use mean of component 2 distribution in each comp 1 bin, requires fit_bins2.
        fit_bins1: array, None
            Bins for component 1 (default=10).
        fit_bins2: array, None
            Bins for component 2 (default=30).
        fit_legend_kwargs: dict
            Additional arguments for plt.legend.
        fit_bindata_kwargs: dict
            Additional arguments for pylab.errorbar.
        fit_plt_kwargs: dict
            Additional arguments for plot of fit pylab.scatter.

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
            # for fit
            **kwargs,
            )
    def plot_density_panel(values1, values2, values_panel, bins_panel,
        bins1=30, bins2=30, ax_rotation=0, rotation_resolution=30,
        xscale='linear', yscale='linear',
        err1=None, err2=None, plt_kwargs={},add_cb=True, cb_kwargs={},
        err_kwargs={}, panel_kwargs_list=None, panel_kwargs_errlist=None,
        fig_kwargs={}, add_label=True, label_format=lambda v: v, **kwargs):

        """
        Scatter plot with errorbars and color based on point density with panels

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
            Bins for component 1 (for density colors).
        bins2: array, int
            Bins for component 2 (for density colors).
        ax_rotation: float
            Angle (in degrees) for rotation of axis of binning. Overwrites use of (bins1, bins2)
        rotation_resolution: int
            Number of bins to be used when ax_rotation!=0.
        xscale: str
            Scale xaxis.
        yscale: str
            Scale yaxis.
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

        Fit Parameters
        --------------
        add_bindata: bool
            Plot binned data used for fit.
        add_fit: bool
            Fit and plot binned dat.
        fit_err2: array
            Error of component 2
        fit_mode: str
            Statistics to be used in fit. Options are:
                `individual` - Use each point
                `mode` - Use mode of component 2 distribution in each comp 1 bin, requires fit_bins2.
                `mean` - Use mean of component 2 distribution in each comp 1 bin, requires fit_bins2.
        fit_bins1: array, None
            Bins for component 1 (default=10).
        fit_bins2: array, None
            Bins for component 2 (default=30).
        fit_legend_kwargs: dict
            Additional arguments for plt.legend.
        fit_bindata_kwargs: dict
            Additional arguments for pylab.errorbar.
        fit_plt_kwargs: dict
            Additional arguments for plot of fit pylab.scatter.

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
            xscale=xscale, yscale=yscale,
            panel_kwargs_list=panel_kwargs_list, panel_kwargs_errlist=panel_kwargs_errlist,
            fig_kwargs=fig_kwargs, add_label=add_label, label_format=label_format,
            # plot_density arguments
            values1=values1, values2=values2, err1=err1, err2=err2, bins1=bins1, bins2=bins2,
            ax_rotation=ax_rotation, rotation_resolution=rotation_resolution,
            plt_kwargs=plt_kwargs, err_kwargs=err_kwargs,
            # for fit
            **kwargs,
            )
    def _plot_metrics(values1, values2, bins=30, mode='redshift', ax=None,
                      bias_kwargs={}, scat_kwargs={}, rotated=False):
        """
        Plot metrics of 1 component.

        Parameters
        ----------
        values1: array
            Component 1
        values2: array
            Component 2
        bins: array, int
            Bins for component 1
        mode: str
            Mode to run. Options are:
            simple - used simple difference
            redshift - metrics for (values2-values1)/(1+values1)
            log - metrics for log of values
        ax: matplotlib.axes
            Ax to add plot
        bias_kwargs: dict
            Arguments for bias plot. Used in pylab.plot
        scat_kwargs: dict
            Arguments for scatter plot. Used in pylab.fill_between
        rotated: bool
            Rotate ax of plot

        Returns
        -------
        ax: matplotlib.axes
            Axis of the plot
        """
        edges1 = autobins(values1, bins, log=mode=='log')
        bmask = np.array(binmasks(values1, edges1))
        safe = [m[m].size>1 for m in bmask]
        diff = {
            'simple': lambda v1, v2: v2-v1,
            'redshift': lambda v1, v2: (v2-v1)/(1+v1),
            'log': lambda v1, v2: np.log10(v2)-np.log10(v1)
            }[mode](values1, values2)
        edges1 = np.log10(edges1) if mode=='log' else edges1
        values1_mid = 0.5*(edges1[1:]+edges1[:-1])
        values1_mid = 10**values1_mid if mode=='log' else values1_mid
        values1_mid = values1_mid[safe]
        bias, scat = np.array([[f(diff[m]) for m in bmask[safe]] for f in (np.mean, np.std)])
        # set for rotation
        ax = plt.axes() if ax is None else ax
        ph.add_grid(ax)
        bias_args = (bias, values1_mid) if rotated else (values1_mid, bias)
        scat_func = ax.fill_betweenx if rotated else ax.fill_between
        set_scale = ax.set_yscale if rotated else ax.set_xscale
        # plot
        bias_kwargs_ = {'color':'C0'}
        bias_kwargs_.update(bias_kwargs)
        scat_kwargs_ = {'alpha':.3, 'color':'C1'}
        scat_kwargs_.update(scat_kwargs)
        ax.plot(*bias_args, **bias_kwargs_)
        scat_func(values1_mid, -scat, scat, **scat_kwargs_)
        return ax
    def plot_metrics(values1, values2, bins1=30, bins2=30, mode='simple',
                     bias_kwargs={}, scat_kwargs={}, fig_kwargs={},
                     legend_kwargs={}):
        """
        Plot metrics of 1 component.

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
        mode: str
            Mode to run. Options are:
            simple - used simple difference
            redshift - metrics for (values2-values1)/(1+values1)
            log - metrics for log of values
        bias_kwargs: dict
            Arguments for bias plot. Used in pylab.plot
        scat_kwargs: dict
            Arguments for scatter plot. Used in pylab.fill_between
        fig_kwargs: dict
            Additional arguments for plt.subplots
        legend_kwargs: dict
            Additional arguments for plt.legend

        Returns
        -------
        ax: matplotlib.axes
            Axis of the plot
        """
        fig_kwargs_ = dict(figsize=(8, 6))
        fig_kwargs_.update(fig_kwargs)
        f, axes = plt.subplots(2, **fig_kwargs_)
        # default args
        bias_kwargs_ = {'label':'bias'}
        bias_kwargs_.update(bias_kwargs)
        scat_kwargs_ = {'label':'scatter'}
        scat_kwargs_.update(scat_kwargs)
        ArrayFuncs._plot_metrics(values1, values2, bins=bins1, mode=mode, ax=axes[0],
                                 bias_kwargs=bias_kwargs_, scat_kwargs=scat_kwargs_)
        ArrayFuncs._plot_metrics(values2, values1, bins=bins2, mode=mode, ax=axes[1],
                                 bias_kwargs=bias_kwargs_, scat_kwargs=scat_kwargs_)
        axes[0].legend(**legend_kwargs)
        axes[0].xaxis.tick_top()
        axes[0].xaxis.set_label_position('top')
        return f, axes

    def plot_density_metrics(values1, values2, bins1=30, bins2=30,
        ax_rotation=0, rotation_resolution=30, xscale='linear', yscale='linear',
        err1=None, err2=None, metrics_mode='simple', plt_kwargs={}, add_cb=True, cb_kwargs={},
        err_kwargs={}, bias_kwargs={}, scat_kwargs={}, fig_kwargs={},
        fig_pos=(0.1, 0.1, 0.95, 0.95), fig_frac=(0.8, 0.01, 0.02), **kwargs):
        """
        Scatter plot with errorbars and color based on point density with scatter and bias panels

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
            Angle (in degrees) for rotation of axis of binning. Overwrites use of (bins1, bins2) on main plot.
        rotation_resolution: int
            Number of bins to be used when ax_rotation!=0.
        xscale: str
            Scale xaxis.
        yscale: str
            Scale yaxis.
        err1: array
            Error of component 1
        err2: array
            Error of component 2
        metrics_mode: str
            Mode to run. Options are:
            simple - used simple difference
            redshift - metrics for (values2-values1)/(1+values1)
            log - metrics for log of values
        plt_kwargs: dict
            Additional arguments for pylab.scatter
        add_cb: bool
            Plot colorbar
        cb_kwargs: dict
            Colorbar arguments
        err_kwargs: dict
            Additional arguments for pylab.errorbar
        bias_kwargs: dict
            Arguments for bias plot. Used in pylab.plot
        scat_kwargs: dict
            Arguments for scatter plot. Used in pylab.fill_between
        fig_kwargs: dict
            Additional arguments for plt.subplots
        fig_pos: tuple
            List with edges for the figure. Must be in format (left, bottom, right, top)
        fig_frac: tuple
            Sizes of each panel in the figure. Must be in the format (main_panel, gap, colorbar)
            and have values: [0, 1]. Colorbar is only used with add_cb key.

        Fit Parameters
        --------------
        add_bindata: bool
            Plot binned data used for fit.
        add_fit: bool
            Fit and plot binned dat.
        fit_err2: array
            Error of component 2
        fit_mode: str
            Statistics to be used in fit. Options are:
                `individual` - Use each point
                `mode` - Use mode of component 2 distribution in each comp 1 bin, requires fit_bins2.
                `mean` - Use mean of component 2 distribution in each comp 1 bin, requires fit_bins2.
        fit_bins1: array, None
            Bins for component 1 (default=10).
        fit_bins2: array, None
            Bins for component 2 (default=30).
        fit_legend_kwargs: dict
            Additional arguments for plt.legend.
        fit_bindata_kwargs: dict
            Additional arguments for pylab.errorbar.
        fit_plt_kwargs: dict
            Additional arguments for plot of fit pylab.scatter.

        Returns
        -------
        fig: matplotlib.figure.Figure
            `matplotlib.figure.Figure` object
        list
            Axes with the panels (main, right, top, label)
        """
        fig_kwargs_ = dict(figsize=(8, 6))
        fig_kwargs_.update(fig_kwargs)
        fig = plt.figure(**fig_kwargs_)
        left, bottom, right, top = fig_pos
        frac, gap, cb = fig_frac
        cb = cb if add_cb else 0
        xmain, xgap, xpanel = (right-left)*np.array([frac, gap, 1-frac-gap-cb])
        ymain, ygap, ypanel, ycb = (top-bottom)*np.array([frac, gap, 1-frac-gap-cb, cb-gap])
        ax_m = fig.add_axes([left, bottom, xmain, ymain]) # main
        ax_v = fig.add_axes([left+xmain+xgap, bottom, xpanel, ymain]) # right
        ax_h = fig.add_axes([left, bottom+ymain+ygap, xmain, ypanel]) # top
        ax_l = fig.add_axes([left+xmain+xgap, bottom+ymain+ygap, xpanel, ypanel]) # label
        ax_cb = fig.add_axes([left, bottom+ymain+2*ygap+ypanel, xmain+xgap+xpanel, ycb])\
                    if add_cb else None
        # Main plot
        cb_kwargs_ = {'cax': ax_cb, 'orientation': 'horizontal'}
        cb_kwargs_.update(cb_kwargs)
        ArrayFuncs.plot_density(values1, values2, bins1=bins1, bins2=bins2,
            ax_rotation=ax_rotation, rotation_resolution=rotation_resolution,
            xscale=xscale, yscale=yscale, err1=err1, err2=err2, ax=ax_m,
            plt_kwargs=plt_kwargs, add_cb=add_cb, cb_kwargs=cb_kwargs_,
            err_kwargs=err_kwargs, **kwargs)
        if add_cb:
            ax_cb.xaxis.tick_top()
            ax_cb.xaxis.set_label_position('top')
        # Metrics plot
        bias_kwargs_ = {'label':'bias'}
        bias_kwargs_.update(bias_kwargs)
        scat_kwargs_ = {'label':'scatter'}
        scat_kwargs_.update(scat_kwargs)
        ArrayFuncs._plot_metrics(values1, values2, bins=bins1, mode=metrics_mode, ax=ax_h,
                                 bias_kwargs=bias_kwargs_, scat_kwargs=scat_kwargs_)
        ArrayFuncs._plot_metrics(values2, values1, bins=bins2, mode=metrics_mode, ax=ax_v,
                                 bias_kwargs=bias_kwargs_, scat_kwargs=scat_kwargs_,
                                 rotated=True)
        # Adjust plots
        ax_l.legend(ax_v.collections+ax_v.lines, ['$\sigma$', '$bias$'])
        ax_m.set_xscale(xscale)
        ax_m.set_yscale(yscale)
        # Horizontal
        ax_h.set_xscale(xscale)
        ax_h.set_xlim(ax_m.get_xlim())
        ax_h.xaxis.set_minor_formatter(NullFormatter())
        ax_h.xaxis.set_major_formatter(NullFormatter())
        # Vertical
        ax_v.set_yscale(yscale)
        ax_v.set_ylim(ax_m.get_ylim())
        ax_v.yaxis.set_minor_formatter(NullFormatter())
        ax_v.yaxis.set_major_formatter(NullFormatter())
        # Label
        ax_l.axis('off')
        return fig, [ax_m, ax_v, ax_h, ax_l]
    def plot_dist(values1, values2, bins1_dist, bins2, values_aux=None, bins_aux=5,
                  log_vals=False, log_aux=False, transpose=False,
                  shape='steps', plt_kwargs={}, line_kwargs_list=None,
                  fig_kwargs={}, legend_kwargs={},
                  add_panel_label=True, panel_label_format=lambda v: v,
                  add_line_label=True, line_label_format=lambda v: v):
        """
        Plot distribution of a parameter, binned by other component in panels,
        and an optional secondary component in lines.

        Parameters
        ----------
        values1: array
            Component 1
        values2: array
            Component 2 (to bin data in panels)
        bins1_dist: array
            Bins for component 1
        bins2: array
            Bins for component 2
        values_aux: array
            Auxiliary component (to bin data in lines)
        bins_aux: array
            Bins for component aux
        log_vals: bool
            Log scale for values (and int bins)
        log_aux: bool
            Log scale for aux values (and int bins)
        transpose: bool
            Invert lines and panels
        shape: str
            Shape of the lines. Can be steps or line.
        plt_kwargs: dict
            Additional arguments for pylab.plot
        line_kwargs_list: list, None
            List of additional arguments for plotting each line (using pylab.plot).
            Must have same size as len(bins_aux)-1
        fig_kwargs: dict
            Additional arguments for plt.subplots
        legend_kwargs: dict
            Additional arguments for plt.legend
        add_panel_label: bool
            Add bin label to panel
        panel_label_format: function
            Function to format the values of the bins
        add_line_label: bool
            Add bin label to line
        line_label_format: function
            Function to format the values of the bins

        Returns
        -------
        fig: matplotlib.figure.Figure
            `matplotlib.figure.Figure` object
        ax: matplotlib.axes
            Axes with the panels
        """
        if transpose and (values_aux is None):
            raise ValueError('transpose=true can only be used with values_aux!=None')
        edges1_dist = autobins(values1, bins1_dist, log=log_vals)
        edges2 = autobins(values2, bins2)
        edges_aux = None if values_aux is None else autobins(values_aux, bins_aux, log_aux)
        masks2 = binmasks(values2, edges2)
        masks_aux = [np.ones(len(values1), dtype=bool)] if values_aux is None\
                    else binmasks(values_aux, edges_aux)
        steps1 = np.log(edges1_dist[1:])-np.log(edges1_dist[:-1]) if log_vals\
                else edges1_dist[1:]-edges1_dist[:-1]
        # Use quantities relative to panel and lines:
        panel_masks, line_masks = (masks_aux, masks2) if transpose else (masks2, masks_aux)
        panel_edges, line_edges = (edges_aux, edges2) if transpose else (edges2, edges_aux)
        nj = int(np.ceil(np.sqrt(panel_edges[:-1].size)))
        ni = int(np.ceil(panel_edges[:-1].size/float(nj)))
        fig_kwargs_ = dict(sharex=True, figsize=(8, 6))
        fig_kwargs_.update(fig_kwargs)
        f, axes = plt.subplots(ni, nj, **fig_kwargs_)
        line_kwargs_list = none_val(line_kwargs_list, [{}] if values_aux is None else
            [{'label': ph.get_bin_label(vb, vt, line_label_format)}
                for vb, vt in zip(line_edges, line_edges[1:])])
        for ax, maskp in zip(axes.flatten(), panel_masks):
            ph.add_grid(ax)
            kwargs = {}
            kwargs.update(plt_kwargs)
            for maskl, p_kwargs in zip(line_masks, line_kwargs_list):
                kwargs.update(p_kwargs)
                hist = np.histogram(values1[maskp*maskl], bins=edges1_dist)[0]
                norm = (hist*steps1).sum()
                norm = norm if norm>0 else 1
                ph.plot_hist_line(hist/norm, edges1_dist,
                                  ax=ax, shape=shape, **kwargs)
            ax.set_xscale('log' if log_vals else 'linear')
            ax.set_yticklabels([])
        for ax in axes.flatten()[len(panel_edges)-1:]:
            ax.axis('off')
        if add_panel_label:
            ph.add_panel_bin_label(axes,  panel_edges[:-1], panel_edges[1:],
                                   format_func=panel_label_format)
        if values_aux is not None:
            axes.flatten()[0].legend(**legend_kwargs)
        return f, axes
    def plot_density_dist(values1, values2, bins1=30, bins2=30,
        ax_rotation=0, rotation_resolution=30, xscale='linear', yscale='linear',
        err1=None, err2=None, metrics_mode='simple', plt_kwargs={}, add_cb=True, cb_kwargs={},
        err_kwargs={}, bias_kwargs={}, scat_kwargs={}, fig_kwargs={},
        fig_pos=(0.1, 0.1, 0.95, 0.95), fig_frac=(0.8, 0.01, 0.02), vline_kwargs={},
        **kwargs):
        """
        Scatter plot with errorbars and color based on point density with distribution panels.

        Parameters
        ----------
        values1: array
            Component 1
        values2: array
            Component 2
        bins1: array, int
            Bins for component 1 (for density colors).
        bins2: array, int
            Bins for component 2 (for density colors).
        ax_rotation: float
            Angle (in degrees) for rotation of axis of binning. Overwrites use of (bins1, bins2) on main plot.
        rotation_resolution: int
            Number of bins to be used when ax_rotation!=0.
        xscale: str
            Scale xaxis.
        yscale: str
            Scale yaxis.
        err1: array
            Error of component 1
        err2: array
            Error of component 2
        plt_kwargs: dict
            Additional arguments for pylab.scatter
        add_cb: bool
            Plot colorbar
        cb_kwargs: dict
            Colorbar arguments
        err_kwargs: dict
            Additional arguments for pylab.errorbar
        bias_kwargs: dict
            Arguments for bias plot. Used in pylab.plot
        scat_kwargs: dict
            Arguments for scatter plot. Used in pylab.fill_between
        fig_kwargs: dict
            Additional arguments for plt.subplots
        fig_pos: tuple
            List with edges for the figure. Must be in format (left, bottom, right, top)
        fig_frac: tuple
            Sizes of each panel in the figure. Must be in the format (main_panel, gap, colorbar)
            and have values: [0, 1]. Colorbar is only used with add_cb key.

        Fit Parameters
        --------------
        add_bindata: bool
            Plot binned data used for fit.
        add_fit: bool
            Fit and plot binned dat.
        fit_err2: array
            Error of component 2
        fit_mode: str
            Statistics to be used in fit. Options are:
                `individual` - Use each point
                `mode` - Use mode of component 2 distribution in each comp 1 bin, requires fit_bins2.
                `mean` - Use mean of component 2 distribution in each comp 1 bin, requires fit_bins2.
        fit_bins1: array, None
            Bins for component 1 (default=10).
        fit_bins2: array, None
            Bins for component 2 (default=30).
        fit_legend_kwargs: dict
            Additional arguments for plt.legend.
        fit_bindata_kwargs: dict
            Additional arguments for pylab.errorbar.
        fit_plt_kwargs: dict
            Additional arguments for plot of fit pylab.scatter.
        vline_kwargs: dict
            Arguments for vlines marking bins in main plot, used in plt.axvline.

        Returns
        -------
        fig: matplotlib.figure.Figure
            `matplotlib.figure.Figure` object
        list
            Axes with the panels (main, right, top, label)
        """
        # Fig
        fig_kwargs_ = dict(figsize=(8, 6))
        fig_kwargs_.update(fig_kwargs)
        fig = plt.figure(**fig_kwargs_)
        left, bottom, right, top = fig_pos
        frac, gap, cb = fig_frac
        cb = cb if add_cb else 0
        xmain, xgap, xpanel = (right-left)*np.array([frac, gap, 1-frac-gap-cb])
        ymain, ygap, ypanel, ycb = (top-bottom)*np.array([frac, gap, 1-frac-gap-cb, cb-gap])
        ax_m = fig.add_axes([left, bottom, xmain, ymain]) # main
        ax_cb = fig.add_axes([left+xmain+xgap, bottom, ycb, ymain]) if add_cb else None # cb
        # Main plot
        cb_kwargs_ = {'cax': ax_cb, 'orientation': 'vertical'}
        cb_kwargs_.update(cb_kwargs)
        ArrayFuncs.plot_density(values1, values2, bins1=bins1, bins2=bins2,
            ax_rotation=ax_rotation, rotation_resolution=rotation_resolution,
            xscale=xscale, yscale=yscale, err1=err1, err2=err2, ax=ax_m,
            plt_kwargs=plt_kwargs, add_cb=add_cb, cb_kwargs=cb_kwargs_,
            err_kwargs=err_kwargs)
        if add_cb:
            ax_cb.xaxis.tick_top()
            ax_cb.xaxis.set_label_position('top')
        ax_m.set_xscale(xscale)
        ax_m.set_yscale(yscale)
        # Add v lines
        ax_m.xaxis.grid(False, which='both')
        fit_bins1 = autobins(values1, kwargs.get('fit_bins1', 10), xscale=='log')
        vline_kwargs_ = {'lw':.5, 'color':'0'}
        vline_kwargs_.update(vline_kwargs)
        for v in fit_bins1:
            ax_m.axvline(v, **vline_kwargs_)
        # Dist plot
        fit_bins2 = autobins(values2, kwargs.get('fit_bins2', 30), yscale=='log')
        masks1 = binmasks(values1, fit_bins1)
        xlims = ax_m.get_xlim()
        if xscale=='log':
            xlims, fit_bins1 = np.log(xlims), np.log(fit_bins1)
        xpos = [xmain*(x-xlims[0])/(xlims[1]-xlims[0]) for x in fit_bins1]
        axes_h = [fig.add_axes([left+xl, bottom+ymain+ygap, xr-xl, ypanel]) # top
                    for xl, xr in zip(xpos, xpos[1:])]
        fit_line_kwargs_list = kwargs.get('fit_line_kwargs_list', [{} for m in masks1])
        dlims = (np.inf, -np.inf)
        for ax, mask, lkwarg in zip(axes_h, masks1, fit_line_kwargs_list):
            ph.add_grid(ax)
            kwargs_ = {}
            kwargs_.update(kwargs.get('fit_plt_kwargs', {}))
            kwargs_.update(lkwarg)
            ph.plot_hist_line(*np.histogram(values2[mask], bins=fit_bins2),
                              ax=ax, shape='line', rotate=True, **kwargs_)
            ax.set_xlim(ax.get_xlim()[::-1])
            ax.set_xticklabels([])
            ax.set_yscale(xscale)
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position('right')
            dlims = min(dlims[0], ax.get_ylim()[0]), max(dlims[1], ax.get_ylim()[1])
        for ax in axes_h:
            ax.set_ylim(dlims)
        for ax in axes_h[:-1]:
            ax.set_yticklabels([])
        # Bindata and fit
        kwargs['fit_err2'] = kwargs.get('fit_err2', err2)
        kwargs['fit_add_fit'] = kwargs.get('add_fit', False)
        kwargs['fit_add_bindata'] = kwargs.get('add_bindata', kwargs['fit_add_fit'])
        ArrayFuncs._add_bindata_and_powlawfit(ax_m, values1, values2,
                                         **{k[4:]:v for k, v in kwargs.items()
                                             if k[:4]=='fit_'})
        return fig, [ax_m, axes_h]

class ClCatalogFuncs():
    """
    Class of plot functions with clevar.ClCatalog as inputs.
    Plot labels and scales are configured by this class.
    """
    class_args = ('xlabel', 'ylabel', 'xscale', 'yscale', 'add_err',
                  'label1', 'label2', 'scale1', 'scale2', 'mask1', 'mask2')
    def _prep_kwargs(cat1, cat2, matching_type, col, kwargs={}):
        """
        Prepare kwargs into args for this class and args for function

        Parameters
        ----------
        cat1: clevar.ClCatalog
            ClCatalog with matching information
        cat2: clevar.ClCatalog
            ClCatalog matched to
        matching_type: str
            Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
        col: str
            Name of column to be plotted
        kwargs: dict
            Input arguments

        Returns
        -------
        class_kwargs: dict
            Arguments for class
        func_kwargs: dict
            Arguments for function
        mp: clevar.match.MatchedPairs
            Matched catalogs
        """
        func_kwargs = {k:v for k, v in kwargs.items() if k not in ClCatalogFuncs.class_args}
        mp = MatchedPairs(cat1, cat2, matching_type,
                          mask1=kwargs.get('mask1', None),
                          mask2=kwargs.get('mask2', None))
        func_kwargs['values1'] = mp.data1[col]
        func_kwargs['values2'] = mp.data2[col]
        func_kwargs['err1'] = mp.data1.get(f'{col}_err') if kwargs.get('add_err', True) else None
        func_kwargs['err2'] = mp.data2.get(f'{col}_err') if kwargs.get('add_err', True) else None
        func_kwargs['fit_err2'] = mp.data2.get(f'{col}_err') if kwargs.get('add_fit_err', True) else None
        class_kwargs = {
            'xlabel': kwargs.get('xlabel', f'${cat1.labels[col]}$'),
            'ylabel': kwargs.get('ylabel', f'${cat2.labels[col]}$'),
            'xscale': kwargs.get('xscale', 'linear'),
            'yscale': kwargs.get('yscale', 'linear'),
        }
        return class_kwargs, func_kwargs, mp
    def _fmt_plot(ax, **kwargs):
        """
        Format plot (scale and label of ax)

        Parameters
        ----------
        ax: matplotlib.axes
            Ax to add plot
        **kwargs
            Other arguments
        """
        ax.set_xlabel(kwargs['xlabel'])
        ax.set_ylabel(kwargs['ylabel'])
        ax.set_xscale(kwargs['xscale'])
        ax.set_yscale(kwargs['yscale'])
    def plot(cat1, cat2, matching_type, col, **kwargs):
        """
        Scatter plot with errorbars and color based on input

        Parameters
        ----------
        cat1: clevar.ClCatalog
            ClCatalog with matching information
        cat2: clevar.ClCatalog
            ClCatalog matched to
        matching_type: str
            Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
        col: str
            Name of column to be plotted
        add_err: bool
            Add errorbars
        mask1: array, None
            Mask for clusters 1 properties, must have size=cat1.size
        mask2: array, None
            Mask for clusters 2 properties, must have size=cat2.size

        Fit Parameters
        --------------
        add_bindata: bool
            Plot binned data used for fit.
        add_fit: bool
            Fit and plot binned dat.
        fit_err2: array
            Error of component 2
        fit_mode: str
            Statistics to be used in fit. Options are:
                `individual` - Use each point
                `mode` - Use mode of component 2 distribution in each comp 1 bin, requires fit_bins2.
                `mean` - Use mean of component 2 distribution in each comp 1 bin, requires fit_bins2.
        fit_bins1: array, None
            Bins for component 1 (default=10).
        fit_bins2: array, None
            Bins for component 2 (default=30).
        fit_legend_kwargs: dict
            Additional arguments for plt.legend.
        fit_bindata_kwargs: dict
            Additional arguments for pylab.errorbar.
        fit_plt_kwargs: dict
            Additional arguments for plot of fit pylab.scatter.

        Other parameters
        ----------------
        ax: matplotlib.axes
            Ax to add plot
        plt_kwargs: dict
            Additional arguments for pylab.scatter
        err_kwargs: dict
            Additional arguments for pylab.errorbar
        xlabel: str
            Label of x axis.
        ylabel: str
            Label of y axis.
        xscale: str
            Scale xaxis.
        yscale: str
            Scale yaxis.

        Returns
        -------
        ax: matplotlib.axes
            Axis of the plot
        """
        cl_kwargs, f_kwargs, mp = ClCatalogFuncs._prep_kwargs(cat1, cat2, matching_type, col, kwargs)
        ax = ArrayFuncs.plot(**f_kwargs)
        ClCatalogFuncs._fmt_plot(ax, **cl_kwargs)
        return ax
    def plot_color(cat1, cat2, matching_type, col, col_color,
                   color1=True, color_log=False, **kwargs):
        """
        Scatter plot with errorbars and color based on input

        Parameters
        ----------
        cat1: clevar.ClCatalog
            ClCatalog with matching information
        cat2: clevar.ClCatalog
            ClCatalog matched to
        matching_type: str
            Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
        col: str
            Name of column to be plotted
        col_color: str
            Name of column for color
        color1: bool
            Use catalog 1 for color. If false uses catalog 2
        color_log: bool
            Use log of col_color
        add_err: bool
            Add errorbars
        mask1: array, None
            Mask for clusters 1 properties, must have size=cat1.size
        mask2: array, None
            Mask for clusters 2 properties, must have size=cat2.size

        Fit Parameters
        --------------
        add_bindata: bool
            Plot binned data used for fit.
        add_fit: bool
            Fit and plot binned dat.
        fit_err2: array
            Error of component 2
        fit_mode: str
            Statistics to be used in fit. Options are:
                `individual` - Use each point
                `mode` - Use mode of component 2 distribution in each comp 1 bin, requires fit_bins2.
                `mean` - Use mean of component 2 distribution in each comp 1 bin, requires fit_bins2.
        fit_bins1: array, None
            Bins for component 1 (default=10).
        fit_bins2: array, None
            Bins for component 2 (default=30).
        fit_legend_kwargs: dict
            Additional arguments for plt.legend.
        fit_bindata_kwargs: dict
            Additional arguments for pylab.errorbar.
        fit_plt_kwargs: dict
            Additional arguments for plot of fit pylab.scatter.

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
        xlabel: str
            Label of x axis.
        ylabel: str
            Label of y axis.
        xscale: str
            Scale xaxis.
        yscale: str
            Scale yaxis.

        Returns
        -------
        ax: matplotlib.axes
            Axis of the plot
        matplotlib.colorbar.Colorbar (optional)
            Colorbar of the recovey rates. Only returned if add_cb=True.
        """
        cl_kwargs, f_kwargs, mp = ClCatalogFuncs._prep_kwargs(cat1, cat2, matching_type, col, kwargs)
        f_kwargs['values_color'] = mp.data1[col_color] if color1 else mp.data2[col_color]
        f_kwargs['values_color'] = np.log10(f_kwargs['values_color']) if color_log\
                                    else f_kwargs['values_color']
        res = ArrayFuncs.plot_color(**f_kwargs)
        ax = res[0] if kwargs.get('add_cb', True) else res
        ClCatalogFuncs._fmt_plot(ax, **cl_kwargs)
        return res
    def plot_density(cat1, cat2, matching_type, col, **kwargs):
        """
        Scatter plot with errorbars and color based on point density

        Parameters
        ----------
        cat1: clevar.ClCatalog
            ClCatalog with matching information
        cat2: clevar.ClCatalog
            ClCatalog matched to
        matching_type: str
            Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
        col: str
            Name of column to be plotted
        bins1: array, int
            Bins of component 1 for density
        bins2: array, int
            Bins of component 2 for density
        add_err: bool
            Add errorbars
        mask1: array, None
            Mask for clusters 1 properties, must have size=cat1.size
        mask2: array, None
            Mask for clusters 2 properties, must have size=cat2.size

        Fit Parameters
        --------------
        add_bindata: bool
            Plot binned data used for fit.
        add_fit: bool
            Fit and plot binned dat.
        fit_err2: array
            Error of component 2
        fit_mode: str
            Statistics to be used in fit. Options are:
                `individual` - Use each point
                `mode` - Use mode of component 2 distribution in each comp 1 bin, requires fit_bins2.
                `mean` - Use mean of component 2 distribution in each comp 1 bin, requires fit_bins2.
        fit_bins1: array, None
            Bins for component 1 (default=10).
        fit_bins2: array, None
            Bins for component 2 (default=30).
        fit_legend_kwargs: dict
            Additional arguments for plt.legend.
        fit_bindata_kwargs: dict
            Additional arguments for pylab.errorbar.
        fit_plt_kwargs: dict
            Additional arguments for plot of fit pylab.scatter.

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
        xlabel: str
            Label of x axis.
        ylabel: str
            Label of y axis.
        xscale: str
            Scale xaxis.
        yscale: str
            Scale yaxis.

        Returns
        -------
        ax: matplotlib.axes
            Axis of the plot
        matplotlib.colorbar.Colorbar (optional)
            Colorbar of the recovey rates. Only returned if add_cb=True.
        """
        cl_kwargs, f_kwargs, mp = ClCatalogFuncs._prep_kwargs(cat1, cat2, matching_type, col, kwargs)
        f_kwargs['xscale'] = kwargs.get('xscale', 'linear')
        f_kwargs['yscale'] = kwargs.get('yscale', 'linear')
        res = ArrayFuncs.plot_density(**f_kwargs)
        ax = res[0] if kwargs.get('add_cb', True) else res
        ClCatalogFuncs._fmt_plot(ax, **cl_kwargs)
        return res
    def _get_panel_args(cat1, cat2, matching_type, col,
        col_panel, bins_panel, panel_cat1=True, log_panel=False,
        **kwargs):
        """
        Prepare args for panel

        Parameters
        ----------
        panel_plot_function: function
            Plot function
        cat1: clevar.ClCatalog
            ClCatalog with matching information
        cat2: clevar.ClCatalog
            ClCatalog matched to
        matching_type: str
            Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
        col: str
            Name of column to be plotted
        col_panel: str
            Name of column to make panels
        bins_panel: array
            Bins to make panels
        panel_cat1: bool
            Used catalog 1 for col_panel. If false uses catalog 2
        log_panel: bool
            Scale of the panel values
        add_err: bool
            Add errorbars
        **kwargs
            Other arguments

        Returns
        -------
        class_kwargs: dict
            Arguments for class
        func_kwargs: dict
            Arguments for function
        mp: clevar.match.MatchedPairs
            Matched catalogs
        """
        cl_kwargs, f_kwargs, mp = ClCatalogFuncs._prep_kwargs(cat1, cat2, matching_type, col, kwargs)
        f_kwargs['values_panel'] = mp.data1[col_panel] if panel_cat1 else mp.data2[col_panel]
        f_kwargs['bins_panel'] = autobins(f_kwargs['values_panel'], bins_panel, log_panel)
        ph._set_label_format(f_kwargs, 'label_format', 'label_fmt', log_panel)
        return cl_kwargs, f_kwargs, mp
    def plot_panel(cat1, cat2, matching_type, col,
        col_panel, bins_panel, panel_cat1=True, log_panel=False,
        **kwargs):
        """
        Scatter plot with errorbars and color based on input with panels

        Parameters
        ----------
        cat1: clevar.ClCatalog
            ClCatalog with matching information
        cat2: clevar.ClCatalog
            ClCatalog matched to
        matching_type: str
            Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
        col: str
            Name of column to be plotted
        col_panel: str
            Name of column to make panels
        bins_panel: array
            Bins to make panels
        panel_cat1: bool
            Used catalog 1 for col_panel. If false uses catalog 2
        log_panel: bool
            Scale of the panel values
        add_err: bool
            Add errorbars
        mask1: array, None
            Mask for clusters 1 properties, must have size=cat1.size
        mask2: array, None
            Mask for clusters 2 properties, must have size=cat2.size

        Fit Parameters
        --------------
        add_bindata: bool
            Plot binned data used for fit.
        add_fit: bool
            Fit and plot binned dat.
        fit_err2: array
            Error of component 2
        fit_mode: str
            Statistics to be used in fit. Options are:
                `individual` - Use each point
                `mode` - Use mode of component 2 distribution in each comp 1 bin, requires fit_bins2.
                `mean` - Use mean of component 2 distribution in each comp 1 bin, requires fit_bins2.
        fit_bins1: array, None
            Bins for component 1 (default=10).
        fit_bins2: array, None
            Bins for component 2 (default=30).
        fit_legend_kwargs: dict
            Additional arguments for plt.legend.
        fit_bindata_kwargs: dict
            Additional arguments for pylab.errorbar.
        fit_plt_kwargs: dict
            Additional arguments for plot of fit pylab.scatter.

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
        xlabel: str
            Label of x axis.
        ylabel: str
            Label of y axis.
        xscale: str
            Scale xaxis.
        yscale: str
            Scale yaxis.

        Returns
        -------
        fig: matplotlib.figure.Figure
            `matplotlib.figure.Figure` object
        ax: matplotlib.axes
            Axes with the panels
        """
        cl_kwargs, f_kwargs, mp = ClCatalogFuncs._get_panel_args(cat1, cat2, matching_type, col,
            col_panel, bins_panel, panel_cat1=panel_cat1, log_panel=log_panel, **kwargs)
        fig, axes = ArrayFuncs.plot_panel(**f_kwargs)
        ph.nice_panel(axes, **cl_kwargs)
        return fig, axes
    def plot_color_panel(cat1, cat2, matching_type, col, col_color,
        col_panel, bins_panel, panel_cat1=True, color1=True, log_panel=False, **kwargs):
        """
        Scatter plot with errorbars and color based on input with panels

        Parameters
        ----------
        pltfunc: function
            ArrayFuncs function
        cat1: clevar.ClCatalog
            ClCatalog with matching information
        cat2: clevar.ClCatalog
            ClCatalog matched to
        matching_type: str
            Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
        col: str
            Name of column to be plotted
        col_color: str
            Name of column for color
        col_panel: str
            Name of column to make panels
        bins_panel: array
            Bins to make panels
        panel_cat1: bool
            Used catalog 1 for col_panel. If false uses catalog 2
        color1: bool
            Use catalog 1 for color. If false uses catalog 2
        log_panel: bool
            Scale of the panel values
        add_err: bool
            Add errorbars
        mask1: array, None
            Mask for clusters 1 properties, must have size=cat1.size
        mask2: array, None
            Mask for clusters 2 properties, must have size=cat2.size

        Fit Parameters
        --------------
        add_bindata: bool
            Plot binned data used for fit.
        add_fit: bool
            Fit and plot binned dat.
        fit_err2: array
            Error of component 2
        fit_mode: str
            Statistics to be used in fit. Options are:
                `individual` - Use each point
                `mode` - Use mode of component 2 distribution in each comp 1 bin, requires fit_bins2.
                `mean` - Use mean of component 2 distribution in each comp 1 bin, requires fit_bins2.
        fit_bins1: array, None
            Bins for component 1 (default=10).
        fit_bins2: array, None
            Bins for component 2 (default=30).
        fit_legend_kwargs: dict
            Additional arguments for plt.legend.
        fit_bindata_kwargs: dict
            Additional arguments for pylab.errorbar.
        fit_plt_kwargs: dict
            Additional arguments for plot of fit pylab.scatter.

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
            Function
        xlabel: str
            Label of x axis.
        ylabel: str
            Label of y axis.
        xscale: str
            Scale xaxis.
        yscale: str
            Scale yaxis.

        Returns
        -------
        fig: matplotlib.figure.Figure
            `matplotlib.figure.Figure` object
        ax: matplotlib.axes
            Axis of the plot
        """
        cl_kwargs, f_kwargs, mp = ClCatalogFuncs._get_panel_args(cat1, cat2, matching_type, col,
            col_panel, bins_panel, panel_cat1=panel_cat1, log_panel=log_panel, **kwargs)
        f_kwargs['values_color'] = mp.data1[col_color] if color1 else mp.data2[col_color]
        fig, axes = ArrayFuncs.plot_color_panel(**f_kwargs)
        ph.nice_panel(axes, **cl_kwargs)
        return fig, axes
    def plot_density_panel(cat1, cat2, matching_type, col,
        col_panel, bins_panel, panel_cat1=True, log_panel=False, **kwargs):
        """
        Scatter plot with errorbars and color based on point density with panels

        Parameters
        ----------
        cat1: clevar.ClCatalog
            ClCatalog with matching information
        cat2: clevar.ClCatalog
            ClCatalog matched to
        matching_type: str
            Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
        col: str
            Name of column to be plotted
        col_panel: str
            Name of column to make panels
        bins_panel: array
            Bins to make panels
        panel_cat1: bool
            Used catalog 1 for col_panel. If false uses catalog 2
        bins1: array, int
            Bins of component 1 for density
        bins2: array, int
            Bins of component 2 for density
        log_panel: bool
            Scale of the panel values
        add_err: bool
            Add errorbars
        mask1: array, None
            Mask for clusters 1 properties, must have size=cat1.size
        mask2: array, None
            Mask for clusters 2 properties, must have size=cat2.size

        Fit Parameters
        --------------
        add_bindata: bool
            Plot binned data used for fit.
        add_fit: bool
            Fit and plot binned dat.
        fit_err2: array
            Error of component 2
        fit_mode: str
            Statistics to be used in fit. Options are:
                `individual` - Use each point
                `mode` - Use mode of component 2 distribution in each comp 1 bin, requires fit_bins2.
                `mean` - Use mean of component 2 distribution in each comp 1 bin, requires fit_bins2.
        fit_bins1: array, None
            Bins for component 1 (default=10).
        fit_bins2: array, None
            Bins for component 2 (default=30).
        fit_legend_kwargs: dict
            Additional arguments for plt.legend.
        fit_bindata_kwargs: dict
            Additional arguments for pylab.errorbar.
        fit_plt_kwargs: dict
            Additional arguments for plot of fit pylab.scatter.

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
        xlabel: str
            Label of x axis.
        ylabel: str
            Label of y axis.
        xscale: str
            Scale xaxis.
        yscale: str
            Scale yaxis.

        Returns
        -------
        fig: matplotlib.figure.Figure
            `matplotlib.figure.Figure` object
        ax: matplotlib.axes
            Axis of the plot
        """
        cl_kwargs, f_kwargs, mp = ClCatalogFuncs._get_panel_args(cat1, cat2, matching_type, col,
            col_panel, bins_panel, panel_cat1=panel_cat1, log_panel=log_panel, **kwargs)
        f_kwargs['xscale'] = kwargs.get('xscale', 'linear')
        f_kwargs['yscale'] = kwargs.get('yscale', 'linear')
        fig, axes = ArrayFuncs.plot_density_panel(**f_kwargs)
        ph.nice_panel(axes, **cl_kwargs)
        return fig, axes
    def plot_metrics(cat1, cat2, matching_type, col, bins1=30, bins2=30, **kwargs):
        """
        Plot metrics.

        Parameters
        ----------
        cat1: clevar.ClCatalog
            ClCatalog with matching information
        cat2: clevar.ClCatalog
            ClCatalog matched to
        matching_type: str
            Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
        col: str
            Name of column to be plotted
        bins1: array, int
            Bins for catalog 1
        bins2: array, int
            Bins for catalog 2
        mode: str
            Mode to run. Options are:
            simple - used simple difference
            redshift - metrics for (values2-values1)/(1+values1)
            log - metrics for log of values
        mask1: array, None
            Mask for clusters 1 properties, must have size=cat1.size
        mask2: array, None
            Mask for clusters 2 properties, must have size=cat2.size

        Other parameters
        ----------------
        bias_kwargs: dict
            Arguments for bias plot. Used in pylab.plot
        scat_kwargs: dict
            Arguments for scatter plot. Used in pylab.fill_between
        fig_kwargs: dict
            Additional arguments for plt.subplots
        legend_kwargs: dict
            Additional arguments for plt.legend
        label1: str
            Label of component from catalog 1.
        label2: str
            Label of component from catalog 2.
        scale1: str
            Scale of component from catalog 1.
        scale2: str
            Scale of component from catalog 2.

        Returns
        -------
        fig: matplotlib.figure.Figure
            `matplotlib.figure.Figure` object
        ax: matplotlib.axes
            Axis of the plot
        """
        cl_kwargs, f_kwargs, mp = ClCatalogFuncs._prep_kwargs(cat1, cat2, matching_type, col, kwargs)
        f_kwargs.pop('fit_err2', None)
        f_kwargs.pop('err1', None)
        f_kwargs.pop('err2', None)
        fig, axes = ArrayFuncs.plot_metrics(**f_kwargs)
        axes[0].set_ylabel(cat1.name)
        axes[1].set_ylabel(cat2.name)
        axes[0].set_xlabel(kwargs.get('label1', cl_kwargs['xlabel']))
        axes[1].set_xlabel(kwargs.get('label2', cl_kwargs['ylabel']))
        axes[0].set_xscale(kwargs.get('scale1', cl_kwargs['xscale']))
        axes[1].set_xscale(kwargs.get('scale2', cl_kwargs['yscale']))
        return fig, axes
    def plot_density_metrics(cat1, cat2, matching_type, col, bins1=30, bins2=30, **kwargs):
        """
        Scatter plot with errorbars and color based on point density with scatter and bias panels

        Parameters
        ----------
        cat1: clevar.ClCatalog
            ClCatalog with matching information
        cat2: clevar.ClCatalog
            ClCatalog matched to
        matching_type: str
            Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
        col: str
            Name of column to be plotted
        bins1: array, int
            Bins for component 1
        bins2: array, int
            Bins for component 2
        metrics_mode: str
            Mode to run. Options are:
            simple - used simple difference
            redshift - metrics for (values2-values1)/(1+values1)
            log - metrics for log of values
        add_err: bool
            Add errorbars
        mask1: array, None
            Mask for clusters 1 properties, must have size=cat1.size
        mask2: array, None
            Mask for clusters 2 properties, must have size=cat2.size

        Fit Parameters
        --------------
        add_bindata: bool
            Plot binned data used for fit.
        add_fit: bool
            Fit and plot binned dat.
        fit_err2: array
            Error of component 2
        fit_mode: str
            Statistics to be used in fit. Options are:
                `individual` - Use each point
                `mode` - Use mode of component 2 distribution in each comp 1 bin, requires fit_bins2.
                `mean` - Use mean of component 2 distribution in each comp 1 bin, requires fit_bins2.
        fit_bins1: array, None
            Bins for component 1 (default=10).
        fit_bins2: array, None
            Bins for component 2 (default=30).
        fit_legend_kwargs: dict
            Additional arguments for plt.legend.
        fit_bindata_kwargs: dict
            Additional arguments for pylab.errorbar.
        fit_plt_kwargs: dict
            Additional arguments for plot of fit pylab.scatter.

        Other parameters
        ----------------
        fig_kwargs: dict
            Additional arguments for plt.subplots
        ax_rotation: float
            Angle (in degrees) for rotation of axis of binning. Overwrites use of (bins1, bins2) on main plot.
        rotation_resolution: int
            Number of bins to be used when ax_rotation!=0.
        plt_kwargs: dict
            Additional arguments for pylab.scatter
        add_cb: bool
            Plot colorbar
        cb_kwargs: dict
            Colorbar arguments
        err_kwargs: dict
            Additional arguments for pylab.errorbar
        bias_kwargs: dict
            Arguments for bias plot. Used in pylab.plot
        scat_kwargs: dict
            Arguments for scatter plot. Used in pylab.fill_between
        xscale: str
            Scale xaxis.
        yscale: str
            Scale yaxis.
        fig_pos: tuple
            List with edges for the figure. Must be in format (left, bottom, right, top)
        fig_frac: tuple
            Sizes of each panel in the figure. Must be in the format (main_panel, gap, colorbar)
            and have values: [0, 1]. Colorbar is only used with add_cb key.

        Returns
        -------
        fig: matplotlib.figure.Figure
            `matplotlib.figure.Figure` object
        list
            Axes with the panels (main, right, top, label)
        """
        cl_kwargs, f_kwargs, mp = ClCatalogFuncs._prep_kwargs(cat1, cat2, matching_type, col, kwargs)
        f_kwargs['xscale'] = kwargs.get('scale1', cl_kwargs['xscale'])
        f_kwargs['yscale'] = kwargs.get('scale2', cl_kwargs['yscale'])
        fig, axes = ArrayFuncs.plot_density_metrics(**f_kwargs)
        axes[0].set_xlabel(kwargs.get('label1', cl_kwargs['xlabel']))
        axes[0].set_ylabel(kwargs.get('label2', cl_kwargs['ylabel']))
        return fig, axes
    def plot_dist(cat1, cat2, matching_type, col, bins1=30, bins2=5, col_aux=None, bins_aux=5,
                  log_vals=False, log_aux=False, transpose=False, **kwargs):
        """
        Plot distribution of a cat1 column, binned by the cat2 column in panels,
        with option for a second cat2 column in lines.

        Parameters
        ----------
        cat1: clevar.ClCatalog
            ClCatalog with matching information
        cat2: clevar.ClCatalog
            ClCatalog matched to
        matching_type: str
            Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
        col: str
            Name of column to be plotted
        bins1: array
            Bins for distribution of the cat1 column
        bins2: array
            Bins for cat2 column
        col_aux: array
            Auxiliary colum of cat2 (to bin data in lines)
        bins_aux: array
            Bins for component aux
        log_vals: bool
            Log scale for values (and int bins)
        log_aux: bool
            Log scale for aux values (and int bins)
        transpose: bool
            Invert lines and panels

        Other parameters
        ----------------
        fig_kwargs: dict
            Additional arguments for plt.subplots
        shape: str
            Shape of the lines. Can be steps or line.
        plt_kwargs: dict
            Additional arguments for pylab.plot
        line_kwargs_list: list, None
            List of additional arguments for plotting each line (using pylab.plot).
            Must have same size as len(bins_aux)-1
        legend_kwargs: dict
            Additional arguments for plt.legend
        add_panel_label: bool
            Add bin label to panel
        panel_label_format: function
            Function to format the values of the bins
        add_line_label: bool
            Add bin label to line
        line_label_format: function
            Function to format the values of the bins

        Returns
        -------
        fig: matplotlib.figure.Figure
            `matplotlib.figure.Figure` object
        ax: matplotlib.axes
            Axes with the panels
        """
        cl_kwargs, f_kwargs, mp = ClCatalogFuncs._prep_kwargs(cat1, cat2, matching_type, col, kwargs)
        f_kwargs.pop('err1', None)
        f_kwargs.pop('err2', None)
        f_kwargs.pop('fit_err2', None)
        f_kwargs['values_aux'] = None if col_aux is None else mp.data2[col_aux]
        f_kwargs['bins1_dist'] = bins1
        f_kwargs['bins2'] = bins2
        f_kwargs['bins_aux'] = bins_aux
        f_kwargs['log_vals'] = log_vals
        f_kwargs['log_aux'] = log_aux
        f_kwargs['transpose'] = transpose
        log_panel, log_line = (log_aux, log_vals) if transpose else (log_vals, log_aux)
        ph._set_label_format(f_kwargs, 'panel_label_format', 'panel_label_fmt', log_panel)
        ph._set_label_format(f_kwargs, 'line_label_format', 'line_label_fmt', log_line)
        fig, axes = ArrayFuncs.plot_dist(**f_kwargs)
        xlabel = kwargs.get('label', f'${cat1.labels[col]}$')
        for ax in (axes[-1,:] if len(axes.shape)>1 else axes):
            ax.set_xlabel(xlabel)
        return fig, axes
    def plot_dist_self(cat, col, bins1=30, bins2=5, col_aux=None, bins_aux=5,
                       log_vals=False, log_aux=False, transpose=False, mask=None, **kwargs):
        """
        Plot distribution of a cat1 column, binned by the same column in panels,
        with option for a second column in lines. Is is useful to compare with plot_dist results.

        Parameters
        ----------
        cat: clevar.ClCatalog
            Input catalog
        col: str
            Name of column to be plotted
        bins1: array
            Bins for distribution of the column
        bins2: array
            Bins for panels
        col_aux: array
            Auxiliary colum (to bin data in lines)
        bins_aux: array
            Bins for component aux
        log_vals: bool
            Log scale for values (and int bins)
        log_aux: bool
            Log scale for aux values (and int bins)
        transpose: bool
            Invert lines and panels
        mask: ndarray
            Mask for catalog

        Other parameters
        ----------------
        fig_kwargs: dict
            Additional arguments for plt.subplots
        shape: str
            Shape of the lines. Can be steps or line.
        plt_kwargs: dict
            Additional arguments for pylab.plot
        line_kwargs_list: list, None
            List of additional arguments for plotting each line (using pylab.plot).
            Must have same size as len(bins_aux)-1
        legend_kwargs: dict
            Additional arguments for plt.legend
        add_panel_label: bool
            Add bin label to panel
        panel_label_format: function
            Function to format the values of the bins
        add_line_label: bool
            Add bin label to line
        line_label_format: function
            Function to format the values of the bins

        Returns
        -------
        fig: matplotlib.figure.Figure
            `matplotlib.figure.Figure` object
        ax: matplotlib.axes
            Axes with the panels
        """
        f_kwargs = {k:v for k, v in kwargs.items() if k not in ClCatalogFuncs.class_args}
        mask = np.ones(cat.size, dtype=bool) if mask is None else mask
        f_kwargs['values1'] = cat[col][mask]
        f_kwargs['values2'] = cat[col][mask]
        f_kwargs['values_aux'] = None if col_aux is None else cat[col_aux][mask]
        f_kwargs['bins1_dist'] = bins1
        f_kwargs['bins2'] = bins2
        f_kwargs['bins_aux'] = bins_aux
        f_kwargs['log_vals'] = log_vals
        f_kwargs['log_aux'] = log_aux
        f_kwargs['transpose'] = transpose
        log_panel, log_line = (log_aux, log_vals) if transpose else (log_vals, log_aux)
        ph._set_label_format(f_kwargs, 'panel_label_format', 'panel_label_fmt', log_panel)
        ph._set_label_format(f_kwargs, 'line_label_format', 'line_label_fmt', log_line)
        fig, axes = ArrayFuncs.plot_dist(**f_kwargs)
        xlabel = kwargs.get('label', f'${cat.labels[col]}$')
        for ax in (axes[-1,:] if len(axes.shape)>1 else axes):
            ax.set_xlabel(xlabel)
        return fig, axes
    def plot_density_dist(cat1, cat2, matching_type, col, **kwargs):
        """
        Scatter plot with errorbars and color based on point density with distribution panels.

        Parameters
        ----------
        cat1: clevar.ClCatalog
            ClCatalog with matching information
        cat2: clevar.ClCatalog
            ClCatalog matched to
        matching_type: str
            Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
        col: str
            Name of column to be plotted
        bins1: array, int
            Bins for component 1 (for density colors).
        bins2: array, int
            Bins for component 2 (for density colors).
        add_err: bool
            Add errorbars
        mask1: array, None
            Mask for clusters 1 properties, must have size=cat1.size
        mask2: array, None
            Mask for clusters 2 properties, must have size=cat2.size

        Fit Parameters
        --------------
        add_bindata: bool
            Plot binned data used for fit.
        add_fit: bool
            Fit and plot binned dat.
        fit_err2: array
            Error of component 2
        fit_mode: str
            Statistics to be used in fit. Options are:
                `individual` - Use each point
                `mode` - Use mode of component 2 distribution in each comp 1 bin, requires fit_bins2.
                `mean` - Use mean of component 2 distribution in each comp 1 bin, requires fit_bins2.
        fit_bins1: array, None
            Bins for component 1 (default=10).
        fit_bins2: array, None
            Bins for component 2 (default=30).
        fit_legend_kwargs: dict
            Additional arguments for plt.legend.
        fit_bindata_kwargs: dict
            Additional arguments for pylab.errorbar.
        fit_plt_kwargs: dict
            Additional arguments for plot of fit pylab.scatter.
        vline_kwargs: dict
            Arguments for vlines marking bins in main plot, used in plt.axvline.

        Other parameters
        ----------------
        fig_kwargs: dict
            Additional arguments for plt.subplots
        ax_rotation: float
            Angle (in degrees) for rotation of axis of binning. Overwrites use of (bins1, bins2) on main plot.
        rotation_resolution: int
            Number of bins to be used when ax_rotation!=0.
        plt_kwargs: dict
            Additional arguments for pylab.scatter
        add_cb: bool
            Plot colorbar
        cb_kwargs: dict
            Colorbar arguments
        err_kwargs: dict
            Additional arguments for pylab.errorbar
        bias_kwargs: dict
            Arguments for bias plot. Used in pylab.plot
        scat_kwargs: dict
            Arguments for scatter plot. Used in pylab.fill_between
        xscale: str
            Scale xaxis.
        yscale: str
            Scale yaxis.
        fig_pos: tuple
            List with edges for the figure. Must be in format (left, bottom, right, top)
        fig_frac: tuple
            Sizes of each panel in the figure. Must be in the format (main_panel, gap, colorbar)
            and have values: [0, 1]. Colorbar is only used with add_cb key.

        Returns
        -------
        fig: matplotlib.figure.Figure
            `matplotlib.figure.Figure` object
        list
            Axes with the panels (main, right, top, label)
        """
        cl_kwargs, f_kwargs, mp = ClCatalogFuncs._prep_kwargs(cat1, cat2, matching_type, col, kwargs)
        f_kwargs['xscale'] = kwargs.get('scale1', cl_kwargs['xscale'])
        f_kwargs['yscale'] = kwargs.get('scale2', cl_kwargs['yscale'])
        fig, axes = ArrayFuncs.plot_density_dist(**f_kwargs)
        axes[0].set_xlabel(kwargs.get('label1', cl_kwargs['xlabel']))
        axes[0].set_ylabel(kwargs.get('label2', cl_kwargs['ylabel']))
        axes[1][-1].set_ylabel(kwargs.get('label2', cl_kwargs['ylabel']))
        return fig, axes

############################################################################################
### Redshift Plots #########################################################################
############################################################################################

def redshift(cat1, cat2, matching_type, **kwargs):
    """
    Scatter plot with errorbars and color based on input

    Parameters
    ----------
    cat1: clevar.ClCatalog
        ClCatalog with matching information
    cat2: clevar.ClCatalog
        ClCatalog matched to
    matching_type: str
        Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
    add_err: bool
        Add errorbars
    mask1: array, None
        Mask for clusters 1 properties, must have size=cat1.size
    mask2: array, None
        Mask for clusters 2 properties, must have size=cat2.size

    Other parameters
    ----------------
    ax: matplotlib.axes
        Ax to add plot
    plt_kwargs: dict
        Additional arguments for pylab.scatter
    err_kwargs: dict
        Additional arguments for pylab.errorbar
    xlabel: str
        Label of x axis.
    ylabel: str
        Label of y axis.

    Returns
    -------
    ax: matplotlib.axes
        Axis of the plot
    """
    return ClCatalogFuncs.plot(cat1, cat2, matching_type, col='z', **kwargs)
def redshift_density(cat1, cat2, matching_type, **kwargs):
    """
    Scatter plot with errorbars and color based on point density

    Parameters
    ----------
    cat1: clevar.ClCatalog
        ClCatalog with matching information
    cat2: clevar.ClCatalog
        ClCatalog matched to
    matching_type: str
        Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
    col: str
        Name of column to be plotted
    bins1: array, int
        Bins of redshift 1 for density
    bins2: array, int
        Bins of redshift 2 for density
    add_err: bool
        Add errorbars
    mask1: array, None
        Mask for clusters 1 properties, must have size=cat1.size
    mask2: array, None
        Mask for clusters 2 properties, must have size=cat2.size

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
    xlabel: str
        Label of x axis.
    ylabel: str
        Label of y axis.
    xscale: str
        Scale xaxis.
    yscale: str
        Scale yaxis.

    Returns
    -------
    ax: matplotlib.axes
        Axis of the plot
    matplotlib.colorbar.Colorbar (optional)
        Colorbar of the recovey rates. Only returned if add_cb=True.
    """
    return ClCatalogFuncs.plot_density(cat1, cat2, matching_type, col='z', **kwargs)
def redshift_masscolor(cat1, cat2, matching_type, log_mass=True, color1=True, **kwargs):
    """
    Scatter plot with errorbars and color based on input

    Parameters
    ----------
    cat1: clevar.ClCatalog
        ClCatalog with matching information
    cat2: clevar.ClCatalog
        ClCatalog matched to
    matching_type: str
        Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
    log_mass: bool
        Log scale for mass
    color1: bool
        Use catalog 1 for color. If false uses catalog 2
    add_err: bool
        Add errorbars
    mask1: array, None
        Mask for clusters 1 properties, must have size=cat1.size
    mask2: array, None
        Mask for clusters 2 properties, must have size=cat2.size

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
    xlabel: str
        Label of x axis.
    ylabel: str
        Label of y axis.

    Returns
    -------
    ax: matplotlib.axes
        Axis of the plot
    matplotlib.colorbar.Colorbar (optional)
        Colorbar of the recovey rates. Only returned if add_cb=True.
    """
    return ClCatalogFuncs.plot_color(cat1, cat2, matching_type, col='z', col_color='mass',
            color1=color1, color_log=log_mass, **kwargs)
def redshift_masspanel(cat1, cat2, matching_type, mass_bins=5, log_mass=True, **kwargs):
    """
    Scatter plot with errorbars and color based on input with panels

    Parameters
    ----------
    cat1: clevar.ClCatalog
        ClCatalog with matching information
    cat2: clevar.ClCatalog
        ClCatalog matched to
    matching_type: str
        Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
    mass_bins: int, array
        Mass bins to make panels
    log_mass: bool
        Log scale for mass
    panel_cat1: bool
        Used catalog 1 for col_panel. If false uses catalog 2
    add_err: bool
        Add errorbars
    mask1: array, None
        Mask for clusters 1 properties, must have size=cat1.size
    mask2: array, None
        Mask for clusters 2 properties, must have size=cat2.size

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
    label_fmt: str
        Format the values of binedges (ex: '.2f')
    xlabel: str
        Label of x axis.
    ylabel: str
        Label of y axis.

    Returns
    -------
    fig: matplotlib.figure.Figure
        `matplotlib.figure.Figure` object
    ax: matplotlib.axes
        Axes with the panels
    """
    kwargs['label_fmt'] = kwargs.get('label_fmt', '.1f')
    return ClCatalogFuncs.plot_panel(cat1, cat2, matching_type, col='z',
            col_panel='mass', bins_panel=mass_bins, log_panel=log_mass,
            **kwargs)
def redshift_density_masspanel(cat1, cat2, matching_type, mass_bins=5, log_mass=True, **kwargs):
    """
    Scatter plot with errorbars and color based on point density with panels

    Parameters
    ----------
    cat1: clevar.ClCatalog
        ClCatalog with matching information
    cat2: clevar.ClCatalog
        ClCatalog matched to
    matching_type: str
        Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
    mass_bins: int, array
        Mass bins to make panels
    log_mass: bool
        Log scale for mass
    panel_cat1: bool
        Used catalog 1 for col_panel. If false uses catalog 2
    bins1: array, int
        Bins of redshift 1 for density
    bins2: array, int
        Bins of redshift 2 for density
    add_err: bool
        Add errorbars
    mask1: array, None
        Mask for clusters 1 properties, must have size=cat1.size
    mask2: array, None
        Mask for clusters 2 properties, must have size=cat2.size

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
    label_fmt: str
        Format the values of binedges (ex: '.2f')
    xlabel: str
        Label of x axis.
    ylabel: str
        Label of y axis.

    Returns
    -------
    ax: matplotlib.axes
        Axis of the plot
    matplotlib.colorbar.Colorbar (optional)
        Colorbar of the recovey rates. Only returned if add_cb=True.
    """
    kwargs['label_fmt'] = kwargs.get('label_fmt', '.1f')
    return ClCatalogFuncs.plot_density_panel(cat1, cat2, matching_type, col='z',
            col_panel='mass', bins_panel=mass_bins, log_panel=log_mass,
            **kwargs)
def redshift_metrics(cat1, cat2, matching_type, **kwargs):
    """
    Plot metrics.

    Parameters
    ----------
    cat1: clevar.ClCatalog
        ClCatalog with matching information
    cat2: clevar.ClCatalog
        ClCatalog matched to
    matching_type: str
        Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
    bins1: array, int
        Bins for catalog 1
    bins2: array, int
        Bins for catalog 2
    mode: str
        Mode to run (default=redshit). Options are:
        simple - used simple difference
        redshift - metrics for (values2-values1)/(1+values1)
        log - metrics for log of values
    mask1: array, None
        Mask for clusters 1 properties, must have size=cat1.size
    mask2: array, None
        Mask for clusters 2 properties, must have size=cat2.size

    Other parameters
    ----------------
    fig_kwargs: dict
        Additional arguments for plt.subplots
    bias_kwargs: dict
        Arguments for bias plot. Used in pylab.plot
    scat_kwargs: dict
        Arguments for scatter plot. Used in pylab.fill_between
    legend_kwargs: dict
        Additional arguments for plt.legend
    label1: str
        Label of component from catalog 1.
    label2: str
        Label of component from catalog 2.
    scale1: str
        Scale of component from catalog 1.
    scale2: str
        Scale of component from catalog 2.

    Returns
    -------
    fig: matplotlib.figure.Figure
        `matplotlib.figure.Figure` object
    ax: matplotlib.axes
        Axis of the plot
    """
    mode = kwargs.pop('mode', 'redshift')
    return ClCatalogFuncs.plot_metrics(cat1, cat2, matching_type, col='z', mode=mode,
                                       **kwargs)
def redshift_density_metrics(cat1, cat2, matching_type, **kwargs):
    """
    Scatter plot with errorbars and color based on point density with scatter and bias panels

    Parameters
    ----------
    cat1: clevar.ClCatalog
        ClCatalog with matching information
    cat2: clevar.ClCatalog
        ClCatalog matched to
    matching_type: str
        Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
    col: str
        Name of column to be plotted
    bins1: array, int
        Bins for component 1
    bins2: array, int
        Bins for component 2
    metrics_mode: str
        Mode to run (default=redshit). Options are:
        simple - used simple difference
        redshift - metrics for (values2-values1)/(1+values1)
        log - metrics for log of values
    add_err: bool
        Add errorbars
    mask1: array, None
        Mask for clusters 1 properties, must have size=cat1.size
    mask2: array, None
        Mask for clusters 2 properties, must have size=cat2.size

    Other parameters
    ----------------
    fig_kwargs: dict
        Additional arguments for plt.subplots
    ax_rotation: float
        Angle (in degrees) for rotation of axis of binning. Overwrites use of (bins1, bins2) on main plot.
    rotation_resolution: int
        Number of bins to be used when ax_rotation!=0.
    plt_kwargs: dict
        Additional arguments for pylab.scatter
    add_cb: bool
        Plot colorbar
    cb_kwargs: dict
        Colorbar arguments
    err_kwargs: dict
        Additional arguments for pylab.errorbar
    bias_kwargs: dict
        Arguments for bias plot. Used in pylab.plot
    scat_kwargs: dict
        Arguments for scatter plot. Used in pylab.fill_between
    xscale: str
        Scale xaxis.
    yscale: str
        Scale yaxis.
    fig_pos: tuple
        List with edges for the figure. Must be in format (left, bottom, right, top)
    fig_frac: tuple
        Sizes of each panel in the figure. Must be in the format (main_panel, gap, colorbar)
        and have values: [0, 1]. Colorbar is only used with add_cb key.

    Returns
    -------
    fig: matplotlib.figure.Figure
        `matplotlib.figure.Figure` object
    list
        Axes with the panels (main, right, top, label)
    """
    metrics_mode = kwargs.pop('metrics_mode', 'redshift')
    return ClCatalogFuncs.plot_density_metrics(cat1, cat2, matching_type, col='z',
        metrics_mode=metrics_mode, **kwargs)
def redshift_dist(cat1, cat2, matching_type, redshift_bins_dist=30, redshift_bins=5, mass_bins=5,
              log_mass=True, transpose=False, **kwargs):
    """
    Plot distribution of a cat1 redshift, binned by the cat2 redshift (in panels),
    with option for cat2 mass bins (in lines).

    Parameters
    ----------
    cat1: clevar.ClCatalog
        ClCatalog with matching information
    cat2: clevar.ClCatalog
        ClCatalog matched to
    matching_type: str
        Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
    redshift_bins_dist: array
        Bins for distribution of the cat1 redshift
    redshift_bins: array
        Bins for cat2 redshift
    mass_bins: array
        Bins for cat2 mass
    log_mass: bool
        Log scale for mass
    transpose: bool
        Invert lines and panels

    Other parameters
    ----------------
    fig_kwargs: dict
        Additional arguments for plt.subplots
    shape: str
        Shape of the lines. Can be steps or line.
    plt_kwargs: dict
        Additional arguments for pylab.plot
    line_kwargs_list: list, None
        List of additional arguments for plotting each line (using pylab.plot).
        Must have same size as len(bins_aux)-1
    legend_kwargs: dict
        Additional arguments for plt.legend
    add_panel_label: bool
        Add bin label to panel
    panel_label_format: function
        Function to format the values of the bins
    add_line_label: bool
        Add bin label to line
    line_label_format: function
        Function to format the values of the bins

    Returns
    -------
    fig: matplotlib.figure.Figure
        `matplotlib.figure.Figure` object
    ax: matplotlib.axes
        Axes with the panels
    """
    kwargs_ = {}
    kwargs_.update(kwargs)
    kwargs_.update({
        'col': 'z',
        'col_aux': 'mass',
        'bins1': redshift_bins_dist,
        'bins2': redshift_bins,
        'bins_aux': mass_bins,
        'log_vals': False,
        'log_aux': log_mass,
        'transpose': transpose,
    })
    return ClCatalogFuncs.plot_dist(cat1, cat2, matching_type, **kwargs_)
def redshift_dist_self(cat, bins1=30, redshift_bins_dist=30, redshift_bins=5, mass_bins=5,
                   log_mass=True, transpose=False, mask=None, **kwargs):
    """
    Plot distribution of a cat redshift, binned by redshift (in panels),
    with option for mass bins (in lines).
    Is is useful to compare with redshift_dist results.

    Parameters
    ----------
    cat: clevar.ClCatalog
        Input Catalog
    redshift_bins_dist: array
        Bins for distribution of redshift
    redshift_bins: array
        Bins for redshift panels
    mass_bins: array
        Bins for mass
    log_mass: bool
        Log scale for mass
    transpose: bool
        Invert lines and panels

    Other parameters
    ----------------
    fig_kwargs: dict
        Additional arguments for plt.subplots
    shape: str
        Shape of the lines. Can be steps or line.
    plt_kwargs: dict
        Additional arguments for pylab.plot
    line_kwargs_list: list, None
        List of additional arguments for plotting each line (using pylab.plot).
        Must have same size as len(bins_aux)-1
    legend_kwargs: dict
        Additional arguments for plt.legend
    add_panel_label: bool
        Add bin label to panel
    panel_label_format: function
        Function to format the values of the bins
    add_line_label: bool
        Add bin label to line
    line_label_format: function
        Function to format the values of the bins

    Returns
    -------
    fig: matplotlib.figure.Figure
        `matplotlib.figure.Figure` object
    ax: matplotlib.axes
        Axes with the panels
    """
    kwargs_ = {}
    kwargs_.update(kwargs)
    kwargs_.update({
        'col': 'z',
        'col_aux': 'mass',
        'bins1': redshift_bins_dist,
        'bins2': redshift_bins,
        'bins_aux': mass_bins,
        'log_vals': False,
        'log_aux': log_mass,
        'transpose': transpose,
        'mask': mask,
    })
    return ClCatalogFuncs.plot_dist_self(cat, **kwargs_)
def redshift_density_dist(cat1, cat2, matching_type, **kwargs):
    """
    Scatter plot with errorbars and color based on point density with scatter and bias panels

    Parameters
    ----------
    cat1: clevar.ClCatalog
        ClCatalog with matching information
    cat2: clevar.ClCatalog
        ClCatalog matched to
    matching_type: str
        Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
    col: str
        Name of column to be plotted
    bins1: array, int
        Bins for component 1 (for density colors).
    bins2: array, int
        Bins for component 2 (for density colors).
    add_err: bool
        Add errorbars
    mask1: array, None
        Mask for clusters 1 properties, must have size=cat1.size
    mask2: array, None
        Mask for clusters 2 properties, must have size=cat2.size

    Fit Parameters
    --------------
    add_bindata: bool
        Plot binned data used for fit.
    add_fit: bool
        Fit and plot binned dat.
    fit_err2: array
        Error of component 2
    fit_mode: str
        Statistics to be used in fit. Options are:
            `individual` - Use each point
            `mode` - Use mode of component 2 distribution in each comp 1 bin, requires fit_bins2.
            `mean` - Use mean of component 2 distribution in each comp 1 bin, requires fit_bins2.
    fit_bins1: array, None
        Bins for component 1 (default=10).
    fit_bins2: array, None
        Bins for component 2 (default=30).
    fit_legend_kwargs: dict
        Additional arguments for plt.legend.
    fit_bindata_kwargs: dict
        Additional arguments for pylab.errorbar.
    fit_plt_kwargs: dict
        Additional arguments for plot of fit pylab.scatter.
    vline_kwargs: dict
        Arguments for vlines marking bins in main plot, used in plt.axvline.

    Other parameters
    ----------------
    fig_kwargs: dict
        Additional arguments for plt.subplots
    ax_rotation: float
        Angle (in degrees) for rotation of axis of binning. Overwrites use of (bins1, bins2) on main plot.
    rotation_resolution: int
        Number of bins to be used when ax_rotation!=0.
    plt_kwargs: dict
        Additional arguments for pylab.scatter
    add_cb: bool
        Plot colorbar
    cb_kwargs: dict
        Colorbar arguments
    err_kwargs: dict
        Additional arguments for pylab.errorbar
    bias_kwargs: dict
        Arguments for bias plot. Used in pylab.plot
    scat_kwargs: dict
        Arguments for scatter plot. Used in pylab.fill_between
    xscale: str
        Scale xaxis.
    yscale: str
        Scale yaxis.
    fig_pos: tuple
        List with edges for the figure. Must be in format (left, bottom, right, top)
    fig_frac: tuple
        Sizes of each panel in the figure. Must be in the format (main_panel, gap, colorbar)
        and have values: [0, 1]. Colorbar is only used with add_cb key.

    Returns
    -------
    fig: matplotlib.figure.Figure
        `matplotlib.figure.Figure` object
    list
        Axes with the panels (main, right, top, label)
    """
    return ClCatalogFuncs.plot_density_dist(cat1, cat2, matching_type, col='z', **kwargs)

############################################################################################
### Mass Plots #############################################################################
############################################################################################
def mass(cat1, cat2, matching_type, log_mass=True, **kwargs):
    """
    Scatter plot with errorbars and color based on input

    Parameters
    ----------
    cat1: clevar.ClCatalog
        ClCatalog with matching information
    cat2: clevar.ClCatalog
        ClCatalog matched to
    matching_type: str
        Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
    log_mass: bool
        Log scale for mass
    add_err: bool
        Add errorbars
    mask1: array, None
        Mask for clusters 1 properties, must have size=cat1.size
    mask2: array, None
        Mask for clusters 2 properties, must have size=cat2.size

    Other parameters
    ----------------
    ax: matplotlib.axes
        Ax to add plot
    plt_kwargs: dict
        Additional arguments for pylab.scatter
    err_kwargs: dict
        Additional arguments for pylab.errorbar
    xlabel: str
        Label of x axis.
    ylabel: str
        Label of y axis.

    Returns
    -------
    ax: matplotlib.axes
        Axis of the plot
    """
    return ClCatalogFuncs.plot(cat1, cat2, matching_type, col='mass',
            xscale='log' if log_mass else 'linear',
            yscale='log' if log_mass else 'linear',
            **kwargs)
def mass_zcolor(cat1, cat2, matching_type, log_mass=True, color1=True, **kwargs):
    """
    Scatter plot with errorbars and color based on input

    Parameters
    ----------
    cat1: clevar.ClCatalog
        ClCatalog with matching information
    cat2: clevar.ClCatalog
        ClCatalog matched to
    matching_type: str
        Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
    log_mass: bool
        Log scale for mass
    color1: bool
        Use catalog 1 for color. If false uses catalog 2
    add_err: bool
        Add errorbars
    mask1: array, None
        Mask for clusters 1 properties, must have size=cat1.size
    mask2: array, None
        Mask for clusters 2 properties, must have size=cat2.size

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
    xlabel: str
        Label of x axis.
    ylabel: str
        Label of y axis.

    Returns
    -------
    ax: matplotlib.axes
        Axis of the plot
    matplotlib.colorbar.Colorbar (optional)
        Colorbar of the recovey rates. Only returned if add_cb=True.
    """
    return ClCatalogFuncs.plot_color(cat1, cat2, matching_type, col='mass', col_color='z',
            xscale='log' if log_mass else 'linear',
            yscale='log' if log_mass else 'linear',
            color1=color1, **kwargs)
def mass_density(cat1, cat2, matching_type, log_mass=True, **kwargs):
    """
    Scatter plot with errorbars and color based on point density

    Parameters
    ----------
    cat1: clevar.ClCatalog
        ClCatalog with matching information
    cat2: clevar.ClCatalog
        ClCatalog matched to
    matching_type: str
        Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
    log_mass: bool
        Log scale for mass
    bins1: array, int
        Bins of mass 1 for density
    bins2: array, int
        Bins of mass 2 for density
    add_err: bool
        Add errorbars
    mask1: array, None
        Mask for clusters 1 properties, must have size=cat1.size
    mask2: array, None
        Mask for clusters 2 properties, must have size=cat2.size

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
    xlabel: str
        Label of x axis.
    ylabel: str
        Label of y axis.

    Returns
    -------
    ax: matplotlib.axes
        Axis of the plot
    matplotlib.colorbar.Colorbar (optional)
        Colorbar of the recovey rates. Only returned if add_cb=True.
    """
    return ClCatalogFuncs.plot_density(cat1, cat2, matching_type, col='mass',
            xscale='log' if log_mass else 'linear',
            yscale='log' if log_mass else 'linear',
            **kwargs)
def mass_zpanel(cat1, cat2, matching_type, redshift_bins=5, log_mass=True, **kwargs):
    """
    Scatter plot with errorbars and color based on input with panels

    Parameters
    ----------
    cat1: clevar.ClCatalog
        ClCatalog with matching information
    cat2: clevar.ClCatalog
        ClCatalog matched to
    matching_type: str
        Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
    redshift_bins: int, array
        Redshift bins to make panels
    log_mass: bool
        Log scale for mass
    panel_cat1: bool
        Used catalog 1 for col_panel. If false uses catalog 2
    add_err: bool
        Add errorbars
    mask1: array, None
        Mask for clusters 1 properties, must have size=cat1.size
    mask2: array, None
        Mask for clusters 2 properties, must have size=cat2.size

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
    label_fmt: str
        Format the values of binedges (ex: '.2f')
    xlabel: str
        Label of x axis.
    ylabel: str
        Label of y axis.

    Returns
    -------
    fig: matplotlib.figure.Figure
        `matplotlib.figure.Figure` object
    ax: matplotlib.axes
        Axes with the panels
    """
    kwargs['label_format'] = kwargs.get('label_format',
        lambda v: f'%{kwargs.pop("label_fmt", ".2f")}'%v)
    return ClCatalogFuncs.plot_panel(cat1, cat2, matching_type, col='mass',
            col_panel='z', bins_panel=redshift_bins,
            xscale='log' if log_mass else 'linear',
            yscale='log' if log_mass else 'linear',
            **kwargs)
def mass_density_zpanel(cat1, cat2, matching_type, redshift_bins=5, log_mass=True, **kwargs):
    """
    Scatter plot with errorbars and color based on point density with panels

    Parameters
    ----------
    cat1: clevar.ClCatalog
        ClCatalog with matching information
    cat2: clevar.ClCatalog
        ClCatalog matched to
    matching_type: str
        Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
    redshift_bins: int, array
        Redshift bins to make panels
    log_mass: bool
        Log scale for mass
    panel_cat1: bool
        Used catalog 1 for col_panel. If false uses catalog 2
    bins1: array, int
        Bins of mass 1 for density
    bins2: array, int
        Bins of mass 2 for density
    add_err: bool
        Add errorbars
    mask1: array, None
        Mask for clusters 1 properties, must have size=cat1.size
    mask2: array, None
        Mask for clusters 2 properties, must have size=cat2.size

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
    label_fmt: str
        Format the values of binedges (ex: '.2f')
    xlabel: str
        Label of x axis.
    ylabel: str
        Label of y axis.

    Returns
    -------
    ax: matplotlib.axes
        Axis of the plot
    matplotlib.colorbar.Colorbar (optional)
        Colorbar of the recovey rates. Only returned if add_cb=True.
    """
    return ClCatalogFuncs.plot_density_panel(cat1, cat2, matching_type, col='mass',
            col_panel='z', bins_panel=redshift_bins,
            xscale='log' if log_mass else 'linear',
            yscale='log' if log_mass else 'linear',
            **kwargs)
def mass_metrics(cat1, cat2, matching_type, log_mass=True, **kwargs):
    """
    Plot metrics.

    Parameters
    ----------
    cat1: clevar.ClCatalog
        ClCatalog with matching information
    cat2: clevar.ClCatalog
        ClCatalog matched to
    matching_type: str
        Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
    log_mass: bool
        Log scale for mass
    bins1: array, int
        Bins for catalog 1
    bins2: array, int
        Bins for catalog 2
    mode: str
        Mode to run (default=log). Options are:
        simple - used simple difference
        redshift - metrics for (values2-values1)/(1+values1)
        log - metrics for log of values
    mask1: array, None
        Mask for clusters 1 properties, must have size=cat1.size
    mask2: array, None
        Mask for clusters 2 properties, must have size=cat2.size

    Other parameters
    ----------------
    fig_kwargs: dict
        Additional arguments for plt.subplots
    bias_kwargs: dict
        Arguments for bias plot. Used in pylab.plot
    scat_kwargs: dict
        Arguments for scatter plot. Used in pylab.fill_between
    legend_kwargs: dict
        Additional arguments for plt.legend
    label1: str
        Label of component from catalog 1.
    label2: str
        Label of component from catalog 2.
    scale1: str
        Scale of component from catalog 1.
    scale2: str
        Scale of component from catalog 2.

    Returns
    -------
    fig: matplotlib.figure.Figure
        `matplotlib.figure.Figure` object
    ax: matplotlib.axes
        Axis of the plot
    """
    mode = kwargs.pop('mode', 'log')
    return ClCatalogFuncs.plot_metrics(cat1, cat2, matching_type, col='mass', mode=mode,
            xscale='log' if log_mass else 'linear',
            yscale='log' if log_mass else 'linear',
            **kwargs)
def mass_density_metrics(cat1, cat2, matching_type, log_mass=True, **kwargs):
    """
    Scatter plot with errorbars and color based on point density with scatter and bias panels

    Parameters
    ----------
    cat1: clevar.ClCatalog
        ClCatalog with matching information
    cat2: clevar.ClCatalog
        ClCatalog matched to
    matching_type: str
        Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
    log_mass: bool
        Log scale for mass
    bins1: array, int
        Bins for component 1
    bins2: array, int
        Bins for component 2
    metrics_mode: str
        Mode to run (default=log). Options are:
        simple - used simple difference
        redshift - metrics for (values2-values1)/(1+values1)
        log - metrics for log of values
    add_err: bool
        Add errorbars
    mask1: array, None
        Mask for clusters 1 properties, must have size=cat1.size
    mask2: array, None
        Mask for clusters 2 properties, must have size=cat2.size

    Other parameters
    ----------------
    fig_kwargs: dict
        Additional arguments for plt.subplots
    ax_rotation: float
        Angle (in degrees) for rotation of axis of binning. Overwrites use of (bins1, bins2) on main plot.
    rotation_resolution: int
        Number of bins to be used when ax_rotation!=0.
    plt_kwargs: dict
        Additional arguments for pylab.scatter
    add_cb: bool
        Plot colorbar
    cb_kwargs: dict
        Colorbar arguments
    err_kwargs: dict
        Additional arguments for pylab.errorbar
    bias_kwargs: dict
        Arguments for bias plot. Used in pylab.plot
    scat_kwargs: dict
        Arguments for scatter plot. Used in pylab.fill_between
    fig_pos: tuple
        List with edges for the figure. Must be in format (left, bottom, right, top)
    fig_frac: tuple
        Sizes of each panel in the figure. Must be in the format (main_panel, gap, colorbar)
        and have values: [0, 1]. Colorbar is only used with add_cb key.

    Returns
    -------
    fig: matplotlib.figure.Figure
        `matplotlib.figure.Figure` object
    list
        Axes with the panels (main, right, top, label)
    """
    metrics_mode = kwargs.pop('metrics_mode', 'log')
    return ClCatalogFuncs.plot_density_metrics(cat1, cat2, matching_type, col='mass',
            xscale='log' if log_mass else 'linear',
            yscale='log' if log_mass else 'linear',
            metrics_mode=metrics_mode, **kwargs)
def mass_dist(cat1, cat2, matching_type, mass_bins_dist=30, mass_bins=5, redshift_bins=5,
              log_mass=True, transpose=False, **kwargs):
    """
    Plot distribution of a cat1 mass, binned by the cat2 mass (in panels),
    with option for cat2 redshift bins (in lines).

    Parameters
    ----------
    cat1: clevar.ClCatalog
        ClCatalog with matching information
    cat2: clevar.ClCatalog
        ClCatalog matched to
    matching_type: str
        Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
    mass_bins_dist: array
        Bins for distribution of the cat1 mass
    mass_bins: array
        Bins for cat2 mass
    redshift_bins: array
        Bins for cat2 redshift
    log_mass: bool
        Log scale for mass
    transpose: bool
        Invert lines and panels

    Other parameters
    ----------------
    fig_kwargs: dict
        Additional arguments for plt.subplots
    shape: str
        Shape of the lines. Can be steps or line.
    plt_kwargs: dict
        Additional arguments for pylab.plot
    line_kwargs_list: list, None
        List of additional arguments for plotting each line (using pylab.plot).
        Must have same size as len(bins_aux)-1
    legend_kwargs: dict
        Additional arguments for plt.legend
    add_panel_label: bool
        Add bin label to panel
    panel_label_format: function
        Function to format the values of the bins
    add_line_label: bool
        Add bin label to line
    line_label_format: function
        Function to format the values of the bins

    Returns
    -------
    fig: matplotlib.figure.Figure
        `matplotlib.figure.Figure` object
    ax: matplotlib.axes
        Axes with the panels
    """
    kwargs_ = {}
    kwargs_.update(kwargs)
    kwargs_.update({
        'col': 'mass',
        'col_aux': 'z',
        'bins1': mass_bins_dist,
        'bins2': mass_bins,
        'bins_aux': redshift_bins,
        'log_vals': log_mass,
        'log_aux': False,
        'transpose': transpose,
    })
    return ClCatalogFuncs.plot_dist(cat1, cat2, matching_type, **kwargs_)
def mass_dist_self(cat, bins1=30, mass_bins_dist=30, mass_bins=5, redshift_bins=5,
                   log_mass=True, transpose=False, mask=None, **kwargs):
    """
    Plot distribution of a cat mass, binned by mass (in panels),
    with option for redshift bins (in lines).
    Is is useful to compare with mass_dist results.

    Parameters
    ----------
    cat: clevar.ClCatalog
        Input Catalog
    mass_bins_dist: array
        Bins for distribution of mass
    mass_bins: array
        Bins for mass panels
    redshift_bins: array
        Bins for redshift
    log_mass: bool
        Log scale for mass
    transpose: bool
        Invert lines and panels

    Other parameters
    ----------------
    fig_kwargs: dict
        Additional arguments for plt.subplots
    shape: str
        Shape of the lines. Can be steps or line.
    plt_kwargs: dict
        Additional arguments for pylab.plot
    line_kwargs_list: list, None
        List of additional arguments for plotting each line (using pylab.plot).
        Must have same size as len(bins_aux)-1
    legend_kwargs: dict
        Additional arguments for plt.legend
    add_panel_label: bool
        Add bin label to panel
    panel_label_format: function
        Function to format the values of the bins
    add_line_label: bool
        Add bin label to line
    line_label_format: function
        Function to format the values of the bins

    Returns
    -------
    fig: matplotlib.figure.Figure
        `matplotlib.figure.Figure` object
    ax: matplotlib.axes
        Axes with the panels
    """
    kwargs_ = {}
    kwargs_.update(kwargs)
    kwargs_.update({
        'col': 'mass',
        'col_aux': 'z',
        'bins1': mass_bins_dist,
        'bins2': mass_bins,
        'bins_aux': redshift_bins,
        'log_vals': log_mass,
        'log_aux': False,
        'transpose': transpose,
        'mask': mask,
    })
    return ClCatalogFuncs.plot_dist_self(cat, **kwargs_)

def mass_density_dist(cat1, cat2, matching_type, log_mass=True, **kwargs):
    """
    Scatter plot with errorbars and color based on point density with scatter and bias panels

    Parameters
    ----------
    cat1: clevar.ClCatalog
        ClCatalog with matching information
    cat2: clevar.ClCatalog
        ClCatalog matched to
    matching_type: str
        Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
    col: str
        Name of column to be plotted
    bins1: array, int
        Bins for component 1 (for density colors).
    bins2: array, int
        Bins for component 2 (for density colors).
    add_err: bool
        Add errorbars
    mask1: array, None
        Mask for clusters 1 properties, must have size=cat1.size
    mask2: array, None
        Mask for clusters 2 properties, must have size=cat2.size
    log_mass: bool
        Log scale for mass

    Fit Parameters
    --------------
    add_bindata: bool
        Plot binned data used for fit.
    add_fit: bool
        Fit and plot binned dat.
    fit_err2: array
        Error of component 2
    fit_mode: str
        Statistics to be used in fit. Options are:
            `individual` - Use each point
            `mode` - Use mode of component 2 distribution in each comp 1 bin, requires fit_bins2.
            `mean` - Use mean of component 2 distribution in each comp 1 bin, requires fit_bins2.
    fit_bins1: array, None
        Bins for component 1 (default=10).
    fit_bins2: array, None
        Bins for component 2 (default=30).
    fit_legend_kwargs: dict
        Additional arguments for plt.legend.
    fit_bindata_kwargs: dict
        Additional arguments for pylab.errorbar.
    fit_plt_kwargs: dict
        Additional arguments for plot of fit pylab.scatter.
    vline_kwargs: dict
        Arguments for vlines marking bins in main plot, used in plt.axvline.

    Other parameters
    ----------------
    fig_kwargs: dict
        Additional arguments for plt.subplots
    ax_rotation: float
        Angle (in degrees) for rotation of axis of binning. Overwrites use of (bins1, bins2) on main plot.
    rotation_resolution: int
        Number of bins to be used when ax_rotation!=0.
    plt_kwargs: dict
        Additional arguments for pylab.scatter
    add_cb: bool
        Plot colorbar
    cb_kwargs: dict
        Colorbar arguments
    err_kwargs: dict
        Additional arguments for pylab.errorbar
    bias_kwargs: dict
        Arguments for bias plot. Used in pylab.plot
    scat_kwargs: dict
        Arguments for scatter plot. Used in pylab.fill_between
    fig_pos: tuple
        List with edges for the figure. Must be in format (left, bottom, right, top)
    fig_frac: tuple
        Sizes of each panel in the figure. Must be in the format (main_panel, gap, colorbar)
        and have values: [0, 1]. Colorbar is only used with add_cb key.

    Returns
    -------
    fig: matplotlib.figure.Figure
        `matplotlib.figure.Figure` object
    list
        Axes with the panels (main, right, top, label)
    """
    kwargs['fit_log'] = kwargs.get('fit_log', True)
    kwargs['xscale'] = 'log' if log_mass else 'linear'
    kwargs['yscale'] = 'log' if log_mass else 'linear'
    return ClCatalogFuncs.plot_density_dist(cat1, cat2, matching_type, col='mass', **kwargs)
