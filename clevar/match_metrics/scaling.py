# Set mpl backend run plots on github actions
import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == 'test':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
from matplotlib.ticker import ScalarFormatter, NullFormatter
import pylab as plt
import numpy as np

from ..utils import none_val, autobins, binmasks
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
        ph.add_grid(ax)
        plt_kwargs_ = {'s':1}
        plt_kwargs_.update(plt_kwargs)
        ax.scatter(values1, values2, **plt_kwargs_)
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
                add_cb=add_cb, cb_kwargs=cb_kwargs, err_kwargs=err_kwargs)
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
                   fig_kwargs={}, add_label=True, label_format=lambda v: v):
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
                   fig_kwargs={}, add_label=True, label_format=lambda v: v):
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
        xscale='linear', yscale='linear',
        err1=None, err2=None, plt_kwargs={},add_cb=True, cb_kwargs={},
        err_kwargs={}, panel_kwargs_list=None, panel_kwargs_errlist=None,
        fig_kwargs={}, add_label=True, label_format=lambda v: v):

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
            Bins for component 1
        bins2: array, int
            Bins for component 2
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
        set_scale({'log':'log'}.get(mode, 'linear'))
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
        err_kwargs={}, bias_kwargs={}, scat_kwargs={}, fig_kwargs={}):
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

        Returns
        -------
        fig: matplotlib.figure.Figure
            `matplotlib.figure.Figure` object
        list
            Axes with the panels (main, top, right, label)
        """
        fig_kwargs_ = dict(figsize=(8, 6))
        fig_kwargs_.update(fig_kwargs)
        fig = plt.figure(**fig_kwargs_)
        left, bottom, right, top = (0.2, 0.2, 0.99, 0.99) # left, bottom, right, top
        frac, gap = 0.8, 0.01 #
        xmain, xgap, xpanel = (right-left)*np.array([frac, gap, 1-frac-gap])
        ymain, ygap, ypanel = (top-bottom)*np.array([frac, gap, 1-frac-gap])
        ax_m = fig.add_axes([left, bottom, xmain, ymain]) # main
        ax_v = fig.add_axes([left+xmain+xgap, bottom, xpanel, ymain]) # right
        ax_h = fig.add_axes([left, bottom+ymain+ygap, xmain, ypanel]) # top
        ax_l = fig.add_axes([left+xmain+xgap, bottom+ymain+ygap, xpanel, ypanel]) # label
        # Main plot
        add_cb = False
        cb_kwargs_ = {'ax': ax_l}
        cb_kwargs_.update(cb_kwargs)
        ArrayFuncs.plot_density(values1, values2, bins1=bins1, bins2=bins2,
            ax_rotation=ax_rotation, rotation_resolution=rotation_resolution,
            xscale=xscale, yscale=yscale, err1=err1, err2=err2, ax=ax_m,
            plt_kwargs=plt_kwargs, add_cb=add_cb, cb_kwargs=cb_kwargs_,
            err_kwargs=err_kwargs)
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
        ax_m.set_xscale('log' if metrics_mode=='log' else 'linear')
        ax_m.set_yscale('log' if metrics_mode=='log' else 'linear')
        ax_h.set_xlim(ax_m.get_xlim())
        ax_v.set_ylim(ax_m.get_ylim())
        ax_h.set_xticklabels([])
        ax_v.set_yticklabels([])
        ax_l.axis('off')
        return fig, [ax_m, ax_v, ax_h, ax_l]

class ClCatalogFuncs():
    """
    Class of plot functions with clevar.ClCatalog as inputs.
    Plot labels and scales are configured by this class.
    """
    class_args = ('xlabel', 'ylabel', 'xscale', 'yscale', 'add_err',
                  'label1', 'label2', 'scale1', 'scale2')
    def _prep_kwargs(cat1, cat2, matching_type, col, kwargs):
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
        mp = MatchedPairs(cat1, cat2, matching_type)
        func_kwargs['values1'] = mp.data1[col]
        func_kwargs['values2'] = mp.data2[col]
        func_kwargs['err1'] = mp.data1.get(f'{col}_err') if kwargs.get('add_err', True) else None
        func_kwargs['err2'] = mp.data2.get(f'{col}_err') if kwargs.get('add_err', True) else None
        class_kwargs = {
            'label1': none_val(kwargs.get('xlabel', None), f'${col}_{{{cat1.name}}}$'),
            'label2': none_val(kwargs.get('ylabel', None), f'${col}_{{{cat2.name}}}$'),
            'scale1': kwargs.get('xscale', 'linear'),
            'scale2': kwargs.get('yscale', 'linear'),
        }
        for col in ('label1', 'label2', 'scale1', 'scale2'):
            class_kwargs[col] = kwargs.get(col, class_kwargs[col])
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
        ax.set_xlabel(kwargs['label1'])
        ax.set_ylabel(kwargs['label2'])
        ax.set_xscale(kwargs['scale1'])
        ax.set_yscale(kwargs['scale2'])
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
        label_fmt = f_kwargs.pop("label_fmt", ".2f")
        f_kwargs['label_format'] = f_kwargs.get('label_format',
            lambda v: f'10^{{%{label_fmt}}}'%np.log10(v) if log_panel else  f'%{label_fmt}'%v)
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
        ax: matplotlib.axes
            Axis of the plot
        matplotlib.colorbar.Colorbar (optional)
            Colorbar of the recovey rates. Only returned if add_cb=True.
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
        ax: matplotlib.axes
            Axis of the plot
        matplotlib.colorbar.Colorbar (optional)
            Colorbar of the recovey rates. Only returned if add_cb=True.
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
        ax: matplotlib.axes
            Axis of the plot
        """
        cl_kwargs, f_kwargs, mp = ClCatalogFuncs._prep_kwargs(cat1, cat2, matching_type, col, kwargs)
        f_kwargs.pop('err1', None)
        f_kwargs.pop('err2', None)
        fig, axes = ArrayFuncs.plot_metrics(**f_kwargs)
        axes[0].set_ylabel(cat1.name)
        axes[1].set_ylabel(cat2.name)
        axes[0].set_xlabel(cl_kwargs['label1'])
        axes[1].set_xlabel(cl_kwargs['label2'])
        axes[0].set_xscale(cl_kwargs['scale1'])
        axes[1].set_xscale(cl_kwargs['scale2'])
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

        Returns
        -------
        fig: matplotlib.figure.Figure
            `matplotlib.figure.Figure` object
        """
        cl_kwargs, f_kwargs, mp = ClCatalogFuncs._prep_kwargs(cat1, cat2, matching_type, col, kwargs)
        f_kwargs.pop('err1', None)
        f_kwargs.pop('err2', None)
        fig, axes = ArrayFuncs.plot_density_metrics(**f_kwargs)
        axes[0].set_xlabel(cat1.name)
        axes[0].set_ylabel(cat2.name)
        axes[0].set_xlabel(cl_kwargs['label1'])
        axes[0].set_ylabel(cl_kwargs['label2'])
        return fig, axes
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
