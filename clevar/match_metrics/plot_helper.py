"""@file clevar/match_metrics/plot_helper.py
Helper functions for plotting.
"""
# Set mpl backend run plots on github actions
import os
import matplotlib as mpl

if os.environ.get("DISPLAY", "") == "test":
    print("no display found. Using non-interactive Agg backend")
    mpl.use("Agg")

# pylint: disable=wrong-import-position
import pylab as plt
import numpy as np
from scipy.interpolate import interp2d
from matplotlib.ticker import ScalarFormatter, NullFormatter

from ..utils import none_val, hp, updated_dict

########################################################################
########## Monkeypatching matplotlib ###################################
########################################################################

from ..utils import smooth_line


def _plot_smooth(self, *args, scheme=(1, 2, 1), n_increase=0, **kwargs):
    """Function to apply loops in plots.

    Parameters
    ----------
    self: class
        To be used by mpl
    *args
        Main function positional arguments
    scheme: tuple
        Scheme to be used for smoothening. Newton's binomial coefficients work better.
    n_increase: int
        Number of loops for the algorithm.
    *8kwargs
        main function keyword arguments

    Returns
    -------
    output of self.plot
    """
    return self.plot(
        *smooth_line(*np.array(args[:2]), scheme=scheme, n_increase=n_increase), *args[2:], **kwargs
    )


plt.plot_smooth = _plot_smooth
plt.Axes.plot_smooth = _plot_smooth

########################################################################
########################################################################
########################################################################


def rm_axis_ticklabels(axis):
    """Remove ticklabels from axis"""
    axis.set_minor_formatter(NullFormatter())
    axis.set_major_formatter(NullFormatter())


def add_grid(ax, major_lw=0.5, minor_lw=0.1):
    """
    Adds a grid to ax

    Parameters
    ----------
    ax: matplotlib.axes
        Ax to add plot
    major_lw: float
        Line width of major axes
    minor_lw: float
        Line width of minor axes
    """
    ax.minorticks_on()
    ax.xaxis.grid(True, which="major", lw=major_lw)
    ax.yaxis.grid(True, which="major", lw=major_lw)
    ax.xaxis.grid(True, which="minor", lw=minor_lw)
    ax.yaxis.grid(True, which="minor", lw=minor_lw)


def plot_hist_line(hist_values, bins, ax, shape="steps", rotate=False, **kwargs):
    """
    Plot recovey rate as lines. Can be in steps or continuous

    Parameters
    ----------
    hist_values: array
        Values of each bin in the histogram
    bins: array, int
        Bins of histogram
    ax: matplotlib.axes
        Ax to add plot
    shape: str
        Shape of the line. Can be steps or line.
    rotate: bool
        Invert x-y axes in plot
    kwargs: parameters
        Additional parameters for plt.plot.
        It also includes the possibility of smoothening the line with `n_increase, scheme`
        arguments. See `clevar.utils.smooth_line` for details.
    """
    if shape == "steps":
        data = (
            np.transpose([bins[:-1], bins[1:]]).flatten(),
            np.transpose([hist_values, hist_values]).flatten(),
        )
    elif shape == "line":
        data = (0.5 * (bins[:-1] + bins[1:]), hist_values)
    else:
        raise ValueError(f"shape ({shape}) must be 'steps' or 'line'")
    if rotate:
        data = data[::-1]
    ax.plot_smooth(*data, **kwargs)


def get_bin_label(edge_lower, edge_higher, format_func=lambda v: v, prefix=""):
    """
    Get label with bin range

    Parameters
    ----------
    edge_lower: float
        Lower values of bin
    edge_higher: float
        Higher values of bin
    format_func: function
        Function to format the values of the bins
    prefix: str
        Prefix to add to labels

    Returns
    -------
    srt
        Label of bin
    """
    return f"${prefix}[{format_func(edge_lower)}$ : ${format_func(edge_higher)}]$"


def add_panel_bin_label(axes, edges_lower, edges_higher, format_func=lambda v: v, prefix=""):
    """
    Adds label with bin range on the top of panel

    Parameters
    ----------
    axes: matplotlib.axes
        Axes with the panels
    edges_lower: array
        Lower values of bins
    edges_higher: array
        Higher values of bins
    format_func: function
        Function to format the values of the bins
    prefix: str
        Prefix to add to labels
    """
    for ax, val_lower, val_higher in zip(axes.flatten(), edges_lower, edges_higher):
        topax = ax.twiny()
        topax.set_xticks([])
        topax.set_xlabel(get_bin_label(val_lower, val_higher, format_func, prefix))


def get_density_colors(
    xvals,
    yvals,
    xbins,
    ybins,
    ax_rotation=0,
    rotation_resolution=30,
    xscale="linear",
    yscale="linear",
):
    """
    Get colors of point based on density

    Parameters
    ----------
    xvals: array
        Values for x coordinate
    yvals: array
        Values for y coordinate
    xbins: array, int
        Bins for x
    ybins: array, int
        Bins for y
    ax_rotation: float
        Angle (in degrees) for rotation of axis of binning. Overwrites use of xbins, ybins
    rotation_resolution: int
        Number of bins to be used when ax_rotation!=0.
    xscale: str
        Scale xaxis.
    yscale: str
        Scale yaxis.

    Returns
    -------
    ndarray
        Density value at location of each point
    """
    # Rotated points around anlgle
    sin, cos = np.sin(np.radians(ax_rotation)), np.cos(np.radians(ax_rotation))
    scalefuncs = {"linear": lambda x: x, "log": np.log10}
    xvals2, yvals2 = scalefuncs[xscale](xvals), scalefuncs[yscale](yvals)
    xvals2 = np.array(xvals2) * cos - np.array(yvals2) * sin
    yvals2 = np.array(xvals2) * sin + np.array(yvals2) * cos
    if ax_rotation == 0:
        bins = (xbins, ybins)
    else:
        bins = (
            np.linspace(xvals2.min(), xvals2.max(), rotation_resolution),
            np.linspace(yvals2.min(), yvals2.max(), rotation_resolution),
        )
    # Compute 2D rotated histogram
    hist, xedges, yedges = np.histogram2d(xvals2, yvals2, bins=bins)
    hist = hist.T
    # Interpolate histogram
    xmid = 0.5 * (xedges[:-1] + xedges[1:])
    ymid = 0.5 * (yedges[:-1] + yedges[1:])
    funcz = interp2d(xmid, ymid, hist, kind="cubic")
    return np.array([funcz(*coord)[0] for coord in zip(xvals2, yvals2)])


def nice_panel(axes, xlabel=None, ylabel=None, xscale="linear", yscale="linear"):
    """
    Add nice labels and ticks to panel plot

    Parameters
    ----------
    axes: array
        Axes with the panels
    bins1: array, int
        Bins for component 1
    bins2: array, int
        Bins for component 2

    Other parameters
    ----------------
    ax: matplotlib.axes
        Ax to add plot
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
    axes: array
        Axes with the panels
    """
    log_xticks = [np.log10(ax.get_xticks()[ax.get_xticks() > 0]) for ax in axes.flatten()]
    for ax in axes[-1, :] if len(axes.shape) > 1 else axes:
        ax.set_xlabel(xlabel)
        ax.set_xscale(xscale)
    if xscale == "log":
        for ax, xticks in zip(axes.flatten() if len(axes.shape) > 1 else axes, log_xticks):
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xticks(10**xticks)
            ax.set_xticklabels(
                [f"${10**(t-int(t)):.0f}\\times 10^{{{np.floor(t):.0f}}}$" for t in xticks],
                rotation=-45,
            )
    log_yticks = [np.log10(ax.get_yticks()[ax.get_yticks() > 0]) for ax in axes.flatten()]
    for ax in axes[:, 0] if len(axes.shape) > 1 else axes[:1]:
        ax.set_ylabel(ylabel)
        ax.set_yscale(yscale)
    if yscale == "log":
        for ax, yticks in zip(axes.flatten() if len(axes.shape) > 1 else axes, log_yticks):
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_yticks(10**yticks)
            ax.set_yticklabels(
                [f"${10**(t-int(t)):.0f}\\times 10^{{{np.floor(t):.0f}}}$" for t in yticks],
                rotation=-45,
            )


def _set_label_format(kwargs, label_format_key, label_fmt_key, log, default_fmt=".2f"):
    """
    Set function for label formatting from dictionary and removes label_fmt_key.

    Parameters
    ----------
    kwargs: dict
        Dictionary with the input values
    label_format_key: str
        Name of the format function entry
    label_fmt_key: str
        Name of entry with format of values (ex: '.2f').
        It is only used if label_format_key not in kwargs.
    log: bool
        Format labels with 10^log10(val) format.
        It is only used if label_format_key not in kwargs.
    default_fmt: str
        Format of linear values (ex: '.2f') when (label_format_key, label_fmt_key) not in kwargs.


    Returns
    -------
    function
        Label format function
    """
    label_fmt = kwargs.pop(label_fmt_key, default_fmt)
    kwargs[label_format_key] = kwargs.get(
        label_format_key,
        lambda v: f"10^{{%{label_fmt}}}" % np.log10(v) if log else f"%{label_fmt}" % v,
    )


def plot_histograms(
    histogram,
    edges1,
    edges2,
    ax,
    shape="steps",
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
    histogram: array
        Histogram 2D with dimention (edges2, edges1).
    edges1, edges2: array
        Edges of histogram.
    ax: matplotlib.axes
        Ax to add plot
    shape: str
        Shape of the lines. Can be steps or line.
    plt_kwargs: dict, None
        Additional arguments for pylab.plot
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
    ax
    """
    add_grid(ax)
    for hist_line, l_kwargs, edges in zip(
        histogram,
        none_val(lines_kwargs_list, iter(lambda: {}, 1)),
        zip(edges2, edges2[1:]),
    ):
        kwargs = updated_dict(
            {"label": get_bin_label(*edges, legend_format) if add_legend else None},
            plt_kwargs,
            l_kwargs,
        )
        plot_hist_line(hist_line, edges1, ax, shape, **kwargs)
    if add_legend:
        ax.legend(**updated_dict(legend_kwargs))
    return ax


def plot_healpix_map(
    healpix_map,
    nest=True,
    auto_lim=False,
    bad_val=None,
    ra_lim=None,
    dec_lim=None,
    fig=None,
    figsize=None,
    **kwargs,
):
    """
    Plot healpix map.

    Parameters
    ----------
    healpix_map: numpy array
        Healpix map (must be 12*(2*n)**2 size).
    nest: bool
        If ordering is nested
    auto_lim: bool
        Set automatic limits for ra/dec, requires bad_val.
    bad_val: float, None
        Values for pixels outside footprint.
    ra_lim: None, list
        Min/max RA for plot.
    dec_lim: None, list
        Min/max DEC for plot.
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
    fig: matplotlib.pyplot.figure
        Figure of the plot.
    ax: matplotlib.axes
        Ax to add plot
    cb: matplotlib.pyplot.colorbar, None
        Colorbar
    """
    nside = hp.npix2nside(len(healpix_map))
    kwargs_ = updated_dict({"flip": "geo", "title": None, "cbar": True, "nest": nest}, kwargs)
    if auto_lim:
        ra, dec = hp.pix2ang(
            nside,
            np.arange(len(healpix_map))[(healpix_map != bad_val) * ~np.isnan(healpix_map)],
            nest=nest,
            lonlat=True,
        )
        # pylint: disable=chained-comparison
        if ra.min() < 180.0 and ra.max() > 180.0:
            if (360.0 - (ra.max() - ra.min())) < (  # crossing 0 gap
                ra[ra > 180.0].min() - ra[ra < 180.0].max()  # normal gap
            ):
                ra[ra > 180.0] -= 360.0

        edge = 2 * (hp.nside2resol(nside, arcmin=True) / 60)
        kwargs_["lonra"] = [max(-360, ra.min() - edge), min(360, ra.max() + edge)]
        kwargs_["latra"] = [max(-90, dec.min() - edge), min(90, dec.max() + edge)]

    kwargs_["lonra"] = ra_lim if ra_lim else kwargs_.get("lonra")
    kwargs_["latra"] = dec_lim if dec_lim else kwargs_.get("latra")

    if (kwargs_["lonra"] is None) != (kwargs_["latra"] is None):
        raise ValueError("When auto_lim=False, ra_lim and dec_lim must be provided together.")

    if fig is None:
        fig = plt.figure()
    hp.cartview(healpix_map, hold=True, **kwargs_)
    ax = fig.axes[-2 if kwargs_["cbar"] else -1]
    ax.axis("on")
    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    if figsize:
        ax.set_aspect("auto")
        fig.set_size_inches(figsize)

    cbar = None
    if kwargs_["cbar"]:
        cbar = fig.axes[-1]

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    xticks = ax.get_xticks()
    xticks[xticks >= 360] -= 360
    if all(int(i) == i for i in xticks):
        xticks = np.array(xticks, dtype=int)
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(xticks)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("RA")
    ax.set_ylabel("DEC")

    return fig, ax, cbar
