# Set mpl backend run plots on github actions
import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == 'test':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import numpy as np
from scipy.interpolate import interp2d
from matplotlib.ticker import ScalarFormatter, NullFormatter

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
    ax.xaxis.grid(True, which='major', lw=major_lw)
    ax.yaxis.grid(True, which='major', lw=major_lw)
    ax.xaxis.grid(True, which='minor', lw=minor_lw)
    ax.yaxis.grid(True, which='minor', lw=minor_lw)
def plot_hist_line(hist_values, bins, ax, shape='steps', **kwargs):
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
def get_bin_label(edge_lower, edge_higher,
                  format_func=lambda v:v):
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

    Returns
    -------
    srt
        Label of bin
    """
    return f'[${format_func(edge_lower)}$ : ${format_func(edge_higher)}$]'
def add_panel_bin_label(axes, edges_lower, edges_higher,
                        format_func=lambda v:v):
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
    """
    for ax, vb, vt in zip(axes.flatten(), edges_lower, edges_higher):
        topax = ax.twiny()
        topax.set_xticks([])
        topax.set_xlabel(get_bin_label(vb, vt, format_func))
def get_density_colors(x, y, xbins, ybins, ax_rotation=0,
                rotation_resolution=30, xscale='linear', yscale='linear'):
    """
    Get colors of point based on density

    Parameters
    ----------
    x: array
        Values for x coordinate
    y: array
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
    sr, cr = np.sin(np.radians(ax_rotation)), np.cos(np.radians(ax_rotation))
    scalefuncs = {'linear': lambda x:x, 'log': lambda x: np.log10(x)}
    x2, y2 = scalefuncs[xscale](x), scalefuncs[yscale](y)
    x2 = np.array(x2)*cr-np.array(y2)*sr
    y2 = np.array(x2)*sr+np.array(y2)*cr
    if ax_rotation == 0:
        bins = (xbins, ybins)
    else:
        bins = (np.linspace(x2.min(), x2.max(), rotation_resolution),
                np.linspace(y2.min(), y2.max(), rotation_resolution))
    # Compute 2D rotated histogram
    hist, xedges, yedges = np.histogram2d(x2, y2, bins=bins)
    hist = hist.T
    # Interpolate histogram
    xm = .5*(xedges[:-1]+ xedges[1:])
    ym = .5*(yedges[:-1]+ yedges[1:])
    fz = interp2d(xm, ym, hist, kind='cubic')
    return np.array([fz(*coord)[0] for coord in zip(x2, y2)])
def nice_panel(axes, xlabel=None, ylabel=None, xscale='linear', yscale='linear'):
    """
    Add nice labels and ticks to panel plot

    Parameters
    ----------
    axes: array
        Axes with the panels
    bins1: array
        Bins for component 1
    bins2: array
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
    log_xticks = [np.log10(ax.get_xticks()[ax.get_xticks()>0])
                    for ax in axes.flatten()]
    for ax in (axes[-1,:] if len(axes.shape)>1 else axes):
        ax.set_xlabel(xlabel)
        ax.set_xscale(xscale)
    if xscale=='log':
        for ax, xticks in zip(axes.flatten() if len(axes.shape)>1 else axes, log_xticks):
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xticks(10**xticks)
            ax.set_xticklabels([f'${10**(t-int(t)):.0f}\\times 10^{{{np.floor(t):.0f}}}$'
                                for t in xticks], rotation=-45)
    log_yticks = [np.log10(ax.get_yticks()[ax.get_yticks()>0])
                    for ax in axes.flatten()]
    for ax in (axes[:,0] if len(axes.shape)>1 else axes[:1]):
        ax.set_ylabel(ylabel)
        ax.set_yscale(yscale)
    if yscale=='log':
        for ax, yticks in zip(axes.flatten() if len(axes.shape)>1 else axes, log_yticks):
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_yticks(10**yticks)
            ax.set_yticklabels([f'${10**(t-int(t)):.0f}\\times 10^{{{np.floor(t):.0f}}}$'
                                for t in yticks], rotation=-45)
    return
