import numpy as np
from scipy.interpolate import interp2d

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
def plot_hist_line(hist_values, bins, ax, shape='steps', kwargs={}):
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
                rotation_resolution=30):
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

    Returns
    -------
    ndarray
        Density value at location of each point
    """
    # Rotated points around anlgle
    sr, cr = np.sin(np.radians(ax_rotation)), np.cos(np.radians(ax_rotation))
    x2 = np.array(x)*cr-np.array(y)*sr
    y2 = np.array(x)*sr+np.array(y)*cr
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
