import numpy as np
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
    """
    return f'[${format_func(edge_lower)}$:${format_func(edge_higher)}$]'
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
