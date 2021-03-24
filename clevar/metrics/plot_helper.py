import numpy as np
def add_grid(ax):
    ax.xaxis.grid(True, which='major', lw=.5)
    ax.yaxis.grid(True, which='major', lw=.5)
    ax.xaxis.grid(True, which='minor', lw=.1)
    ax.yaxis.grid(True, which='minor', lw=.1)
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
