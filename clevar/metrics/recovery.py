import numpy as np
import pylab as plt

def plot_hist_line(values, bins, is_matched, ax, **kwargs):
    hist_all = np.histogram(values, bins=bins)[0]
    hist_matched = np.histogram(values[is_matched], bins=bins)[0]
    recovery = np.zeros(hist_all.size)
    recovery[:] = np.nan
    recovery[hist_all>0] = hist_matched[hist_all>0]/hist_all[hist_all>0]
    ax.plot(np.transpose([bins[:-1], bins[1:]]).flatten(),
            np.transpose([recovery, recovery]).flatten(),
            **kwargs)
def plot_recovery(quantity1, bins1, is_matched,
                  quantity2=None, bins2=None, ax=None,
                  plt_kwargs={}):
    ax = plt.axes() if ax is None else ax
    plot_hist_line(quantity1, bins1, is_matched, ax, **plt_kwargs)
