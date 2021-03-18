import numpy as np
import pylab as plt

from ..utils import none_val, bin_masks

def plot_hist_line(values, bins, is_matched, ax, **kwargs):
    hist_all = np.histogram(values, bins=bins)[0]
    hist_matched = np.histogram(values[is_matched], bins=bins)[0]
    recovery = np.zeros(hist_all.size)
    recovery[:] = np.nan
    recovery[hist_all>0] = hist_matched[hist_all>0]/hist_all[hist_all>0]
    ax.plot(np.transpose([bins[:-1], bins[1:]]).flatten(),
            np.transpose([recovery, recovery]).flatten(),
            **kwargs)
def plot_recovery_raw(quantity, bins, is_matched,
                  split_masks=None, ax=None,
                  plt_kwargs={}, plt_mask_kwargs_list=None):
    ax = plt.axes() if ax is None else ax
    masks = none_val(split_masks, [np.ones(len(quantity), dtype=bool)])
    plt_mask_kwargs_list = none_val(plt_mask_kwargs_list, [{} for m in masks])
    for m, mask_kwargs in zip(masks, plt_mask_kwargs_list):
        kwargs = {}
        kwargs.update(plt_kwargs)
        kwargs.update(mask_kwargs)
        plot_hist_line(quantity[m], bins, is_matched[m], ax, **kwargs)
def plot_recovery(cat, matching_type, redshift_bins, mass_bins,
                  ax=None, transpose=False, lines_kwargs=None):
    ax = plt.axes() if ax is None else ax
    if transpose:
        vals, bins, split_masks = cat.data['mass'], mass_bins, bin_masks(cat.data['z'], redshift_bins)
    else:
        vals, bins, split_masks = cat.data['z'], redshift_bins, bin_masks(cat.data['mass'], mass_bins)
    is_matched = matching_masks[matching_type]
    plot_recovery_raw(vals, bins, cat.get_matching_mask(matching_type), split_masks,
                      ax=ax, plt_kwargs=plt_kwargs, plt_mask_kwargs_list=lines_kwargs)
