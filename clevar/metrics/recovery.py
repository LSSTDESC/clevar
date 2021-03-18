import numpy as np
import pylab as plt

from ..utils import none_val, bin_masks
from . import plot_helper as ph

def get_recovery_rate(vals1, vals2, bins1, bins2, is_matched):
    hist_all = np.histogram2d(vals1, vals2, bins=(bins1, bins2))[0]
    hist_matched = np.histogram2d(vals1[is_matched], vals2[is_matched], bins=(bins1, bins2))[0]
    recovery = np.zeros(hist_all.shape)
    recovery[:] = np.nan
    recovery[hist_all>0] = hist_matched[hist_all>0]/hist_all[hist_all>0]
    return recovery
def plot_hist_line(hist_values, bins, ax, kwargs={}):
    ax.plot(np.transpose([bins[:-1], bins[1:]]).flatten(),
            np.transpose([hist_values, hist_values]).flatten(),
            **kwargs)
class RecoveryArray():
    def plot(values1, values2, bins1, bins2, is_matched,
                      split_masks=None, ax=None,
                      plt_kwargs={}, lines_kwargs_list=None):
        for i in (values1, values2, bins1, bins2, is_matched):
            print(list(i))
        ax = plt.axes() if ax is None else ax
        ph.add_grid(ax)
        recovery = get_recovery_rate(values1, values2, bins1, bins2, is_matched).T
        lines_kwargs_list = none_val(lines_kwargs_list, [{} for m in bins2[:-1]])
        for rec_line, l_kwargs in zip(recovery, lines_kwargs_list):
            kwargs = {}
            kwargs.update(plt_kwargs)
            kwargs.update(l_kwargs)
            plot_hist_line(rec_line, bins1, ax, kwargs)
    def plot_panel(values1, values2, bins1, bins2, is_matched,
                   transpose=False, plt_kwargs={}, panel_kwargs_list=None,
                   fig_kwargs={}):
        ni = int(np.floor(np.sqrt(len(bins2))))
        nj = ni if ni**2>=len(bins2) else ni+1
        f, axes = plt.subplots(ni, nj, sharex=True, sharey=True, **fig_kwargs)
        recovery = get_recovery_rate(values1, values2, bins1, bins2, is_matched).T
        panel_kwargs_list = none_val(panel_kwargs_list, [{} for m in bins2[:-1]])
        for ax, rec_line, p_kwargs in zip(axes, recovery, panel_kwargs_list):
            ph.add_grid(ax)
            kwargs = {}
            kwargs.update(plt_kwargs)
            kwargs.update(p_kwargs)
            plot_hist_line(rec_line, bins1, ax, kwargs)
        return f, axes
    def plot2D(values1, values2, bins1, bins2, is_matched,
                transpose=False, ax=None, plt_kwargs={},
                add_cb=True, cb_kwargs={}):
        recovery = get_recovery_rate(values1, values2, bins1, bins2, is_matched).T
        ax = plt.axes() if ax is None else ax
        c = ax.pcolor(bins1, bins2, recovery, **plt_kwargs)
        return plt.colorbar(c, **cb_kwargs)
class RecoveryCatalog():
    def _prep_recovery_data(cat, col1, col2, bins1, bins2, transpose):
        if transpose:
            return cat.data[col2], cat.data[col1], bins2, bins1
        else:
            return cat.data[col1], cat.data[col2], bins1, bins2
    def plot(cat, col1, col2, bins1, bins2, matching_type,
             ax=None, transpose=False, plt_kwargs={}, lines_kwargs_list=None):
        RecoveryArray.plot(*RecoveryCatalog._prep_recovery_data(cat, col1, col2, bins1, bins2, transpose),
              is_matched=cat.get_matching_mask(matching_type), ax=ax,
              plt_kwargs=plt_kwargs, lines_kwargs_list=lines_kwargs_list)
    def plot_panel(cat, col1, col2, bins1, bins2, matching_type,
                   transpose=False, plt_kwargs={}, panel_kwargs_list=None,
                   fig_kwargs={}):
        return RecoveryArray.plot_panel(
              *RecoveryCatalog._prep_recovery_data(cat, col1, col2, bins1, bins2, transpose),
              is_matched=cat.get_matching_mask(matching_type),
              plt_kwargs=plt_kwargs, panel_kwargs_list=panel_kwargs_list, fig_kwargs=fig_kwargs)
    def plot2D(cat, col1, col2, bins1, bins2, matching_type,
               ax=None, transpose=False, plt_kwargs={},
               add_cb=True, cb_kwargs={}):
        return RecoveryArray.plot2D(
            *RecoveryCatalog._prep_recovery_data(cat, col1, col2, bins1, bins2, transpose),
             is_matched=cat.get_matching_mask(matching_type), ax=ax,
             plt_kwargs=plt_kwargs, add_cb=add_cb, cb_kwargs=cb_kwargs)
def plot(cat, matching_type, redshift_bins, mass_bins,
                  ax=None, transpose=False, plt_kwargs={}, lines_kwargs_list=None):
    RecoveryCatalog.plot(cat, col1='z', col2='mass', bins1=redshift_bins, bins2=mass_bins,
                         matching_type=matching_type, transpose=transpose, ax=ax,
                         plt_kwargs=plt_kwargs, lines_kwargs_list=lines_kwargs_list)
def plot_panel(cat, matching_type, redshift_bins, mass_bins,
               transpose=False, plt_kwargs={}, panel_kwargs_list=None,
               fig_kwargs={}):
    return RecoveryCatalog.plot_panel(cat, col1='z', col2='mass', bins1=redshift_bins, bins2=mass_bins,
                         matching_type=matching_type, transpose=transpose,
                         plt_kwargs=plt_kwargs, panel_kwargs_list=panel_kwargs_list,
                         fig_kwargs=fig_kwargs)
def plot2D(cat, matching_type, redshift_bins, mass_bins,
            transpose=False, ax=None, plt_kwargs={},
            add_cb=True, cb_kwargs={}):
    return RecoveryCatalog.plot2D(cat, col1='z', col2='mass', bins1=redshift_bins, bins2=mass_bins,
                         matching_type=matching_type, transpose=transpose, ax=ax,
                         plt_kwargs=plt_kwargs, add_cb=add_cb, cb_kwargs=cb_kwargs)
