"""@file clevar/match_metrics/recovery/funcs.py

Main recovery functions for mass and redshift plots,
wrapper of catalog_funcs functions
"""
import numpy as np
from scipy.integrate import quad_vec

from .. import plot_helper as ph
from . import catalog_funcs

def _plot_base(pltfunc, cat, matching_type, redshift_bins, mass_bins,
               transpose=False, **kwargs):
    """
    Adapts a ClCatalogFuncs function for main functions using mass and redshift.

    Parameters
    ----------
    pltfunc: function
        ClCatalogFuncs function
    cat: clevar.ClCatalog
        ClCatalog with matching information
    matching_type: str
        Type of matching to be considered. Must be in:
        'cross', 'cat1', 'cat2', 'multi_cat1', 'multi_cat2', 'multi_join'
    redshift_bins: array, int
        Bins for redshift
    mass_bins: array, int
        Bins for mass
    transpose: bool
        Transpose mass and redshift in plots
    **kwargs:
        Additional arguments to be passed to pltfunc

    Returns
    -------
    Same as pltfunc
    """
    args = ('mass', 'z', mass_bins, redshift_bins) if transpose\
        else ('z', 'mass', redshift_bins, mass_bins)
    return pltfunc(cat, *args, matching_type, **kwargs)


def _intrinsic_comp(p_m1_m2, min_mass2, ax, transpose, bins1, bins2,
                    max_mass2=1e16, min_mass2_norm=1e12):
    """
    Plots instrinsic completeness given by a mass threshold on the other catalog.

    Parameters
    ----------
    p_m1_m2: function
        `P(M1|M2)` - Probability of having a catalog 1 cluster with mass M1 given
        a catalog 2 cluster with mass M2.
    min_mass2: float
        Threshold mass in integration.
    ax: matplotlib.axes
        Ax to add plot
    transpose: bool
        Transpose mass and redshift in plots
    bins1: array
        Redshift bins (or mass if transpose).
    bins2: array
        Mass bins (or redshift if transpose).
    max_mass2: float
        Maximum mass2 for integration.
    max_mass2_norm: float
        Minimum mass2 to be used in normalization integral.
    """
    min_logmass2_norm, max_logmass2 = np.log10(min_mass2_norm), np.log10(max_mass2)
    integ = lambda m1, min_mass2: quad_vec(
        lambda logm2: p_m1_m2(m1, 10**logm2),
        np.log10(min_mass2), np.log10(max_mass2),
        epsabs=1e-50)[0]
    comp = lambda m1, m2_th: integ(m1, m2_th)/integ(m1, min_mass2_norm)
    if transpose:
        _kwargs = {'alpha':.2, 'color':'0.3', 'lw':0, 'zorder':0}
        ax.fill_between(bins1, 0, comp(bins1, min_mass2), **_kwargs)
    else:
        for m_inf, m_sup, line in zip(bins2, bins2[1:], ax.lines):
            _kwargs = {'alpha':.2, 'color':line._color, 'lw':0}
            ax.fill_between((bins1[0], bins1[-1]), comp(m_inf, min_mass2),
                            comp(m_sup, min_mass2), **_kwargs)

def plot(cat, matching_type, redshift_bins, mass_bins, transpose=False, log_mass=True,
         redshift_label=None, mass_label=None, recovery_label=None, p_m1_m2=None,
         min_mass2=1, **kwargs):
    """
    Plot recovery rate as lines, with each line binned by redshift inside a mass bin.

    Parameters
    ----------
    cat: clevar.ClCatalog
        ClCatalog with matching information
    matching_type: str
        Type of matching to be considered. Must be in:
        'cross', 'cat1', 'cat2', 'multi_cat1', 'multi_cat2', 'multi_join'
    redshift_bins: array, int
        Bins for redshift
    mass_bins: array, int
        Bins for mass
    transpose: bool
        Transpose mass and redshift in plots
    log_mass: bool
        Plot mass in log scale
    mask: array
        Mask of unwanted clusters
    mask_unmatched: array
        Mask of unwanted unmatched clusters (ex: out of footprint)
    p_m1_m2: function
        `P(M1|M2)` - Probability of having a catalog 1 cluster with mass M1 given
        a catalog 2 cluster with mass M2. If provided, computes the intrinsic completeness.
    min_mass2: float
        Threshold mass for intrinsic completeness computation.


    Other parameters
    ----------------
    shape: str
        Shape of the lines. Can be steps or line.
    mass_label: str
        Label for mass.
    redshift_label: str
        Label for redshift.
    recovery_label: str
        Label for recovery rate.
    ax: matplotlib.axes
        Ax to add plot
    plt_kwargs: dict
        Additional arguments for pylab.plot
    lines_kwargs_list: list, None
        List of additional arguments for plotting each line (using pylab.plot).
        Must have same size as len(bins2)-1
    add_legend: bool
        Add legend of bins
    legend_format: function
        Function to format the values of the bins in legend
    legend_fmt: str
        Format the values of binedges (ex: '.2f')
    legend_kwargs: dict
        Additional arguments for pylab.legend

    Returns
    -------
    info: dict
        Information of data in the plots, it contains the sections:

            * `ax`: ax used in the plot.
            * `data`: Binned data used in the plot. It has the sections:

                * `recovery`: Recovery rate binned with (bin1, bin2).\
                bins where no cluster was found have nan value.
                * `edges1`: The bin edges along the first dimension.
                * `edges2`: The bin edges along the second dimension.
                * `counts`: Counts of all clusters in bins.
                * `matched`: Counts of matched clusters in bins.
    """
    legend_fmt = kwargs.pop("legend_fmt", ".1f" if log_mass*(not transpose) else ".2f")
    kwargs['legend_format'] = kwargs.get('legend_format',
        lambda v: f'10^{{%{legend_fmt}}}'%np.log10(v) if log_mass*(not transpose)\
             else f'%{legend_fmt}'%v)
    info = _plot_base(catalog_funcs.plot, cat, matching_type,
                      redshift_bins, mass_bins, transpose,
                      scale1='log' if log_mass*transpose else 'linear',
                      xlabel=mass_label if transpose else redshift_label,
                      ylabel=recovery_label,
                      **kwargs)
    if p_m1_m2 is not None:
        _intrinsic_comp(p_m1_m2, min_mass2, info['ax'], transpose,
                        info['data']['edges1'], info['data']['edges2'], max_mass2=1e16)
    return info


def plot_panel(cat, matching_type, redshift_bins, mass_bins, transpose=False, log_mass=True,
               redshift_label=None, mass_label=None, recovery_label=None, p_m1_m2=None,
               min_mass2=1, **kwargs):

    """
    Plot recovery rate as lines in panels, with each line binned by redshift
    and each panel is based on the data inside a mass bin.

    Parameters
    ----------
    cat: clevar.ClCatalog
        ClCatalog with matching information
    matching_type: str
        Type of matching to be considered. Must be in:
        'cross', 'cat1', 'cat2', 'multi_cat1', 'multi_cat2', 'multi_join'
    redshift_bins: array, int
        Bins for redshift
    mass_bins: array, int
        Bins for mass
    transpose: bool
        Transpose mass and redshift in plots
    log_mass: bool
        Plot mass in log scale
    mask: array
        Mask of unwanted clusters
    mask_unmatched: array
        Mask of unwanted unmatched clusters (ex: out of footprint)

    Other parameters
    ----------------
    shape: str
        Shape of the lines. Can be steps or line.
    mass_label: str
        Label for mass.
    redshift_label: str
        Label for redshift.
    recovery_label: str
        Label for recovery rate.
    ax: matplotlib.axes
        Ax to add plot
    plt_kwargs: dict
        Additional arguments for pylab.plot
    panel_kwargs_list: list, None
        List of additional arguments for plotting each panel (using pylab.plot).
        Must have same size as len(bins2)-1
    fig_kwargs: dict
        Additional arguments for plt.subplots
    add_label: bool
        Add bin label to panel
    label_format: function
        Function to format the values of the bins
    label_fmt: str
        Format the values of binedges (ex: '.2f')
    p_m1_m2: function
        `P(M1|M2)` - Probability of having a catalog 1 cluster with mass M1 given
        a catalog 2 cluster with mass M2. If provided, computes the intrinsic completeness.
    min_mass2: float
        Threshold mass for intrinsic completeness computation.

    Returns
    -------
    info: dict
        Information of data in the plots, it contains the sections:

            * `fig`: `matplotlib.figure.Figure` object.
            * `axes`: `matplotlib.axes` used in the plot.
            * `data`: Binned data used in the plot. It has the sections:

                * `recovery`: Recovery rate binned with (bin1, bin2).\
                bins where no cluster was found have nan value.
                * `edges1`: The bin edges along the first dimension.
                * `edges2`: The bin edges along the second dimension.
                * `counts`: Counts of all clusters in bins.
                * `matched`: Counts of matched clusters in bins.
    """
    log = log_mass*(not transpose)
    ph._set_label_format(kwargs, 'label_format', 'label_fmt',
                         log=log, default_fmt=".1f" if log else ".2f")
    info = _plot_base(catalog_funcs.plot_panel, cat, matching_type,
                      redshift_bins, mass_bins, transpose,
                      scale1='log' if log_mass*transpose else 'linear',
                      xlabel=mass_label if transpose else redshift_label,
                      ylabel=recovery_label,
                      **kwargs)
    if p_m1_m2 is not None:
        bins2_generator = zip(info['data']['edges2'], info['data']['edges2'][1:])
        for ax in info['axes'].flatten():
            bins2 = info['data']['edges2'] if transpose else next(bins2_generator)
            _intrinsic_comp(
                p_m1_m2, min_mass2, ax, transpose,
                info['data']['edges1'], bins2, max_mass2=1e16)
    return info


def plot2D(cat, matching_type, redshift_bins, mass_bins, transpose=False, log_mass=True,
           redshift_label=None, mass_label=None, recovery_label=None, **kwargs):
    """
    Plot recovery rate as in 2D (redshift, mass) bins.

    Parameters
    ----------
    cat: clevar.ClCatalog
        ClCatalog with matching information
    matching_type: str
        Type of matching to be considered. Must be in:
        'cross', 'cat1', 'cat2', 'multi_cat1', 'multi_cat2', 'multi_join'
    redshift_bins: array, int
        Bins for redshift
    mass_bins: array, int
        Bins for mass
    transpose: bool
        Transpose mass and redshift in plots
    log_mass: bool
        Plot mass in log scale
    mask: array
        Mask of unwanted clusters
    mask_unmatched: array
        Mask of unwanted unmatched clusters (ex: out of footprint)

    Other parameters
    ----------------
    mass_label: str
        Label for mass.
    redshift_label: str
        Label for redshift.
    recovery_label: str
        Label for recovery rate.
    ax: matplotlib.axes
        Ax to add plot
    plt_kwargs: dict
        Additional arguments for pylab.plot
    add_cb: bool
        Plot colorbar
    cb_kwargs: dict
        Colorbar arguments
    add_num: int
        Add numbers in each bin
    num_kwargs: dict
        Arguments for number plot (used in plt.text)

    Returns
    -------
    info: dict
        Information of data in the plots, it contains the sections:

            * `ax`: ax used in the plot.
            * `cb` (optional): colorbar.
            * `data`: Binned data used in the plot. It has the sections:

                * `recovery`: Recovery rate binned with (bin1, bin2).\
                bins where no cluster was found have nan value.
                * `edges1`: The bin edges along the first dimension.
                * `edges2`: The bin edges along the second dimension.
                * `counts`: Counts of all clusters in bins.
                * `matched`: Counts of matched clusters in bins.
    """
    return _plot_base(catalog_funcs.plot2D, cat, matching_type,
                      redshift_bins, mass_bins, transpose,
                      scale1='log' if log_mass*transpose else 'linear',
                      scale2='log' if log_mass*(not transpose) else 'linear',
                      xlabel=mass_label if transpose else redshift_label,
                      ylabel=redshift_label if transpose else mass_label,
                      **kwargs)
