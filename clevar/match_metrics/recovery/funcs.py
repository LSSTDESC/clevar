"""@file clevar/match_metrics/recovery/funcs.py

Main recovery functions for mass and redshift plots,
wrapper of catalog_funcs functions
"""
import numpy as np
from scipy.integrate import quad_vec

from .. import plot_helper as ph
from . import catalog_funcs


def _plot_base(pltfunc, cat, matching_type, redshift_bins, mass_bins, transpose=False, **kwargs):
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
    args = (
        ("mass", "z", mass_bins, redshift_bins)
        if transpose
        else ("z", "mass", redshift_bins, mass_bins)
    )
    return pltfunc(cat, *args, matching_type, **kwargs)


def _intrinsic_comp(
    p_m1_m2,
    min_mass2,
    ax,
    transpose,
    mass_bins,
    redshift_bins=None,
    max_mass2=1e16,
    min_mass2_norm=1e12,
):
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
    mass_bins: array
        Mass bins.
    redshift_bins: array
        Redshift bins (required if transpose=False).
    max_mass2: float
        Maximum mass2 for integration. If none, estimated from p_m1_m2.
    min_mass2_norm: float
        Minimum mass2 to be used in normalization integral. If none, estimated from p_m1_m2.
    """
    # For detemining min/max mass
    lim_mass_vals = np.logspace(-5, 20, 2501)
    if min_mass2_norm is None:
        p_vals = p_m1_m2(mass_bins[0], lim_mass_vals)
        p_max, arg_max = p_vals.max(), p_vals.argmax()
        p_vals = p_vals[:arg_max]
        v_min = max(p_vals.min(), p_max * 1e-10)
        min_mass2_norm = lim_mass_vals[:arg_max][p_vals < v_min][-1]
    if max_mass2 is None:
        p_vals = p_m1_m2(mass_bins[-1], lim_mass_vals)
        p_max, arg_max = p_vals.max(), p_vals.argmax()
        p_vals = p_vals[arg_max + 1 :]
        v_min = max(p_vals.min(), p_max * 1e-10)
        max_mass2 = lim_mass_vals[arg_max + 1 :][p_vals < v_min][0]
    integ = lambda m1, min_mass2: quad_vec(
        lambda logm2: p_m1_m2(m1, 10**logm2),
        np.log10(min_mass2),
        np.log10(max_mass2),
        epsabs=1e-50,
    )[0]
    comp = lambda m1, m2_th: integ(m1, m2_th) / integ(m1, min_mass2_norm)
    if transpose:
        _kwargs = {"alpha": 0.2, "color": "0.3", "lw": 0, "zorder": 0}
        ax.fill_between(mass_bins, 0, comp(mass_bins, min_mass2), **_kwargs)
    else:
        for m_inf, m_sup, line in zip(mass_bins, mass_bins[1:], ax.lines):
            _kwargs = {"alpha": 0.2, "color": line._color, "lw": 0}
            ax.fill_between(
                (redshift_bins[0], redshift_bins[-1]),
                comp(m_inf, min_mass2),
                comp(m_sup, min_mass2),
                **_kwargs,
            )


def plot(
    cat,
    matching_type,
    redshift_bins,
    mass_bins,
    transpose=False,
    log_mass=True,
    redshift_label=None,
    mass_label=None,
    recovery_label=None,
    p_m1_m2=None,
    min_mass2=1,
    max_mass2=None,
    min_mass2_norm=None,
    **kwargs,
):
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
    max_mass2: float
        Maximum mass2 for integration. If none, estimated from p_m1_m2.
    min_mass2_norm: float
        Minimum mass2 to be used in normalization integral. If none, estimated from p_m1_m2.


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
    plt_kwargs: dict, None
        Additional arguments for pylab.plot.
        It also includes the possibility of smoothening the line with `n_increase, scheme`
        arguments. See `clevar.utils.smooth_line` for details.
    lines_kwargs_list: list, None
        List of additional arguments for plotting each line (using pylab.plot).
        Must have same size as len(bins2)-1
    add_legend: bool
        Add legend of bins
    legend_format: function
        Function to format the values of the bins in legend
    legend_fmt: str
        Format the values of binedges (ex: '.2f')
    legend_kwargs: dict, None
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
    legend_fmt = kwargs.pop("legend_fmt", ".1f" if log_mass * (not transpose) else ".2f")
    kwargs["legend_format"] = kwargs.get(
        "legend_format",
        lambda v: f"10^{{%{legend_fmt}}}" % np.log10(v)
        if log_mass * (not transpose)
        else f"%{legend_fmt}" % v,
    )
    info = _plot_base(
        catalog_funcs.plot,
        cat,
        matching_type,
        redshift_bins,
        mass_bins,
        transpose,
        scale1="log" if log_mass * transpose else "linear",
        xlabel=mass_label if transpose else redshift_label,
        ylabel=recovery_label,
        **kwargs,
    )
    if p_m1_m2 is not None:
        mass_bins = info["data"]["edges1"] if transpose else info["data"]["edges2"]
        redshift_bins = info["data"]["edges2"] if transpose else info["data"]["edges1"]
        _intrinsic_comp(
            p_m1_m2,
            min_mass2,
            info["ax"],
            transpose,
            mass_bins,
            redshift_bins,
            max_mass2,
            min_mass2_norm,
        )
    return info


def plot_panel(
    cat,
    matching_type,
    redshift_bins,
    mass_bins,
    transpose=False,
    log_mass=True,
    redshift_label=None,
    mass_label=None,
    recovery_label=None,
    p_m1_m2=None,
    min_mass2=1,
    max_mass2=None,
    min_mass2_norm=None,
    **kwargs,
):
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
    plt_kwargs: dict, None
        Additional arguments for pylab.plot.
        It also includes the possibility of smoothening the line with `n_increase, scheme`
        arguments. See `clevar.utils.smooth_line` for details.
    panel_kwargs_list: list, None
        List of additional arguments for plotting each panel (using pylab.plot).
        Must have same size as len(bins2)-1
    fig_kwargs: dict, None
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
    max_mass2: float
        Maximum mass2 for integration. If none, estimated from p_m1_m2.
    min_mass2_norm: float
        Minimum mass2 to be used in normalization integral. If none, estimated from p_m1_m2.

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
    log = log_mass * (not transpose)
    ph._set_label_format(
        kwargs, "label_format", "label_fmt", log=log, default_fmt=".1f" if log else ".2f"
    )
    info = _plot_base(
        catalog_funcs.plot_panel,
        cat,
        matching_type,
        redshift_bins,
        mass_bins,
        transpose,
        scale1="log" if log_mass * transpose else "linear",
        xlabel=mass_label if transpose else redshift_label,
        ylabel=recovery_label,
        **kwargs,
    )
    if p_m1_m2 is not None:
        for ax, bins2 in zip(
            info["axes"].flatten(), zip(info["data"]["edges2"], info["data"]["edges2"][1:])
        ):
            mass_bins = info["data"]["edges1"] if transpose else bins2
            redshift_bins = bins2 if transpose else info["data"]["edges1"]
            _intrinsic_comp(
                p_m1_m2,
                min_mass2,
                ax,
                transpose,
                mass_bins,
                redshift_bins,
                max_mass2,
                min_mass2_norm,
            )
    return info


def plot2D(
    cat,
    matching_type,
    redshift_bins,
    mass_bins,
    transpose=False,
    log_mass=True,
    redshift_label=None,
    mass_label=None,
    recovery_label=None,
    **kwargs,
):
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
    plt_kwargs: dict, None
        Additional arguments for pylab.pcolor.
    add_cb: bool
        Plot colorbar
    cb_kwargs: dict, None
        Colorbar arguments
    add_num: int
        Add numbers in each bin
    num_kwargs: dict, None
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
    return _plot_base(
        catalog_funcs.plot2D,
        cat,
        matching_type,
        redshift_bins,
        mass_bins,
        transpose,
        scale1="log" if log_mass * transpose else "linear",
        scale2="log" if log_mass * (not transpose) else "linear",
        xlabel=mass_label if transpose else redshift_label,
        ylabel=redshift_label if transpose else mass_label,
        **kwargs,
    )


def skyplot(
    cat,
    matching_type,
    nside=256,
    nest=True,
    mask=None,
    mask_unmatched=None,
    auto_lim=False,
    ra_lim=None,
    dec_lim=None,
    recovery_label="Recovery Rate",
    fig=None,
    figsize=None,
    **kwargs,
):
    """
    Plot recovery rate in healpix pixels.

    Parameters
    ----------
    cat: clevar.ClCatalog
        ClCatalog with matching information
    matching_type: str
        Type of matching to be considered. Must be in:
        'cross', 'cat1', 'cat2', 'multi_cat1', 'multi_cat2', 'multi_join'
    nside: int
        Healpix nside
    nest: bool
        If ordering is nested
    mask: array
        Mask of unwanted clusters
    mask_unmatched: array
        Mask of unwanted unmatched clusters (ex: out of footprint)
    auto_lim: bool
        Set automatic limits for ra/dec.
    ra_lim: None, list
        Min/max RA for plot.
    dec_lim: None, list
        Min/max DEC for plot.
    recovery_label: str
        Lable for colorbar. Default: 'recovery rate'
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
    info: dict
        Information of data in the plots, it contains the sections:

            * `fig` (matplotlib.pyplot.figure): Figure of the plot. The main can be accessed at\
            `info['fig'].axes[0]`, and the colorbar at `info['fig'].axes[1]`.
            * `nc_pix`: Dictionary with the number of clusters in each pixel.
            * `nc_mt_pix`: Dictionary with the number of matched clusters in each pixel.
    """
    return catalog_funcs.skyplot(
        cat,
        matching_type,
        nside=nside,
        nest=nest,
        mask=mask,
        mask_unmatched=mask_unmatched,
        auto_lim=auto_lim,
        ra_lim=ra_lim,
        dec_lim=dec_lim,
        recovery_label=recovery_label,
        fig=fig,
        figsize=figsize,
        **kwargs,
    )
