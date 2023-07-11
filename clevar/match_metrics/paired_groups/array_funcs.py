"""@file clevar/match_metrics/paired_groups/array_funcs.py

Functions that use arrays.
"""
import numpy as np

from ..plot_helper import plt
from ...utils import none_val, updated_dict
from .. import plot_helper as ph


def get_groups_2dhistoram(group1, group2):
    nmax = group1.max()
    group_bins = np.arange(1, nmax + 2)
    groups_sizes1 = np.histogram(
        group1,
        bins=group_bins,
    )[0]
    groups_sizes2 = np.histogram(
        group2,
        bins=group_bins,
    )[0]

    groups_sizes_max1 = groups_sizes1.max()
    groups_sizes_max2 = groups_sizes2.max()
    return np.histogram2d(
        groups_sizes1,
        groups_sizes2,
        bins=(
            np.arange(1, groups_sizes_max1 + 2),
            np.arange(1, groups_sizes_max2 + 2),
        ),
    )


def _get_matrix_split_numbers(hist):
    size1, size2 = hist.shape
    return np.array(
        [
            hist[0, 0],
            np.diag(hist)[1:].sum(),
            hist[np.triu_indices(n=size1, k=1, m=size2)].sum(),
            hist[np.tril_indices(n=size1, k=-1, m=size2)].sum(),
        ]
    )


def _get_groups_to_keep(group, mask_group, mask_group_exclusive=False):
    if mask_group_exclusive:
        groups_keep = np.ones(group.max(), dtype=bool)
        groups_keep[group[~mask_group] - 1] = False
    else:
        groups_keep = np.zeros(group.max(), dtype=bool)
        groups_keep[group[mask_group] - 1] = True
    return groups_keep


#############################################
#### Plots ##################################
#############################################

_frag_overm_labels = [
    r"$1\rightarrow 1$",
    r"$n\rightarrow n$",
    r"$n\rightarrow (<n)$",
    r"$n\rightarrow (>n)$",
]


def _plot_fragmentation_overmerging_fraction(
    data,
    ax=None,
    plt_kwargs=None,
    lines_kwargs_list=None,
):
    """
    Plot recovery rate as lines, with each line binned by bins1 inside a bin of bins2.

    Parameters
    ----------
    ax: matplotlib.axes
        Ax to add plot
    plt_kwargs: dict, None
        Additional arguments for pylab.plot
    lines_kwargs_list: list, None
        List of additional arguments for plotting each line (using pylab.plot).
        Must have same size as len(bins2)-1

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
    info = {
        "data": data,
        "ax": plt.axes() if ax is None else ax,
    }
    ph.add_grid(info["ax"])
    for line, lab_kw, l_kwargs in zip(
        (data["numbers1"] / data["numbers1"].sum(axis=1)[:, None]).T,
        [{"label": v} for v in _frag_overm_labels],
        none_val(lines_kwargs_list, iter(lambda: {}, 1)),
    ):
        kwargs = updated_dict(
            {},
            plt_kwargs,
            lab_kw,
            l_kwargs,
        )
        info["ax"].plot(data["bins_mid"], line, **kwargs)
    return info


def _plot_fragmentation_overmerging_ratio(
    data,
    ax=None,
    plt_kwargs=None,
    lines_kwargs_list=None,
):
    """
    Plot recovery rate as lines, with each line binned by bins1 inside a bin of bins2.

    Parameters
    ----------
    ax: matplotlib.axes
        Ax to add plot
    plt_kwargs: dict, None
        Additional arguments for pylab.plot
    lines_kwargs_list: list, None
        List of additional arguments for plotting each line (using pylab.plot).
        Must have same size as len(bins2)-1

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
    info = {
        "data": data,
        "ax": plt.axes() if ax is None else ax,
    }
    ph.add_grid(info["ax"])
    for line, lab_kw, l_kwargs in zip(
        [
            data["numbers2"].sum(axis=1) / data["numbers1"].sum(axis=1),
            *(data["numbers2"] / data["numbers1"]).T[2:],
        ],
        [
            {"label": "Total", "color": "0"},
            *[{"label": _frag_overm_labels[i], "color": f"C{i}"} for i in (2, 3)],
        ],
        none_val(lines_kwargs_list, iter(lambda: {}, 1)),
    ):
        kwargs = updated_dict(
            {},
            plt_kwargs,
            lab_kw,
            l_kwargs,
        )
        info["ax"].plot(data["bins_mid"], line, **kwargs)
    return info
