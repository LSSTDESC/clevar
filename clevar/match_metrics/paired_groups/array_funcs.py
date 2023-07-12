"""@file clevar/match_metrics/paired_groups/array_funcs.py

Functions that use arrays.
"""
import numpy as np

from ..plot_helper import plt
from ...utils import none_val, updated_dict
from .. import plot_helper as ph


def get_groups_2dhistoram(group1, group2):
    """Get the histogram for the number of groups with a given number of members.

    Parameters
    ----------
    group1, group2: array
        Id of group per cluster.

    Returns
    -------
    hist: 2d array
        Histogram 2d
    bins1, bins2: array
        Number of group members for catalog 1(2).
    """
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


def get_groups_counts(group1, group2):
    """Get number of clusters in each group.

    Parameters
    ----------
    group1, group2: array
        Id of group per cluster.

    Retruns
    -------
    counts1: array
        Numbers of objects in each group per cluster for catalog1.
    counts2: array
        Numbers of objects in each group per cluster for catalog2.
    """
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
    inds1 = (
        np.digitize(
            group1,
            bins=group_bins,
        )
        - 1
    )
    inds2 = (
        np.digitize(
            group2,
            bins=group_bins,
        )
        - 1
    )

    counts1 = np.zeros(group1.size)
    counts2 = np.zeros(group2.size)

    counts1 = groups_sizes1[inds1]
    counts2 = groups_sizes2[inds2]

    return counts1, counts2


def _get_matrix_split_numbers(matrix):
    """Get number of objects in matrix split by the diagonal.

    Parameters
    ----------
    matrix: array 2d
        Matrix

    Retruns
    -------
    array
        Numbers in matrix from: first cell, diagonal (minus first cell), above diagonal,
        below diagonal.
    """
    size1, size2 = matrix.shape
    return np.array(
        [
            matrix[0, 0],
            np.diag(matrix)[1:].sum(),
            matrix[np.triu_indices(n=size1, k=1, m=size2)].sum(),
            matrix[np.tril_indices(n=size1, k=-1, m=size2)].sum(),
        ]
    )


def _get_groups_to_keep(group, mask_group, mask_group_exclusive=False):
    """Get ids of groups to keep given a mask.

    Parameters
    ----------
    group: array
        Id of group per cluster.
    mask_group: array, None
        Masks for groups in catalog.
    mask_group_exclusive: bool
        If true, groups with any masked members are excluded,
        else only groups with all masked members are excluded.

    Returns
    -------
    array
        Ids of groups to keep
    """
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


def plot_fragmentation_overmerging_fraction(
    data,
    ax=None,
    plt_kwargs=None,
    lines_kwargs_list=None,
):
    """Plot fraction of clusters with fragmentation and overmerging.

    Parameters
    ----------
    data: dict
        Binned data used in the plot. Must contain `numbers1`, `numbers2`, `bins_mid`.
    ax: matplotlib.axes
        Ax to add plot
    plt_kwargs: dict, None
        Additional arguments for pylab.plot
    lines_kwargs_list: list, None
        List of additional arguments for plotting each line (using pylab.plot).
        Must have same size as len(bins2)-1
    legend_kwargs: dict, None
        Additional arguments for pylab.legend

    Returns
    -------
    info: dict
        Information of data in the plots, it contains the sections:

            * `ax`: ax used in the plot.
            * `data`: Binned data used in the plot. It has the sections:

                * `numbers1` (2d array): Numbers of objects in bins with configuration:
                1->1, n->n, n->(<n), n->(>n).
                * `numbers2` (2d array): Numbers of objects in bins with configuration:
                1->1, n->n, n->(>n), n->(<n).
                * `bins_mid` (array): mid values of bins

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


def plot_fragmentation_overmerging_ratio(
    data,
    ax=None,
    plt_kwargs=None,
    lines_kwargs_list=None,
):
    """Plot ratio of cluster2/cluster1 counts in samples: total, with fraction and overmerging.

    Parameters
    ----------
    data: dict
        Binned data used in the plot. Must contain `numbers1`, `numbers2`, `bins_mid`.
    ax: matplotlib.axes
        Ax to add plot
    plt_kwargs: dict, None
        Additional arguments for pylab.plot
    lines_kwargs_list: list, None
        List of additional arguments for plotting each line (using pylab.plot).
        Must have same size as len(bins2)-1
    legend_kwargs: dict, None
        Additional arguments for pylab.legend

    Returns
    -------
    info: dict
        Information of data in the plots, it contains the sections:

            * `ax`: ax used in the plot.
            * `data`: Binned data used in the plot. It has the sections:

                * `numbers1` (2d array): Numbers of objects in bins with configuration:
                1->1, n->n, n->(<n), n->(>n).
                * `numbers2` (2d array): Numbers of objects in bins with configuration:
                1->1, n->n, n->(>n), n->(<n).
                * `bins_mid` (array): mid values of bins

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
