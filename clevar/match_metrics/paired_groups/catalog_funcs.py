"""@file clevar/match_metrics/paired_groups/catalog_funcs.py

Functions that use catalogs.
"""
import numpy as np

from ..plot_helper import plt
from ...utils import updated_dict
from . import array_funcs


def _find_grup(cat1, cat2, group_id1, group_id2, mask1, mask2, id1):
    ind1 = cat1.id_dict[id1]
    if id1 not in group_id1 and mask1[ind1]:
        group_id1.add(id1)
        for id2 in cat1["mt_multi_self"][ind1]:
            _find_grup(cat2, cat1, group_id2, group_id1, mask2, mask1, id2)


def add_grups(cat1, cat2, mask1=None, mask2=None):
    """Finds groups and adds group id to catalogs

    Parameters
    ----------
    cat1, cat2: clevar.ClCatalog
        ClCatalogs with multiple matching information.
    mask1, mask2: array, None
        Masks for clusters 1(2), must have size=cat1(2).size
    """
    cat1["group"] = -1
    cat2["group"] = -1
    if mask1 is None:
        mask1 = np.ones(cat1.size, dtype=bool)
    if mask2 is None:
        mask2 = np.ones(cat2.size, dtype=bool)
    group_id = 0
    for cand1 in cat1:
        # simplified comparison for cand1["group"] < 0 and len(cand1["mt_multi_self"]) > 0
        if (cand1["group"] * len(cand1["mt_multi_self"])) < 0:
            group_id += 1
            current_group_id1 = set()
            current_group_id2 = set()
            _find_grup(
                cat1,
                cat2,
                current_group_id1,
                current_group_id2,
                mask1,
                mask2,
                cand1[cat1.tags["id"]],
            )
            cat1["group"][cat1.ids2inds(current_group_id1)] = group_id
            cat2["group"][cat2.ids2inds(current_group_id2)] = group_id


def get_fragmentation_overmerging_numbers(
    cat1,
    cat2,
    mask_group1=None,
    mask_group2=None,
    mask_group1_exclusive=False,
    mask_group2_exclusive=False,
):
    """Get number of clusters with fragmentation and overmerging.

    Parameters
    ----------
    cat1, cat2: clevar.ClCatalog
        ClCatalogs with multiple matching information.
    mask_group1, mask_group2: array, None
        Masks for groups in catalog 1(2).
    mask_group1_exclusive, mask_group2_exclusive: bool
        If true, groups with any masked members are excluded,
        else only groups with all masked members are excluded.

    Retruns
    -------
    number1: array
        Numbers of objects with configuration: 1->1, n->n, n->(<n), n->(>n).
    number2: array
        Numbers of objects with configuration: 1->1, n->n, n->(>n), n->(<n).
    """
    if mask_group1 is None:
        mask_group1 = np.ones(cat1.size, dtype=bool)
    if mask_group2 is None:
        mask_group2 = np.ones(cat2.size, dtype=bool)

    groups1 = cat1["group"][cat1["group"] > 0]
    groups2 = cat2["group"][cat2["group"] > 0]

    # pylint: disable=protected-access
    groups_keep = array_funcs._get_groups_to_keep(
        groups1, mask_group1[cat1["group"] > 0], mask_group1_exclusive
    ) * array_funcs._get_groups_to_keep(
        groups2, mask_group2[cat2["group"] > 0], mask_group2_exclusive
    )

    hist, vals1, vals2 = array_funcs.get_groups_2dhistoram(
        groups1[groups_keep[groups1 - 1]],
        groups2[groups_keep[groups2 - 1]],
    )

    numbers1 = array_funcs._get_matrix_split_numbers(hist * vals1[:-1, None])
    numbers2 = array_funcs._get_matrix_split_numbers(hist * vals2[:-1])

    return numbers1, numbers2


def get_fragmentation_overmerging_numbers_binned(
    cat1,
    cat2,
    quantity1,
    bins1,
    mask_group1_exclusive=False,
):
    """Get number of clusters with fragmentation and overmerging in bins.

    Parameters
    ----------
    cat1, cat2: clevar.ClCatalog
        ClCatalogs with multiple matching information.
    quantity1: str
        Column of catalog 1 to bin the data
    bins1: array
        Bins for quantity1
    mask_group1_exclusive: bool
        If true, only groups no masked members are kept in bins,
        else groups with any unmasked members are kept in bins.

    Retruns
    -------
    numbers1: 2d array
        Numbers of objects in bins with configuration: 1->1, n->n, n->(<n), n->(>n).
    numbers2: 2d array
        Numbers of objects in bins with configuration: 1->1, n->n, n->(>n), n->(<n).
    """
    numbers1, numbers2 = np.ones(4), np.ones(4)
    for v_lower, v_upper in zip(bins1, bins1[1:]):
        numbers = get_fragmentation_overmerging_numbers(
            cat1,
            cat2,
            mask_group1=(cat1[quantity1] >= v_lower) * (cat1[quantity1] < v_upper),
            mask_group1_exclusive=mask_group1_exclusive,
        )
        numbers1 = np.vstack([numbers1, numbers[0]])
        numbers2 = np.vstack([numbers2, numbers[1]])
    return numbers1[1:], numbers2[1:]


def get_groups_counts(cat1, cat2, mask1=None, mask2=None):
    """Get number of clusters in each group.

    Parameters
    ----------
    cat1, cat2: clevar.ClCatalog
        ClCatalogs with multiple matching information.
    mask_group1, mask_group2: array, None
        Masks for groups in catalog 1(2).

    Retruns
    -------
    counts1: array
        Numbers of objects in each group per cluster for catalog1.
    counts2: array
        Numbers of objects in each group per cluster for catalog2.
    """
    if mask1 is None:
        mask1 = np.ones(cat1.size, dtype=bool)
    if mask2 is None:
        mask2 = np.ones(cat2.size, dtype=bool)

    msk1 = mask1 * (cat1["group"] > 0)
    msk2 = mask1 * (cat2["group"] > 0)

    counts1 = np.zeros(msk1.size)
    counts2 = np.zeros(msk2.size)

    ct1, ct2 = array_funcs.get_groups_counts(
        group1=cat1["group"][msk1],
        group2=cat2["group"][msk2],
    )

    counts1[msk1] = ct1
    counts2[msk2] = ct2

    return counts1, counts2


#############################################
#### Plots ##################################
#############################################


def _get_fragmentation_overmerging_data(
    cat1,
    cat2,
    quantity1,
    bins1,
    mask_group1_exclusive=False,
):
    """Get fragmentation and overmerging data.

    Parameters
    ----------
    cat1, cat2: clevar.ClCatalog
        ClCatalogs with multiple matching information.
    quantity1: str
        Column of catalog 1 to bin the data
    bins1: array
        Bins for quantity1
    mask_group1_exclusive: bool
        If true, only groups no masked members are kept in bins,
        else groups with any unmasked members are kept in bins.

    Returns
    -------
    dict
        Binned data used in the plot. It has the sections:

            * `numbers1` (2d array): Numbers of objects in bins with configuration:
            1->1, n->n, n->(<n), n->(>n).
            * `numbers2` (2d array): Numbers of objects in bins with configuration:
            1->1, n->n, n->(>n), n->(<n).
            * `bins_mid` (array): mid values of bins
    """
    numbers1, numbers2 = get_fragmentation_overmerging_numbers_binned(
        cat1,
        cat2,
        quantity1,
        bins1,
        mask_group1_exclusive,
    )
    bins_mid = 0.5 * (bins1[:-1] + bins1[1:])
    return {
        "numbers1": numbers1,
        "numbers2": numbers2,
        "bins_mid": bins_mid,
    }


def plot_fragmentation_overmerging_fraction(
    cat1,
    cat2,
    quantity1,
    bins1,
    mask_group1_exclusive=False,
    ax=None,
    plt_kwargs=None,
    lines_kwargs_list=None,
    legend_kwargs=None,
):
    """Plot fraction of clusters with fragmentation and overmerging.

    Parameters
    ----------
    cat1, cat2: clevar.ClCatalog
        ClCatalogs with multiple matching information.
    mask_group1, mask_group2: array, None
        Masks for groups in catalog 1(2).
    mask_group1_exclusive, mask_group2_exclusive: bool
        If true, groups with any masked members are excluded,
        else only groups with all masked members are excluded.
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
    data = _get_fragmentation_overmerging_data(
        cat1=cat1,
        cat2=cat2,
        quantity1=quantity1,
        bins1=bins1,
        mask_group1_exclusive=mask_group1_exclusive,
    )
    # pylint: disable=protected-access
    info = array_funcs.plot_fragmentation_overmerging_fraction(
        data=data,
        ax=ax,
        plt_kwargs=plt_kwargs,
        lines_kwargs_list=lines_kwargs_list,
    )
    info["ax"].legend(**updated_dict(legend_kwargs))
    info["ax"].set_ylabel(f"{cat1.name} sample fraction")
    info["ax"].set_xlabel(f"${cat1.labels[quantity1]}$")
    return info


def plot_fragmentation_overmerging_ratio(
    cat1,
    cat2,
    quantity1,
    bins1,
    mask_group1_exclusive=False,
    ax=None,
    plt_kwargs=None,
    lines_kwargs_list=None,
    legend_kwargs=None,
):
    """Plot ratio of cluster2/cluster1 counts in samples: total, with fraction and overmerging.

    Parameters
    ----------
    cat1, cat2: clevar.ClCatalog
        ClCatalogs with multiple matching information.
    mask_group1, mask_group2: array, None
        Masks for groups in catalog 1(2).
    mask_group1_exclusive, mask_group2_exclusive: bool
        If true, groups with any masked members are excluded,
        else only groups with all masked members are excluded.
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
    data = _get_fragmentation_overmerging_data(
        cat1=cat1,
        cat2=cat2,
        quantity1=quantity1,
        bins1=bins1,
        mask_group1_exclusive=mask_group1_exclusive,
    )
    # pylint: disable=protected-access
    info = array_funcs.plot_fragmentation_overmerging_ratio(
        data=data,
        ax=ax,
        plt_kwargs=plt_kwargs,
        lines_kwargs_list=lines_kwargs_list,
    )
    info["ax"].legend(**updated_dict(legend_kwargs))
    info["ax"].set_ylabel(f"$n_{{{cat2.name}}}/n_{{{cat1.name}}}$")
    info["ax"].set_xlabel(f"${cat1.labels[quantity1]}$")
    return info


def plot_fragmentation_overmerging(
    cat1,
    cat2,
    quantity1,
    bins1,
    mask_group1_exclusive=False,
    fig=None,
    plt_kwargs=None,
    lines_kwargs_list=None,
    legend_kwargs=None,
    fig_kwargs=None,
):
    """Plot fraction of clusters with fragmentation and overmerging,
    and ratio of cluster2/cluster1 counts in samples: total, with fraction and overmerging.

    Parameters
    ----------
    cat1, cat2: clevar.ClCatalog
        ClCatalogs with multiple matching information.
    mask_group1, mask_group2: array, None
        Masks for groups in catalog 1(2).
    mask_group1_exclusive, mask_group2_exclusive: bool
        If true, groups with any masked members are excluded,
        else only groups with all masked members are excluded.
    fig: matplotlib.figure.Figure, None
        Matplotlib figure object. If not provided a new one is created.
    plt_kwargs: dict, None
        Additional arguments for pylab.plot
    lines_kwargs_list: list, None
        List of additional arguments for plotting each line (using pylab.plot).
        Must have same size as len(bins2)-1
    legend_kwargs: dict, None
        Additional arguments for pylab.legend
    fig_kwargs: dict, None
        Additional arguments for plt.subplots

    Returns
    -------
    info: dict
        Information of data in the plots, it contains the sections:

            * `fig` (matplotlib.pyplot.figure): Figure of the plot.
            * `data`: Binned data used in the plot. It has the sections:

                * `numbers1` (2d array): Numbers of objects in bins with configuration:
                1->1, n->n, n->(<n), n->(>n).
                * `numbers2` (2d array): Numbers of objects in bins with configuration:
                1->1, n->n, n->(>n), n->(<n).
                * `bins_mid` (array): mid values of bins

    """
    if fig is None:
        fig = plt.subplots(2, **updated_dict({"sharex": True}, fig_kwargs))[0]
    data = _get_fragmentation_overmerging_data(
        cat1=cat1,
        cat2=cat2,
        quantity1=quantity1,
        bins1=bins1,
        mask_group1_exclusive=mask_group1_exclusive,
    )
    info = {"data": data, "fig": fig}

    # pylint: disable=protected-access
    array_funcs.plot_fragmentation_overmerging_fraction(
        data=data,
        ax=fig.axes[0],
        plt_kwargs=plt_kwargs,
        lines_kwargs_list=lines_kwargs_list,
    )

    array_funcs.plot_fragmentation_overmerging_ratio(
        data=data,
        ax=fig.axes[1],
        plt_kwargs=plt_kwargs,
        lines_kwargs_list=lines_kwargs_list,
    )
    fig.axes[0].legend(**updated_dict(legend_kwargs))
    fig.axes[0].set_ylabel(f"{cat1.name} sample fraction")

    fig.axes[1].legend(**updated_dict(legend_kwargs))
    fig.axes[1].set_ylabel(f"$n_{{{cat2.name}}}/n_{{{cat1.name}}}$")
    fig.axes[1].set_xlabel(f"${cat1.labels[quantity1]}$")

    return info


def plot_groups_hist(
    cat1,
    cat2,
    fig_kwargs=None,
    pcolor_kwargs=None,
    text_kwargs=None,
    add_cb=True,
):
    """Plots numbers of clusters with different number of group pairings.

    Parameters
    ----------
    cat1, cat2: clevar.ClCatalog
        ClCatalogs with multiple matching information.
    fig_kwargs: dict, None
        Additional arguments for plt.subplots
    pcolor_kwargs: dict, None
        Additional arguments for pylab.pcolor
    text_kwargs: dict, None
        Additional arguments for pylab.text
    add_cb: bool
        Plot colorbar

    Returns
    -------
    info: dict
        Information of data in the plots, it contains the sections:

            * `fig` (matplotlib.pyplot.figure): Figure of the plot. The main can be accessed at\
            `info['fig'].axes[0]`, and the colorbar at `info['fig'].axes[1]`.
            * `data`: Binned data used in the plot. It has the sections:

                * `values1` (array): Number of group members for catalog 1.
                * `values2` (array): Number of group members for catalog 2.
                * `table` (2d array): Numbers of groups in by (`values1`, `values2`).


    """
    info = array_funcs.plot_groups_hist(
        groups1=cat1["group"][cat1["group"] > 0],
        groups2=cat2["group"][cat2["group"] > 0],
        fig_kwargs=fig_kwargs,
        pcolor_kwargs=pcolor_kwargs,
        text_kwargs=text_kwargs,
        add_cb=add_cb,
    )
    ax = info["fig"].axes[0]
    ax.set_xlabel(f"# of members in groups ({cat1.name})")
    ax.set_ylabel(f"# of members in groups ({cat2.name})")
    return info
