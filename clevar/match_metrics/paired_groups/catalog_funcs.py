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
    """
    Plot recovery rate as lines, with each line binned by bins1 inside a bin of bins2.

    Parameters
    ----------

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
    data = _get_fragmentation_overmerging_data(
        cat1=cat1,
        cat2=cat2,
        quantity1=quantity1,
        bins1=bins1,
        mask_group1_exclusive=mask_group1_exclusive,
    )
    # pylint: disable=protected-access
    info = array_funcs._plot_fragmentation_overmerging_fraction(
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
    data = _get_fragmentation_overmerging_data(
        cat1=cat1,
        cat2=cat2,
        quantity1=quantity1,
        bins1=bins1,
        mask_group1_exclusive=mask_group1_exclusive,
    )
    # pylint: disable=protected-access
    info = array_funcs._plot_fragmentation_overmerging_ratio(
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
    if fig is None:
        fig = plt.subplots(2, sharex=True)[0]
    data = _get_fragmentation_overmerging_data(
        cat1=cat1,
        cat2=cat2,
        quantity1=quantity1,
        bins1=bins1,
        mask_group1_exclusive=mask_group1_exclusive,
    )
    info = {"data": data, "fig": fig}

    # pylint: disable=protected-access
    array_funcs._plot_fragmentation_overmerging_fraction(
        data=data,
        ax=fig.axes[0],
        plt_kwargs=plt_kwargs,
        lines_kwargs_list=lines_kwargs_list,
    )

    array_funcs._plot_fragmentation_overmerging_ratio(
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
