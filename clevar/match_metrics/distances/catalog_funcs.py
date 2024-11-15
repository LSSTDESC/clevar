"""@file clevar/match_metrics/distances/catalog_funcs.py
Main distances functions using catalogs.
"""

import numpy as np

from ...geometry import convert_units
from ...match import get_matched_pairs
from .. import plot_helper as ph
from ..plot_helper import plt


def _histograms(
    distances,
    distance_bins,
    quantity2=None,
    bins2=None,
    shape="steps",
    ax=None,
    plt_kwargs=None,
    lines_kwargs_list=None,
    add_legend=True,
    legend_format=lambda v: v,
    legend_kwargs=None,
):
    """
    Plot histograms for distances.

    Parameters
    ----------
    distances: array
        Distances to be bined
    distance_bins: array, int
        Bins for distance
    quantity2: str
        Name of quantity 2 to bin
    bins2: array, int
        Bins for quantity 2
    log2: bool
        Log scale for quantity 2
    shape: str
        Shape of the lines. Can be steps or line.
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
    legend_kwargs: dict, None
        Additional arguments for pylab.legend

    Returns
    -------
    info: dict
        Information of data in the plots, it contains the sections:

            * `ax`: ax used in the plot.
            * `data`: Binned data used in the plot. It has the sections:

                * `hist`: Binned distances with (distance_bins, bin2).\
                bins where no cluster was found have nan value.
                * `distance_bins`: The bin edges for distances.
                * `bins2` (optional): The bin edges along the second dimension.
    """
    info = {}
    if quantity2 is not None:
        info["data"] = {
            key: value
            for value, key in zip(
                np.histogram2d(distances, quantity2, bins=(distance_bins, bins2)),
                ("hist", "distance_bins", "bins2"),
            )
        }
        hist = info["data"]["hist"].T
        edges1 = info["data"]["distance_bins"]
        edges2 = info["data"]["bins2"]
    else:
        info["data"] = {
            key: value
            for value, key in zip(
                np.histogram(distances, bins=distance_bins), ("hist", "distance_bins")
            )
        }
        hist = [info["data"]["hist"]]
        edges1 = info["data"]["distance_bins"]
        edges2 = [0, 1]
        add_legend = False
    info["ax"] = plt.axes() if ax is None else ax
    ph.plot_histograms(
        hist,
        edges1,
        edges2,
        shape=shape,
        ax=info["ax"],
        plt_kwargs=plt_kwargs,
        lines_kwargs_list=lines_kwargs_list,
        add_legend=add_legend,
        legend_format=legend_format,
        legend_kwargs=legend_kwargs,
    )
    return info


def central_position(
    cat1,
    cat2,
    matching_type,
    radial_bins=20,
    radial_bin_units="degrees",
    cosmo=None,
    col2=None,
    bins2=None,
    **kwargs,
):
    """
    Plot distance between central position of matched clusters, binned by a second quantity.

    Parameters
    ----------
    cat1, cat2: clevar.ClCatalog
        ClCatalogs with matching information.
    matching_type: str
        Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
    radial_bins: array, int
        Bins for radial distances
    radial_bin_units: str
        Units of radial bins
    cosmo: clevar.Cosmology
        Cosmology (used if physical units required)
    col2: str
        Name of quantity 2 (of cat1) to bin
    bins2: array, int
        Bins for quantity 2
    log2: bool
        Log scale for quantity 2
    mask1, mask2: array, None
        Masks for clusters 1(2), must have size=cat1(2).size


    Other parameters
    ----------------
    shape: str
        Shape of the lines. Can be steps or line.
    ax: matplotlib.axes
        Ax to add plot
    plt_kwargs: dict, None
        Additional arguments for pylab.plot.
        It also includes the possibility of smoothening the line with `n_increase, scheme`
        arguments. See `clevar.utils.smooth_line` for details.
    lines_kwargs_list: list, None
        List of additional arguments for plotting each line (using pylab.plot).
        Must have same size as len(bins2)-1

    Returns
    -------
    info: dict
        Information of data in the plots, it contains the sections:

            * `ax`: ax used in the plot.
            * `distances`: values of distances.
            * `data`: Binned data used in the plot. It has the sections:

                * `hist`: Binned distances with (distance_bins, bin2).\
                bins where no cluster was found have nan value.
                * `distance_bins`: The bin edges for distances.
                * `bins2` (optional): The bin edges along the second dimension.
    """
    mt1, mt2 = get_matched_pairs(
        cat1.raw(),
        cat2.raw(),
        matching_type,
        mask1=kwargs.pop("mask1", None),
        mask2=kwargs.pop("mask2", None),
    )
    info = {
        "distances": convert_units(
            mt1["SkyCoord"].separation(mt2["SkyCoord"]).deg,
            "degrees",
            radial_bin_units,
            redshift=mt1["z"],
            cosmo=cosmo,
        )
    }
    info.update(
        _histograms(
            distances=info["distances"],
            distance_bins=radial_bins,
            quantity2=mt1[col2] if (col2 in mt1.data.colnames or col2 in mt1.tags) else None,
            bins2=bins2,
            **kwargs,
        )
    )
    dist_labels = {
        "degrees": "deg",
        "arcmin": "arcmin",
        "arcsec": "arcsec",
        "pc": "pc",
        "kpc": "kpc",
        "mpc": "Mpc",
    }
    info["ax"].set_xlabel(rf"$\Delta\theta$ [{dist_labels[radial_bin_units.lower()]}]")
    info["ax"].set_ylabel("Number of matches")
    return info


def redshift(
    cat1, cat2, matching_type, redshift_bins=20, col2=None, bins2=None, normalize=None, **kwargs
):
    """
    Plot redshift distance between matched clusters, binned by a second quantity.

    Parameters
    ----------
    cat1, cat2: clevar.ClCatalog
        ClCatalogs with matching information.
    matching_type: str
        Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
    redshift_bins: array, int
        Bins for redshift distances
    col2: str
        Name of quantity 2 to bin
    bins2: array, int
        Bins for quantity 2
    normalize: str, None
        Normalize difference by (1+z). Can be 'cat1' for (1+z1), 'cat2' for (1+z2)
        or 'mean' for (1+(z1+z2)/2).
    mask1, mask2: array, None
        Masks for clusters 1(2), must have size=cat1(2).size

    Other parameters
    ----------------
    shape: str
        Shape of the lines. Can be steps or line.
    ax: matplotlib.axes
        Ax to add plot
    plt_kwargs: dict, None
        Additional arguments for pylab.plot.
        It also includes the possibility of smoothening the line with `n_increase, scheme`
        arguments. See `clevar.utils.smooth_line` for details.
    lines_kwargs_list: list, None
        List of additional arguments for plotting each line (using pylab.plot).
        Must have same size as len(bins2)-1

    Returns
    -------
    info: dict
        Information of data in the plots, it contains the sections:

            * `ax`: ax used in the plot.
            * `distances`: values of distances.
            * `data`: Binned data used in the plot. It has the sections:

                * `hist`: Binned distances with (distance_bins, bin2).\
                bins where no cluster was found have nan value.
                * `distance_bins`: The bin edges for distances.
                * `bins2` (optional): The bin edges along the second dimension.
    """
    mt1, mt2 = get_matched_pairs(
        cat1.raw(),
        cat2.raw(),
        matching_type,
        mask1=kwargs.pop("mask1", None),
        mask2=kwargs.pop("mask2", None),
    )
    norm = {
        None: 1,
        "cat1": (1 + mt1["z"]),
        "cat2": (1 + mt2["z"]),
        "mean": 0.5 * (2 + mt1["z"] + mt2["z"]),
    }[normalize]
    info = {"distances": (mt1["z"] - mt2["z"]) / norm}
    info.update(
        _histograms(
            info["distances"],
            distance_bins=redshift_bins,
            quantity2=mt1[col2] if (col2 in mt1.data.colnames or col2 in mt1.tags) else None,
            bins2=bins2,
            **kwargs,
        )
    )
    zl1, zl2 = cat1.labels["z"], cat2.labels["z"]
    dz_label = f"{zl1}-{zl2}"
    dist_labels = {
        None: f"${dz_label}$",
        "cat1": f"$({dz_label})/(1+{zl1})$",
        "cat2": f"$({dz_label})/(1+{zl2})$",
        "mean": f"$({dz_label})/(1+z_m)$",
    }
    info["ax"].set_xlabel(dist_labels[normalize])
    info["ax"].set_ylabel("Number of matches")
    return info
