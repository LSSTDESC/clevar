import numpy as np
import pylab as plt

from ..utils import none_val, bin_masks
from ..geometry import convert_units
from ..match import MatchedPairs
from . import plot_helper as ph

def get_central_distances(cat1, cat2, matching_type, units='degrees', cosmo=None):
    """
    Get distance of centers from cat1 to cat2.

    Parameters
    ----------
    cat1: clevar.Catalog
        Catalog with matching information
    cat2: clevar.Catalog
        Catalog matched to
    matching_type: str
        Type of matching to be considered. Must be in:
        'cross', 'self', 'other'
    units: str
        Units of output distance
    cosmo: clevar.Cosmology
        Cosmology (used if physical units required)

    Returns
    -------
    array
        Distances of cluster centers in input units
    """
    mp = MatchedPairs(cat1, cat2, matching_type)
    sk1 = mp.data1['SkyCoord']
    sk2 = mp.data2['SkyCoord']
    return convert_units(sk1.separation(sk2).deg, 'degrees', units,
        redshift=mp.data1['z'], cosmo=cosmo)
def get_redshift_distances(cat1, cat2, matching_type, normalize=None):
    """
    Get redshift distances between cat1 and cat2.

    Parameters
    ----------
    cat1: clevar.Catalog
        Catalog with matching information
    cat2: clevar.Catalog
        Catalog matched to
    matching_type: str
        Type of matching to be considered. Must be in:
        'cross', 'self', 'other'
    normalize: str, None
        Normalize difference by (1+z). Can be 'cat1' for (1+z1), 'cat2' for (1+z2)
        or 'mean' for (1+(z1+z2)/2).

    Returns
    -------
    array
        Redshift difference: z1-z2 (can be normalized)
    """
    mp = MatchedPairs(cat1, cat2, matching_type)
    z1 = mp.data1['z']
    z2 = mp.data2['z']
    return (z1-z2)/{None:1, 'cat1':(1+z1), 'cat2':(1+z2), 'mean':.5*(2+z1+z2)}[normalize]
class CatalogFuncs():
    def _histograms(distances, distance_bins, quantity2=None, bins2=None,
                    shape='steps', ax=None, plt_kwargs={}, lines_kwargs_list=None):
        """
        Plot histograms for distances.
    
        Parameters
        ----------
        distances: array
            Distances to be bined
        distance_bins: array
            Bins for distance
        quantity2: str
            Name of quantity 2 to bin
        bins2: array
            Bins for quantity 2
        shape: str
            Shape of the lines. Can be steps or line.
        ax: matplotlib.axes
            Ax to add plot
        plt_kwargs: dict
            Additional arguments for pylab.plot
        lines_kwargs_list: list, None
            List of additional arguments for plotting each line (using pylab.plot).
            Must have same size as len(bins2)-1

        Returns
        -------
        ax: matplotlib.axes
            Axis of the plot
        """
        bins2 = none_val(bins2, [quantity2.min(), quantity2.max()+1])
        bin_masks = [(quantity2>=b0)*(quantity2<b1) for b0, b1 in zip(bins2, bins2[1:])]
        lines_kwargs_list = none_val(lines_kwargs_list, [{} for m in bin_masks])
        ax = none_val(ax, plt.axes())
        ph.add_grid(ax)
        for m, l_kwargs in zip(bin_masks, lines_kwargs_list):
            print(m[m].size, distances[m].min(), distances[m].max())
            kwargs = {}
            kwargs.update(plt_kwargs)
            kwargs.update(l_kwargs)
            ph.plot_hist_line(*np.histogram(distances[m], bins=distance_bins),
                              ax=ax, shape=shape, **kwargs)
        return ax
    def central_position(cat1, cat2, matching_type, radial_bins=20, radial_bin_units='degrees', cosmo=None,
                         col2=None, bins2=None, **kwargs):
        """
        Plot recovery rate as lines, with each line binned by redshift inside a mass bin.
    
        Parameters
        ----------
        cat1: clevar.Catalog
            Catalog with matching information
        cat2: clevar.Catalog
            Catalog matched to
        matching_type: str
            Type of matching to be considered. Must be in:
            'cross', 'self', 'other', 'multi_self', 'multi_other', 'multi_join'
        radial_bins: array
            Bins for radial distances
        radial_bin_units: str
            Units of radial bins
        cosmo: clevar.Cosmology
            Cosmology (used if physical units required)
        col2: str
            Name of quantity 2 (of cat1) to bin
        bins2: array
            Bins for quantity 2

        Other parameters
        ----------------
        shape: str
            Shape of the lines. Can be steps or line.
        ax: matplotlib.axes
            Ax to add plot
        plt_kwargs: dict
            Additional arguments for pylab.plot
        lines_kwargs_list: list, None
            List of additional arguments for plotting each line (using pylab.plot).
            Must have same size as len(bins2)-1

        Returns
        -------
        ax: matplotlib.axes
            Axis of the plot
        """
        ax = CatalogFuncs._histograms(distances=get_central_distances(cat1, cat2, matching_type,
                                                                      units=radial_bin_units,
                                                                      cosmo=cosmo),
                                      distance_bins=radial_bins, 
                                      quantity2=cat1.data[col2][cat1.get_matching_mask(matching_type)],
                                      bins2=bins2, **kwargs)
        dist_labels = {'degrees':'deg', 'arcmin': 'arcmin', 'arcsec':'arcsec',
                        'pc':'pc', 'kpc':'kpc', 'mpc': 'Mpc'}
        ax.set_xlabel(f'$\Delta\\theta$ [{dist_labels[radial_bin_units.lower()]}]')
        ax.set_ylabel('Number of matches')
        return ax
    def redshift(cat1, cat2, matching_type, redshift_bins=20, col2=None, bins2=None,
                 normalize=None, **kwargs):
        """
        Plot recovery rate as lines, with each line binned by redshift inside a mass bin.
    
        Parameters
        ----------
        cat1: clevar.Catalog
            Catalog with matching information
        cat2: clevar.Catalog
            Catalog matched to
        matching_type: str
            Type of matching to be considered. Must be in:
            'cross', 'self', 'other', 'multi_self', 'multi_other', 'multi_join'
        redshift_bins: array
            Bins for redshift distances
        col2: str
            Name of quantity 2 to bin
        bins2: array
            Bins for quantity 2
        normalize: str, None
            Normalize difference by (1+z). Can be 'cat1' for (1+z1), 'cat2' for (1+z2)
            or 'mean' for (1+(z1+z2)/2).

        Other parameters
        ----------------
        shape: str
            Shape of the lines. Can be steps or line.
        ax: matplotlib.axes
            Ax to add plot
        plt_kwargs: dict
            Additional arguments for pylab.plot
        lines_kwargs_list: list, None
            List of additional arguments for plotting each line (using pylab.plot).
            Must have same size as len(bins2)-1

        Returns
        -------
        ax: matplotlib.axes
            Axis of the plot
        """
        ax = CatalogFuncs._histograms(distances=get_redshift_distances(cat1, cat2,
                                                                       matching_type, normalize),
                                      distance_bins=redshift_bins, 
                                      quantity2=cat1.data[col2][cat1.get_matching_mask(matching_type)],
                                      bins2=bins2, **kwargs)
        dz = 'z_1-z_2'
        dist_labels = {None:f'${dz}$', 'cat1': f'$({dz})/(1+z_1)$',
                       'cat2': f'$({dz})/(1+z_2)$', 'mean': f'$({dz})/(1+z_m)$',}
        ax.set_xlabel(dist_labels[normalize])
        ax.set_ylabel('Number of matches')
        return ax

def central_position(cat1, cat2, matching_type, radial_bins=20, radial_bin_units='degrees', cosmo=None,
                    mass_bins=None, mass_label=None, ax=None, **kwargs):
    """
    Plot recovery rate as lines, with each line binned by redshift inside a mass bin.

    Parameters
    ----------
    cat1: clevar.Catalog
        Catalog with matching information
    cat2: clevar.Catalog
        Catalog matched to
    matching_type: str
        Type of matching to be considered. Must be in:
        'cross', 'self', 'other', 'multi_self', 'multi_other', 'multi_join'
    radial_bins: array
        Bins for radial distances
    radial_bin_units: str
        Units of radial bins
    cosmo: clevar.Cosmology
        Cosmology (used if physical units required)
    mass_bins: array
        Bins for mass

    Other parameters
    ----------------
    shape: str
        Shape of the lines. Can be steps or line.
    ax: matplotlib.axes
        Ax to add plot
    plt_kwargs: dict
        Additional arguments for pylab.plot
    lines_kwargs_list: list, None
        List of additional arguments for plotting each line (using pylab.plot).
        Must have same size as len(bins2)-1

    Returns
    -------
    ax: matplotlib.axes
        Axis of the plot
    """
    return CatalogFuncs.central_position(cat1, cat2, matching_type, radial_bins=radial_bins,
            radial_bin_units=radial_bin_units, cosmo=cosmo, col2='mass', bins2=mass_bins,
            ax=ax, **kwargs)

def redshift(cat1, cat2, matching_type, redshift_bins=20, normalize=None,
             mass_bins=None, mass_label=None, ax=None, **kwargs):
    """
    Plot recovery rate as lines, with each line binned by redshift inside a mass bin.

    Parameters
    ----------
    cat1: clevar.Catalog
        Catalog with matching information
    cat2: clevar.Catalog
        Catalog matched to
    matching_type: str
        Type of matching to be considered. Must be in:
        'cross', 'self', 'other', 'multi_self', 'multi_other', 'multi_join'
    redshift_bins: array
        Bins for redshift distances
    normalize: str, None
        Normalize difference by (1+z). Can be 'cat1' for (1+z1), 'cat2' for (1+z2)
        or 'mean' for (1+(z1+z2)/2).
    mass_bins: array
        Bins for mass

    Other parameters
    ----------------
    shape: str
        Shape of the lines. Can be steps or line.
    ax: matplotlib.axes
        Ax to add plot
    plt_kwargs: dict
        Additional arguments for pylab.plot
    lines_kwargs_list: list, None
        List of additional arguments for plotting each line (using pylab.plot).
        Must have same size as len(bins2)-1

    Returns
    -------
    ax: matplotlib.axes
        Axis of the plot
    """
    return CatalogFuncs.redshift(cat1, cat2, matching_type, redshift_bins=redshift_bins,
            normalize=normalize, col2='mass', bins2=mass_bins, ax=ax, **kwargs)

