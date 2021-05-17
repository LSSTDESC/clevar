# Set mpl backend run plots on github actions
import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == 'test':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import pylab as plt
import numpy as np

from ..utils import none_val, autobins, binmasks
from ..geometry import convert_units
from ..match import MatchedPairs
from . import plot_helper as ph

class ClCatalogFuncs():
    def _histograms(distances, distance_bins, quantity2=None, bins2=None, log2=False,
                    shape='steps', ax=None, plt_kwargs={}, lines_kwargs_list=None,
                    add_legend=True, legend_format=lambda v: v, legend_kwargs={}):
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
        plt_kwargs: dict
            Additional arguments for pylab.plot
        lines_kwargs_list: list, None
            List of additional arguments for plotting each line (using pylab.plot).
            Must have same size as len(bins2)-1
        add_legend: bool
            Add legend of bins
        legend_format: function
            Function to format the values of the bins in legend
        legend_kwargs: dict
            Additional arguments for pylab.legend

        Returns
        -------
        ax: matplotlib.axes
            Axis of the plot
        """
        if quantity2 is not None:
            bins2 = autobins(quantity2, bins2, log2)
            bin_masks = binmasks(quantity2, bins2)
        else:
            bins2 = [0, 1]
            bin_masks = [np.ones(len(distances), dtype=bool)]
            add_legend = False
        lines_kwargs_list = none_val(lines_kwargs_list, [{} for m in bin_masks])
        ax = plt.axes() if ax is None else ax
        ph.add_grid(ax)
        distance_bins_ = np.histogram(distances, bins=distance_bins)[1]
        for m, l_kwargs, e0, e1 in zip(bin_masks, lines_kwargs_list, bins2, bins2[1:]):
            kwargs = {}
            kwargs['label'] = ph.get_bin_label(e0, e1, legend_format) if add_legend else None
            kwargs.update(plt_kwargs)
            kwargs.update(l_kwargs)
            ph.plot_hist_line(*np.histogram(distances[m], bins=distance_bins_),
                              ax=ax, shape=shape, **kwargs)
        if add_legend:
            ax.legend(**legend_kwargs)
        return ax
    def central_position(cat1, cat2, matching_type, radial_bins=20, radial_bin_units='degrees', cosmo=None,
                         col2=None, bins2=None, **kwargs):
        """
        Plot recovery rate as lines, with each line binned by redshift inside a mass bin.

        Parameters
        ----------
        cat1: clevar.ClCatalog
            ClCatalog with matching information
        cat2: clevar.ClCatalog
            ClCatalog matched to
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
        mask1: array, None
            Mask for clusters 1 properties, must have size=cat1.size
        mask2: array, None
            Mask for clusters 2 properties, must have size=cat2.size


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
        mp = MatchedPairs(cat1, cat2, matching_type,
                          mask1=kwargs.pop('mask1', None),
                          mask2=kwargs.pop('mask2', None))
        sk1, sk2 = mp.data1['SkyCoord'], mp.data2['SkyCoord']
        distances = convert_units(sk1.separation(sk2).deg, 'degrees',
                                  radial_bin_units, redshift=mp.data1['z'],
                                  cosmo=cosmo)
        ax = ClCatalogFuncs._histograms(distances=distances,
                                      distance_bins=radial_bins,
                                      quantity2=mp.data1[col2] if col2 in mp.data1.colnames else None,
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
        cat1: clevar.ClCatalog
            ClCatalog with matching information
        cat2: clevar.ClCatalog
            ClCatalog matched to
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
        mask1: array, None
            Mask for clusters 1 properties, must have size=cat1.size
        mask2: array, None
            Mask for clusters 2 properties, must have size=cat2.size

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
        mp = MatchedPairs(cat1, cat2, matching_type,
                          mask1=kwargs.pop('mask1', None),
                          mask2=kwargs.pop('mask2', None))
        z1, z2 = mp.data1['z'], mp.data2['z']
        norm = {
                None:1,
                'cat1':(1+z1),
                'cat2':(1+z2),
                'mean':.5*(2+z1+z2),
                }[normalize]
        ax = ClCatalogFuncs._histograms(distances=(z1-z2)/norm,
                                      distance_bins=redshift_bins,
                                      quantity2=mp.data1[col2] if col2 in mp.data1.colnames else None,
                                      bins2=bins2, **kwargs)
        zl1, zl2 = cat1.labels['z'], cat2.labels['z']
        dz = f'{zl1}-{zl2}'
        dist_labels = {None:f'${dz}$', 'cat1': f'$({dz})/(1+{zl1})$',
                       'cat2': f'$({dz})/(1+{zl2})$', 'mean': f'$({dz})/(1+z_m)$',}
        ax.set_xlabel(dist_labels[normalize])
        ax.set_ylabel('Number of matches')
        return ax

def central_position(cat1, cat2, matching_type, radial_bins=20, radial_bin_units='degrees', cosmo=None,
                     quantity_bins=None, bins=None, log_quantity=False, ax=None, **kwargs):
    """
    Plot recovery rate as lines, with each line binned by redshift inside a mass bin.

    Parameters
    ----------
    cat1: clevar.ClCatalog
        ClCatalog with matching information
    cat2: clevar.ClCatalog
        ClCatalog matched to
    matching_type: str
        Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
    radial_bins: array, int
        Bins for radial distances
    radial_bin_units: str
        Units of radial bins
    cosmo: clevar.Cosmology
        Cosmology (used if physical units required)
    quantity_bins: str
        Column to bin the data
    bins: array, int
        Bins for quantity
    log_quantity: bool
        Display label in log fmt
    mask1: array, None
        Mask for clusters 1 properties, must have size=cat1.size
    mask2: array, None
        Mask for clusters 2 properties, must have size=cat2.size

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
    ax: matplotlib.axes
        Axis of the plot
    """
    legend_fmt = kwargs.pop("legend_fmt", ".1f" if log_quantity else ".2f")
    kwargs['legend_format'] = kwargs.get('legend_format',
        lambda v: f'10^{{%{legend_fmt}}}'%np.log10(v) if log_quantity else f'%{legend_fmt}'%v)
    kwargs['add_legend'] = kwargs.get('add_legend', True)*(bins is not None)
    return ClCatalogFuncs.central_position(cat1, cat2, matching_type, radial_bins=radial_bins,
            radial_bin_units=radial_bin_units, cosmo=cosmo, col2=quantity_bins, bins2=bins,
            ax=ax, **kwargs)

def redshift(cat1, cat2, matching_type, redshift_bins=20, normalize=None,
             quantity_bins=None, bins=None, log_quantity=False,
             ax=None, **kwargs):
    """
    Plot recovery rate as lines, with each line binned by redshift inside a mass bin.

    Parameters
    ----------
    cat1: clevar.ClCatalog
        ClCatalog with matching information
    cat2: clevar.ClCatalog
        ClCatalog matched to
    matching_type: str
        Type of matching to be considered. Must be in: 'cross', 'cat1', 'cat2'
    redshift_bins: array, int
        Bins for redshift distances
    normalize: str, None
        Normalize difference by (1+z). Can be 'cat1' for (1+z1), 'cat2' for (1+z2)
        or 'mean' for (1+(z1+z2)/2).
    quantity_bins: str
        Column to bin the data
    bins: array, int
        Bins for quantity
    log_quantity: bool
        Display label in log fmt
    mask1: array, None
        Mask for clusters 1 properties, must have size=cat1.size
    mask2: array, None
        Mask for clusters 2 properties, must have size=cat2.size

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
    ax: matplotlib.axes
        Axis of the plot
    """
    legend_fmt = kwargs.pop("legend_fmt", ".1f" if log_quantity else ".2f")
    kwargs['legend_format'] = kwargs.get('legend_format',
        lambda v: f'10^{{%{legend_fmt}}}'%np.log10(v) if log_quantity else f'%{legend_fmt}'%v)
    kwargs['add_legend'] = kwargs.get('add_legend', True)*(bins is not None)
    return ClCatalogFuncs.redshift(cat1, cat2, matching_type, redshift_bins=redshift_bins,
            normalize=normalize, col2=quantity_bins, bins2=bins, ax=ax, **kwargs)
