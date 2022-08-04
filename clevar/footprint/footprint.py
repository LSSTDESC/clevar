#!/usr/bin/env python
import warnings
import numpy as np

from ..catalog import ClData, TagData, ClCatalog
from ..geometry import convert_units, angular_bank, physical_bank
from ..utils import none_val, hp, updated_dict
from astropy.coordinates import SkyCoord
from astropy import units as u
from .nfw_funcs import nfw2D_profile_flatcore
from ..match_metrics import plot_helper as ph
from ..match_metrics.plot_helper import plt

class Footprint(TagData):
    '''
    Functions for footprint management

    Attributes
    ----------
    nside: int
        Heapix NSIDE
    nest: bool
        If ordering is nested
    data: clevar.ClData
        Data with columns: pixel, detfrac, zmax
    pixel_dict: dict
        Dictionary to point to pixel in data
    size: int
        Number of objects in the catalog
    tags: LoweCaseDict
        Tag for main quantities used in matching and plots (ex: pixel, detfrac, zmax)
    '''

    @property
    def pixel_dict(self):
        return self.__pixel_dict

    @property
    def nside(self):
        return self.data.meta['nside']
    
    @property
    def nest(self):
        return self.data.meta['nest']
    
    @nside.setter
    def nside(self, nside):
        self.data.meta['nside'] = nside
    
    @nest.setter
    def nest(self, nest):
        self.data.meta['nest'] = nest

    def __init__(self, nside=None, tags=None, nest=False, **kwargs):
        '''
        Parameters
        ----------
        nside: int
            Heapix NSIDE
        nest: bool
            If ordering is nested
        pixel: array
            Pixels inside the footprint
        detfrac_vals: array, None
            Detection fraction, if None is set to 1
        zmax_vals: array, None
            Zmax, if None is set to 99
        '''
        self.__pixel_dict = {}
        tags = updated_dict({'pixel':'pixel'}, tags)
        if len(kwargs)>0:
            kwargs['nside'] = nside
            kwargs['nest'] = nest
        TagData.__init__(self, tags=tags,
                         default_tags=['pixel', 'detfrac', 'zmax'],
                         **kwargs)

    def _add_values(self, nside=None, nest=False, **columns):
        '''
        Adds provided values for attribues and assign default values to rest

        Parameters
        ----------
        nside: int
            Heapix NSIDE
        nest: bool
            If ordering is nested
        pixel: array
            Pixels inside the footprint
        detfrac: array
            Detection fraction. If None, value 1 is assigned.
        zmax: array
            Zmax. If None, value 99 is assigned.
        '''
        if not hp.isnsideok(nside):
            raise ValueError(f'nside (={nside}) must be a power of 2.')
        self.nside = nside
        self.nest = nest
        TagData._add_values(self, **columns)
        self['pixel'] = self['pixel'].astype(int)
        if self.tags.get('detfrac', None) not in self.colnames:
            self['detfrac'] = 1.
        if self.tags.get('zmax', None) not in self.colnames:
            self['zmax'] = 99.
        ra, dec = hp.pix2ang(nside, self['pixel'], lonlat=True, nest=nest)
        self['SkyCoord'] = SkyCoord(ra*u.deg, dec*u.deg, frame='icrs')
        self.pixel_dict.update(self._make_col_dict('pixel'))
    def get_map(self, data, bad_val=0):
        '''
        Transforms a internal quantity into a map

        Parameters
        ----------
        data: str
            Name of internal data to be used.
        bad_val: float, None
            Values for pixels outside footprint.

        Returns
        -------
        Map
            Healpix map with the given values, other pixels are zero
        '''
        return hp.pix2map(self.nside, self['pixel'], self[data], bad_val)
    def get_values_in_pixels(self, data, pixels, bad_val, transform=lambda x:x):
        '''
        Transforms a internal quantity into a map

        Parameters
        ----------
        values: array
            Values in each pixel.
        data: str
            Name of internal data to be used.
        pixels: list
            List of pixels
        bad_val: float, None
            Values for pixels outside footprint.
        transform: function
            Function to be applied to data for value in pixel. Default is f(data)=data.

        Returns
        -------
        Map
            Healpix map with the given values, other pixels are zero
        '''
        return np.array([transform(self[data][self.pixel_dict[p]])
                        if p in self.pixel_dict else bad_val for p in pixels])
    def zmax_masks_from_footprint(self, ra, dec, z):
        '''
        Create a zmax mask for a catalog based on a footprint

        Parameters
        ----------
        ra: numpy array
            Ra array in degrees
        dec: numpy array
            Dec array in degrees
        z: numpy array
            Redshift array

        Returns
        -------
        ndarray
            Arrays of booleans of objects in footprint
        '''
        pixels = hp.ang2pix(self.nside, ra, dec, nest=self.nest, lonlat=True)
        print("## creating visibility mask ##")
        #zmax_vals = self.get_map(self['zmax'])[pixels] old method
        zmax_vals = self.get_values_in_pixels('zmax', pixels, 0)
        return z<=zmax_vals
    @classmethod
    def read(self, filename, nside, pixel_name, detfrac_name=None, zmax_name=None,
             nest=False):
        """
        Loads fits file and converst to FootprintZmax object

        Parameters
        ----------
        filename: str
            Name of input file
        nside: int
            Healpix nside
        pixel_name: str
            Name of pixel column
        detfrac_name: str, None
            Name of detection fraction colum. If None value 1 is assigned.
        zmax_name:
            Name of maximum redshit colum. If None value 99 is assigned.
        """
        self = Footprint()
        values = ClData.read(filename)
        self._add_values(nside=nside, nest=nest,
            pixel=values[pixel_name],
            detfrac=values[detfrac_name] if detfrac_name is not None else None,
            zmax=values[zmax_name] if zmax_name is not None else None)
        return self
    def __repr__(self):
        out = f"FootprintZmax object with {len(self.data):,} pixels\n"
        out += "zmax: [%g, %g]\n"%(self['zmax'].min(), self['zmax'].max())
        out += "detfrac: [%g, %g]"%(self['detfrac'].min(), self['detfrac'].max())
        return out
    def _get_coverfrac(self, cl_sk, cl_z, aperture_radius, aperture_radius_unit,
                       cosmo=None, wtfunc=lambda pixels, sk: np.ones(len(pixels))):
        '''
        Get cover fraction

        Parameters
        ----------
        cl_sk: astropy.coordinates.SkyCoord
            Cluster SkyCoord [degrees, frame='icrs']
        cl_z: float
            Cluster redshift
        aperture_radius: float
            Radius of aperture
        aperture_radius_unit: str
            Unit of aperture radius
        cosmo: clevar.Cosmology object
            Cosmology object for when aperture has physical units.
        wtfunc: function
            Window function, must take (pixels, SkyCoord) as input.
            Default is flat window.

        Returns
        -------
        float
            Cover fraction
        '''
        if aperture_radius_unit in physical_bank and cosmo is None:
            raise TypeError('A cosmology is necessary if aperture in physical units.')
        pix_list = hp.query_disc(
            nside=self.nside, inclusive=True, nest=self.nest,
            vec=hp.pixelfunc.ang2vec(cl_sk.ra.value, cl_sk.dec.value, lonlat=True),
            radius=convert_units(aperture_radius, aperture_radius_unit, 'radians',
                                 redshift=cl_z, cosmo=cosmo)
            )
        weights = np.array(wtfunc(pix_list, cl_sk))
        detfrac_vals = self.get_values_in_pixels('detfrac', pix_list, 0)
        zmax_vals = self.get_values_in_pixels('zmax', pix_list, 0)
        values = detfrac_vals*np.array(cl_z<=zmax_vals, dtype=float)
        return (weights*values).sum()/weights.sum()
    def _get_coverfrac_nfw2D(self, cl_sk, cl_z, cl_radius, cl_radius_unit,
                             aperture_radius, aperture_radius_unit, cosmo):
        '''
        Cover fraction with NFW 2D flatcore window. It is computed using:

        Parameters
        ----------
        cl_sk: astropy.coordinates.SkyCoord
            Cluster SkyCoord [degrees, frame='icrs']
        cl_z: float
            Cluster redshift
        cl_radius: float
            Cluster radius
        cl_radius_unit: str
            Unit of cluster radius
        aperture_radius: float
            Radius of aperture
        aperture_radius_unit: str
            Unit of aperture radius
        cosmo: clevar.Cosmology object
            Cosmology object.

        Returns
        -------
        float
        '''
        cl_radius_mpc = convert_units(cl_radius, cl_radius_unit, 'mpc', redshift=cl_z, cosmo=cosmo)
        return self._get_coverfrac(cl_sk, cl_z, aperture_radius, aperture_radius_unit, cosmo=cosmo,
            wtfunc=lambda pix_list, cl_sk: self._nfw_flatcore_window_func(
                pix_list, cl_sk, cl_z, cl_radius_mpc, cosmo))
    def get_coverfrac(self, cl_ra, cl_dec, cl_z, aperture_radius, aperture_radius_unit,
                      cosmo=None, wtfunc=lambda pixels, sk: np.ones(len(pixels))):
        r'''
        Get cover fraction with a given window.

        .. math::
            CF(R) = \frac{\sum_{r_i<R}w(r_i)df(r_i)}{\sum_{r_i<R}w(r_i)}

        where the index `i` represents pixels of the footprint,
        :math:`r_i` is the distance between the cluster center and the pixel center,
        :math:`R` is the aperture radius to be considered
        and :math:`w` is the window function.

        Parameters
        ----------
        cl_ra: float
            Cluster RA in deg
        cl_dec: float
            Cluster DEC in deg
        cl_z: float
            Cluster redshift
        aperture_radius: float
            Radius of aperture
        aperture_radius_unit: str
            Unit of aperture radius
        cosmo: clevar.Cosmology object
            Cosmology object for when aperture has physical units.
        wtfunc: function
            Window function, must take (pixels, SkyCoord) as input.
            Default is flat window.

        Returns
        -------
        float
            Cover fraction
        '''
        return self._get_coverfrac(SkyCoord(cl_ra*u.deg, cl_dec*u.deg, frame='icrs'),
                                   cl_z, aperture_radius, aperture_radius_unit,
                                   cosmo=cosmo, wtfunc=wtfunc)
    def get_coverfrac_nfw2D(self, cl_ra, cl_dec, cl_z, cl_radius, cl_radius_unit,
                            aperture_radius, aperture_radius_unit, cosmo):
        r'''
        Cover fraction with NFW 2D flatcore window.

        .. math::
            CF(R) = \frac{\sum_{r_i<R}w_{NFW}(r_i)df(r_i)}{\sum_{r_i<R}w_{NFW}(r_i)}

        where the index `i` represents pixels of the footprint,
        :math:`r_i` is the distance between the cluster center and the pixel center,
        :math:`R` is the aperture radius to be considered
        and :math:`w_{nfw}` is the NFW 2D flatcore window function.

        Parameters
        ----------
        cl_ra: float
            Cluster RA in deg
        cl_dec: float
            Cluster DEC in deg
        cl_z: float
            Cluster redshift
        cl_radius: float
            Cluster radius
        cl_radius_unit: str
            Unit of cluster radius
        aperture_radius: float
            Radius of aperture
        aperture_radius_unit: str
            Unit of aperture radius
        cosmo: clevar.Cosmology object
            Cosmology object.

        Returns
        -------
        float

        Notes
        -----
            The NFW 2D function was taken from section 3.1 of arXiv:1104.2089 and was
            validated with Rs = 0.15 Mpc/h (0.214 Mpc) and Rcore = 0.1 Mpc/h (0.142 Mpc).
        '''
        return self._get_coverfrac_nfw2D(SkyCoord(cl_ra*u.deg, cl_dec*u.deg, frame='icrs'),
                                         cl_z, cl_radius, cl_radius_unit,
                                         aperture_radius, aperture_radius_unit, cosmo=cosmo)
    def _nfw_flatcore_window_func(self, pix_list, cl_sk, cl_z, cl_radius, cosmo):
        '''
        Get aperture function for NFW 2D Profile with a top-hat core

        Parameters
        ----------
        pix_list: list
            List of pixels in the aperture
        cl_sk: astropy.coordinates.SkyCoord
            Cluster SkyCoord [degrees, frame='icrs']
        cl_z: float
            Cluster redshift
        cl_radius: float
            Cluster radius in Mpc

        Returns
        -------
        array
            Value of the aperture function at each pixel
        '''
        Rs = 0.15/cosmo['h'] # 0.214Mpc
        Rcore = 0.1/cosmo['h'] # 0.142Mpc
        R = self.get_values_in_pixels('SkyCoord', pix_list, Rcore,
            transform=lambda x: convert_units(cl_sk.separation(x).value,
                                              'degrees', 'mpc', cl_z, cosmo)
            )
        return nfw2D_profile_flatcore(R, cl_radius, Rs, Rcore)
    def plot(self, data, bad_val=hp.UNSEEN, auto_lim=False, ra_lim=None, dec_lim=None,
             cluster=None, cluster_kwargs=None, cosmo=None, fig=None, figsize=None, **kwargs):
        """
        Plot footprint. It can also overlay clusters with their radial sizes.

        Parameters
        ----------
        data: str
            Name of internal data to be used.
        bad_val: float
            Values for pixels outside footprint.
        auto_lim: bool
            Set automatic limits for ra/dec.
        ra_lim: list
            RA limits in degrees.
        dec_lim: list
            DEC limits in degrees.
        cluster: clevar.ClCatalog
            Clusters to be overlayed on footprint.
        cluster_kwargs: dict, None
            Keyword arguments to plot clusters. If cluster radius used, arguments
            for plt.Circle function, if not arguments for plt.scatter.
        cosmo: clevar.Cosmology object
            Cosmology object for when cluster radius has physical units.
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
        fig: matplotlib.pyplot.figure
            Figure of the plot. The main can be accessed at fig.axes[0], and the colorbar
            at fig.axes[1].
        """
        fig, ax, cb = ph.plot_healpix_map(
            self.get_map(data, bad_val), nest=self.nest, auto_lim=auto_lim, bad_val=bad_val,
            ra_lim=ra_lim, dec_lim=dec_lim, fig=fig, figsize=figsize, **kwargs)
        if cb:
            cb.set_xlabel(data)

        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        if cluster is not None:
            if isinstance(cluster, ClCatalog):
                if 'radius' in cluster.tags and cluster.radius_unit is not None:
                    if (cluster.radius_unit in physical_bank and
                            (cosmo is None or 'z' not in cluster.tags)):
                        raise TypeError(
                            'A cosmology and cluster redsift is necessary if '
                            'cluster radius in physical units.')
                    rad_deg = convert_units(cluster['radius'], cluster.radius_unit, 'degrees',
                                            redshift=cluster['z'], cosmo=cosmo)
                    plt_cl = lambda ra, dec, radius: [
                        ax.add_patch(plt.Circle(
                            (ra_, dec_), radius=radius_,
                            **updated_dict({'color':'b', 'fill':0, 'lw':1}, cluster_kwargs)))
                        for ra_, dec_, radius_ in np.transpose([ra, dec, radius])[
                            (ra+radius>=xlim[0])*(ra-radius<xlim[1])
                            *(dec+radius>=ylim[0])*(dec-radius<ylim[1])]
                        ]
                else:
                    warnings.warn("Column 'radius' or radius_unit of cluster not set up. "
                                  "Plotting clusters as points with plt.scatter.")
                    rad_deg = np.ones(cluster.size)
                    lims_mask = lambda ra, dec: ((ra>=xlim[0])*(ra<xlim[1])*(dec>=ylim[0])*(dec<ylim[1]))
                    plt_cl = lambda ra, dec, radius: \
                        ax.scatter(*np.transpose([ra, dec])[lims_mask(ra, dec)].T,
                                   **updated_dict({'color':'b', 's':5}, cluster_kwargs))
                # Plot clusters in regular range
                plt_cl(cluster['ra'], cluster['dec'], rad_deg)
                # Plot clusters using -180<ra<0
                if ax.get_xlim()[0]<=0.:
                    ra2, dec2, r2 = np.transpose([cluster['ra'], cluster['dec'], rad_deg])[
                        (cluster['ra']>=180)].T
                    ra2 -= 360.
                    plt_cl(ra2, dec2, r2)
                # Plot clusters using 180<ra<360
                if ax.get_xlim()[1]>=180.:
                    ra2, dec2, r2 = np.transpose([cluster['ra'], cluster['dec'], rad_deg])[
                        (cluster['ra']<=0)].T
                    ra2 += 360.
                    plt_cl(ra2, dec2, r2)
                # Plot clusters using ra>360 (for xlim [360<, >0])
                if ax.get_xlim()[1]>=360.:
                    ra2, dec2, r2 = np.transpose([cluster['ra'], cluster['dec'], rad_deg])[
                        (cluster['ra']>=0)].T
                    ra2 += 360.
                    plt_cl(ra2, dec2, r2)
            else:
                raise TypeError(f'cluster argument (={cluster}) must be a ClCatalog.')

        return fig
