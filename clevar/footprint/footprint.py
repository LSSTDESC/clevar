#!/usr/bin/env python
import numpy as np

from ..catalog import ClData
from ..geometry import convert_units
from ..utils import none_val, hp
from astropy.coordinates import SkyCoord
from astropy import units as u
from .nfw_funcs import nfw2D_profile_flatcore

class Footprint():
    '''
    Functions for footprint management

    Attributes
    ----------
    nside: int
        Heapix NSIDE
    nest: bool
        If ordering is nested
    pixels: array
        Pixels inside the footprint
    detfrac: array
        Detection fraction
    zmax: array
        Zmax
    detfrac_map: None, array
        Map of Detection fraction, added by add_maps function
    zmax_map: None, array
        Map of Zmax, added by add_maps function
    '''
    def __init__(self, *args, **kargs):
        '''
        Parameters
        ----------
        nside: int
            Heapix NSIDE
        nest: bool
            If ordering is nested
        pixels: array
            Pixels inside the footprint
        detfrac_vals: array
            Detection fraction
        zmax_vals: array
            Zmax
        '''
        self.data = ClData()
        self.nside = None
        self.nest = None
        if len(args)>0 or len(kargs)>0:
            self._add_values(*args, **kargs)
    def _add_values(self, nside, pixels, detfrac=None, zmax=None,
            nest=False):
        '''
        Adds provided values for attribues and assign default values to rest

        Parameters
        ----------
        nside: int
            Heapix NSIDE
        nest: bool
            If ordering is nested
        pixels: array
            Pixels inside the footprint
        detfrac: array
            Detection fraction. If None, value 1 is assigned.
        zmax: array
            Zmax. If None, value 99 is assigned.
        '''
        self.nside = nside
        self.nest = nest
        self.data.meta.update({'nside':nside, 'nest':nest})
        self.data['pixel'] = np.array(pixels, dtype=int)
        self.data['detfrac'] = none_val(detfrac, 1)
        self.data['zmax'] = none_val(zmax, 99)
        ra, dec = hp.pix2ang(nside, pixels, lonlat=True)
        self.data['SkyCoord'] = SkyCoord(ra*u.deg, dec*u.deg, frame='icrs')
        self.pixel_dict = {p:i for i, p in enumerate(self['pixel'])}
    def __getitem__(self, item):
        return self.data[item]
    def get_map(self, data):
        '''
        Transforms a internal quantity into a map

        Parameters
        ----------
        values: array
            Values in each pixel

        Returns
        -------
        Map
            Healpix map with the given values, other pixels are zero
        '''
        return hp.pix2map(self.nside, self['pixel'], data, 0)
    def get_values_in_pixels(self, data, pixel_vals, bad_val, transform=lambda x:x):
        '''
        Transforms a internal quantity into a map

        Parameters
        ----------
        values: array
            Values in each pixel

        Returns
        -------
        Map
            Healpix map with the given values, other pixels are zero
        '''
        return np.array([transform(self[data][self.pixel_dict[p]]) if p in self.pixel_dict else bad_val
                    for p in pixel_vals])
    def zmax_masks_from_footprint(self, ra, dec, z):
        '''
        Create a mask for a catalog based on a footprint

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
        mask, mask
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
            pixels=values[pixel_name],
            detfrac=values[detfrac_name] if detfrac_name is not None else None,
            zmax=values[zmax_name] if zmax_name is not None else None)
        return self
    def __repr__(self,):
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
            Cosmology object for when radius has angular units
        wtfunc: function
            Window function

        Returns
        -------
        float
            Cover fraction
        '''
        pix_list = hp.query_disc(
            nside=self.nside, inclusive=True,
            vec=hp.pixelfunc.ang2vec(cl_sk.ra.value, cl_sk.dec.value, lonlat=True),
            radius=convert_units(aperture_radius, aperture_radius_unit, 'radians',
                                 redshift=cl_z, cosmo=cosmo)
            )
        weights = wtfunc(pix_list, cl_sk)
        detfrac_vals = self.get_values_in_pixels('detfrac', pix_list, 0)
        zmax_vals = self.get_values_in_pixels('zmax', pix_list, 0)
        values = detfrac_vals*np.array(cl_z<=zmax_vals, dtype=float)
        return sum(weights*values)/sum(weights)
    def _get_coverfrac_nfw2D(self, cl_sk, cl_z, cl_radius, cl_radius_unit,
                             aperture_radius, aperture_radius_unit, cosmo=None):
        '''
        Cover fraction with NFW 2D flatcore window

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
            Cosmology object for when radius has angular units

        Returns
        -------
        float
        '''
        cl_radius_mpc = convert_units(cl_radius, cl_radius_unit, 'mpc',
                                      redshift=cl_z, cosmo=cosmo)
        return self._get_coverfrac(cl_sk, cl_z, aperture_radius, aperture_radius_unit, cosmo=cosmo,
            wtfunc=lambda pix_list, cl_sk: self._nfw_flatcore_window_func(pix_list, cl_sk, cl_z,
                                                                       cl_radius_mpc, cosmo))
    def get_coverfrac(self, cl_ra, cl_dec, cl_z, aperture_radius, aperture_radius_unit,
                      cosmo=None, wtfunc=lambda pixels, sk: np.ones(len(pixels))):
        '''
        Get cover fraction

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
            Cosmology object for when radius has angular units
        wtfunc: function
            Window function

        Returns
        -------
        float
            Cover fraction
        '''
        return self._get_coverfrac(SkyCoord(cl_ra*u.deg, cl_dec*u.deg, frame='icrs'),
                                   cl_z, aperture_radius, aperture_radius_unit,
                                   cosmo=cosmo, wtfunc=wtfunc)
    def get_coverfrac_nfw2D(self, cl_ra, cl_dec, cl_z, cl_radius, cl_radius_unit,
                            aperture_radius, aperture_radius_unit, cosmo=None):
        '''
        Cover fraction with NFW 2D flatcore window

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
            Cosmology object for when radius has angular units

        Returns
        -------
        float
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
