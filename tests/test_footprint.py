"""Tests for clevar/footprint"""
import os
import numpy as np
import healpy as hp
from clevar.catalog import ClCatalog
from clevar.cosmology import AstroPyCosmology
from clevar.footprint import Footprint
from clevar.footprint.artificial import create_footprint
from numpy.testing import assert_raises, assert_allclose, assert_equal

def get_test_data():
    ra_vals = np.linspace(0, 30, 10)
    dec_vals = np.linspace(0, 30, 10)
    input0 = {
        'ra': np.outer(ra_vals, dec_vals*0+1).flatten(),
        'dec': np.outer(ra_vals*0+1, dec_vals).flatten(),
        'z': np.ones(100),
        'radius': np.ones(100),
    }
    c1 = ClCatalog('Cat1', radius_unit='mpc', **{k:v[::2] for k, v in input0.items()})
    c2 = ClCatalog('Cat2', radius_unit='mpc', **{k:v[1::2] for k, v in input0.items()})
    return c1, c2
def test_footprint():
    c1, c2 = get_test_data()
    cosmo = AstroPyCosmology()
    # Gen footprint
    nside = 128
    pixels1 = hp.ang2pix(nside, c1['ra'], c1['dec'], lonlat=True)
    ftpt1 = Footprint(nside=nside, pixels=list(set(pixels1)))
    pixels2 = hp.ang2pix(nside, c2['ra'], c2['dec'], lonlat=True)
    ftpt2 = Footprint(nside=nside, pixels=list(set(pixels2)))
    # Footprint functions
    print(ftpt1)
    ftpt1.get_map(ftpt1['zmax'])
    assert(1>ftpt1.get_coverfrac(0, 0, 0.1, 1, 'arcsec', cosmo=cosmo))
    assert(1>ftpt1.get_coverfrac_nfw2D(0, 0, 0.1, 1, 'arcsec', 1, 'arcsec', cosmo=cosmo))
    # Load external
    ftpt1.data.write('ftpt1.fits', overwrite=True)
    ftpt1 = Footprint.read('ftpt1.fits', pixel_name='pixel', nside=nside)
    os.system('rm -f ftpt1.fits')
    # Add quantities to catalog
    c1.add_ftpt_masks(ftpt1, ftpt2)
    c1.add_ftpt_coverfrac(ftpt2, 1, 'mpc', cosmo=cosmo)
    c1.add_ftpt_coverfrac(ftpt2, 1, 'mpc', cosmo=cosmo, window='flat')
    assert_raises(ValueError, c1.add_ftpt_coverfrac, ftpt2, 1, 'mpc', window='unknown')
    # save footprint quantities of catalog
    c1.save_footprint_quantities('cat1_ftq.fits', overwrite=True)
    c1.load_footprint_quantities('cat1_ftq.fits')
    os.system('rm -f cat1_ftq.fits')

def test_artificial_footprint():
    c1, c2 = get_test_data()
    cosmo = AstroPyCosmology()
    # baseline footprint
    nside = 128
    pixels1 = hp.ang2pix(nside, c1['ra'], c1['dec'], lonlat=True)
    ftpt1 = Footprint(nside=nside, pixels=list(set(pixels1)))
    # Artificial footprint
    ftpt_test = create_footprint(c1['ra'], c1['dec'], nside=nside)
    assert_equal(sorted(ftpt1['pixel']), sorted(ftpt_test['pixel']))
    # Other functions
    ftpt_test = create_footprint(c1['ra'], c1['dec'], nside=None, min_density=2, neighbor_fill=None)
    assert_equal(ftpt_test.nside, 16)
    ftpt_test2 = create_footprint(c1['ra'], c1['dec'], nside=None, min_density=2, neighbor_fill=5)
    assert(ftpt_test['pixel'].size<ftpt_test2['pixel'].size)
    assert_raises(ValueError, create_footprint, c1['ra'], c1['dec'], nside=None, min_density=0)
