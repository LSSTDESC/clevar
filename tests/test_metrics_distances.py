"""Tests for clevar/metrics/distances"""
import numpy as np
from clevar.catalog import Catalog
from clevar.cosmology import AstroPyCosmology as CosmoClass
from clevar.match import ProximityMatch
from clevar.metrics import distances as dt

def get_test_data():
    input1 = {
        'id': [f'CL{i}' for i in range(5)],
        'ra': [0., .0001, 0.00011, 25, 20],
        'dec': [0., 0, 0, 0, 0],
        'z': [.2, .3, .25, .4, .35],
        'mass': [10**13.5, 10**13.4, 10**13.3, 10**13.8, 10**14],
    }
    input2 = {k:v[:4] for k, v in input1.items()}
    input2['z'][:2] = [.3, .2]
    input2['mass'][:3] = input2['mass'][:3][::-1]
    c1 = Catalog('Cat1', **input1)
    c2 = Catalog('Cat2', **input2)
    cosmo = CosmoClass()
    mt = ProximityMatch()
    mt_config1 = {'delta_z':.2,
                'match_radius': '1 mpc',
                'cosmo':cosmo}
    mt_config2 = {'delta_z':.2,
                'match_radius': '1 arcsec'}
    mt.prep_cat_for_match(c1, **mt_config1)
    mt.prep_cat_for_match(c2, **mt_config2)
    mt.multiple(c1, c2)
    mt.multiple(c2, c1)
    mt.unique(c1, c2, 'angular_proximity')
    mt.unique(c2, c1, 'angular_proximity')
    return c1, c2
def test_central_position():
    cat1, cat2 = get_test_data()
    dt.central_position(cat1, cat2, 'cat1', radial_bins=20, radial_bin_units='degrees', cosmo=None,
                    mass_bins=None, mass_label=None, ax=None)
def test_redshift():
    cat1, cat2 = get_test_data()
    dt.redshift(cat1, cat2, 'cat1', redshift_bins=20, normalize=None,
             mass_bins=None, mass_label=None, ax=None)
    dt.redshift(cat1, cat2, 'cat1', redshift_bins=20, normalize=None,
             mass_bins=[1e14, 1e16], mass_label=None, ax=None)
