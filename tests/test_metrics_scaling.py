"""Tests for clevar/match_metrics/recovery"""
import numpy as np
from clevar.catalog import ClCatalog
from clevar.cosmology import AstroPyCosmology as CosmoClass
from clevar.match import ProximityMatch
from clevar.match_metrics import scaling
from numpy.testing import assert_raises

def get_test_data():
    input1 = {
        'id': [f'CL{i}' for i in range(5)],
        'ra': [0., .0001, 0.00011, 25, 20],
        'dec': [0., 0, 0, 0, 0],
        'z': [.2, .3, .25, .4, .35],
        'mass': [10**13.5, 10**13.4, 10**13.3, 10**13.8, 10**14],
        'mass_err': [10**12.5, 10**12.4, 10**12.3, 10**12.8, 10**13],
    }
    input2 = {k:v[:4] for k, v in input1.items()}
    input2['z'][:2] = [.3, .2]
    input2['mass'][:3] = input2['mass'][:3][::-1]
    c1 = ClCatalog('Cat1', **input1)
    c2 = ClCatalog('Cat2', **input2)
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
def test_z_simple():
    c1, c2 = get_test_data()
    ax = scaling.redshift(c1, c2, 'cat1')
    ax = scaling.redshift(c1, c2, 'cat1', add_err=True)
def test_z_color():
    c1, c2 = get_test_data()
    ax, cb = scaling.redshift_masscolor(c1, c2, 'cat1', add_err=True)
    ax = scaling.redshift_masscolor(c1, c2, 'cat1', add_cb=False)
def test_z_density():
    c1, c2 = get_test_data()
    ax, cb = scaling.redshift_density(c1, c2, 'cat1', add_err=True)
    ax, cb = scaling.redshift_density(c1, c2, 'cat1', ax_rotation=45)
def test_z_panel():
    c1, c2 = get_test_data()
    fig, axes = scaling.redshift_masspanel(c1, c2, 'cat1', add_err=True, mass_bins=4)
def test_z_density_panel():
    c1, c2 = get_test_data()
    fig, axes = scaling.redshift_density_masspanel(c1, c2, 'cat1', add_err=True)
def test_z_metrics():
    c1, c2 = get_test_data()
    fig, axes = scaling.redshift_metrics(c1, c2, 'cat1')
def test_z_density_panel():
    c1, c2 = get_test_data()
    fig, axes = scaling.redshift_density_metrics(c1, c2, 'cat1')
def test_m_simple():
    c1, c2 = get_test_data()
    ax = scaling.mass(c1, c2, 'cat1', add_err=True)
def test_m_color():
    c1, c2 = get_test_data()
    ax, cb = scaling.mass_zcolor(c1, c2, 'cat1', add_err=True)
def test_m_density():
    c1, c2 = get_test_data()
    ax, cb = scaling.mass_density(c1, c2, 'cat1', add_err=True)
def test_m_panel():
    c1, c2 = get_test_data()
    fig, axes = scaling.mass_zpanel(c1, c2, 'cat1', add_err=True)
def test_m_density_panel():
    c1, c2 = get_test_data()
    fig, axes = scaling.mass_density_zpanel(c1, c2, 'cat1', add_err=True)
def test_color_panel():
    c1, c2 = get_test_data()
    fig, axes = scaling.ClCatalogFuncs.plot_color_panel(c1, c2, 'cat1', 'mass',
        col_color='z', col_panel='z', bins_panel=3)
def test_m_metrics():
    c1, c2 = get_test_data()
    fig, axes = scaling.mass_metrics(c1, c2, 'cat1')
def test_m_density_panel():
    c1, c2 = get_test_data()
    fig, axes = scaling.mass_density_metrics(c1, c2, 'cat1',
                    mask1=c1['mass']>0, mask2=c2['mass']>0)
