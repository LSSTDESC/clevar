"""Tests for clevar/match_metrics/recovery"""
import numpy as np
from clevar.catalog import ClCatalog
from clevar.cosmology import AstroPyCosmology as CosmoClass
from clevar.match import ProximityMatch
from clevar.match_metrics import scaling
from numpy.testing import assert_raises

##############################
#### Input data ##############
##############################
class _test_data():
    input1 = {
        'ra': np.arange(30),
        'dec': np.arange(30),
        'z': np.linspace(0.01, 2, 30),
        'mass': 10**np.linspace(13, 15, 30),
        'mass_err': 10**(np.ones(30)*.5),
    }
    c1 = ClCatalog('Cat1', **input1)
    c2 = ClCatalog('Cat2', **input1)
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

##############################
### Test #####################
##############################
def test_z_simple():
    c1, c2 = _test_data.c1, _test_data.c2
    ax = scaling.redshift(c1, c2, 'cat1')
    ax = scaling.redshift(c1, c2, 'cat1', add_err=True)
def test_z_color():
    c1, c2 = _test_data.c1, _test_data.c2
    info = scaling.redshift_masscolor(c1, c2, 'cat1', add_err=True)
    ax = scaling.redshift_masscolor(c1, c2, 'cat1', add_cb=False)
def test_z_density():
    c1, c2 = _test_data.c1, _test_data.c2
    info = scaling.redshift_density(c1, c2, 'cat1', add_err=True)
    info = scaling.redshift_density(c1, c2, 'cat1', ax_rotation=45)
def test_z_panel():
    c1, c2 = _test_data.c1, _test_data.c2
    info = scaling.redshift_masspanel(c1, c2, 'cat1', add_err=True, mass_bins=4)
def test_z_density_panel():
    c1, c2 = _test_data.c1, _test_data.c2
    info = scaling.redshift_density_masspanel(c1, c2, 'cat1', add_err=True)
def test_z_metrics():
    c1, c2 = _test_data.c1, _test_data.c2
    info = scaling.redshift_metrics(c1, c2, 'cat1')
def test_z_density_metrics():
    c1, c2 = _test_data.c1, _test_data.c2
    info = scaling.redshift_density_metrics(c1, c2, 'cat1')
def test_z_dist():
    c1, c2 = _test_data.c1, _test_data.c2
    info = scaling.redshift_dist(c1, c2, 'cat1')
def test_z_dist_self():
    c1, c2 = _test_data.c1, _test_data.c2
    info = scaling.redshift_dist_self(c1)
def test_z_density_dist():
    c1, c2 = _test_data.c1, _test_data.c2
    info = scaling.redshift_density_dist(c1, c2, 'cat1')
def test_m_simple():
    c1, c2 = _test_data.c1, _test_data.c2
    info = scaling.mass(c1, c2, 'cat1', add_err=True)
def test_m_color():
    c1, c2 = _test_data.c1, _test_data.c2
    info = scaling.mass_zcolor(c1, c2, 'cat1', add_err=True)
def test_m_density():
    c1, c2 = _test_data.c1, _test_data.c2
    info = scaling.mass_density(c1, c2, 'cat1', add_err=True)
    scaling.mass_density(c1, c2, 'cat1', add_bindata=True, fit_bins1=[1, 2], fit_bins2=3)
def test_m_panel():
    c1, c2 = _test_data.c1, _test_data.c2
    info = scaling.mass_zpanel(c1, c2, 'cat1', add_err=True)
def test_m_density_panel():
    c1, c2 = _test_data.c1, _test_data.c2
    info = scaling.mass_density_zpanel(c1, c2, 'cat1', add_err=True)
def test_color_panel():
    c1, c2 = _test_data.c1, _test_data.c2
    info = scaling.ClCatalogFuncs.plot_panel(c1, c2, 'cat1', 'mass', col_color='z', col_panel='z',
                                             bins_panel=3)
    info = scaling.ClCatalogFuncs.plot_panel(c1, c2, 'cat1', 'mass', col_color='z', col_panel='z',
                                             bins_panel=[3, 4, 5])
def test_m_metrics():
    c1, c2 = _test_data.c1, _test_data.c2
    info = scaling.mass_metrics(c1, c2, 'cat1')
    info = scaling.mass_metrics(c1, c2, 'cat1', metrics=['p_68'])
    assert_raises(ValueError, scaling.mass_metrics, c1, c2, 'cat1', metrics=['xxx'])
    info = scaling.mass_metrics(c1, c2, 'cat1', metrics_mode='diff_log')
def test_m_density_metrics():
    c1, c2 = _test_data.c1, _test_data.c2
    info = scaling.mass_density_metrics(c1, c2, 'cat1', mask1=c1['mass']>0, mask2=c2['mass']>0)
def test_m_dist():
    c1, c2 = _test_data.c1, _test_data.c2
    info = scaling.mass_dist(c1, c2, 'cat1', mass_bins=6)
def test_m_dist_self():
    c1, c2 = _test_data.c1, _test_data.c2
    info = scaling.mass_dist_self(c1)
    assert_raises(ValueError, scaling.ClCatalogFuncs.plot_dist_self,
                  c1, col='mass', transpose=True, col_aux=None)
def test_m_density_dist():
    c1, c2 = _test_data.c1, _test_data.c2
    info = scaling.mass_density_dist(c1, c2, 'cat1', add_fit=True, fit_bins1=5, fit_bins2=3)
    info = scaling.mass_density_dist(c1, c2, 'cat1', add_fit=True, fit_bins1=5, fit_bins2=3,
                                     fit_statistics='mean')
    info = scaling.mass_density_dist(c1, c2, 'cat1', add_fit=True, fit_bins1=5, fit_bins2=3,
                                     fit_statistics='individual')
    assert_raises(ValueError, scaling.mass_density_dist, c1, c2, 'cat1', add_fit=True,
                  fit_statistics='unknown')
    info = scaling.mass_density_dist(c1, c2, 'cat1', add_fit=True, fit_bins1=2, fit_bins2=3,
                                     fit_statistics='mode')
    info = scaling.mass_density_dist(c1, c2, 'cat1', add_fit=True, fit_bins1=[1e16,1e18], fit_bins2=3)
