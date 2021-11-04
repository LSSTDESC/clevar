"""Tests for clevar/match_metrics/recovery"""
import numpy as np
from clevar.catalog import ClCatalog
from clevar.cosmology import AstroPyCosmology as CosmoClass
from clevar.utils import gaussian
from clevar.match import ProximityMatch
from clevar.match_metrics import recovery as rc
from numpy.testing import assert_raises

##############################
#### Input data ##############
##############################
class _test_data():
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
    c1 = ClCatalog('Cat1', **input1)
    c2 = ClCatalog('Cat2', **input2)
    cosmo = CosmoClass()
    mt = ProximityMatch()
    mt_config1 = {'delta_z':.2, 'match_radius': '1 mpc', 'cosmo':cosmo}
    mt_config2 = {'delta_z':.2, 'match_radius': '1 arcsec'}
    mt.prep_cat_for_match(c1, **mt_config1)
    mt.prep_cat_for_match(c2, **mt_config2)
    mt.multiple(c1, c2)
    mt.multiple(c2, c1)
    mt.unique(c1, c2, 'angular_proximity')
    mt.unique(c2, c1, 'angular_proximity')

##############################
### Test #####################
##############################
def test_plot():
    cat = _test_data.c1
    matching_type = 'cat1'
    redshift_bins = [0, 0.5, 1]
    mass_bins = [1e13, 1e16]
    rc.plot(cat, matching_type, redshift_bins, mass_bins, transpose=False, log_mass=True,
            redshift_label=None, mass_label=None, recovery_label=None)
    rc.plot(cat, matching_type, redshift_bins, mass_bins, shape='line')
    rc.plot(cat, matching_type, redshift_bins, mass_bins, add_legend=True)
    assert_raises(ValueError, rc.plot, cat, matching_type, redshift_bins, mass_bins,
                  shape='unknown')
    rc.plot(cat, matching_type, redshift_bins, mass_bins, add_legend=True,
            p_m1_m2=lambda  m1, m2: gaussian(np.log10(m2), np.log10(m1), 0.1))
    rc.plot(cat, matching_type, redshift_bins, mass_bins, add_legend=True, transpose=True,
            p_m1_m2=lambda  m1, m2: gaussian(np.log10(m2), np.log10(m1), 0.1))


def test_plot_panel():
    cat = _test_data.c1
    matching_type = 'cat1'
    redshift_bins = [0, 0.5, 1]
    mass_bins = [1e13, 1e14, 1e15, 1e16]
    rc.plot_panel(
        cat, matching_type, redshift_bins, mass_bins, transpose=False, log_mass=True,
        redshift_label=None, mass_label=None, recovery_label=None,)
    rc.plot_panel(cat, matching_type, redshift_bins, mass_bins, add_label=True, label_fmt='.2f')
    rc.plot_panel(cat, matching_type, redshift_bins, mass_bins, transpose=True)
    rc.plot_panel(cat, matching_type, redshift_bins, mass_bins,
                  p_m1_m2=lambda  m1, m2: gaussian(np.log10(m2), np.log10(m1), 0.1))
    rc.plot_panel(cat, matching_type, redshift_bins, mass_bins, transpose=True,
                  p_m1_m2=lambda  m1, m2: gaussian(np.log10(m2), np.log10(m1), 0.1))


def test_plot2D():
    cat = _test_data.c1
    matching_type = 'cat1'
    redshift_bins = [0, 0.5, 1]
    mass_bins = [1e13, 1e14, 1e15, 1e16]
    rc.plot2D(cat, matching_type, redshift_bins, mass_bins, transpose=False, log_mass=True,
              redshift_label=None, mass_label=None, recovery_label=None,)
    rc.plot2D(cat, matching_type, redshift_bins, mass_bins, add_num=True)


def test_skyplot():
    cat = _test_data.c1
    matching_type = 'cat1'
    rc.skyplot(cat, matching_type, recovery_label=None,)
    # for monocromatic map
    rc.skyplot(cat[cat['mt_self']!=None], matching_type, recovery_label=None,)
