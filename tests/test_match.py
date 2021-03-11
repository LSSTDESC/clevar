# pylint: disable=no-member, protected-access
""" Tests for match.py """
import os
import numpy as np
from numpy.testing import assert_raises, assert_allclose, assert_equal

from clevar.catalog import Catalog
from clevar.match.parent import Match
from clevar.match import ProximityMatch
from clevar.cosmology import AstroPyCosmology, CCLCosmology

def test_parent():
    mt = Match()
    assert_raises(NotImplementedError, mt._prep_for_match, None)
    assert_raises(NotImplementedError, mt.multiple, None, None)

def test_proximity():
    _test_proximity(AstroPyCosmology)
    _test_proximity(CCLCosmology)
def _test_proximity(Cosmology):
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
    print(c1.data)
    print(c2.data)
    # init match
    cosmo =  AstroPyCosmology()
    mt = ProximityMatch()
    mt_config1 = {'delta_z':.2,
                'match_radius': '1 mpc',
                'cosmo':cosmo}
    mt_config2 = {'delta_z':.2,
                'match_radius': '1 arcsec'}
    mt.prep_cat_for_match(c1, **mt_config1)
    mt.prep_cat_for_match(c2, **mt_config2)
    # Check prep cat
    assert_allclose(c2.mt_input['ang'], np.ones(c2.size)/3600.)
    # Check multiple match
    mt.multiple(c1, c2)
    mt.multiple(c2, c1)
    mmt = [
        ['CL0', 'CL1', 'CL2'],
        ['CL0', 'CL1', 'CL2'],
        ['CL0', 'CL1', 'CL2'],
        ['CL3'],
        [],
        ]
    assert_equal(c1.match['multi_self'], mmt)
    assert_equal(c1.match['multi_other'], mmt)
    assert_equal(c2.match['multi_self'], mmt[:-1])
    assert_equal(c2.match['multi_other'], mmt[:-1])
    # Check unique with ang preference
    mt.unique(c1, c2, 'ang')
    mt.unique(c2, c1, 'ang')
    smt = ['CL0', 'CL1', 'CL2', 'CL3', None]
    assert_equal(c1.match['self'], smt)
    assert_equal(c1.match['other'], smt)
    assert_equal(c2.match['self'], smt[:-1])
    assert_equal(c2.match['other'], smt[:-1])
    # Check unique with mass preference
    for col in ('self', 'other'):
        c1.match[col] = None
        c2.match[col] = None
    mt.unique(c1, c2, 'mproxy')
    mt.unique(c2, c1, 'mproxy')
    smt = ['CL2', 'CL1', 'CL0', 'CL3', None]
    assert_equal(c1.match['self'], smt)
    assert_equal(c1.match['other'], smt)
    assert_equal(c2.match['self'], smt[:-1])
    assert_equal(c2.match['other'], smt[:-1])
    # Check unique with z preference
    for col in ('self', 'other'):
        c1.match[col] = None
        c2.match[col] = None
    c2.match['other'][0] = 'CL3' # to force a replacement
    mt.unique(c1, c2, 'z')
    mt.unique(c2, c1, 'z')
    smt = ['CL1', 'CL0', 'CL2', 'CL3', None]
    assert_equal(c1.match['self'], smt)
    assert_equal(c1.match['other'], smt)
    assert_equal(c2.match['self'], smt[:-1])
    assert_equal(c2.match['other'], smt[:-1])
    # Error for unkown preference
    assert_raises(ValueError, mt.unique, c1, c2, 'unknown')
    # Check save and load matching
    mt.save_matches(c1, c2, {'out_dir':'temp'})
    c1_v2 = Catalog('Cat1', **input1)
    c2_v2 = Catalog('Cat2', **input2)
    mt.load_matches(c1_v2, c2_v2, {'out_dir':'temp'})
    for col in ('self', 'other', 'multi_self', 'multi_other'):
        assert_equal(c1.match[col], c1_v2.match[col])
        assert_equal(c2.match[col], c2_v2.match[col])
    os.system('rm -rf temp')
    # Other config of prep for matching
        # No redshift use
    mt.prep_cat_for_match(c1, delta_z=None, match_radius='1 mpc', cosmo=cosmo)
    assert all(c1.mt_input['zmin']<c1.data['z'].min())
    assert all(c1.mt_input['zmax']>c1.data['z'].max())
        # missing all zmin/zmax info in catalog
    assert_raises(ValueError, mt.prep_cat_for_match, c1, delta_z='cat',
                    match_radius='1 mpc', cosmo=cosmo)
        # zmin/zmax in catalog
    c1.data['zmin'] = c1.data['z']-.2
    c1.data['zmax'] = c1.data['z']+.2
    c1.data['z_err'] = 0.1
    mt.prep_cat_for_match(c1, delta_z='cat', match_radius='1 mpc', cosmo=cosmo)
    assert_allclose(c1.mt_input['zmin'], c1.data['zmin'])
    assert_allclose(c1.mt_input['zmax'], c1.data['zmax'])
        # z_err in catalog
    del c1.data['zmin']
    del c1.data['zmax']
    mt.prep_cat_for_match(c1, delta_z='cat', match_radius='1 mpc', cosmo=cosmo)
    assert_allclose(c1.mt_input['zmin'], c1.data['z']-c1.data['z_err'])
    assert_allclose(c1.mt_input['zmax'], c1.data['z']+c1.data['z_err'])
        # zmin/zmax from aux file
    zv = np.linspace(0, 5, 10)
    np.savetxt('zvals.dat', [zv, zv-.22, zv+.33])
    mt.prep_cat_for_match(c1, delta_z='zvals.dat', match_radius='1 mpc', cosmo=cosmo)
    assert_allclose(c1.mt_input['zmin'], c1.data['z']-.22)
    assert_allclose(c1.mt_input['zmax'], c1.data['z']+.33)
    os.system('rm -rf zvals.dat')
        # radus in catalog
    c1.data['rad'] = 1
    c1.radius_unit = 'Mpc'
    mt.prep_cat_for_match(c1, delta_z='cat', match_radius='cat', cosmo=cosmo)
        # radus in unknown unit
    assert_raises(ValueError, mt.prep_cat_for_match, c1, delta_z='cat',
                    match_radius='1 unknown', cosmo=cosmo)
    # Other multiple match configs
    mt.prep_cat_for_match(c1, **mt_config1)
    mt.multiple(c1, c2, radius_selection='self')
    mt.multiple(c1, c2, radius_selection='other')
    mt.multiple(c1, c2, radius_selection='min')
