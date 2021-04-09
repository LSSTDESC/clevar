# pylint: disable=no-member, protected-access
""" Tests for match.py """
import os
import numpy as np
from numpy.testing import assert_raises, assert_allclose, assert_equal

from clevar.catalog import Catalog
from clevar.match.parent import Match
from clevar.match import ProximityMatch

def test_parent():
    mt = Match()
    assert_raises(NotImplementedError, mt._prep_for_match, None)
    assert_raises(NotImplementedError, mt.multiple, None, None)
    assert_raises(NotImplementedError, mt.match_from_config, None, None, None, None)

def _test_mt_results(mt, c1, c2, multiple_res1, multiple_res2,
                    unique_res1, unique_res2, cross_res1, cross_res2):
    # Check multiple match
    assert_equal(c1['mt_multi_self'], multiple_res1)
    assert_equal(c1['mt_multi_other'], multiple_res1)
    assert_equal(c2['mt_multi_self'], multiple_res2)
    assert_equal(c2['mt_multi_other'], multiple_res2)
    # Check unique
    assert_equal(c1['mt_self'], unique_res1)
    assert_equal(c1['mt_other'], unique_res1)
    assert_equal(c2['mt_self'], unique_res2)
    assert_equal(c2['mt_other'], unique_res2)
    # Check cross
    assert_equal(c1['mt_cross'], cross_res1)
    assert_equal(c2['mt_cross'], cross_res2)

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
    return input1, input2
def test_proximity(CosmoClass):
    input1, input2 = get_test_data()
    c1 = Catalog('Cat1', **input1)
    c2 = Catalog('Cat2', **input2)
    print(c1.data)
    print(c2.data)
    # init match
    cosmo =  CosmoClass()
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
    mmt = [
        ['CL0', 'CL1', 'CL2'],
        ['CL0', 'CL1', 'CL2'],
        ['CL0', 'CL1', 'CL2'],
        ['CL3'],
        [],
        ]
    smt = ['CL0', 'CL1', 'CL2', 'CL3', None]
    mt.multiple(c1, c2)
    mt.multiple(c2, c1)
    mt.unique(c1, c2, 'angular_proximity')
    mt.unique(c2, c1, 'angular_proximity')
    c1.cross_match()
    c2.cross_match()
    _test_mt_results(mt, c1, c2, multiple_res1=mmt, unique_res1=smt, cross_res1=smt,
                     multiple_res2=mmt[:-1], unique_res2=smt[:-1], cross_res2=smt[:-1])
    # Check unique with mass preference
    for col in ('mt_self', 'mt_other'):
        c1[col] = None
        c2[col] = None
    mt.unique(c1, c2, 'more_massive')
    mt.unique(c2, c1, 'more_massive')
    mt.cross_match(c1)
    mt.cross_match(c2)
    smt = ['CL2', 'CL1', 'CL0', 'CL3', None]
    _test_mt_results(mt, c1, c2, multiple_res1=mmt, unique_res1=smt, cross_res1=smt,
                     multiple_res2=mmt[:-1], unique_res2=smt[:-1], cross_res2=smt[:-1])
    # Check unique with z preference
    for col in ('mt_self', 'mt_other'):
        c1[col] = None
        c2[col] = None
    c2['mt_other'][0] = 'CL3' # to force a replacement
    mt.unique(c1, c2, 'redshift_proximity')
    mt.unique(c2, c1, 'redshift_proximity')
    mt.cross_match(c1)
    mt.cross_match(c2)
    smt = ['CL1', 'CL0', 'CL2', 'CL3', None]
    _test_mt_results(mt, c1, c2, multiple_res1=mmt, unique_res1=smt, cross_res1=smt,
                     multiple_res2=mmt[:-1], unique_res2=smt[:-1], cross_res2=smt[:-1])
    # Error for unkown preference
    assert_raises(ValueError, mt.unique, c1, c2, 'unknown')
    # Check save and load matching
    mt.save_matches(c1, c2, out_dir='temp', overwrite=True)
    c1_v2 = Catalog('Cat1', **input1)
    c2_v2 = Catalog('Cat2', **input2)
    mt.load_matches(c1_v2, c2_v2, out_dir='temp')
    for col in ('mt_self', 'mt_other', 'mt_multi_self', 'mt_multi_other'):
        assert_equal(c1[col], c1_v2[col])
        assert_equal(c2[col], c2_v2[col])
    os.system('rm -rf temp')
    # Other config of prep for matching
        # No redshift use
    mt.prep_cat_for_match(c1, delta_z=None, match_radius='1 mpc', cosmo=cosmo)
    assert all(c1.mt_input['zmin']<c1['z'].min())
    assert all(c1.mt_input['zmax']>c1['z'].max())
        # missing all zmin/zmax info in catalog
    assert_raises(ValueError, mt.prep_cat_for_match, c1, delta_z='cat',
                    match_radius='1 mpc', cosmo=cosmo)
        # zmin/zmax in catalog
    c1['zmin'] = c1['z']-.2
    c1['zmax'] = c1['z']+.2
    c1['z_err'] = 0.1
    mt.prep_cat_for_match(c1, delta_z='cat', match_radius='1 mpc', cosmo=cosmo)
    assert_allclose(c1.mt_input['zmin'], c1['zmin'])
    assert_allclose(c1.mt_input['zmax'], c1['zmax'])
        # z_err in catalog
    del c1['zmin']
    del c1['zmax']
    mt.prep_cat_for_match(c1, delta_z='cat', match_radius='1 mpc', cosmo=cosmo)
    assert_allclose(c1.mt_input['zmin'], c1['z']-c1['z_err'])
    assert_allclose(c1.mt_input['zmax'], c1['z']+c1['z_err'])
        # zmin/zmax from aux file
    zv = np.linspace(0, 5, 10)
    np.savetxt('zvals.dat', [zv, zv-.22, zv+.33])
    mt.prep_cat_for_match(c1, delta_z='zvals.dat', match_radius='1 mpc', cosmo=cosmo)
    assert_allclose(c1.mt_input['zmin'], c1['z']-.22)
    assert_allclose(c1.mt_input['zmax'], c1['z']+.33)
    os.system('rm -rf zvals.dat')
        # radus in catalog
    c1['rad'] = 1
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
def test_proximity_cfg(CosmoClass):
    input1, input2 = get_test_data()
    c1 = Catalog('Cat1', **input1)
    c2 = Catalog('Cat2', **input2)
    print(c1.data)
    print(c2.data)
    # init match
    cosmo =  CosmoClass()
    mt = ProximityMatch()
    # test wrong matching config
    assert_raises(ValueError, mt.match_from_config, c1, c2, {'type':'unknown'}, cosmo=cosmo)
    ### test 0 ###
    mt_config = {
        'which_radius': 'max',
        'type': 'cross',
        'preference': 'angular_proximity',
        'catalog1': {
            'delta_z':.2,
            'match_radius': '1 mpc'},
        'catalog2': {
            'delta_z':.2,
            'match_radius': '1 arcsec'},
    }
    # Check multiple match
    mmt = [
        ['CL0', 'CL1', 'CL2'],
        ['CL0', 'CL1', 'CL2'],
        ['CL0', 'CL1', 'CL2'],
        ['CL3'],
        [],
        ]
    smt = ['CL0', 'CL1', 'CL2', 'CL3', None]
    ### test 0 ###
    mt.match_from_config(c1, c2, mt_config, cosmo=cosmo)
        # Check prep cat
    assert_allclose(c2.mt_input['ang'], np.ones(c2.size)/3600.)
        # Check match
    _test_mt_results(mt, c1, c2, multiple_res1=mmt, unique_res1=smt, cross_res1=smt,
                     multiple_res2=mmt[:-1], unique_res2=smt[:-1], cross_res2=smt[:-1])
    ### test 1 ###
    mt_config['which_radius'] = 'cat1'
    c1._init_match_vals()
    c2._init_match_vals()
    mt.match_from_config(c1, c2, mt_config, cosmo=cosmo)
    _test_mt_results(mt, c1, c2, multiple_res1=mmt, unique_res1=smt, cross_res1=smt,
                     multiple_res2=mmt[:-1], unique_res2=smt[:-1], cross_res2=smt[:-1])
    ### test 2 ###
    mt_config['which_radius'] = 'cat2'
    c1._init_match_vals()
    c2._init_match_vals()
    mt.match_from_config(c1, c2, mt_config, cosmo=cosmo)
    _test_mt_results(mt, c1, c2, multiple_res1=mmt, unique_res1=smt, cross_res1=smt,
                     multiple_res2=mmt[:-1], unique_res2=smt[:-1], cross_res2=smt[:-1])
