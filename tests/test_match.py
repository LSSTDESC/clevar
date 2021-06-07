# pylint: disable=no-member, protected-access
""" Tests for match.py """
import os
import numpy as np
from numpy.testing import assert_raises, assert_allclose, assert_equal

from clevar.catalog import ClCatalog, MemCatalog
from clevar.match.parent import Match
from clevar.match import ProximityMatch, MembershipMatch

def test_parent():
    mt = Match()
    assert_raises(NotImplementedError, mt.prep_cat_for_match, None)
    assert_raises(NotImplementedError, mt.multiple, None, None)
    assert_raises(NotImplementedError, mt.match_from_config, None, None, None, None)

def _test_mt_results(cat, multi_self, self, cross, multi_other=None, other=None):
    multi_other = multi_self if multi_other is None else multi_other
    other = self if other is None else other
    # Check multiple match
    slists = lambda mmt: [sorted(l) for l in mmt]
    assert_equal(slists(cat['mt_multi_self']), slists(multi_self))
    assert_equal(slists(cat['mt_multi_other']), slists(multi_other))
    # Check unique
    assert_equal(cat['mt_self'], self)
    assert_equal(cat['mt_other'], other)
    # Check cross
    assert_equal(cat['mt_cross'], cross)

def get_test_data_prox():
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
    input1, input2 = get_test_data_prox()
    c1 = ClCatalog('Cat1', **input1)
    c2 = ClCatalog('Cat2', **input2)
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
    _test_mt_results(c1, multi_self=mmt, self=smt, cross=smt)
    _test_mt_results(c2, multi_self=mmt[:-1], self=smt[:-1], cross=smt[:-1])
    # Check unique with mass preference
    for col in ('mt_self', 'mt_other'):
        c1[col] = None
        c2[col] = None
    mt.unique(c1, c2, 'more_massive')
    mt.unique(c2, c1, 'more_massive')
    mt.cross_match(c1)
    mt.cross_match(c2)
    smt = ['CL2', 'CL1', 'CL0', 'CL3', None]
    _test_mt_results(c1, multi_self=mmt, self=smt, cross=smt)
    _test_mt_results(c2, multi_self=mmt[:-1], self=smt[:-1], cross=smt[:-1])
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
    _test_mt_results(c1, multi_self=mmt, self=smt, cross=smt)
    _test_mt_results(c2, multi_self=mmt[:-1], self=smt[:-1], cross=smt[:-1])
    # Error for unkown preference
    assert_raises(ValueError, mt.unique, c1, c2, 'unknown')
    # Check save and load matching
    mt.save_matches(c1, c2, out_dir='temp', overwrite=True)
    c1_v2 = ClCatalog('Cat1', **input1)
    c2_v2 = ClCatalog('Cat2', **input2)
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
    c1['radius'] = 1
    c1.radius_unit = 'Mpc'
    mt.prep_cat_for_match(c1, delta_z='cat', match_radius='cat', cosmo=cosmo)
        # radus in catalog - mass units
    c1['rad'] = 1e14
    c1.radius_unit = 'M200c'
    mt.prep_cat_for_match(c1, delta_z='cat', match_radius='cat', cosmo=cosmo)
    c1.radius_unit = 'M200'
    assert_raises(ValueError, mt.prep_cat_for_match, c1, delta_z='cat',
                    match_radius='cat', cosmo=cosmo)
    c1.radius_unit = 'MXXX'
    assert_raises(ValueError, mt.prep_cat_for_match, c1, delta_z='cat',
                    match_radius='cat', cosmo=cosmo)
        # radus in unknown unit
    assert_raises(ValueError, mt.prep_cat_for_match, c1, delta_z='cat',
                    match_radius='1 unknown', cosmo=cosmo)
    # Other multiple match configs
    mt.prep_cat_for_match(c1, **mt_config1)
    mt.multiple(c1, c2, radius_selection='self')
    mt.multiple(c1, c2, radius_selection='other')
    mt.multiple(c1, c2, radius_selection='min')
def test_proximity_cfg(CosmoClass):
    input1, input2 = get_test_data_prox()
    c1 = ClCatalog('Cat1', **input1)
    c2 = ClCatalog('Cat2', **input2)
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
    _test_mt_results(c1, multi_self=mmt, self=smt, cross=smt)
    _test_mt_results(c2, multi_self=mmt[:-1], self=smt[:-1], cross=smt[:-1])
    ### test 1 ###
    mt_config['which_radius'] = 'cat1'
    c1._init_match_vals()
    c2._init_match_vals()
    mt.match_from_config(c1, c2, mt_config, cosmo=cosmo)
    _test_mt_results(c1, multi_self=mmt, self=smt, cross=smt)
    _test_mt_results(c2, multi_self=mmt[:-1], self=smt[:-1], cross=smt[:-1])
    ### test 2 ###
    mt_config['which_radius'] = 'cat2'
    c1._init_match_vals()
    c2._init_match_vals()
    mt.match_from_config(c1, c2, mt_config, cosmo=cosmo)
    _test_mt_results(c1, multi_self=mmt, self=smt, cross=smt)
    _test_mt_results(c2, multi_self=mmt[:-1], self=smt[:-1], cross=smt[:-1])
def get_test_data_mem():
    ncl = 5
    input1 = {
        'id': [f'CL{i}' for i in range(ncl)],
        'mass': [30+i for i in range(ncl)],
    }
    input2 = {k:v[:-1] for k, v in input1.items()}
    # members
    mem_dat = [(f'MEM{imem}', f'CL{icl}') for imem, icl in
        enumerate([i for i in range(ncl) for j in range(i, ncl)])]
    input1_mem = {'id_cluster': [f'CL{i}' for i in range(ncl) for j in range(i, ncl)]}
    input2_mem = {'id_cluster': [f'CL{i}' for i in range(ncl) for j in range(i, ncl)][:-1]}
    input1_mem['id'] = [f'MEM{i}' for i in range(len(input1_mem['id_cluster']))]
    input2_mem['id'] = [f'MEM{i}' for i in range(len(input2_mem['id_cluster']))]
    input1_mem['ra'] = np.arange(len(input1_mem['id_cluster']))
    input2_mem['ra'] = np.arange(len(input2_mem['id_cluster']))
    input1_mem['dec'] = np.zeros(len(input1_mem['id_cluster']))
    input2_mem['dec'] = np.zeros(len(input2_mem['id_cluster']))
    input2_mem['id_cluster'][0] = f'CL{ncl-2}'
    c1 = ClCatalog('Cat1', **input1)
    c2 = ClCatalog('Cat2', **input2)
    mem1 = MemCatalog('Mem1', **input1_mem)
    mem2 = MemCatalog('Mem2', **input2_mem)
    return c1, c2, mem1, mem2
def test_membership():
    c1, c2, mem1, mem2 = get_test_data_mem()
    print(c1.data)
    print(c2.data)
    # init match
    mt = MembershipMatch()
    # Try to use fill_shared_members before fill_shared_members 
    assert_raises(ValueError, mt.fill_shared_members, c1, c2, mem1, mem2)
    # Check both methods
    mt.match_members(mem1, mem2, method='id')
    mt2 = MembershipMatch()
    mt2.match_members(mem1, mem2, method='angular_distance', radius='1arcsec')
    assert_equal(mt.matched_mems, mt2.matched_mems)
    # Save and load matched members
    mt.save_matched_members('temp_mem.txt')
    mt2.load_matched_members('temp_mem.txt')
    assert_equal(mt.matched_mems, mt2.matched_mems)
    os.system('rm temp_mem.txt')
    # Fill shared members
    mt.fill_shared_members(c1, c2, mem1, mem2)
    # Save and load shared members
    mt.save_shared_members(c1, c2, 'temp')
    c1_test, c2_test = ClCatalog('test1'), ClCatalog('test2')
    mt.load_shared_members(c1_test, c2_test, 'temp')
    assert_equal(c1.mt_input, c1_test.mt_input)
    assert_equal(c2.mt_input, c2_test.mt_input)
    os.system('rm temp.1.p temp.2.p')
    # Check multiple match
    mt.multiple(c1, c2)
    mt.multiple(c2, c1)
    mt.unique(c1, c2, 'shared_member_fraction')
    mt.unique(c2, c1, 'shared_member_fraction')
    c1.cross_match()
    c2.cross_match()
    print(c1)
    print(c2)
    print(c1['mt_multi_self', 'mt_multi_other'])
    print(c2['mt_multi_self', 'mt_multi_other'])
    mmt1 = [['CL0', 'CL3'], ['CL1'], ['CL2'], ['CL3'], []]
    mmt2 = [['CL0'], ['CL1'], ['CL2'], ['CL0', 'CL3'], ]
    smt = ['CL0', 'CL1', 'CL2', 'CL3', None]
    _test_mt_results(c1, multi_self=mmt1, self=smt, cross=smt)
    _test_mt_results(c2, multi_self=mmt2, self=smt[:-1], cross=smt[:-1])
    # Check with minimum_share_fraction
    c1._init_match_vals()
    c2._init_match_vals()
    mt.multiple(c1, c2)
    mt.multiple(c2, c1)
    mt.unique(c1, c2, 'shared_member_fraction', minimum_share_fraction=.7)
    mt.unique(c2, c1, 'shared_member_fraction', minimum_share_fraction=.7)
    c1.cross_match()
    c2.cross_match()
    print('########')
    print(c1)
    print(c2)
    print(c1['mt_multi_self', 'mt_multi_other'])
    print(c2['mt_multi_self', 'mt_multi_other'])
    smt = ['CL0', 'CL1', 'CL2', 'CL3', None]
    cmt = ['CL0', 'CL1', 'CL2', None, None]
    _test_mt_results(c1, multi_self=mmt1, self=smt, cross=cmt, other=cmt)
    _test_mt_results(c2, multi_self=mmt2, self=cmt[:-1], cross=cmt[:-1], other=smt[:-1])
    # Check with minimum_share_fraction
    c1._init_match_vals()
    c2._init_match_vals()
    mt.multiple(c1, c2)
    mt.multiple(c2, c1)
    mt.unique(c1, c2, 'shared_member_fraction', minimum_share_fraction=.9)
    mt.unique(c2, c1, 'shared_member_fraction', minimum_share_fraction=.9)
    c1.cross_match()
    c2.cross_match()
    print('########')
    print(c1)
    print(c2)
    print(c1['mt_multi_self', 'mt_multi_other'])
    print(c2['mt_multi_self', 'mt_multi_other'])
    smt = [None, 'CL1', 'CL2', 'CL3', None]
    omt = ['CL0', 'CL1', 'CL2', None, None]
    cmt = [None, 'CL1', 'CL2', None, None]
    _test_mt_results(c1, multi_self=mmt1, self=smt, cross=cmt, other=omt)
    _test_mt_results(c2, multi_self=mmt2, self=omt[:-1], cross=cmt[:-1], other=smt[:-1])
    # Check save and load matching
    mt.save_matches(c1, c2, out_dir='temp', overwrite=True)
    c1_test, c2_test = get_test_data_mem()[:2]
    mt.load_matches(c1_test, c2_test, out_dir='temp')
    for col in c1.data.colnames:
        if col[:3]=='mt_':
            assert_equal(c1[col], c1_test[col])
            assert_equal(c2[col], c2_test[col])
    os.system('rm -rf temp')
