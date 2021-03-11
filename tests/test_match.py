# pylint: disable=no-member, protected-access
""" Tests for match.py """
import numpy as np
from numpy.testing import assert_raises, assert_allclose, assert_equal
from astropy.table import Table

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
    input1 = Table()
    input1['ID'] = [f'CL{i}' for i in range(5)]
    input1['RA'] = [0., .0001, 10, 25, 20]
    input1['DEC'] = 0.
    input1['Z'] = [.2, .3, .25, .4, .35]
    
    input2 = input1[:4]
    input2['Z'][:2] = [.3, .2]

    c1 = Catalog('Cat1', id=input1['ID'], ra=input1['RA'], dec=input1['DEC'], z=input1['Z'])
    c2 = Catalog('Cat2', id=input2['ID'], ra=input2['RA'], dec=input2['DEC'], z=input2['Z'])
    # init match
    mt = ProximityMatch()
    mt_config1 = {'delta_z':.2,
                'match_radius': '1 mpc',
                'cosmo':AstroPyCosmology()}
    mt_config2 = {'delta_z':.2,
                'match_radius': '1 arcsec'}
    mt.prep_cat_for_match(c1, **mt_config1)
    mt.prep_cat_for_match(c2, **mt_config2)
    #
    assert_allclose(c2.mt_input['ang'], np.ones(c2.size)/3600.)
    #
    mt.multiple(c1, c2)
    mt.multiple(c2, c1)
    mmt = [
        ['CL0', 'CL1'],
        ['CL0', 'CL1'],
        ['CL2'],
        ['CL3'],
        [],
        ]
    assert_equal(c1.match['multi_self'], mmt)
    assert_equal(c1.match['multi_other'], mmt)
    assert_equal(c2.match['multi_self'], mmt[:-1])
    assert_equal(c2.match['multi_other'], mmt[:-1])
    #
    mt.unique(c1, c2, 'ang')
    mt.unique(c2, c1, 'ang')
    smt = ['CL0', 'CL1', 'CL2', 'CL3', None]
    assert_equal(c1.match['self'], smt)
    assert_equal(c1.match['other'], smt)
    assert_equal(c2.match['self'], smt[:-1])
    assert_equal(c2.match['other'], smt[:-1])
    #
    mt.save_matches(c1, c2, {'out_dir':'temp'})
    c1_v2 = Catalog('Cat1', id=input1['ID'], ra=input1['RA'], dec=input1['DEC'], z=input1['Z'])
    c2_v2 = Catalog('Cat2', id=input2['ID'], ra=input2['RA'], dec=input2['DEC'], z=input2['Z'])
    mt.load_matches(c1_v2, c2_v2, {'out_dir':'temp'})
    for col in ('self', 'other', 'multi_self', 'multi_other'):
        assert_equal(c1.match[col], c1_v2.match[col])
        assert_equal(c2.match[col], c2_v2.match[col])
