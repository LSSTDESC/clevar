from clevar import ClCatalog, MemCatalog, ClData
from clevar.catalog import Catalog
from clevar.utils import NameList, LowerCaseDict, updated_dict
import numpy as np
from numpy.testing import assert_raises, assert_allclose, assert_equal
import pytest
import os

def test_lowercasedict():
    for key in ('Key', 'key', 'KEY', 'keY'):
        nlist = NameList([key, 2])
        d = LowerCaseDict()
        for key2 in ('Key', 'key', 'KEY', 'keY'):
            assert key2 in nlist
            assert 2 in nlist
            d.setdefault(key, 'value')
            assert_equal(d[key2], 'value')
            del d[key2]
            assert key2 not in d
            d[key] = 'value'
            assert_equal(d[key2], 'value')
            d.pop(key2)
            assert key2 not in d
            d.update({key:'value'})
            assert_equal(d[key2], 'value')
    assert_raises(ValueError, updated_dict, 'temp')

def _base_cat_test(**quantities):
    c = Catalog(name='test', **quantities)
    test_vals = quantities.get('data', quantities)
    for k, v in test_vals.items():
        assert_equal(c[k], v)
        assert_equal(c[:1][k], v[:1])
        assert_equal(c.get(k), v)
    assert_equal(c.get('ra2'), None)
    assert_equal(len(c), len(test_vals['ra']))
    c['ra', 'dec']
    # test read
    c.write('cat_with_header.fits', overwrite=True)
    assert_raises(ValueError, Catalog.read, 'cat_with_header.fits', name='temp', tags='x')
    c_read = Catalog.read_full('cat_with_header.fits')
    os.system('rm -f cat_with_header.fits')
    # test removing column
    del c['ra']
    assert_raises(KeyError, c.__getitem__, 'ra')
    assert_raises(ValueError, c.tag_column, 'XXX', 'newtag')
    # test warnings
    c.tag_column('ra', 'newtag')
    with pytest.warns(None) as record:
        c.tag_column('z', 'newtag')
    assert f'{record._list[0].message}'=='tag newtag:ra being replaced by newtag:z'
    with pytest.warns(None) as record:
        c.tag_column('dec', 'z')
    assert f'{record._list[0].message}'==("There is a column with the same name as the tag setup."
                                          " cat['z'] calls cat['dec'] now."
                                          " To get 'z' column, use cat.data['z'].")
    # tag columns
    assert_raises(ValueError, c.tag_columns, ['2', '3'], ['1'])
    c.tag_columns(['id', 'dec'], ['id', 'dec'])

def test_catalog():
    quantities = {'id': ['a', 'b'], 'ra': [10, 20], 'dec': [20, 30], 'z': [0.5, 0.6]}
    _base_cat_test(**quantities)
    _base_cat_test(data=quantities)
    # fail to instantiance object
    assert_raises(ValueError, Catalog.__init__, Catalog, name=None)
    assert_raises(ValueError, Catalog.__init__, Catalog, name='test', labels='x')
    assert_raises(ValueError, Catalog.__init__, Catalog, name='test', tags='x')
    c_ = Catalog('null')
    assert_raises(TypeError, c_.__setitem__, 'mass', 1)
    assert_raises(TypeError, c_._add_values, data=1)
    assert_raises(KeyError, c_._add_values, data=[1], mass=[1])
    # Check creation of id col
    with pytest.warns(None) as record:
        c_no_id = Catalog(name='no id', ra=[1, 2])
    assert f'{record._list[0].message}'=='id column missing, additional one is being created.'
    assert_equal(c_no_id['id'], ['0', '1'])
    with pytest.warns(None) as record:
        c_no_id = Catalog(name='no id')
        c_no_id['ra'] = [1, 2]
    assert f'{record._list[0].message}'=='id column missing, additional one is being created.'
    assert_equal(c_no_id['id'], ['0', '1'])

def test_clcatalog():
    quantities = {'id': ['a', 'b'], 'ra': [10, 20], 'dec': [20, 30], 'z': [0.5, 0.6]}
    c = ClCatalog(name='test', **quantities)
    # Check init values
    assert_equal(c.name, 'test')
    for k, v in quantities.items():
        assert_equal(c[k], v)
    assert_equal(c.size, len(quantities['ra']))
    c.__str__()
    c._repr_html_()
    assert_raises(KeyError, c.__getitem__, 'unknown')
    c.mt_input = ClData({'test':[1, 1]})
    c._repr_html_()
    # test repeated ids
    with pytest.warns(None) as record:
        c0 = ClCatalog(name='test', id=['a', 'a'], z=[0, 0])
    assert f'{record._list[0].message}'== \
            'Repeated ID\'s in id column, adding suffix _r# to them.'
    assert_equal(c0['id'], ['a_r1', 'a_r2'])
    del c0
    # Check init match values
    empty_list = np.array([None for i in range(c.size)], dtype=np.ndarray)
    for i in range(c.size):
        empty_list[i] = []
    for n in ('self', 'other'):
        assert all(c[f'mt_{n}']==None)
        assert_equal(c[f'mt_multi_{n}'], empty_list)
    c.cross_match()
    assert all(c['mt_cross']==None)
    # Check ind2inds
    assert_equal(c.ids2inds(['b', 'a']), [1, 0])
    # Check resolve multiple
    c['mt_multi_self'][0] = ['x', 'x']
    c.remove_multiple_duplicates()
    assert_equal(c['mt_multi_self'][0], ['x'])
    # Check cross match
    c.data['mt_self'] = ['a', 'b']
    c.data['mt_other'] = ['a', 'c']
    c.cross_match()
    assert_equal(c['mt_cross'], ['a', None])
    # Check fails
    assert_raises(ValueError, ClCatalog, name='test', ra=[0, 0], dec=[0])
    # Check create ids
    c = ClCatalog('test', ra=[0, 0, 0])
    assert_equal(c['id'], ['0', '1', '2'])
    # Check inexistent mask
    assert_raises(ValueError, c.get_matching_mask, 'made up mask')
    # Reading function
    assert_raises(TypeError, ClCatalog.read, 'demo/cat1.fits', tags={'id': 'ID'})
    assert_raises(KeyError, ClCatalog.read, 'demo/cat1.fits', 'test')
    assert_raises(KeyError, ClCatalog.read, 'demo/cat1.fits', 'test', tags={'id': 'ID2'})
    assert_raises(ValueError, ClCatalog.read, 'demo/cat1.fits', 'test', tags='x')
    c = ClCatalog.read('demo/cat1.fits', 'test', tags={'id': 'ID'})
    c.write('cat1_with_header.fits', overwrite=True)
    c_read = ClCatalog.read_full('cat1_with_header.fits')
    os.system('rm -f cat1_with_header.fits')
    # Check add members
    c = ClCatalog(name='test', **{'id': ['a', 'b'], 'ra': [10, 20],
                                  'dec': [20, 30], 'z': [0.5, 0.6]})
    mem_dat = {'id': ['m_a1', 'm_a2', 'm_b1', 'm_c1'], 'id_cluster': ['a', 'a', 'b', 'c']}
    c.add_members(**mem_dat)
    mem = MemCatalog('mem', **mem_dat)
    c.add_members(members_catalog=mem, **mem_dat)
    assert_raises(TypeError, c.add_members, members_catalog={})
    # Check read members
    c.read_members('demo/cat1_mem.fits', tags={'id':'ID', 'id_cluster':'ID_CLUSTER'})
    # Check raw function
    c_raw = c.raw()
    assert_equal(c_raw.members, None)
    assert_equal(c_raw.leftover_members, None)
    # Del members
    c.remove_members()
    assert_equal(c.members, None)
    assert_equal(c.leftover_members, None)

def test_memcatalog():
    quantities = {'id': ['a', 'b'], 'ra': [10, 20], 'dec': [20, 30],
        'z': [0.5, 0.6], 'id_cluster': ['c1', 'c1']}
    c = MemCatalog(name='test', **quantities)
    # Check init values
    assert_equal(c.name, 'test')
    for k, v in quantities.items():
        assert_equal(c[k], v)
        assert_equal(c.get(k), v)
    assert_equal(c.size, len(quantities['ra']))
    c.__str__()
    c._repr_html_()
    assert_raises(KeyError, c.__getitem__, 'unknown')
    # Check ind2inds
    assert_equal(c.ids2inds(['b', 'a']), [1, 0])
    # Check init match values
    c._init_match_vals()
    empty_list = np.array([None for i in range(c.size)], dtype=np.ndarray)
    for i in range(c.size):
        empty_list[i] = []
    for n in ('self', 'other'):
        assert all(c[f'mt_{n}']==None)
        assert_equal(c[f'mt_multi_{n}'], empty_list)
    c.cross_match()
    # test mt col remains
    mt_self = [0, 1]
    c['mt_self'] = mt_self
    assert_equal(c['dec', ]['mt_self'], mt_self)
    assert all(c['mt_cross']==None)
    # Check resolve multiple
    c['mt_multi_self'][0] = ['x', 'x']
    c.remove_multiple_duplicates()
    assert_equal(c['mt_multi_self'][0], ['x'])
    # Check cross match
    c.data['mt_self'] = ['a', 'b']
    c.data['mt_other'] = ['a', 'c']
    c.cross_match()
    assert_equal(c['mt_cross'], ['a', None])
    # Check fails
    assert_raises(ValueError, MemCatalog, name='test', id=[0, 0])
    assert_raises(ValueError, MemCatalog, name='test', id=[0, 0], id_cluster=[0])
    assert_raises(ValueError, MemCatalog, name='test', id=[0, 0], id_cluster=[0], tags='x')
    # Check create ids
    c = MemCatalog('test', ra=[0, 0, 0], id_cluster=[0, 0, 0])
    assert_equal(c['id'], ['0', '1', '2'])
    # Check inexistent mask
    assert_raises(ValueError, c.get_matching_mask, 'made up mask')
    # Reading function
    assert_raises(KeyError, MemCatalog.read, 'demo/cat1_mem.fits', 'test')
    assert_raises(KeyError, MemCatalog.read, 'demo/cat1_mem.fits', 'test', tags={'id':'ID2'})
    assert_raises(ValueError, MemCatalog.read, 'demo/cat1_mem.fits', 'test', tags={'id':'ID'})
    c = MemCatalog.read('demo/cat1_mem.fits', 'test', tags={'id':'ID', 'id_cluster':'ID_CLUSTER'})
