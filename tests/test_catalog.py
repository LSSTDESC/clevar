from clevar import ClCatalog, MemCatalog, ClData
from clevar.catalog import Catalog
import numpy as np
from numpy.testing import assert_raises, assert_allclose, assert_equal
import os

def test_catalog():
    quantities = {'id': ['a', 'b'], 'ra': [10, 20], 'dec': [20, 30], 'z': [0.5, 0.6]}
    c = Catalog(name='test', **quantities)
    for k, v in quantities.items():
        assert_equal(c[k], v)
        assert_equal(c[:1][k], v[:1])
        assert_equal(c.get(k), v)
    assert_equal(c.get('ra2'), None)
    assert_equal(len(c), len(quantities['ra']))
    del c['ra']
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
    assert_raises(ValueError, ClCatalog.read, 'demo/cat1.fits', id='ID')
    assert_raises(ValueError, ClCatalog.read, 'demo/cat1.fits', 'test')
    assert_raises(KeyError, ClCatalog.read, 'demo/cat1.fits', 'test', id='ID2')
    c = ClCatalog.read('demo/cat1.fits', 'test', id='ID')
    c.write('cat1_with_header.fits', overwrite=True)
    c_read = ClCatalog.read('cat1_with_header.fits', id='ID')
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
    c.read_members('demo/cat1_mem.fits', id='ID', id_cluster='ID_CLUSTER')
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
    # Check create ids
    c = MemCatalog('test', ra=[0, 0, 0], id_cluster=[0, 0, 0])
    assert_equal(c['id'], ['0', '1', '2'])
    # Check inexistent mask
    assert_raises(ValueError, c.get_matching_mask, 'made up mask')
    # Reading function
    assert_raises(ValueError, MemCatalog.read, 'demo/cat1_mem.fits', 'test')
    assert_raises(KeyError, MemCatalog.read, 'demo/cat1_mem.fits', 'test', id='ID2')
    assert_raises(ValueError, MemCatalog.read, 'demo/cat1_mem.fits', 'test', id='ID')
    c = MemCatalog.read('demo/cat1_mem.fits', 'test', id='ID', id_cluster='ID_CLUSTER')
