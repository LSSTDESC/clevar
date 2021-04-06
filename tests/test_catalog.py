from clevar import Catalog
import numpy as np
from numpy.testing import assert_raises, assert_allclose, assert_equal

def test_adding_quantities():
    quantities = {'id': ['a', 'b'], 'ra': [10, 20], 'dec': [20, 30], 'z': [0.5, 0.6]}
    c = Catalog(name='test', **quantities)
    # Check init values
    assert_equal(c.name, 'test')
    for k, v in quantities.items():
        assert_equal(c[k], v)
    assert_equal(c.size, len(quantities['ra']))
    c.__str__()
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
    quantities_fail = {'id': ['1'], 'ra': [12, 30], 'dec': [20], 'z': [0.5]}
    assert_raises(ValueError, Catalog, name='test', ra=[0, 0], dec=[0])
    # Check create ids
    c = Catalog('test', ra=[0, 0, 0])
    assert_equal(c['id'], ['0', '1', '2'])
    # Check inexistent mask
    assert_raises(ValueError, c.get_matching_mask, 'made up mask')
