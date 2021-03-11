from clevar import Catalog
import numpy as np
from numpy.testing import assert_raises, assert_allclose, assert_equal

def test_adding_quantities():
    quantities = {'id': ['1'], 'ra': [12], 'dec': [20], 'z': [0.5]}
    c = Catalog(name='test', **quantities)
    # Check init values
    assert_equal(c.name, 'test')
    for k, v in quantities.items():
        assert_equal(c.data[k], v)
    assert_equal(c.size, len(quantities['ra']))
    # Check init match values
    empty_list = np.array([None for i in range(c.size)], dtype=np.ndarray)
    for i in range(c.size):
        empty_list[i] = []
    for n in ('self', 'other'):
        assert all(c.match[n]==None)
        assert_equal(c.match[f'multi_{n}'], empty_list)
    for i in range(0):
            self.match['multi_self'][i] = []
            self.match['multi_other'][i] = []
