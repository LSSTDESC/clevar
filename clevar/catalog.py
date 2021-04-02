"""@file catalog.py
The Catalog and improved Astropy table classes
"""
import numpy as np
from astropy.table import Table as APtable
from astropy.coordinates import SkyCoord
from astropy import units as u

from .utils import veclen

class ClData(APtable):
    """
    ClData: A data object. It behaves as an astropy table but case independent.

    Attributes
    ----------
    meta: dict
        Dictionary with metadata for this object

    Same as astropy tables
    """
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        *args, **kwargs: Same used for astropy tables
        """
        APtable.__init__(self, *args, **kwargs)
    def __getitem__(self, item):
        """
        Makes sure GCData keeps its properties after [] operations are used.
        It also makes all letter casings accepted

        Returns
        -------
        GCData
            Data with [] operations applied
        """
        if isinstance(item, str):
            name_dict = {n.lower():n for n in self.colnames}
            item = item.lower()
            item = ','.join([name_dict[i] for i in item.split(',')])
        out = APtable.__getitem__(self, item)
        return out
    def get(self, key, default=None):
        """
        Return the column for key if key is in the dictionary, else default
        """
        return self[key] if key in self.colnames else default
_matching_mask_funcs = {
    'cross': lambda match: match['cross']!=None,
    'self': lambda match: match['self']!=None,
    'other': lambda match: match['other']!=None,
    'multi_self': lambda match: veclen(match['multi_self'])>0,
    'multi_other': lambda match: veclen(match['multi_other'])>0,
    'multi_join': lambda match: (veclen(match['multi_self'])>0)+(veclen(match['multi_other'])>0),
}
class Catalog():
    """
    Object to handle catalogs

    Attributes
    ----------
    name: str
        Catalog name
    data: ClData
        Main catalog data (ex: id, ra, dec, z). Fixed values.
    data: ClData
        Mathing data (self, other, cross, multi_self, multi_other)
    mt_input: object
        Constains the necessary inputs for the match (added by Match objects)
    size: int
        Number of objects in the catalog
    id_dict: dict
        Dictionary of indicies given the cluster id
    radius_unit: str, None
        Unit of the radius column
    """
    def __init__(self, name, **kwargs):
        self.name = name
        self.data = ClData()
        self.match = ClData()
        self.mt_input = None
        self.size = None
        self.id_dict = {}
        self.radius_unit = None
        if len(kwargs)>0:
            self._add_values(**kwargs)
    def _add_values(self, **columns):
        """Add values for all attributes. If id is not provided, one is created"""
        # Check all columns have same size
        names = [n for n in columns]
        sizes = [len(v) for v in columns.values()]
        if self.size is None:
            self.size = sizes[0]
        tab = " "*12
        if any(self.size!=s for s in sizes):
            raise ValueError(f"Column sizes inconsistent:\n"+
                f"{tab}{'Catalog':10}: {self.size:,}\n"+
                "\n".join([f"{tab}{k:10}: {l:,}" for k, l in zip(names, sizes)])
                )
        if 'id' not in columns:
            self.data['id'] = np.array(range(self.size), dtype=str)
        else:
            self.data['id'] = np.array(columns['id'], dtype=str)
        self.match['id'] = self.data['id']
        for k, v in columns.items():
            if k!='id':
                self.data[k] = v
        if 'ra' in self.data.colnames and 'dec' in self.data.colnames:
            self.data['SkyCoord'] = SkyCoord(self.data['ra']*u.deg, self.data['dec']*u.deg, frame='icrs')
        self.id_dict = {i:ind for ind, i in enumerate(self.data['id'])}
        self._init_match_vals()
    def _init_match_vals(self):
        """Fills self.match with default values"""
        self.match['self'] = None
        self.match['other'] = None
        self.match['multi_self']  = None
        self.match['multi_other'] = None
        for i in range(self.size):
            self.match['multi_self'][i] = []
            self.match['multi_other'][i] = []
    def ids2inds(self, ids):
        """Returns the indicies of objects given an id list.

        Parameters
        ----------
        ids: list
            List of object ids
        """
        return np.array([self.id_dict[i] for i in ids])
    def remove_multiple_duplicates(self):
        """Removes duplicates in multiple match columns"""
        for i in range(self.size):
            for col in ('multi_self', 'multi_other'):
                if self.match[col][i]:
                    self.match[col][i] = list(set(self.match[col][i]))
    def cross_match(self):
        """Makes cross matches, requires unique matches to be done first."""
        self.match['cross'] = None
        cross_mask = self.match['self']==self.match['other']
        self.match['cross'][cross_mask] = self.match['self'][cross_mask]
    def get_matching_mask(self, matching_type):
        if matching_type not in _matching_mask_funcs:
            raise ValueError(f'matching_type ({matching_type}) must be in {list(_matching_mask_funcs.keys())}')
        return _matching_mask_funcs[matching_type](self.match)
