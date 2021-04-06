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
    'mt_cross': lambda match: match['mt_cross']!=None,
    'mt_self': lambda match: match['mt_self']!=None,
    'mt_other': lambda match: match['mt_other']!=None,
    'mt_multi_self': lambda match: veclen(match['mt_multi_self'])>0,
    'mt_multi_other': lambda match: veclen(match['mt_multi_other'])>0,
    'mt_multi_join': lambda match: (veclen(match['mt_multi_self'])>0)+(veclen(match['mt_multi_other'])>0),
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
        Mathing data (mt_self, mt_other, mt_cross, mt_multi_self, mt_multi_other)
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
        self.mt_input = None
        self.size = None
        self.id_dict = {}
        self.radius_unit = None
        if len(kwargs)>0:
            self._add_values(**kwargs)
    def __setitem__(self, item, value):
        self.data[item] = value
    def __getitem__(self, item):
        return self.data[item]
    def __delitem__(self, item):
        del self.data[item]
    def __str__(self):
        return f'{self.name}:\n{self.data.__str__()}'
    def _repr_html_(self):
        return f'<b>{self.name}</b><br>{self.data._repr_html_()}'
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
        for k, v in columns.items():
            if k!='id':
                self.data[k] = v
        if 'ra' in self.data.colnames and 'dec' in self.data.colnames:
            self.data['SkyCoord'] = SkyCoord(self['ra']*u.deg, self['dec']*u.deg, frame='icrs')
        self.id_dict = {i:ind for ind, i in enumerate(self['id'])}
        self._init_match_vals()
    def _init_match_vals(self):
        """Fills self.match with default values"""
        self.data['mt_self'] = None
        self.data['mt_other'] = None
        self.data['mt_multi_self']  = None
        self.data['mt_multi_other'] = None
        for i in range(self.size):
            self.data['mt_multi_self'][i] = []
            self.data['mt_multi_other'][i] = []
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
            for col in ('mt_multi_self', 'mt_multi_other'):
                if self[col][i]:
                    self.data[col][i] = list(set(self[col][i]))
    def cross_match(self):
        """Makes cross matches, requires unique matches to be done first."""
        self.data['mt_cross'] = None
        cross_mask = self['mt_self']==self['mt_other']
        self.data['mt_cross'][cross_mask] = self['mt_self'][cross_mask]
    def get_matching_mask(self, matching_type):
        if matching_type not in _matching_mask_funcs:
            raise ValueError(f'matching_type ({matching_type}) must be in {list(_matching_mask_funcs.keys())}')
        return _matching_mask_funcs[matching_type](self.data)
