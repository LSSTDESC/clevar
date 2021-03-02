"""@file catalog.py
The Catalog and improved Astropy table classes
"""
import numpy as np
from astropy.table import Table as APtable
from astropy.coordinates import SkyCoord
from astropy import units as u

class ClData(APtable):
    """
    Data: A data object. It behaves as an astropy table but case independent.

    Parameters
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

class Catalog():
    def __init__(self, **kwargs):
        self.data = ClData()
        self.match = ClData()
        self.mt_input = None
        self.size = None
        self.id_dict = {}
        if len(kwargs)>0:
            self._add_values(**kwargs)
    def _add_values(self, **columns):
        """Add values for all attributes"""
        # Check all columns have same size
        names = [n for n in columns]
        sizes = [len(v) for v in columns.values()]
        if self.size is None:
            self.size = sizes[0]
        tab = " "*12
        if any(self.size!=s for s in sizes):
            raise TypeError(f"Column sizes inconsistent:\n"+
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
        self.match['self'] = None
        self.match['other'] = None
        self.match['multi_self']  = None
        self.match['multi_other'] = None
        for i in range(self.size):
            self.match['multi_self'][i] = []
            self.match['multi_other'][i] = []
    def ids2inds(self, ids):
        return np.array([self.id_dict[i] for i in ids])
    def remove_multiple_duplicates(self):
        for i in range(self.size):
            for col in ('multi_self', 'multi_other'):
                if self.match[col][i]:
                    self.match[col][i] = list(set(self.match[col][i]))
    def cross_match(self):
        self.match['cross'] = self.match['self'][self.match['self']==self.match['other']]
