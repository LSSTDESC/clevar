"""@file catalog.py
The ClCatalog and improved Astropy table classes
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
            missing_cols = [i for i in item.split(',') if i not in name_dict]
            if len(missing_cols)>0:
                missing_cols = ', '.join(missing_cols)
                raise ValueError(f"Columns '{missing_cols}' not found in {name_dict.keys()}")
            item = ','.join([name_dict[i] for i in item.split(',')])
        out = APtable.__getitem__(self, item)
        return out
    def get(self, key, default=None):
        """
        Return the column for key if key is in the dictionary, else default
        """
        return self[key] if key in self.colnames else default
_matching_mask_funcs = {
    'cross': lambda match: match['mt_cross']!=None,
    'self': lambda match: match['mt_self']!=None,
    'other': lambda match: match['mt_other']!=None,
    'multi_self': lambda match: veclen(match['mt_multi_self'])>0,
    'multi_other': lambda match: veclen(match['mt_multi_other'])>0,
    'multi_join': lambda match: (veclen(match['mt_multi_self'])>0)+(veclen(match['mt_multi_other'])>0),
}
class ClCatalog():
    """
    Object to handle catalogs

    Attributes
    ----------
    name: str
        ClCatalog name
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
        data = self[[c for c in self.data.colnames if c!='SkyCoord']]
        return f'<b>{self.name}</b><br>Radius unit: {self.radius_unit}<br>{data._repr_html_()}'
    def _add_values(self, **columns):
        """Add values for all attributes. If id is not provided, one is created"""
        self.radius_unit = columns.pop('radius_unit', None)
        # Check all columns have same size
        names = [n for n in columns]
        sizes = [len(v) for v in columns.values()]
        if self.size is None:
            self.size = sizes[0]
        tab = " "*12
        if any(self.size!=s for s in sizes):
            raise ValueError(f"Column sizes inconsistent:\n"+
                f"{tab}{'ClCatalog':10}: {self.size:,}\n"+
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
    def _add_ftpt_mask(self, ftpt, maskname):
        """
        Adds a mask based on the cluster position relative to a footprint.
        It also considers zmax values on the footprint if available.

        Parameters
        ----------
        ftpt: clevar.mask.Footprint object
            Footprint
        maskname: str
            Name of mask to be added
        """
        self[f'ft_{maskname}'] = ftpt.zmax_masks_from_footprint(self['ra'], self['dec'],
            self['z'] if 'z' in self.data.colnames else 1e-10)
    def add_ftpt_masks(self, ftpt_self, ftpt_other):
        """
        Add masks based on the cluster position relative to both footprints.
        It also considers zmax values on the footprint if available.

        Parameters
        ----------
        ftpt_self: clevar.mask.Footprint object
            Footprint of this catalog
        ftpt_other: clevar.mask.Footprint object
            Footprint of the other catalog
        """
        self._add_ftpt_mask(ftpt_self, 'self')
        self._add_ftpt_mask(ftpt_other, 'other')
    def add_ftpt_coverfrac(self, ftpt, aperture, aperture_unit, cosmo=None):
        """
        Computes and adds a cover fraction value. It considers zmax and detection fraction
        when available in the footprint.

        Parameters
        ----------
        ftpt: clevar.mask.Footprint object
            Footprint used to compute the coverfration
        ftpt_other: clevar.mask.Footprint object
            Footprint of the other catalog
        raduis: float
            Radial aperture to compute the coverfraction
        aperture_unit: float
            Unit of radial aperture
        cosmo: clevar.Cosmology object
            Cosmology object for when aperture has physical units
        """
        num = f'{aperture}'
        num = f'{aperture:.2f}' if len(num)>6 else num
        self[f'cf_{num}_{aperture_unit}'] = [
            ftpt._get_coverfrac(c['SkyCoord'], c['z'], aperture, aperture_unit, cosmo=cosmo)
            for c in self]
    def add_ftpt_coverfrac_nfw2D(self, ftpt, aperture, aperture_unit, cosmo=None):
        """
        Computes and adds a cover fraction value weighted by a nfw 2D flatcore window function.
        It considers zmax and detection fraction when available in the footprint.

        Parameters
        ----------
        ftpt: clevar.mask.Footprint object
            Footprint used to compute the coverfration
        ftpt_other: clevar.mask.Footprint object
            Footprint of the other catalog
        raduis: float
            Radial aperture to compute the coverfraction
        aperture_unit: float
            Unit of radial aperture
        cosmo: clevar.Cosmology object
            Cosmology object for physical and angular convertions
        """
        num = f'{aperture}'
        num = f'{aperture:.2f}' if len(num)>6 else num
        self[f'cf_nfw_{num}_{aperture_unit}'] = [
            ftpt._get_coverfrac_nfw2D(c['SkyCoord'], c['z'], c['radius'], self.radius_unit,
                                      aperture, aperture_unit, cosmo=cosmo)
            for c in self]
    def save_match(self, filename, overwrite=False):
        """
        Saves the matching results of one catalog

        Parameters
        ----------
        filename: str
            Name of file
        overwrite: bool
            Overwrite saved files
        """
        out = ClData()
        out['id'] = self['id']
        for col in ('mt_self', 'mt_other'):
            out[col] = [c if c else '' for c in self[col]]
        for col in ('mt_multi_self', 'mt_multi_other'):
            out[col] = [','.join(c) if c else '' for c in self[col]]
        out.write(filename, overwrite=overwrite)
    def load_match(self, filename):
        """
        Load matching results to catalogs

        Parameters
        ----------
        filename: str
            Name of file with matching results
        """
        mt = ClData.read(filename)
        for col in ('mt_self', 'mt_other'):
            self[col] = np.array([c if c!='' else None for c in mt[col]], dtype=np.ndarray)
        for col in ('mt_multi_self', 'mt_multi_other'):
            self[col] = np.array([None for c in mt[col]], dtype=np.ndarray)
            for i, c in enumerate(mt[col]):
                if len(c)>0:
                    self[col][i] = c.split(',')
                else:
                    self[col][i] = []
        self.cross_match()
    def save_footprint_quantities(self, filename, overwrite=False):
        """
        Saves the matching results of one catalog

        Parameters
        ----------
        filename: str
            Name of file
        overwrite: bool
            Overwrite saved files
        """
        out = self[['id']+[c for c in self.data.colnames if c[:2] in ('ft', 'cf')]]
        out.write(filename, overwrite=overwrite)
    def load_footprint_quantities(self, filename):
        """
        Load matching results to catalogs

        Parameters
        ----------
        filename: str
            Name of file with matching results
        """
        ftq = ClData.read(filename)
        for col in ftq.colnames:
            if col!='id':
                self[col] = ftq[col]
