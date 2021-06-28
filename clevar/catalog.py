"""@file catalog.py
The ClCatalog and improved Astropy table classes
"""
import numpy as np
from astropy.table import Table as APtable
from astropy.coordinates import SkyCoord
from astropy import units as u

from .utils import veclen, none_val

class ClData(APtable):
    """
    ClData: A data object. It behaves as an astropy table but case independent.

    Attributes
    ----------
    meta: dict
        Dictionary with metadata for this object
    namedict: dict
        Dictionary for making ClData case insensitive

    Same as astropy tables
    """
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        *args, **kwargs: Same used for astropy tables
        """
        self.namedict = {}
        APtable.__init__(self, *args, **kwargs)
    def __getitem__(self, item):
        """
        To make case insensitive
        """
        if isinstance(item, str):
            item = self.namedict.get(item.lower(), item)
        return APtable.__getitem__(self, item)
    def __setitem__(self, item, value):
        """
        To make case insensitive
        """
        if isinstance(item, str):
            item = self.namedict.get(item.lower(), item)
            self.namedict[item.lower()] = item
        APtable.__setitem__(self, item, value)
    def get(self, key, default=None):
        """
        Return the column for key if key is in the dictionary, else default
        """
        return self[key] if key.lower() in self.namedict else default
    def _repr_html_(self):
        return APtable._repr_html_(self[[c for c in self.colnames if c!='SkyCoord']])
    @classmethod
    def read(self, filename, **kwargs):
        tab = APtable.read(filename, **kwargs)
        out = ClData(meta=tab.meta)
        for c in tab.colnames:
            out[c] = tab[c]
        return out
    def _check_cols(self, columns):
        """
        Checks if required columns exist

        Parameters
        ----------
        columns: list
            Names of columns to find.
        """
        missing = [f"'{k}'" for k in columns
                   if k.lower() not in self.namedict]
        if len(missing)>0:
            missing = ", ".join(missing)
            raise KeyError(
                f"Column(s) '{missing}' not found "
                "in catalog {data.colnames}")

_matching_mask_funcs = {
    'cross': lambda match: match['mt_cross']!=None,
    'self': lambda match: match['mt_self']!=None,
    'other': lambda match: match['mt_other']!=None,
    'multi_self': lambda match: veclen(match['mt_multi_self'])>0,
    'multi_other': lambda match: veclen(match['mt_multi_other'])>0,
    'multi_join': lambda match: (veclen(match['mt_multi_self'])>0)+(veclen(match['mt_multi_other'])>0),
}

class Catalog():
    """
    Parent object to handle catalogs.

    Attributes
    ----------
    name: str
        ClCatalog name
    data: ClData
        Main catalog data (ex: id, ra, dec, z). Fixed values.
        Matching data (mt_self, mt_other, mt_cross, mt_multi_self, mt_multi_other)
    mt_input: object
        Contains the necessary inputs for the match (added by Match objects)
    size: int
        Number of objects in the catalog
    id_dict: dict
        Dictionary of indicies given the object id
    labels: dict
        Labels of data columns for plots
    """
    def __init__(self, name, **kwargs):
        self.name = name
        self.data = ClData()
        self.mt_input = None
        self.size = None
        self.id_dict = {}
        self.labels = {}
        self.colnames = []
        if len(kwargs)>0:
            self._add_values(**kwargs)
    def __setitem__(self, item, value):
        if isinstance(item, str):
            if item[:3]!='mt_':
                self.labels[item] = self.labels.get(item, f'{item}_{{{self.name}}}')
            if item not in self.colnames:
                self.colnames.append(item)
        self.data[item] = value
    def __getitem__(self, item):
        data = self.data[item]
        if isinstance(item, (str, int, np.int64)):
            return data
        else:
            return Catalog(name=self.name, labels=self.labels,
                **{c:data[c] for c in data.colnames})
    def __len__(self):
        return self.size
    def __delitem__(self, item):
        del self.data[item]
    def __str__(self):
        return f'{self.name}:\n{self.data.__str__()}'
    def _repr_html_(self):
        return f'<b>{self.name}</b><br>{self.data._repr_html_()}'
    def _add_values(self, **columns):
        """Add values for all attributes. If id is not provided, one is created"""
        self.labels.update(columns.pop('labels', {}))
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
            self['id'] = np.array(range(self.size), dtype=str)
        else:
            self['id'] = np.array(columns['id'], dtype=str)
        for k, v in columns.items():
            if k!='id':
                self[k] = v
        if 'ra' in self.data.colnames and 'dec' in self.data.colnames:
            self['SkyCoord'] = SkyCoord(self['ra']*u.deg, self['dec']*u.deg, frame='icrs')
        self.id_dict = {i:ind for ind, i in enumerate(self['id'])}
    def _init_match_vals(self):
        """Fills self.match with default values"""
        self['mt_self'] = None
        self['mt_other'] = None
        self['mt_multi_self']  = None
        self['mt_multi_other'] = None
        for i in range(self.size):
            self['mt_multi_self'][i] = []
            self['mt_multi_other'][i] = []
    def ids2inds(self, ids, missing=None):
        """Returns the indicies of objects given an id list.

        Parameters
        ----------
        ids: list
            List of object ids
        missing: None
            Value added to position of missing id.
        """
        return np.array([self.id_dict.get(i, missing) for i in ids])
    def remove_multiple_duplicates(self):
        """Removes duplicates in multiple match columns"""
        for i in range(self.size):
            for col in ('mt_multi_self', 'mt_multi_other'):
                if self[col][i]:
                    self[col][i] = list(set(self[col][i]))
    def cross_match(self):
        """Makes cross matches, requires unique matches to be done first."""
        self['mt_cross'] = None
        cross_mask = self['mt_self']==self['mt_other']
        self['mt_cross'][cross_mask] = self['mt_self'][cross_mask]
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
    def add_ftpt_coverfrac(self, ftpt, aperture, aperture_unit, window='flat',
        cosmo=None, colname=None):
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
        window: str
            Window to weight corverfrac. Options are: flat, nfw2D (with flat core)
        cosmo: clevar.Cosmology object
            Cosmology object for when aperture has physical units. Required if nfw2D window used.
        colname: str, None
            Name of coverfrac column.
        """
        num = f'{aperture}'
        num = f'{aperture:.2f}' if len(num)>6 else num
        window_cfg = {
            'flat': {
                'func': ftpt._get_coverfrac,
                'get_args': lambda c: [c['SkyCoord'], c['z'], aperture, aperture_unit],
                'colname': f'{num}_{aperture_unit}',
            },
            'nfw2D': {
                'func': ftpt._get_coverfrac_nfw2D,
                'get_args': lambda c: [c['SkyCoord'], c['z'], c['radius'],
                                   self.radius_unit, aperture, aperture_unit],
                'colname': f'nfw_{num}_{aperture_unit}',
            },
        }.get(window, None)
        if window_cfg is None:
            raise ValueError(f"window ({window}) must be either 'flat' of 'nfw2D'")
        self[f"cf_{none_val(colname, window_cfg['colname'])}"] = [
            window_cfg['func'](*window_cfg['get_args'](c), cosmo=cosmo)
            for c in self]
    def get(self, key, default=None):
        """
        Return the column for key if key is in the dictionary, else default
        """
        return ClData.get(self.data, key, default)
    def write(self, filename, add_header=True, overwrite=False):
        """
        Write catalog.

        Parameters
        ----------
        filename: str
            Name of file
        add_header: bool
            Saves catalog name and labels.
        overwrite: bool
            Overwrite saved files
        """
        out = ClData()
        if add_header:
            out.meta['name'] = self.name
            out.meta.update({f'hierarch label_{k}':v for k, v in self.labels.items()})
        for col in self.colnames:
            if col in ('mt_self', 'mt_other'):
                out[col] = [c if c else '' for c in self[col]]
            elif col in ('mt_multi_self', 'mt_multi_other'):
                out[col] = [','.join(c) if c else '' for c in self[col]]
            else:
                out[col] = self[col]
        out.write(filename, overwrite=overwrite)
    def _fmt_loaded_mt(self):
        """
        Format matching columns read from file.
        """
        for col in self.colnames:
            if col in ('mt_self', 'mt_other'):
                self[col] = np.array(self[col], dtype=np.ndarray)
                in_vals = np.array(self[col], dtype=str)
                none_vals = in_vals==''
                self[col][none_vals] = None
                self[col][~none_vals] = in_vals[~none_vals]
            if col in ('mt_multi_self', 'mt_multi_other'):
                self[col] = np.array(self[col], dtype=np.ndarray)
                for i, c in enumerate(self[col]):
                    self[col][i] = c.split(',') if len(c)>0 else []

    @classmethod
    def _read(self, data, name=None, **kwargs):
        """Does the main execution for reading catalog.

        Parameters
        ----------
        data: clevar.ClData
            Input data.
        name: str, None
            Catalog name, if none reads from file.
        labels: dict
            Labels of data columns for plots (default vals from file header).
        **kwargs: keyword argumens
            All columns to be added must be passes with named argument,
            the name is used in the Catalog data and the value must
            be the column name in your input file (ex: z='REDSHIFT').
        """
        columns = {k: v for k, v in kwargs.items()
            if k not in ('labels', 'radius_unit')}
        if len(columns)==0:
            raise ValueError('At least one column must be provided.')
        # Catalog name
        if name is None:
            if 'NAME' not in data.meta:
                raise ValueError('Name not found in file, please provide as argument.')
            name = data.meta['NAME']
        # labels and radius unit
        kwargs_info = {
            'labels': {k[6:].lower():v
                for k, v in data.meta.items()
                if k[:6]=='LABEL_'},
        }
        kwargs_info['labels'].update(kwargs.get('labels', {}))
        radius_unit = kwargs.get('radius_unit', data.meta.get('RADIUS_UNIT'))
        kwargs_info.update({} if radius_unit is None else {'radius_unit': radius_unit})
        # Missing cols
        data._check_cols(columns.values())
        # out data
        mt_cols = ('mt_self', 'mt_other', 'mt_multi_self', 'mt_multi_other')
        out = self(name, **kwargs_info,
            **{k:data[v] for k, v in columns.items()
                if k not in mt_cols})
        for k, v in columns.items():
            if k in mt_cols: # matching cols
                out[k] = np.array(data[v], dtype=str)
        out._fmt_loaded_mt()
        return out
    @classmethod
    def read(self, filename, name=None, **kwargs):
        """Read catalog from fits file.

        Parameters
        ----------
        filename: str
            Input file.
        name: str, None
            Catalog name, if none reads from file.
        labels: dict
            Labels of data columns for plots (default vals from file header).
        **kwargs: keyword argumens
            All columns to be added must be passes with named argument,
            the name is used in the Catalog data and the value must
            be the column name in your input file (ex: z='REDSHIFT').
        """
        data = ClData.read(filename)
        return self._read(data, name=name, **kwargs)
    @classmethod
    def read_full(self, filename):
        """Read fits file catalog saved by clevar with all information.
        The catalog must contain name information.

        Parameters
        ----------
        filename: str
            Input file.
        """
        data = ClData.read(filename)
        out = self._read(data, **{c:c for c in data.colnames})
        return out
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
        cols = ['id', 'mt_self', 'mt_other',
            'mt_multi_self', 'mt_multi_other']
        cols += [col for col in self.data.colnames
            if (col[:3]=='mt_' and col not in cols+['mt_cross'])]
        self[cols].write(filename, overwrite=overwrite)
    def load_match(self, filename):
        """
        Load matching results to catalogs

        Parameters
        ----------
        filename: str
            Name of file with matching results
        """
        mt = self.read_full(filename)
        for col in mt.colnames:
            if col!='id':
                self[col] = mt[col]
        self.cross_match()
        print(f' * Total objects:    {self.size:,}')
        print(f' * multiple (self):  {len(self[veclen(self["mt_multi_self"])>0]):,}')
        print(f' * multiple (other): {len(self[veclen(self["mt_multi_other"])>0]):,}')
        print(f' * unique (self):    {len(self[self["mt_self"]!=None]):,}')
        print(f' * unique (other):   {len(self[self["mt_other"]!=None]):,}')
        print(f' * cross:            {len(self[self["mt_cross"]!=None]):,}')
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
        out = self.data[['id']+[c for c in self.data.colnames if c[:2] in ('ft', 'cf')]]
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

class ClCatalog(Catalog):
    """
    Object to handle cluster catalogs.

    Attributes
    ----------
    name: str
        ClCatalog name
    data: ClData
        Main catalog data (ex: id, ra, dec, z). Fixed values.
        Mathing data (mt_self, mt_other, mt_cross, mt_multi_self, mt_multi_other)
    mt_input: object
        Contains the necessary inputs for the match (added by Match objects)
    size: int
        Number of objects in the catalog
    id_dict: dict
        Dictionary of indicies given the cluster id
    radius_unit: str, None
        Unit of the radius column
    labels: dict
        Labels of data columns for plots
    """
    def __init__(self, name, **kwargs):
        radius_unit = kwargs.pop('radius_unit', None)
        Catalog.__init__(self, name, **kwargs)
        self.radius_unit = radius_unit
    def _repr_html_(self):
        return f'<b>{self.name}</b><br>Radius unit: {self.radius_unit}<br>{self.data._repr_html_()}'
    def _add_values(self, **columns):
        """Add values for all attributes. If id is not provided, one is created"""
        Catalog._add_values(self, **columns)
        self.radius_unit = columns.pop('radius_unit', None)
        self._init_match_vals()
    @classmethod
    def read(self, filename, name=None, **kwargs):
        """Read catalog from fits file

        Parameters
        ----------
        filename: str
            Input file.
        name: str, None
            ClCatalog name, if none reads from file.
        labels: dict
            Labels of data columns for plots (default vals from file header).
        radius_unit: str, None
            Unit of the radius column (default read from file).
        **kwargs: keyword argumens
            All columns to be added must be passes with named argument,
            the name is used in the ClCatalog data and the value must
            be the column name in your input file (ex: z='REDSHIFT').
        """
        data = ClData.read(filename)
        return self._read(data, name=name, **kwargs)

class MemCatalog(Catalog):
    """
    Object to handle member catalogs.

    Attributes
    ----------
    name: str
        ClCatalog name
    data: ClData
        Main catalog data (ex: id, id_cluster, pmem). Fixed values.
    mt_input: object
        Contains the necessary inputs for the match (added by Match objects)
    size: int
        Number of objects in the catalog
    id_dict: dict
        Dictionary of indicies given the member id
    id_dict_list: dict
        Dictionary of indicies given the member id, returns list allowing for repeated ids.
    labels: dict
        Labels of data columns for plots
    """
    def __init__(self, name, **kwargs):
        if all('id_cluster'!=n.lower() for n in kwargs):
            raise ValueError("Members catalog must have a 'id_cluster' column!")
        Catalog.__init__(self, name, **kwargs)
    def _add_values(self, **columns):
        """Add values for all attributes. If id is not provided, one is created"""
        Catalog._add_values(self, **columns)
        self['id_cluster'] = np.array(columns['id_cluster'], dtype=str)
        self.id_dict_list = {}
        for ind, i in enumerate(self['id']):
            self.id_dict_list[i] = self.id_dict_list.get(i, [])+[ind]
