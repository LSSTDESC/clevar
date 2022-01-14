"""@file catalog.py
The ClCatalog and improved Astropy table classes
"""
import warnings
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
    def _init_match_vals(self, overwrite=False):
        """Fills self.match with default values

        Paramters
        ---------
        overwrite: bool
            Overwrite values of pre-existing columns.
        """
        for col in ('mt_self', 'mt_other', 'mt_multi_self', 'mt_multi_other'):
            if overwrite or col not in self.colnames:
                self[col] = None
                if col in ('mt_multi_self', 'mt_multi_other'):
                    for i in range(self.size):
                        self[col][i] = []
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
            if col in ('mt_self', 'mt_other', 'mt_cross'):
                out[col] = [c if c else '' for c in self[col]]
            elif col in ('mt_multi_self', 'mt_multi_other'):
                out[col] = [','.join(c) if c else '' for c in self[col]]
            else:
                out[col] = self[col]
        out.write(filename, overwrite=overwrite)
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
        mt_cols = ('mt_self', 'mt_other', 'mt_cross', 'mt_multi_self', 'mt_multi_other')
        out = self(name, **kwargs_info,
            **{k:data[v] for k, v in columns.items()
                if k.lower() not in mt_cols})
        # matching cols
        for k, v in columns.items():
            kl = k.lower()
            if kl in mt_cols: # matching cols
                col = np.array(data[v], dtype=str)
                if kl in ('mt_self', 'mt_other', 'mt_cross'):
                    out[kl] = np.array(col, dtype=np.ndarray)
                    out[kl][out[kl]==''] = None
                if kl in ('mt_multi_self', 'mt_multi_other'):
                    out[kl] = [c.split(',') if len(c)>0 else [] for c in col]
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
    members: MemCatalog
        Catalog of members associated to the clusters
    leftover_members: MemCatalog
        Catalog of members not associated to the clusters
    """
    def __init__(self, name, **kwargs):
        self.members = None
        self.leftover_members = None
        radius_unit = kwargs.pop('radius_unit', None)
        mt_input = kwargs.pop('mt_input', None)
        members = kwargs.pop('members', None)
        members_warning = kwargs.pop('members_warning', True)
        Catalog.__init__(self, name, **kwargs)
        self.radius_unit = radius_unit
        self.mt_input = mt_input
        if members is not None:
            self.add_members(members_catalog=members,
                             members_warning=members_warning)
    def _repr_html_(self):
        print_data = ClData()
        show_data_cols = [c for c in self.colnames if c!='SkyCoord']
        for col in show_data_cols:
            print_data[col] = self.data[col]
        table = print_data._repr_html_()
        if self.mt_input is not None:
            for col in self.mt_input.colnames:
                print_data[col] = self.mt_input[col]
            table = print_data._repr_html_()
            table = table.split('<thead><tr><th')
            style = 'text-align:left; background-color:grey; color:white'
            table.insert(1, (
                f''' colspan={len(show_data_cols)}></th>'''
                f'''<th colspan={len(self.mt_input.colnames)} style='{style}'>mt_input</th>'''
                '''</tr></thread>''')
            )
            table='<thead><tr><th'.join(table)

        return f'<b>{self.name}</b><br>Radius unit: {self.radius_unit}<br>{table}'
    def _add_values(self, **columns):
        """Add values for all attributes. If id is not provided, one is created"""
        Catalog._add_values(self, **columns)
        self.radius_unit = columns.pop('radius_unit', None)
        self._init_match_vals()
    def __getitem__(self, item):
        data = self.data[item]
        if isinstance(item, (str, int, np.int64)):
            return data
        else:
            mt_input = self.mt_input
            if (mt_input is not None and not
                    (isinstance(item, (tuple, list)) and item
                    and all(isinstance(x, str) for x in item))):
                # Check if item is not a tuple or list of strings
                mt_input = mt_input[item]
            return ClCatalog(
                name=self.name, labels=self.labels, radius_unit=self.radius_unit,
                **{c:data[c] for c in data.colnames},
                mt_input=mt_input, members=self.members, members_warning=False)
    def raw(self):
        """
        Get a copy of the catalog without members.
        """
        if self.members is not None:
            out = ClCatalog(
                name=self.name, labels=self.labels, radius_unit=self.radius_unit,
                **{c:self.data[c] for c in self.data.colnames},
                mt_input=self.mt_input)
        else:
            out = self
        return out
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

    def add_members(self, members_consistency=True, members_warning=True,
                    members_catalog=None, **kwargs):
        """
        Add members to clusters

        Parameters
        ----------
        members_consistency: bool
            Require that all input members belong to this cluster catalog.
        members_warning: bool
            Raise warning if members are do not belong to this cluster catalog,
            and save them in leftover_members attribute.
        members_catalog: clevar.MemCatalog, None
            Members catalog if avaliable.
        **kwargs: keyword arguments
            Arguments to initialize member catalog if members_catalog=None. For details, see:
            https://lsstdesc.org/clevar/compiled-examples/catalogs.html#adding-members-to-cluster-catalogs
        """
        self.leftover_members = None # clean any previous mem info
        if members_catalog is None:
            members = MemCatalog('members', **kwargs)
        elif isinstance(members_catalog, MemCatalog):
            members = members_catalog[:]
            if len(kwargs)>0:
                warnings.warn(f'leftover input arguments ignored: {kwargs.keys()}')
        else:
            raise TypeError(
                f'members_catalog type is {type(members_catalog)},'
                ' it must be a MemCatalog object.')
        members['ind_cl'] = [self.id_dict.get(ID, -1) for ID in members['id_cluster']]
        if members_consistency:
            mem_in_cl = members['ind_cl']>=0
            if not all(mem_in_cl):
                if members_warning:
                    warnings.warn(
                        'Some galaxies were not members of the cluster catalog.'
                        ' They are stored in leftover_members attribute.')
                    self.leftover_members = members[~mem_in_cl]
                    self.leftover_members.name = 'leftover members'
            members = members[mem_in_cl]
        self.members = members
    def read_members(self, filename, members_consistency=True,
                     members_warning=True, **kwargs):
        """Read members catalog from fits file.

        Parameters
        ----------
        filename: str
            Input file.
        members_consistency: bool
            Require that all input members belong to this cluster catalog.
        members_warning: bool
            Raise warning if members are do not belong to this cluster catalog,
            and save them in leftover_members attribute.
        labels: dict
            Labels of data columns for plots (default vals from file header).
        **kwargs: keyword argumens
            All columns to be added must be passes with named argument,
            the name is used in the Catalog data and the value must
            be the column name in your input file (ex: z='REDSHIFT').
        """
        self.add_members(
            members_catalog=MemCatalog.read(filename, 'members', **kwargs),
            members_consistency=members_consistency, members_warning=members_warning)
    def remove_members(self):
        """
        Remove member catalogs.
        """
        self.members = None
        self.leftover_members = None


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
    def __getitem__(self, item):
        data = self.data[item]
        if isinstance(item, (str, int, np.int64)):
            return data
        else:
            return MemCatalog(name=self.name, labels=self.labels,
                              **{c:data[c] for c in data.colnames})
