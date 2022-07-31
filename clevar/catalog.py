"""@file catalog.py
The ClCatalog and improved Astropy table classes
"""
import warnings
import numpy as np
from astropy.table import Table as APtable
from astropy.coordinates import SkyCoord
from astropy import units as u

from .utils import (veclen, none_val, NameList, LowerCaseDict, updated_dict,
                    pack_mt_col, unpack_mt_col, pack_mmt_col, unpack_mmt_col)


class ClData(APtable):
    """
    ClData: A data object. It behaves as an astropy table but case independent.

    Attributes
    ----------
    meta: dict
        Dictionary with metadata for this object
    namedict: dict
        Dictionary for making ClData case insensitive
    size: int
        Number of objects in the catalog

    Same as astropy tables
    """

    @property
    def namedict(self):
        return self.__namedict

    @property
    def size(self):
        return len(self)

    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        *args, **kwargs: Same used for astropy tables
        """
        self.__namedict = LowerCaseDict()
        super().__init__(*args, **kwargs)
        for col in self.colnames:
            self.namedict[col] = col

    def __getitem__(self, item):
        """
        To make case insensitive
        """
        # get one column
        if isinstance(item, str):
            return super().__getitem__(self.namedict.get(item, item))

        # get one row
        elif isinstance(item, (int, np.int64)):
            out = super().__getitem__(item)
            out.namedict = self.namedict
            return out

        # Get sub cols
        elif isinstance(item, (tuple, list)) and all(isinstance(x, str) for x in item):
            item_ = NameList(map(lambda i: self.namedict[i], item))

        # Get sub rows
        else:
            item_ = item

        return ClData(super().__getitem__(item_))

    def __setitem__(self, item, value):
        """
        To make case insensitive
        """
        if isinstance(item, str):
            item = self.namedict.get(item, item)
            self.namedict[item] = item
        super().__setitem__(item, value)

    def __delitem__(self, item):
        if isinstance(item, str):
            del self.namedict[item]
        super().__delitem__(item)

    def get(self, key, default=None):
        """
        Return the column for key if key is in the dictionary, else default
        """
        return self[key] if key in self.namedict else default

    def _repr_html_(self):
        return APtable._repr_html_(self[[c for c in self.colnames if c!='SkyCoord']])

    @classmethod
    def read(self, *args, **kwargs):
        return ClData(APtable.read(*args, **kwargs))

    def _check_cols(self, columns):
        """
        Checks if required columns exist

        Parameters
        ----------
        columns: list
            Names of columns to find.
        """
        missing = [f"'{k}'" for k in columns if k not in self.namedict]
        if len(missing)>0:
            missing = ", ".join(missing)
            raise KeyError(
                f"Column(s) '{missing}' not found "
                "in catalog {data.colnames}")


class TagData():
    """
    Parent object to implement tag to catalogs.

    Attributes
    ----------
    data: ClData
        Main catalog data (ex: id, ra, dec, z).
    size: int
        Number of objects in the catalog
    tags: LoweCaseDict
        Tag for main quantities used in matching and plots (ex: id, ra, dec, z)
    default_tags: NameList
        List of keys that generate tags automatically.
    colnames: NameList
        Names of columns in data.
    """

    @property
    def tags(self):
        return self.__tags

    @property
    def data(self):
        return self.__data

    @property
    def size(self):
        return self.data.size

    @property
    def default_tags(self):
        return self.__default_tags

    @property
    def colnames(self):
        return NameList(self.data.colnames)

    def __init__(self, tags=None, default_tags=None, **kwargs):
        if tags is not None and not isinstance(tags, dict):
            raise ValueError('tags must be dict.')
        self.__data = ClData()
        self.__tags = LowerCaseDict(none_val(tags, {}))
        self.__default_tags = NameList(none_val(default_tags, []))
        if len(kwargs)>0:
            self._add_values(**kwargs)
            # make sure columns don't overwrite tags
            if tags is not None:
                for colname, coltag in tags.items():
                    self.tag_column(coltag, colname, skip_warn=True)

    def __setitem__(self, item, value):
        """ Adds items to tags if in default_tags"""
        if isinstance(item, str):
            if item in self.default_tags:
                name = self.tags.get(item, item)
                self.tags[item] = self.data.namedict.get(name, name)
            self.data[item] = value
        else:
            raise ValueError(f'can only set with str item (={item}) argument.')

    def _getitem_base(self, item, DataType, **kwargs):
        """ Base function to also be used by child classes"""

        # Get one row/col
        if isinstance(item, (str, int, np.int64)):
            return self.data[self.tags.get(item, item) if isinstance(item, str)
                             else item]

        # Get sub cols
        if isinstance(item, (tuple, list)) and all(isinstance(x, str) for x in item):
            item_ = NameList(map(lambda i: self.tags.get(i, i), item))
            tags = dict(filter(lambda key_val: key_val[1] in item_, self.tags.items()))

        # Get sub rows
        else:
            item_ = item
            tags = self.tags

        return DataType(tags=tags, data=self.data[item_], **kwargs)

    def __getitem__(self, item):
        return self._getitem_base(item, TagData)

    def __len__(self):
        return self.size

    def __delitem__(self, item):
        del self.data[item]
        if item in self.tags:
            del self.tags[item]

    def __str__(self):
        return self.data.__str__()

    def _prt_tags(self):
        return ', '.join([f'{v}({k})' for k, v in self.tags.items()])

    def _prt_table_tags(self, table):
        tags_inv = {v:f' ({k})' for k, v in self.tags.items() if k!=v}
        table_list = table._repr_html_().split('<tr>')
        new_header = '</th>'.join([c+tags_inv.get(c[4:], '') if c[:4]=='<th>' else c
                                    for c in table_list[1].split('</th>')])
        table_list[1] = new_header
        return '<tr>'.join(table_list)

    def _repr_html_(self):
        return (f'<b>tags:</b> {self._prt_tags()}'
                f'<br>{self._prt_table_tags(self.data)}')

    def _add_values(self, must_have_id=False, first_cols=None, **columns):
        """Add values for all attributes."""
        if 'data' in columns:
            if len(columns)>1:
                extra_cols = ', '.join([cols for cols in columns if cols!='data'])
                raise KeyError(f'data and columns (={extra_cols}) cannot be passed together.')
            if not hasattr(columns['data'], '__getitem__'):
                raise TypeError('data must be interactable (i. e. have __getitem__ function.)')
            data = ClData(columns['data'])
        else:
            # Check all columns have same size
            names = [n for n in columns]
            sizes = [len(v) for v in columns.values()]
            if any(sizes[0]!=s for s in sizes):
                raise ValueError(f"Column sizes inconsistent:\n"+
                    "\n".join([f"{' '*12}{k:10}: {l:,}" for k, l in zip(names, sizes)])
                    )
            data = ClData(columns)
        # Add columns
        cols = list(data.colnames)
        if first_cols:
            for col in first_cols[::-1]:
                cols.insert(0, cols.pop(cols.index(col)))
        if must_have_id:
            if self.tags['id'] not in data.namedict:
                self._create_id(len(data))
            else:
                self[self.tags['id']] = data[self.tags['id']]
                cols.pop(cols.index(self.tags['id']))
        for colname in cols:
            self[colname] = data[colname]

    def _create_id(self, size):
        id_name = 'id' if self.tags['id']=='id' else f'id ({self.tags["id"]})'
        warnings.warn(
            f'{id_name} column missing, additional one is being created.')
        TagData.__setitem__(self, self.tags['id'], np.array(range(size), dtype=str))

    def _make_col_dict(self, colname):
        return dict(map(lambda v:v[::-1], enumerate(self[colname])))

    def _make_col_dict_list(self, colname):
        dict_list = {}
        for ind, i in enumerate(self[colname]):
            dict_list[i] = dict_list.get(i, [])+[ind]
        return dict_list

    def tag_column(self, colname, coltag, skip_warn=False):
        """
        Tag column

        Parameters
        ----------
        colname: str
            Name of column
        coltag: str
            Tag for column
        skip_warn: bool
            Skip overwriting warning
        """
        if colname not in self.data.namedict:
            raise ValueError(
                f'setting tag {coltag}:{colname} to column ({colname}) missing in catalog')
        if coltag in self.data.namedict and coltag.lower()!=colname.lower():
            warnings.warn(
                f'There is a column with the same name as the tag setup.'
                f' cat[\'{coltag}\'] calls cat[\'{colname}\'] now.'
                f' To get \'{coltag}\' column, use cat.data[\'{coltag}\'].'
                )
        if coltag in self.tags and self.tags.get(coltag, None)!=colname and not skip_warn:
            warnings.warn(
                f'tag {coltag}:{self.tags[coltag]} being replaced by {coltag}:{colname}')
        self.tags[coltag] = self.data.namedict[colname]

    def tag_columns(self, colnames, coltags):
        """
        Tag columns

        Parameters
        ----------
        colname: list
            Name of columns
        coltag: list
            Tag for columns
        """
        if len(colnames)!=len(coltags):
            raise ValueError(
                f'Size of colnames ({len(colnames)} and coltags {len(coltags)} must be the same.')
        for colname, coltag in zip(colnames, coltags):
            self.tag_column(colname, coltag)

    def get(self, key, default=None):
        """
        Return the column for key if key is in the dictionary, else default
        """
        key_ = self.tags.get(key, key)
        return ClData.get(self.data, key_, default)

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
            out.meta.update({f'hierarch TAG_{k}':v for k, v in self.tags.items()})
        for col in filter(lambda c: c!='SkyCoord', self.data.colnames):
            out[col] = self[col]
        out.write(filename, overwrite=overwrite)

_matching_mask_funcs = {
    'cross': lambda match: match['mt_cross']!=None,
    'self': lambda match: match['mt_self']!=None,
    'other': lambda match: match['mt_other']!=None,
    'multi_self': lambda match: veclen(match['mt_multi_self'])>0,
    'multi_other': lambda match: veclen(match['mt_multi_other'])>0,
    'multi_join': lambda match: (veclen(match['mt_multi_self'])>0)+(veclen(match['mt_multi_other'])>0),
}

class Catalog(TagData):
    """
    Parent object to handle catalogs.

    Attributes
    ----------
    name: str
        ClCatalog name
    data: ClData
        Main catalog data (ex: id, ra, dec, z). Fixed values.
        Matching data (mt_self, mt_other, mt_cross, mt_multi_self, mt_multi_other)
    id_dict: dict
        Dictionary of indicies given the cluster id
    size: int
        Number of objects in the catalog
    tags: LoweCaseDict
        Tag for main quantities used in matching and plots (ex: id, ra, dec, z)
    labels: LowerCaseDict
        Labels of data columns for plots
    """

    @property
    def id_dict(self):
        return self.__id_dict

    def __init__(self, name, tags=None, labels=None, unique_id=False, default_tags=None, **kwargs):
        if not isinstance(name, str):
            raise ValueError('name must be str.')
        if labels is not None and not isinstance(labels, dict):
            raise ValueError('labels must be dict.')
        self.unique_id = unique_id
        self.name = name
        self.labels = LowerCaseDict()
        self.__id_dict = {}
        TagData.__init__(self, tags=updated_dict({'id':'id'}, tags),
                         default_tags=none_val(default_tags, ['id', 'ra', 'dec']),
                         **kwargs)
        self.labels.update(none_val(labels, {}))

    def __str__(self):
        return f'{self.name}:\n{self.data.__str__()}'

    def _repr_html_(self):
        return (f'<b>{self.name}</b>'
                f'<br>{TagData._repr_html_(self)}')

    def __getitem__(self, item):
        return self._getitem_base(item, Catalog, name=self.name, labels=self.labels)

    def __setitem__(self, item, value):
        value_ = value
        if isinstance(item, str):
            if item[:3]!='mt_':
                self.labels[item] = self.labels.get(item, f'{item}_{{{self.name}}}')
            if item.lower()==self.tags['id'].lower():
                value_ = self._fmt_id_col(value)
            elif len(self.data.colnames)==0:
                if isinstance(value, (int, np.int64)):
                    raise TypeError('Empty table cannot have column set to scalar value')
                self._create_id(len(value))
        TagData.__setitem__(self, item, value_)

    def _fmt_id_col(self, id_col):
        id_out = np.array(id_col, dtype=str) # make id a string
        if self.unique_id:
            unique_vals, counts = np.unique(id_out, return_counts=True)
            if (counts>1).any():
                warnings.warn(
                    f'Repeated ID\'s in {self.tags["id"]} column, adding suffix _r# to them.')
                id_out = np.array(id_out, dtype=np.ndarray)
                for id_ in unique_vals[counts>1]:
                    case = id_out==id_
                    fmt = f'_r%0{len(f"{case.sum()}")}d'
                    id_out[case] += [fmt%(i+1) for i in range(case.sum())]
        return id_out

    def _add_values(self, **columns):
        """Add values for all attributes. If id is not provided, one is created"""
        TagData._add_values(self, must_have_id=True, **columns)
        self._add_skycoord()
        self.id_dict.update(self._make_col_dict('id'))

    def _add_skycoord(self):
        if ('ra' in self.tags and 'dec' in self.tags) and \
                not 'SkyCoord' in self.data.colnames:
            self['SkyCoord'] = SkyCoord(self['ra']*u.deg, self['dec']*u.deg, frame='icrs')

    def _init_match_vals(self, overwrite=False):
        """Fills self.match with default values

        Paramters
        ---------
        overwrite: bool
            Overwrite values of pre-existing columns.
        """
        for col in ('mt_self', 'mt_other', 'mt_multi_self', 'mt_multi_other'):
            if overwrite or col not in self.data.namedict:
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

    def tag_column(self, colname, coltag, skip_warn=False):
        """
        Tag column

        Parameters
        ----------
        colname: str
            Name of column
        coltag: str
            Tag for column
        skip_warn: bool
            Skip overwriting warning
        """
        TagData.tag_column(self, colname, coltag, skip_warn=skip_warn)
        self.labels[coltag] = self.labels.get(coltag, f'{coltag}_{{{self.name}}}')

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
            self['z'] if 'z' in self.tags else 1e-10)

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
        zcol, rcol = self.tags['z'], self.tags['radius']
        window_cfg = {
            'flat': {
                'func': ftpt._get_coverfrac,
                'get_args': lambda c: [c['SkyCoord'], c[zcol], aperture, aperture_unit],
                'colname': f'{num}_{aperture_unit}',
            },
            'nfw2D': {
                'func': ftpt._get_coverfrac_nfw2D,
                'get_args': lambda c: [c['SkyCoord'], c[zcol], c[rcol],
                                   self.radius_unit, aperture, aperture_unit],
                'colname': f'nfw_{num}_{aperture_unit}',
            },
        }.get(window, None)
        if window_cfg is None:
            raise ValueError(f"window ({window}) must be either 'flat' of 'nfw2D'")
        self[f"cf_{none_val(colname, window_cfg['colname'])}"] = [
            window_cfg['func'](*window_cfg['get_args'](c), cosmo=cosmo)
            for c in self]

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
            out.meta.update({f'hierarch LABEL_{k}':v for k, v in self.labels.items()})
            out.meta.update({f'hierarch TAG_{k}':v for k, v in self.tags.items()})
        for col in self.data.colnames:
            if col in ('mt_self', 'mt_other', 'mt_cross'):
                out[col] = pack_mt_col(self[col])
            elif col in ('mt_multi_self', 'mt_multi_other'):
                out[col] = pack_mmt_col(self[col])
            elif col=='SkyCoord':
                pass
            else:
                out[col] = self[col]
        out.write(filename, overwrite=overwrite)

    @classmethod
    def _read(self, data, **kwargs):
        """Does the main execution for reading catalog.

        Parameters
        ----------
        data: clevar.ClData
            Input data.
        name: str, None
            Catalog name, if none reads from file.
        **kwargs: keyword argumens
            All columns to be added must be passes with named argument,
            the name is used in the Catalog data and the value must
            be the column name in your input file (ex: z='REDSHIFT').
        """
        # out data
        mt_cols = NameList(('mt_self', 'mt_other', 'mt_cross', 'mt_multi_self', 'mt_multi_other'))
        non_mt_cols = [c for c in data.colnames if c not in mt_cols]
        out = self(data=data[non_mt_cols], **kwargs)
        # matching cols
        for colname in filter(lambda c:c in mt_cols, data.colnames):
            if colname in NameList(('mt_self', 'mt_other', 'mt_cross')):
                out[colname] = unpack_mt_col(data[colname])
            if colname in NameList(('mt_multi_self', 'mt_multi_other')):
                out[colname] = unpack_mmt_col(data[colname])
        return out

    @classmethod
    def read(self, filename, name, tags=None, labels=None, full=False):
        """Read catalog from fits file. If full=False, only columns in tags are read.

        Parameters
        ----------
        filename: str
            Input file.
        name: str, None
            Catalog name, if none reads from file.
        tags: LoweCaseDict, None
            Tags for table (required if full=False).
        labels: dict, None
            Labels of data columns for plots.
        full: bool
            Reads all columns of the catalog
        """
        data = ClData.read(filename)
        if not full:
            if tags is None:
                raise KeyError('If full=False, tags must be provided.')
            if not isinstance(tags, dict):
                raise ValueError('tags must be dict.')
            data._check_cols(tags.values())
            data = data[list(tags.values())]
        return self._read(data, name=name, labels=labels, tags=tags)

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
        # read labels and radius unit from file
        kwargs = {
            'name': data.meta['NAME'],
            'labels': LowerCaseDict({k[6:]:v for k, v in data.meta.items() if k[:6]=='LABEL_'}),
            'tags': LowerCaseDict({k[4:]:v for k, v in data.meta.items() if k[:4]=='TAG_'}),
        }
        kwargs.update({'radius_unit': data.meta['RADIUS_UNIT']}
                        if 'RADIUS_UNIT' in data.meta else {})
        return self._read(data, **kwargs)

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
        for col in mt.data.colnames:
            if col!='id':
                self[col] = mt[col]
        self.cross_match()
        print(f' * Total objects:    {self.size:,}')
        print(f' * multiple (self):  {(veclen(self["mt_multi_self"])>0).sum():,}')
        print(f' * multiple (other): {(veclen(self["mt_multi_other"])>0).sum():,}')
        print(f' * unique (self):    {(self["mt_self"]!=None).sum():,}')
        print(f' * unique (other):   {(self["mt_other"]!=None).sum():,}')
        print(f' * cross:            {(self["mt_cross"]!=None).sum():,}')

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
    tags: LoweCaseDict
        Tag for main quantities used in matching and plots (ex: id, ra, dec, z, mass,...)
    labels: LowerCaseDict
        Labels of data columns for plots
    members: MemCatalog
        Catalog of members associated to the clusters
    leftover_members: MemCatalog
        Catalog of members not associated to the clusters

    Notes
    -----
    Column names that automatically become tags: id, ra, dec, mass, z, radius, zmin, zmax, z_err
    """

    def __init__(self, name, tags=None, labels=None, radius_unit=None, members=None, **kwargs):
        self.members = None
        self.leftover_members = None
        self.radius_unit = radius_unit
        self.mt_input = kwargs.pop('mt_input', None)
        members_warning = kwargs.pop('members_warning', True)
        Catalog.__init__(self, name, labels=labels, tags=tags,
                         default_tags=['id', 'ra', 'dec', 'mass', 'z',
                                       'radius', 'zmin', 'zmax', 'z_err'],
                         unique_id=True, **kwargs)
        if members is not None:
            self.add_members(members_catalog=members,
                             members_warning=members_warning)

    def _repr_html_(self):
        # remove SkyCoord from print
        show_data_cols = [c for c in self.data.colnames if c!='SkyCoord']
        print_data = self.data[show_data_cols]
        table = self._prt_table_tags(print_data)
        # add mt_input to print
        if self.mt_input is not None:
            for col in self.mt_input.colnames:
                print_data[col] = self.mt_input[col]
            table_list = self._prt_table_tags(print_data).split('<thead><tr><th')
            style = 'text-align:left; background-color:grey; color:white'
            table_list.insert(1, (
                f''' colspan={len(show_data_cols)}></th>'''
                f'''<th colspan={len(self.mt_input.colnames)} style='{style}'>mt_input</th>'''
                '''</tr></thread>''')
            )
            table = '<thead><tr><th'.join(table_list)
        return (f'<b>{self.name}</b>'
                f'<br></b><b>tags:</b> {self._prt_tags()}'
                f'<br><b>Radius unit:</b> {self.radius_unit}'
                f'<br>{table}')

    def _add_values(self, **columns):
        """Add values for all attributes. If id is not provided, one is created"""
        Catalog._add_values(self, **columns)
        self._init_match_vals()

    def __getitem__(self, item):
        kwargs = {'name':self.name, 'labels': self.labels, 'radius_unit': self.radius_unit,
                  'members_warning':False}
        # get one col/row
        if isinstance(item, (str, int, np.int64)):
            kwargs['item'] = item
        else:
            # Get sub cols
            if (isinstance(item, (tuple, list)) and all(isinstance(x, str) for x in item)):
                item_nl = NameList(item)
                kwargs['item'] = [*item, *filter(lambda c: c[:3]=='mt_' and c not in item_nl,
                                    self.data.colnames)]
                kwargs['mt_input'] = self.mt_input
                kwargs['members'] = self.members
            # Get sub rows
            else:
                kwargs['item'] = item
                if self.mt_input is not None:
                    kwargs['mt_input'] = self.mt_input[item]
                if self.members is not None and isinstance(item, (list, np.ndarray)):
                    cl_mask = np.zeros(self.size, dtype=bool)
                    cl_mask[item] = True
                    kwargs['members'] = self.members[cl_mask[self.members['ind_cl']]]
        return self._getitem_base(DataType=ClCatalog, **kwargs)


        if isinstance(item, (str, int, np.int64)):
            item_ = self.tags.get(item, item) if isinstance(item, str) else item
            return self.data[item_]
        else:
            mt_input = None
            members = self.members
            tags = self.tags
            # Check if item_ is not a list of strings
            if (isinstance(item, (tuple, list)) and all(isinstance(x, str) for x in item)):
                item_ = NameList([self.tags.get(i, i) for i in item])
                tags = {k:v for k, v in self.tags.items() if k in item_ or v in item_}
                mt_cols = [c for c in self.data.colnames if c[:3]=='mt_' and c not in item_]
                item_ = [*item_, *mt_cols]
            else:
                item_ = item
                if self.mt_input is not None:
                    mt_input = self.mt_input[item_]
                if members is not None and isinstance(item_, (list, np.ndarray)):
                    cl_mask = np.zeros(self.size, dtype=bool)
                    cl_mask[item_] = True
                    members = members[cl_mask[members['ind_cl']]]
            # generate catalog
            return ClCatalog(name=self.name, labels=self.labels, radius_unit=self.radius_unit,
                             mt_input=mt_input, members=members, members_warning=False,
                             tags=tags, data=self.data[item_])

    def raw(self):
        """
        Get a copy of the catalog without members.
        """
        if self.members is not None:
            out = ClCatalog(
                name=self.name, tags=self.tags, labels=self.labels,
                radius_unit=self.radius_unit, data=self.data,
                mt_input=self.mt_input)
        else:
            out = self
        return out

    @classmethod
    def read(self, filename, name, tags=None, labels=None, radius_unit=None, full=False):
        """Read catalog from fits file. If full=False, only columns in tags are read.

        Parameters
        ----------
        filename: str
            Input file.
        name: str, None
            Catalog name, if none reads from file.
        tags: LoweCaseDict, None
            Tags for table (required if full=False).
        labels: dict, None
            Labels of data columns for plots.
        radius_unit: str, None
            Unit of the radius column (default read from file).
        full: bool
            Reads all columns of the catalog
        """
        data = ClData.read(filename)
        if not full:
            if tags is None:
                raise KeyError('If full=False, tags must be provided.')
            if not isinstance(tags, dict):
                raise ValueError('tags must be dict.')
            data._check_cols(tags.values())
            data = data[list(tags.values())]
        return self._read(data, name=name, labels=labels, tags=tags, radius_unit=radius_unit)

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
            members = members_catalog
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

    def read_members(self, filename, tags=None, labels=None, members_consistency=True,
                     members_warning=True, full=False):
        """Read members catalog from fits file.

        Parameters
        ----------
        filename: str
            Input file.
        tags: LoweCaseDict, None
            Tags for member table.
        labels: dict, None
            Labels of data columns for plots.
        members_consistency: bool
            Require that all input members belong to this cluster catalog.
        members_warning: bool
            Raise warning if members are do not belong to this cluster catalog,
            and save them in leftover_members attribute.
        full: bool
            Reads all columns of the catalog
        """
        self.add_members(
            members_catalog=MemCatalog.read(
                filename, 'members', labels=labels, tags=tags, full=full),
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
    tags: LoweCaseDict
        Tag for main quantities used in matching and plots (ex: id, id_cluster, ra, dec, z,...)
    labels: dict
        Labels of data columns for plots

    Notes
    -----
    Column names that automatically become tags: id, id_cluster, ra, dec, z, radius, pmem
    """

    @property
    def id_dict_list(self):
        return self.__id_dict_list

    def __init__(self, name, tags=None, labels=None, **kwargs):
        if tags is not None and not isinstance(tags, dict):
            raise ValueError('tags must be dict.')
        self.__id_dict_list = {}
        Catalog.__init__(self, name, labels=labels,
                         tags=LowerCaseDict(updated_dict({'id_cluster':'id_cluster'}, tags)),
                         default_tags=['id', 'id_cluster', 'ra', 'dec', 'z', 'radius', 'pmem'],
                         **kwargs)

    def _add_values(self, **columns):
        """Add values for all attributes. If id is not provided, one is created"""
        # create catalog
        Catalog._add_values(self, first_cols=[self.tags['id_cluster']], **columns)
        if self.tags['id_cluster'] not in self.data.colnames:
            idcl_name = ('id_cluster' if self.tags['id_cluster']=='id_cluster'
                            else f'id_cluster ({self.tags["id_cluster"]})')
            raise ValueError(f'Members catalog must have a {idcl_name} column!.')

        self.id_dict_list.update(self._make_col_dict_list('id'))

    def __getitem__(self, item):
        kwargs = {'name':self.name, 'labels': self.labels}
        # Get sub cols
        if isinstance(item, (tuple, list)) and all(isinstance(x, str) for x in item):
            item_nl = NameList(item)
            # also pass main and mt cols
            main_cols = filter(lambda c: self.tags[c] not in item_nl, ['id', 'id_cluster'])
            mt_cols = filter(lambda c: c[:3]=='mt_' and c not in item_nl, self.data.colnames)
            kwargs['item'] = [*item, *main_cols, *mt_cols]
        else:
            kwargs['item'] = item
        return self._getitem_base(DataType=MemCatalog, **kwargs)
