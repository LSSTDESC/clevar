"""@file catalog.py
The improved Astropy table classes
"""
import warnings
import numpy as np
from astropy.table import Table as APtable

from ..utils import (
    none_val,
    NameList,
    LowerCaseDict,
)


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

    @property
    def namedict(self):
        """Case independent dictionary with column names"""
        return self.__namedict

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
        self.skip_cols = ["SkyCoord"]
        self.first_cols = []

    def __getitem__(self, item):
        """
        To make case insensitive
        """
        # get one column
        if isinstance(item, str):
            return super().__getitem__(self.namedict.get(item, item))

        # get one row
        if isinstance(item, (int, np.int64)):
            out = super().__getitem__(item)
            out.namedict = self.namedict
            return out

        # Get sub cols
        if isinstance(item, (tuple, list)) and all(isinstance(x, str) for x in item):
            item_ = NameList(map(lambda i: self.namedict[i], item))

        # Get sub rows
        else:
            item_ = item

        out = ClData(super().__getitem__(item_))
        out.skip_cols = self.skip_cols
        out.first_cols = list(filter(lambda col: col in self.namedict, self.first_cols))

        return out

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

    def _show_cols(self):
        return [
            *self.first_cols,
            *(
                col
                for col in self.colnames
                if col not in self.skip_cols and col not in self.first_cols
            ),
        ]

    def _repr_html_(self):
        print(self._show_cols(), self.first_cols)
        return APtable._repr_html_(self[self._show_cols()])

    @classmethod
    def read(cls, *args, **kwargs):
        """Read data"""
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
        if len(missing) > 0:
            missing = ", ".join(missing)
            raise KeyError(f"Column(s) '{missing}' not found " "in catalog {data.colnames}")


class TagData:
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
        """Tags of catalog"""
        return self.__tags

    @property
    def data(self):
        """Internal data"""
        return self.__data

    @property
    def size(self):
        """Number of rows"""
        return len(self.data)

    @property
    def default_tags(self):
        """Innate tags"""
        return self.__default_tags

    @property
    def colnames(self):
        """Case independent column names"""
        return NameList(self.data.colnames)

    def __init__(self, tags=None, default_tags=None, **kwargs):
        if tags is not None and not isinstance(tags, dict):
            raise ValueError("tags must be dict.")
        self.__data = ClData()
        self.__tags = LowerCaseDict(none_val(tags, {}))
        self.__default_tags = NameList(none_val(default_tags, []))
        if len(kwargs) > 0:
            self._add_values(**kwargs)
            # make sure columns don't overwrite tags
            if tags is not None:
                for colname, coltag in tags.items():
                    self.tag_column(coltag, colname, skip_warn=True)

    def __setitem__(self, item, value):
        """Adds items to tags if in default_tags"""
        if isinstance(item, str):
            cname = self.tags.get(item, item)
            if item in self.default_tags:
                self.tags[item] = self.data.namedict.get(cname, cname)
            self.data[cname] = value
        else:
            raise ValueError(f"can only set with str item (={item}) argument.")

    def _getitem_base(self, item, DataType, **kwargs):
        """Base function to also be used by child classes"""
        # pylint: disable=invalid-name

        # Get one row/col
        if isinstance(item, (str, int, np.int64)):
            return self.data[self.tags.get(item, item) if isinstance(item, str) else item]

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
        return ", ".join([f"{v}({k})" for k, v in self.tags.items()])

    def _prt_table_tags(self, table):
        # pylint: disable=protected-access
        tags_inv = {v: f" ({k})" for k, v in self.tags.items() if k != v}
        table_list = table._repr_html_().split("<tr>")
        new_header = "</th>".join(
            [
                c + tags_inv.get(c[4:], "") if c[:4] == "<th>" else c
                for c in table_list[1].split("</th>")
            ]
        )
        table_list[1] = new_header
        return "<tr>".join(table_list)

    def _repr_html_(self):
        return f"<b>tags:</b> {self._prt_tags()}" f"<br>{self._prt_table_tags(self.data)}"

    def _add_values(self, must_have_id=False, **columns):
        """Add values for all attributes."""
        if "data" in columns:
            if len(columns) > 1:
                extra_cols = ", ".join([cols for cols in columns if cols != "data"])
                raise KeyError(f"data and columns (={extra_cols}) cannot be passed together.")
            if not hasattr(columns["data"], "__getitem__"):
                raise TypeError("data must be interactable (i. e. have __getitem__ function.)")
            data = ClData(columns["data"])
        else:
            # Check all columns have same size
            sizes = [len(v) for v in columns.values()]
            if any(sizes[0] != s for s in sizes):
                raise ValueError(
                    "Column sizes inconsistent:\n"
                    + "\n".join([f"{' '*12}{k:10}: {l:,}" for k, l in zip(columns, sizes)])
                )
            data = ClData(columns)

        # Add columns

        missing = [
            f"'{dtag}': '{self.tags.get(dtag, None)}'"
            for dtag in self.default_tags
            if dtag in self.tags
            and self.tags.get(dtag, None) not in data.namedict
            and dtag.lower() != "id"
        ]
        if len(missing) > 0:
            missing = ", ".join(missing)
            raise KeyError(f"Tagged column(s) ({missing}) not found in catalog {data.colnames}")

        self.__data = data
        if must_have_id and self.tags["id"] not in data.namedict:
            self._create_id(len(data))

        for name in self.colnames:
            if name in self.default_tags:
                colname = self.tags.get(name, name)
                self.tag_column(colname, name)

    def _create_id(self, size):
        id_name = "id" if self.tags["id"] == "id" else f'id ({self.tags["id"]})'
        warnings.warn(f"{id_name} column missing, additional one is being created.")
        TagData.__setitem__(self, self.tags["id"], np.array(range(size), dtype=str))

    def _make_col_dict(self, colname):
        return dict(zip(self[colname], np.arange(self.size, dtype=int)))
        return dict(map(lambda v: v[::-1], enumerate(self[colname])))

    def _make_col_dict_list(self, colname):
        dict_list = {}
        for ind, i in enumerate(self[colname]):
            dict_list[i] = dict_list.get(i, []) + [ind]
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
                f"setting tag {coltag}:{colname} to column ({colname}) missing in catalog"
            )
        if coltag in self.data.namedict and coltag.lower() != colname.lower():
            warnings.warn(
                f"There is a column with the same name as the tag setup."
                f" cat['{coltag}'] calls cat['{colname}'] now."
                f" To get '{coltag}' column, use cat.data['{coltag}']."
            )
        if coltag in self.tags and self.tags.get(coltag, None) != colname and not skip_warn:
            warnings.warn(f"tag {coltag}:{self.tags[coltag]} being replaced by {coltag}:{colname}")
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
        if len(colnames) != len(coltags):
            raise ValueError(
                f"Size of colnames ({len(colnames)} and coltags {len(coltags)} must be the same."
            )
        for colname, coltag in zip(colnames, coltags):
            self.tag_column(colname, coltag)

    def get(self, key, default=None):
        """
        Return the column for key if key is in the dictionary, else default
        """
        # pylint: disable=arguments-out-of-order
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
            out.meta.update({f"hierarch TAG_{k}": v for k, v in self.tags.items()})
        for col in filter(lambda c: c != "SkyCoord", self.data.colnames):
            out[col] = self[col]
        out.write(filename, overwrite=overwrite)

    @classmethod
    def _read_data(cls, filename, tags=None, full=False):
        """Read catalog from fits file. If full=False, only columns in tags are read.

        Parameters
        ----------
        filename: str
            Input file.
        tags: LoweCaseDict, None
            Tags for table (required if full=False).
        full: bool
            Reads all columns of the catalog
        """
        # pylint: disable=protected-access
        data = ClData.read(filename)
        if not full:
            if tags is None:
                raise KeyError("If full=False, tags must be provided.")
            if not isinstance(tags, dict):
                raise ValueError("tags must be dict.")
            data._check_cols(tags.values())
            data = data[list(tags.values())]
        return data

    @classmethod
    def read(cls, filename, tags=None, full=False):
        """Read catalog from fits file. If full=False, only columns in tags are read.

        Parameters
        ----------
        filename: str
            Input file.
        tags: LoweCaseDict, None
            Tags for table (required if full=False).
        full: bool
            Reads all columns of the catalog
        """
        # pylint: disable=protected-access
        data = cls._read_data(filename, tags=tags, full=full)
        return cls(data=data, tags=tags)

    @classmethod
    def read_full(cls, filename):
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
            "tags": LowerCaseDict({k[4:]: v for k, v in data.meta.items() if k[:4] == "TAG_"}),
        }
        return cls(data=data, **kwargs)
