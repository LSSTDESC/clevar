"""@file __init__.py
The Catalogs and improved Astropy tables
"""

from .catalog import Catalog, ClCatalog, MemCatalog
from .tagdata import ClData, TagData

__all__ = [
    "Catalog",
    "ClCatalog",
    "MemCatalog",
    "ClData",
    "TagData",
]
