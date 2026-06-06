"""Footprint module"""

import warnings

from . import artificial
from .artificial import create_footprint as create_artificial_footprint
from .footprint import Footprint

__all__ = [
    "artificial",
    "create_artificial_footprint",
    "Footprint",
]
