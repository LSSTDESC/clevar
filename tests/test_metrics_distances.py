"""Tests for clevar/match_metrics/distances"""
import numpy as np
from clevar.catalog import ClCatalog
from clevar.cosmology import AstroPyCosmology as CosmoClass
from clevar.match import ProximityMatch
from clevar.match_metrics import distances as dt


##############################
#### Input data ##############
##############################
class _test_data:
    input1 = {
        "id": [f"CL{i}" for i in range(5)],
        "ra": [0.0, 0.0001, 0.00011, 25, 20],
        "dec": [0.0, 0, 0, 0, 0],
        "z": [0.2, 0.3, 0.25, 0.4, 0.35],
        "mass": [10**13.5, 10**13.4, 10**13.3, 10**13.8, 10**14],
    }
    input2 = {k: v[:4] for k, v in input1.items()}
    input2["z"][:2] = [0.3, 0.2]
    input2["mass"][:3] = input2["mass"][:3][::-1]
    cat1 = ClCatalog("Cat1", **input1)
    cat2 = ClCatalog("Cat2", **input2)
    cosmo = CosmoClass()
    mt = ProximityMatch()
    mt_config1 = {"delta_z": 0.2, "match_radius": "1 mpc", "cosmo": cosmo}
    mt_config2 = {"delta_z": 0.2, "match_radius": "1 arcsec"}
    mt.prep_cat_for_match(cat1, **mt_config1)
    mt.prep_cat_for_match(cat2, **mt_config2)
    mt.multiple(cat1, cat2)
    mt.multiple(cat2, cat1)
    mt.unique(cat1, cat2, "angular_proximity")
    mt.unique(cat2, cat1, "angular_proximity")


##############################
### Test #####################
##############################
def test_central_position():
    cat1, cat2 = _test_data.cat1, _test_data.cat2
    dt.central_position(
        cat1, cat2, "cat1", radial_bins=20, radial_bin_units="degrees", cosmo=None, ax=None
    )


def test_redshift():
    cat1, cat2 = _test_data.cat1, _test_data.cat2
    dt.redshift(cat1, cat2, "cat1", redshift_bins=20, normalize=None, ax=None)
    dt.redshift(
        cat1,
        cat2,
        "cat1",
        redshift_bins=20,
        normalize=None,
        quantity_bins="mass",
        bins=[1e14, 1e16],
        ax=None,
    )
