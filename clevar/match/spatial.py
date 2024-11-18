"""@file spatial.py
The SpatialMatch class
"""

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as spline

from .parent import Match
from ..geometry import units_bank, convert_units
from ..utils import str2dataunit


class SpatialMatch(Match):
    """
    SpatialMatch Class

    Attributes
    ----------
    type : str
        Type of matching object. Set to "Spatial"
    history : list
        Steps in the matching
    """

    # pylint: disable=abstract-method

    def __init__(self):
        Match.__init__(self)
        self.type = "Spatial"

    def _prep_z_for_match(
        self,
        cat,
        delta_z,
        n_delta_z=1,
    ):
        """
        Adds zmin and zmax to cat.mt_input

        Parameters
        ----------
        cat: clevar.ClCatalog
            Input ClCatalog
        delta_z: float, string
            Defines the zmin, zmax for matching. Options are:

                * `'cat'` - uses redshift properties of the catalog
                * `'spline.filename'` - interpolates data in 'filename' (z, zmin, zmax) fmt
                * `float` - uses delta_z*(1+z)
                * `None` - does not use z

        n_delta_z: float
            Number of delta_z to be used in the matching
        """
        print("### Prep z_cols")
        if delta_z is None:
            # No z use
            print("* zmin|zmax set to -1|10")
            cat.mt_input["zmin"] = -1 * np.ones(cat.size)
            cat.mt_input["zmax"] = 10 * np.ones(cat.size)
        elif delta_z == "cat":
            # Values from catalog
            if "zmin" in cat.tags and "zmax" in cat.tags:
                print("* zmin|zmax from cat cols (zmin, zmax)")
                cat.mt_input["zmin"] = self._rescale_z(cat["z"], cat["zmin"], n_delta_z)
                cat.mt_input["zmax"] = self._rescale_z(cat["z"], cat["zmax"], n_delta_z)
            # create zmin/zmax if not there
            elif "z_err" in cat.tags:
                print("* zmin|zmax from cat cols (err)")
                cat.mt_input["zmin"] = cat["z"] - n_delta_z * cat["z_err"]
                cat.mt_input["zmax"] = cat["z"] + n_delta_z * cat["z_err"]
            else:
                raise ValueError("ClCatalog must contain zmin, zmax or z_err for this matching.")
        elif isinstance(delta_z, str):
            # zmin/zmax in auxiliar file
            print("* zmin|zmax from aux file")
            order = 3
            zval, zvmin, zvmax = np.loadtxt(delta_z)
            zvmin = self._rescale_z(zval, zvmin, n_delta_z)
            zvmax = self._rescale_z(zval, zvmax, n_delta_z)
            cat.mt_input["zmin"] = spline(zval, zvmin, k=order)(cat["z"])
            cat.mt_input["zmax"] = spline(zval, zvmax, k=order)(cat["z"])
        elif isinstance(delta_z, (int, float)):
            # zmin/zmax from sigma_z*(1+z)
            print("* zmin|zmax from config value")
            cat.mt_input["zmin"] = cat["z"] - delta_z * n_delta_z * (1.0 + cat["z"])
            cat.mt_input["zmax"] = cat["z"] + delta_z * n_delta_z * (1.0 + cat["z"])

    def _rescale_z(self, z, zlim, n_rescale):
        """Rescale zmin/zmax by a factor n_rescale

        Parameters
        ----------
        z: float, array
            Redshift value
        zlim: float, array
            Redshift limit
        n_rescale: float
            Value to rescale zlim

        Returns
        -------
        float, array
            Rescaled z limit
        """
        return z + n_rescale * (zlim - z)

    def _prep_radius_for_match(self, cat, match_radius, n_match_radius=1, cosmo=None):
        """
        Adds radius to cat.mt_input

        Parameters
        ----------
        cat: clevar.ClCatalog
            Input ClCatalog

        match_radius: string
            Defines the radius for matching. Options are:

                * `'cat'` - uses the radius in the catalog
                * `'value unit'` - used fixed value (ex: `1 arcsec`, `1 Mpc`)

        n_match_radius: float
            Multiplies the radius of the matchingi
        cosmo: clevar.Cosmology object
            Cosmology object for when radius has physical units
        """
        print("### Prep ang_cols")
        if match_radius == "cat":
            print("* ang radius from cat")
            in_rad, in_rad_unit = cat["radius"], cat.radius_unit
            # when mass is passed to radius: m#b or m#c - background/critical
            if in_rad_unit.lower() != "mpc" and in_rad_unit[0].lower() == "m":
                print(f"    * Converting mass ({in_rad_unit}) ->radius")
                delta, mtyp = str2dataunit(
                    in_rad_unit[1:],
                    ["b", "c"],
                    err_msg=(
                        f"Mass unit ({in_rad_unit}) must be in format "
                        "'m#b' (background) or 'm#c' (critical)"
                    ),
                )
                in_rad = cosmo.eval_mass2radius(
                    in_rad, cat["z"], delta, mass_type={"b": "background", "c": "critical"}[mtyp]
                )
                in_rad_unit = "mpc"
        else:
            print("* ang radius from set scale")
            in_rad, in_rad_unit = str2dataunit(match_radius, units_bank.keys())
            in_rad *= np.ones(cat.size)
        # convert to degrees
        cat.mt_input["ang"] = convert_units(
            in_rad * n_match_radius,
            in_rad_unit,
            "degrees",
            redshift=cat["z"] if "z" in cat.tags else None,
            cosmo=cosmo,
        )

    def _unique_match_from_config(self, cat1, cat2, match_config):
        """
        Make matching of catalogs based on a configuration dictionary

        Parameters
        ----------
        cat1: clevar.ClCatalog
            ClCatalog 1
        cat2: clevar.ClCatalog
            ClCatalog 2
        match_config: dict
            Dictionary with the matching configuration. Keys must be:

                * `type`: type of matching, can be: `cat1`, `cat2`, `cross`.
                * `preference`: Preference to set best match, can be: `more_massive`,
                  `angular_proximity`, `redshift_proximity`, `shared_member_fraction`.
        """
        if match_config["type"] in ("cat1", "cross"):
            print("\n## Finding unique matches of catalog 1")
            self.unique(cat1, cat2, match_config["preference"])
        if match_config["type"] in ("cat2", "cross"):
            print("\n## Finding unique matches of catalog 2")
            self.unique(cat2, cat1, match_config["preference"])

        if match_config["type"] == "cross":
            self.cross_match(cat1)
            self.cross_match(cat2)

    def _valid_match_input_setup(self, cat1, cat2):
        """
        Validate if catalogs are setup for matching and init match columns.

        Parameters
        ----------
        cat1: clevar.ClCatalog
            ClCatalog 1
        cat2: clevar.ClCatalog
            ClCatalog 2
        """
        if cat1.mt_input is None:
            raise AttributeError("cat1.mt_input is None, run prep_cat_for_match first.")
        if cat2.mt_input is None:
            raise AttributeError("cat2.mt_input is None, run prep_cat_for_match first.")

        # pylint: disable=protected-access
        cat1._init_match_vals()
        cat2._init_match_vals()
