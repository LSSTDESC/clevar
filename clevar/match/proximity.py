"""@file proximity.py
The ProximityMatch class
"""
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as spline

from .parent import Match
from ..geometry import units_bank, convert_units
from ..catalog import ClData
from ..utils import veclen, str2dataunit


class ProximityMatch(Match):
    """
    ProximityMatch Class

    Attributes
    ----------
    type : str
        Type of matching object. Set to "Proximity"
    history : list
        Steps in the matching
    """

    def __init__(self):
        Match.__init__(self)
        self.type = "Proximity"

    def multiple(self, cat1, cat2, radius_selection="max", verbose=True):
        """
        Make the one way multiple matching

        Parameters
        ----------
        cat1: clevar.ClCatalog
            Base catalog
        cat2: clevar.ClCatalog
            Target catalog
        verbose: bool
            Print result for individual matches.
        radius_selection: str (optional)
            Case of radius to be used, can be: max, min, self, other.
        """
        # pylint: disable=arguments-renamed
        # pylint: disable=too-many-locals
        if cat1.mt_input is None:
            raise AttributeError("cat1.mt_input is None, run prep_cat_for_match first.")
        if cat2.mt_input is None:
            raise AttributeError("cat2.mt_input is None, run prep_cat_for_match first.")
        self._cat1_mmt = np.zeros(cat1.size, dtype=bool)  # To add flag in multi step matching
        ra2, dec2, sk2 = (cat2[c] for c in ("ra", "dec", "SkyCoord"))
        ang2, z2min, z2max = (cat2.mt_input[c] for c in ("ang", "zmin", "zmax"))
        ang2max = ang2.max()
        print(f"Finding candidates ({cat1.name})")
        for ind1, (ra1, dec1, sk1, ang1, z1min, z1max) in enumerate(
            zip(
                *(
                    [cat1[c] for c in ("ra", "dec", "SkyCoord")]
                    + [cat1.mt_input[c] for c in ("ang", "zmin", "zmax")]
                )
            )
        ):
            # crop in redshift range
            mask = (z2max >= z1min) * (z2min <= z1max)
            if mask.any():
                # makes square crop with radius
                dist0 = max(ang2max, ang1)
                dist0_cos = dist0 / np.cos(np.radians(dec1))
                mask *= (
                    (ra2 >= ra1 - dist0_cos)
                    * (ra2 < ra1 + dist0_cos)
                    * (dec2 >= dec1 - dist0)
                    * (dec2 < dec1 + dist0)
                )
                if mask.any():
                    # makes circular crop
                    dist = sk1.separation(sk2[mask]).value
                    max_dist = self._max_mt_distance(
                        ang1, ang2[mask], radius_selection=radius_selection
                    )
                    for id2 in cat2["id"][mask][dist <= max_dist]:
                        cat1["mt_multi_self"][ind1].append(id2)
                        ind2 = int(cat2.id_dict[id2])
                        cat2["mt_multi_other"][ind2].append(cat1["id"][ind1])
                        self._cat1_mmt[ind1] = True
            if verbose:
                print(
                    f"  {ind1:,}({cat1.size:,}) - {len(cat1['mt_multi_self'][ind1]):,} candidates",
                    end="\r",
                )
        print(f'* {(veclen(cat1["mt_multi_self"])>0).sum():,}/{cat1.size:,} objects matched.')
        cat1.remove_multiple_duplicates()
        cat2.remove_multiple_duplicates()
        self.history.append(
            {
                "func": "multiple",
                "cats": f"{cat1.name}, {cat2.name}",
                "radius_selection": radius_selection,
            }
        )
        # pylint: disable=protected-access
        cat1._set_mt_hist(self.history)
        cat2._set_mt_hist(self.history)

    def prep_cat_for_match(
        self, cat, delta_z, match_radius, n_delta_z=1, n_match_radius=1, cosmo=None
    ):
        """
        Adds zmin, zmax and radius to cat.mt_input

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

        match_radius: string
            Defines the radius for matching. Options are:

                * `'cat'` - uses the radius in the catalog
                * `'value unit'` - used fixed value (ex: `1 arcsec`, `1 Mpc`)

        n_delta_z: float
            Number of delta_z to be used in the matching
        n_match_radius: float
            Multiplies the radius of the matchingi
        cosmo: clevar.Cosmology object
            Cosmology object for when radius has physical units
        """
        # pylint: disable=arguments-differ
        print("## Prep mt_cols")
        cat.mt_input = ClData()
        # Set zmin, zmax
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

        # Set angular radius
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
        self.history.append(
            {
                "func": "prep_cat_for_match",
                "cat": cat.name,
                "delta_z": delta_z,
                "match_radius": match_radius,
                "n_delta_z": n_delta_z,
                "n_match_radius": n_match_radius,
                "cosmo": cosmo if cosmo is None else cosmo.get_desc(),
            }
        )

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

    def _max_mt_distance(self, radius1, radius2, radius_selection):
        """Get maximum angular distance allowed for the matching

        Parameters
        ----------
        radius1: float, array
            Radius to be used for catalog 1
        radius2: float, array
            Radius to be used for catalog 2
        radius_selection: str
            Case of radius to be used, can be: self, other, min, max.

        Returns
        -------
        float, array
            Maximum angular distance allowed for matching
        """
        if radius_selection == "self":
            coeff1 = np.ones(radius1.size)
            coeff2 = np.zeros(radius2.size)
        elif radius_selection == "other":
            coeff1 = np.zeros(radius1.size)
            coeff2 = np.ones(radius2.size)
        elif radius_selection == "max":
            coeff1 = radius1 >= radius2
            coeff2 = radius1 < radius2
        elif radius_selection == "min":
            coeff1 = radius1 < radius2
            coeff2 = radius1 >= radius2
        return coeff1 * radius1 + coeff2 * radius2

    def match_from_config(self, cat1, cat2, match_config, cosmo=None):
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
                * `catalog1`: `kwargs` used in `prep_cat_for_match(cat1, **kwargs)` (minus `cosmo`).
                * `catalog2`: `kwargs` used in `prep_cat_for_match(cat2, **kwargs)` (minus `cosmo`).
                * `which_radius`: Case of radius to be used, can be: `cat1`, `cat2`, `min`, `max`.
                * `preference`: Preference to set best match, can be: `more_massive`,
                  `angular_proximity`, `redshift_proximity`, `shared_member_fraction`.
                * `verbose`: Print result for individual matches (default=`True`).

        cosmo: clevar.Cosmology object
            Cosmology object for when radius has physical units
        """
        if match_config["type"] not in ("cat1", "cat2", "cross"):
            raise ValueError("config type must be cat1, cat2 or cross")
        if match_config["type"] in ("cat1", "cross"):
            print("\n## ClCatalog 1")
            self.prep_cat_for_match(cat1, cosmo=cosmo, **match_config["catalog1"])
        if match_config["type"] in ("cat2", "cross"):
            print("\n## ClCatalog 2")
            self.prep_cat_for_match(cat2, cosmo=cosmo, **match_config["catalog2"])

        verbose = match_config.get("verbose", True)
        if match_config["type"] in ("cat1", "cross"):
            print("\n## Multiple match (catalog 1)")
            radius_selection = {
                "cat1": "self",
                "cat2": "other",
            }.get(
                match_config["which_radius"],
                match_config["which_radius"],
            )
            self.multiple(cat1, cat2, radius_selection, verbose=verbose)
        if match_config["type"] in ("cat2", "cross"):
            print("\n## Multiple match (catalog 2)")
            radius_selection = {
                "cat1": "other",
                "cat2": "self",
            }.get(
                match_config["which_radius"],
                match_config["which_radius"],
            )
            # pylint: disable=arguments-out-of-order
            self.multiple(cat2, cat1, radius_selection, verbose=verbose)

        if match_config["type"] in ("cat1", "cross"):
            print("\n## Finding unique matches of catalog 1")
            self.unique(cat1, cat2, match_config["preference"])
        if match_config["type"] in ("cat2", "cross"):
            print("\n## Finding unique matches of catalog 2")
            self.unique(cat2, cat1, match_config["preference"])

        if match_config["type"] == "cross":
            self.cross_match(cat1)
            self.cross_match(cat2)
