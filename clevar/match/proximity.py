"""@file proximity.py
The ProximityMatch class
"""

import numpy as np
from astropy import units as u

from ..catalog import ClData
from .spatial import SpatialMatch


# To disable pylint strike: Method '_set_unique_matching_function' is abstract in class 'Match'
# but is not overridden in child class 'ProximityMatch'
# to be fixed latter
# pylint: disable=abstract-method
class ProximityMatch(SpatialMatch):
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
        SpatialMatch.__init__(self)
        self.type = "Proximity"

    def multiple(self, cat1, cat2, radius_selection="max", verbose=True, kdtree_div=None):
        """
        Make the one way multiple matching

        Parameters
        ----------
        cat1: clevar.ClCatalog
            Base catalog
        cat2: clevar.ClCatalog
            Target catalog
        radius_selection: str (optional)
            Case of radius to be used, can be: max, min, self, other.
        verbose: bool
            Print result for individual matches, only used in old matching scheme.
        kdtree_div: int, None
            If None, simple old matching scheme is used where a loop is used,
            else it should be number of redshift to be used in the kdtree split.
        """
        # pylint: disable=arguments-renamed
        # pylint: disable=too-many-locals
        self._valid_match_input_setup(cat1, cat2)

        self._cat1_mmt = np.zeros(cat1.size, dtype=bool)  # To add flag in multi step matching

        if kdtree_div is None:
            self._simple_multiple_match(cat1, cat2, radius_selection, verbose)
        else:
            self._kdtree_multiple_match(cat1, cat2, radius_selection, kdtree_div)

        hist = {
            "func": "multiple",
            "cats": f"{cat1.name}, {cat2.name}",
            "radius_selection": radius_selection,
        }
        self._rm_dup_add_hist(cat1, cat2, hist)

    def _simple_multiple_match(self, cat1, cat2, radius_selection="max", verbose=True):
        """
        Simple way to make the one way multiple matching

        Parameters
        ----------
        cat1: clevar.ClCatalog
            Base catalog
        cat2: clevar.ClCatalog
            Target catalog
        radius_selection: str (optional)
            Case of radius to be used, can be: max, min, self, other.
        verbose: bool
            Print result for individual matches.
        """
        # pylint: disable=arguments-renamed
        # pylint: disable=too-many-locals
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
                self._prt_cand_mt(cat1, ind1)

    def _kdtree_multiple_match(self, cat1, cat2, radius_selection="max", ndiv=1):
        """
        Make the one way multiple matching using kdtree

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
        kdtree_div: int, None
            If None, simple old matching scheme is used where a loop is used,
            else it should be number of redshift to be used in the kdtree split.
        """
        # pylint: disable=too-many-locals

        if "z" not in cat1.colnames:
            inds1_sorted = np.arange(cat1.size)
        else:
            inds1_sorted = np.argsort(cat1["z"])

        # props
        ang1, z1min, z1max = (cat1.mt_input[c] for c in ("ang", "zmin", "zmax"))
        ang2, z2min, z2max = (cat2.mt_input[c] for c in ("ang", "zmin", "zmax"))

        for inds1 in np.array_split(inds1_sorted, ndiv):
            if len(inds1) == 0:
                break

            # pre-select cat2 in redshift
            # crop in redshift range
            inds2 = np.where((z2max >= z1min[inds1].min()) * (z2min <= z1max[inds1].max()))[0]

            kdt_prt = " * kdtree"
            if "z" in cat1.colnames:
                kdt_prt += f" zbin [{cat1['z'][inds1[0]]:.2f}:{cat1['z'][inds1[-1]]:.2f}]"
            print(
                f"{kdt_prt} - {len(inds1):,} x {len(inds2):,}"
                f" = {len(inds1) * len(inds2):,} clusters"
            )

            # coarse selection
            # pairs of indicies for matched clusters
            mt_inds1, mt_inds2, angsep = cat2["SkyCoord"][inds2].search_around_sky(
                cat1["SkyCoord"][inds1], np.maximum(ang1[inds1], ang2[inds2].max()) * u.degree
            )[:3]
            inds1 = inds1[mt_inds1]
            inds2 = inds2[mt_inds2]

            # refine to actual radii and z dist
            mt_msk = (
                (
                    angsep.value
                    <= self._max_mt_distance(
                        ang1[inds1],
                        ang2[inds2],
                        radius_selection=radius_selection,
                    )
                )
                * (z2max[inds2] >= z1min[inds1])
                * (z2min[inds2] <= z1max[inds1])
            )
            inds1 = inds1[mt_msk]
            inds2 = inds2[mt_msk]

            # pair clusters
            inds1_unq = self._smart_pairing(cat1, cat2, inds1, inds2, "mt_multi_self")
            self._smart_pairing(cat2, cat1, inds2, inds1, "mt_multi_other")
            self._cat1_mmt[inds1_unq] = True

    @staticmethod
    def _smart_pairing(cat1, cat2, inds1, inds2, mt_col):
        """
        More efficient pairing for matched catalogs
        """
        # Find how many galaxies are associated to each detection
        inds1_unq, counts = np.unique(inds1, return_counts=True)

        # Find positions to split by inds1 types
        split_pos = counts.cumsum()

        # For this, inds1 must be sorted
        argsort = np.argsort(inds1)

        # add pairings
        cat1[mt_col][inds1_unq] = list(
            map(list, np.split(cat2["id"][inds2[argsort]], split_pos)[:-1])
        )

        return inds1_unq

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
        if cat.mt_input is None:
            cat.mt_input = ClData()
        # Set zmin, zmax
        self._prep_z_for_match(cat, delta_z, n_delta_z)
        # Set angular radius
        self._prep_radius_for_match(cat, match_radius, n_match_radius=n_match_radius, cosmo=cosmo)
        # add conf to history of matching
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
        coeff1, coeff2 = None, None  # for pylint complaining
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

        self._unique_match_from_config(cat1, cat2, match_config)
