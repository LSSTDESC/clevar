"""@file box.py
The BoxMatch class
"""
import numpy as np

from .spatial import SpatialMatch
from ..catalog import ClData

_area_type_funcs = {
    "min": lambda area1, area2: np.min([area1, area2], axis=0),
    "max": lambda area1, area2: np.max([area1, area2], axis=0),
    "self": lambda area1, area2: area1,
    "other": lambda area1, area2: area2,
}


class BoxMatch(SpatialMatch):
    """
    BoxMatch Class

    Attributes
    ----------
    type : str
        Type of matching object. Set to "Box"
    history : list
        Steps in the matching
    """

    def __init__(self):
        SpatialMatch.__init__(self)
        self.type = "Box"

    def multiple(
        self,
        cat1,
        cat2,
        metric="GIoU",
        metric_cut=0.5,
        share_area_frac=0.5,
        verbose=True,
        detailed_print_only=False,
    ):
        """
        Make the one way multiple matching

        Parameters
        ----------
        cat1: clevar.ClCatalog
            Base catalog
        cat2: clevar.ClCatalog
            Target catalog
        metric: str (optional)
            Metric to be used for matching. Can be: GIoU (generalized Intersection over Union).
        metric_cut: float
            Minimum value of metric for match.
        share_area_frac: float
            Minimum relative size of area for match.
        verbose: bool
            Print result for individual matches.
        detailed_print_only: bool
            Only prints detailed comparisons for matching, does not register matches in catalogs.
        """
        # pylint: disable=arguments-renamed
        # pylint: disable=too-many-locals
        if cat1.mt_input is None:
            raise AttributeError("cat1.mt_input is None, run prep_cat_for_match first.")
        if cat2.mt_input is None:
            raise AttributeError("cat2.mt_input is None, run prep_cat_for_match first.")

        if metric == "GIoU":

            def get_metric_value(*args):
                return self._compute_giou(*args)

        elif metric[:3] == "IoA" and metric[3:] in ("min", "max", "self", "other"):

            def get_metric_value(*args):
                return self._compute_intersection_over_area(*args, area_type=metric[3:])

        else:
            raise ValueError("metric must be GIoU.")

        self._cat1_mmt = np.zeros(cat1.size, dtype=bool)  # To add flag in multi step matching
        ra2min, ra2max, dec2min, dec2max = (
            cat2[c] for c in ("ra_min", "ra_max", "dec_min", "dec_max")
        )
        z2min, z2max = (cat2.mt_input[c] for c in ("zmin", "zmax"))
        print(f"Finding candidates ({cat1.name})")
        for ind1, (ra1min, ra1max, dec1min, dec1max, z1min, z1max) in enumerate(
            zip(
                *(cat1[c] for c in ("ra_min", "ra_max", "dec_min", "dec_max")),
                *(cat1.mt_input[c] for c in ("zmin", "zmax")),
            )
        ):
            self._detailed_print(0, locals())
            # crop in redshift range
            mask = (z2max >= z1min) * (z2min <= z1max)
            if mask.any():
                self._detailed_print(1, locals())
                # makes square crop with intersection
                mask[mask] *= self.mask_intersection(
                    ra1min,
                    ra1max,
                    dec1min,
                    dec1max,
                    ra2min[mask],
                    ra2max[mask],
                    dec2min[mask],
                    dec2max[mask],
                )
                if mask.any():
                    self._detailed_print(2, locals())
                    area1, area2, intersection, outter = self._compute_areas(
                        *(
                            [ra1min, ra1max, dec1min, dec1max] * np.ones(mask.sum())[:, None]
                        ).T,  # for vec computation
                        ra2min[mask],
                        ra2max[mask],
                        dec2min[mask],
                        dec2max[mask],
                    )
                    # makes metric crop
                    metric_value = get_metric_value(area1, area2, intersection, outter)
                    self._detailed_print(3, locals())
                    if not detailed_print_only:
                        for id2 in cat2["id"][mask][
                            (metric_value >= metric_cut)
                            * (area1 / area2 >= share_area_frac)
                            * (area2 / area1 >= share_area_frac)
                        ]:
                            cat1["mt_multi_self"][ind1].append(id2)
                            ind2 = int(cat2.id_dict[id2])
                            cat2["mt_multi_other"][ind2].append(cat1["id"][ind1])
                            self._cat1_mmt[ind1] = True
                else:
                    self._detailed_print(4, locals())
            else:
                self._detailed_print(5, locals())
            if verbose:
                self._prt_cand_mt(cat1, ind1)
        if not detailed_print_only:
            hist = {
                "func": "multiple",
                "cats": f"{cat1.name}, {cat2.name}",
            }
            self._rm_dup_add_hist(cat1, cat2, hist)

    def _detailed_print(self, i, locs):
        if not locs["detailed_print_only"]:
            return
        if i == 0:
            print(
                "\n\nCluster:",
                "(",
                ", ".join([f"{locs[v]:.4f}" for v in ("ra1min", "ra1max", "dec1min", "dec1max")]),
                ") (",
                ", ".join([f"{locs[v]:.4f}" for v in ("z1min", "z1max")]),
                ")",
                f"( area: {self._compute_area(locs['ra1min'], locs['ra1max'], locs['dec1min'], locs['dec1max'])*3600:.2f} arcmin2 )",
            )
        elif i == 1:
            print(f" * z pass: {locs['mask'].sum():,}")
        elif i == 2:
            print(f" * intersection pass: {locs['mask'].sum():,}")
        elif i == 3:
            for val in zip(
                locs["cat2"]["id"][locs["mask"]],
                locs["ra2min"][locs["mask"]],
                locs["ra2max"][locs["mask"]],
                locs["dec2min"][locs["mask"]],
                locs["dec2max"][locs["mask"]],
                locs["area1"],
                locs["area2"],
                locs["intersection"],
                locs["outter"],
                locs["metric_value"],
                locs["metric_value"] >= locs["metric_cut"],
                locs["area1"] / locs["area2"] >= locs["share_area_frac"],
                locs["area2"] / locs["area1"] >= locs["share_area_frac"],
            ):
                print("   Candidate:", val[0])
                print(f"        Pass   : {bool(np.prod(val[10:]))} (", *val[10:], ")")
                print(
                    "        Areas  :",
                    ", ".join([f"{v*3600:.2f}" for v in val[6:9]]),
                    f"arcmin2 ( {locs['metric']}: {val[9]:.2g} )",
                )
                print(
                    "        Coords : (",
                    ", ".join([f"{v:.4f}" for v in val[1:5]]),
                    ")",
                )
        elif i == 4:
            print(" * intersection fail")
        elif i == 5:
            print(" * z fail!")
        else:
            raise ValueError(f"i={i} invalid!")

    def mask_intersection(self, ra1min, ra1max, dec1min, dec1max, ra2min, ra2max, dec2min, dec2max):
        """Mask clusters without intersection

        Return
        ------
        mask
            If clusters have intersection
        """
        mask = ra1min <= ra2max
        if not mask.any():
            return mask
        mask *= ra2min <= ra1max
        if not mask.any():
            return mask
        mask *= dec1min <= dec2max
        if not mask.any():
            return mask
        mask *= dec2min <= dec1max
        if not mask.any():
            return mask
        return mask

    def _compute_area(self, ra_min, ra_max, dec_min, dec_max):
        return (ra_max - ra_min) * (dec_max - dec_min)

    def _compute_areas(self, ra1min, ra1max, dec1min, dec1max, ra2min, ra2max, dec2min, dec2max):
        area1 = self._compute_area(ra1max, ra1min, dec1max, dec1min)
        area2 = self._compute_area(ra2max, ra2min, dec2max, dec2min)
        # areas
        intersection = self._compute_area(
            np.max([ra1min, ra2min], axis=0),
            np.min([ra1max, ra2max], axis=0),
            np.max([dec1min, dec2min], axis=0),
            np.min([dec1max, dec2max], axis=0),
        )
        outter = self._compute_area(
            np.min([ra1min, ra2min], axis=0),
            np.max([ra1max, ra2max], axis=0),
            np.min([dec1min, dec2min], axis=0),
            np.max([dec1max, dec2max], axis=0),
        )
        return area1, area2, intersection, outter

    def _compute_giou(self, area1, area2, intersection, outter):
        union = area1 + area2 - intersection
        return intersection / union + union / outter - 1.0

    def _compute_intersection_over_area(self, area1, area2, intersection, outter, area_type):
        return intersection / _area_type_funcs[area_type](area1, area2)

    def prep_cat_for_match(
        self,
        cat,
        delta_z,
        n_delta_z=1,
        ra_min="ra_min",
        ra_max="ra_max",
        dec_min="dec_min",
        dec_max="dec_max",
    ):
        """
        Adds zmin and zmax to cat.mt_input and tags ra/dec min/max cols

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
        # pylint: disable=arguments-differ
        # pylint: disable=unused-argument
        print("## Prep mt_cols")
        if cat.mt_input is None:
            cat.mt_input = ClData()
        self._prep_z_for_match(cat, delta_z, n_delta_z)
        for name in ("ra_min", "ra_max", "dec_min", "dec_max"):
            if name not in cat.tags:
                cat.tag_column(locals()[name], name)
        self.history.append(
            {
                "func": "prep_cat_for_match",
                "cat": cat.name,
                "delta_z": delta_z,
                "n_delta_z": n_delta_z,
            }
        )

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
