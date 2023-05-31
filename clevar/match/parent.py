"""@file parent.py
The Match class
"""
import os
import numpy as np


class Match:
    """
    Matching Class
    """

    def __init__(self):
        self.type = None
        self.history = []
        self._cat1_mt = None
        self._cat1_mmt = None

    def prep_cat_for_match(self, cat, *args):
        """
        Prepare the catalog for matching, will fill the cat.mt_input object.
        Each method must implement its specifics preparations.
        """
        raise NotImplementedError

    def multiple(self, cat1, cat2, verbose=True):
        """Makes multiple matchig

        Parameters
        ----------
        cat1: clevar.ClCatalog
            Base catalog
        cat2: clevar.ClCatalog
            Target catalog
        verbose: bool
            Print result for individual matches.

        Note
        ----
            Not implemented in parent class
        """
        raise NotImplementedError

    def unique(self, cat1, cat2, preference, minimum_share_fraction=0):
        """Makes unique matchig, requires multiple matching to be made first

        Parameters
        ----------
        cat1: clevar.ClCatalog
            Base catalog
        cat2: clevar.ClCatalog
            Target catalog
        preference: str
            Preference to set best match. Options are: `'more_massive'`, `'angular_proximity'`,
            `'redshift_proximity'`, `'shared_member_fraction'`.
        minimum_share_fraction: float
            Parameter for `preference='shared_member_fraction'`.
            Minimum share fraction to consider in matches (default=0).
        """
        self._cat1_mt = np.zeros(cat1.size, dtype=bool)  # To add flag in multi step matching
        if "mass" in cat1.tags:
            i_vals = np.argsort(cat1["mass"])[::-1]
        else:
            i_vals = range(cat1.size)
        if preference == "more_massive":
            def set_unique(*args):
                return  self._match_mpref(*args)
        elif preference == "angular_proximity":
            def set_unique(*args):
                return  self._match_apref(*args, "angular_proximity")
        elif preference == "redshift_proximity":
            def set_unique(*args):
                return  self._match_apref(*args, "redshift_proximity")
        elif preference == "shared_member_fraction":
            cat1["mt_frac_self"] = np.zeros(cat1.size)
            cat2["mt_frac_other"] = np.zeros(cat2.size)
            if "mt_frac_other" not in cat1.colnames:
                cat1["mt_frac_other"] = np.zeros(cat1.size)
            def set_unique(*args):
                return  self._match_sharepref(*args, minimum_share_fraction)
        else:
            raise ValueError(
                "preference must be 'more_massive', 'angular_proximity' or 'redshift_proximity'"
            )
        print(f"Unique Matches ({cat1.name})")
        for ind1 in i_vals:
            if cat1["mt_self"][ind1] is None:
                self._cat1_mt[ind1] = set_unique(cat1, ind1, cat2)
        self._cat1_mt *= cat1.get_matching_mask("self")  # In case ang pref removes a match
        print(f'* {cat1.get_matching_mask("self").sum():,}/{cat1.size:,} objects matched.')
        cfg = {"func": "unique", "cats": f"{cat1.name}, {cat2.name}", "preference": preference}
        if preference == "shared_member_fraction":
            cfg["minimum_share_fraction"] = minimum_share_fraction
        self.history.append(cfg)
        # pylint: disable=protected-access
        cat1._set_mt_hist(self.history)
        cat2._set_mt_hist(self.history)

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
            Dictionary with the matching configuration.
        cosmo: clevar.Cosmology object
            Cosmology object for when radius has physical units

        Note
        ----
            Not implemented in parent class
        """
        raise NotImplementedError

    def _match_mpref(self, cat1, ind1, cat2):
        """
        Make the unique match by mass preference

        Parameters
        ----------
        cat1: clevar.ClCatalog
            Base catalog
        ind1: int
            Index of the cluster from cat1 to be matched
        cat2: clevar.ClCatalog
            Target catalog

        Returns
        -------
        bool
            Tells if the cluster was matched
        """
        inds2 = cat2.ids2inds(cat1["mt_multi_self"][ind1])
        if len(inds2) > 0:
            for ind2 in inds2[np.argsort(cat2["mass"][inds2])][::-1]:
                if cat2["mt_other"][ind2] is None:
                    cat1["mt_self"][ind1] = cat2["id"][ind2]
                    cat2["mt_other"][ind2] = cat1["id"][ind1]
                    return True
        return False

    def _match_apref(self, cat1, ind1, cat2, match_pref):
        """
        Make the unique match by angular (or redshift) distance preference

        Parameters
        ----------
        cat1: clevar.ClCatalog
            Base catalog
        ind1: int
            Index of the cluster from cat1 to be matched
        cat2: clevar.ClCatalog
            Target catalog
        match_pref: str
            Matching preference, can be 'angular_proximity' or 'redshift_proximity'

        Returns
        -------
        bool
            Tells if the cluster was matched
        """
        inds2 = cat2.ids2inds(cat1["mt_multi_self"][ind1])
        dists = self._get_dist_mt(cat1[ind1], cat2[inds2], match_pref)
        sort_d = np.argsort(dists)
        for dist, ind2 in zip(dists[sort_d], inds2[sort_d]):
            i1_replace = cat1.id_dict[cat2["mt_other"][ind2]] if cat2["mt_other"][ind2] else None
            if i1_replace is None:
                cat1["mt_self"][ind1] = cat2["id"][ind2]
                cat2["mt_other"][ind2] = cat1["id"][ind1]
                return True
            if dist < self._get_dist_mt(cat1[i1_replace], cat2[ind2], match_pref):
                cat1["mt_self"][ind1] = cat2["id"][ind2]
                cat2["mt_other"][ind2] = cat1["id"][ind1]
                cat1["mt_self"][i1_replace] = None
                self._match_apref(cat1, i1_replace, cat2, match_pref)
                return True
        return False

    def _match_sharepref(self, cat1, ind1, cat2, minimum_share_fraction=0):
        """
        Make the unique match by shared membership fraction preference

        Parameters
        ----------
        cat1: clevar.ClCatalog
            Base catalog
        ind1: int
            Index of the cluster from cat1 to be matched
        cat2: clevar.ClCatalog
            Target catalog
        minimum_share_fraction: float
            Parameter for `preference='shared_member_fraction'`.
            Minimum share fraction to consider in matches (default=0).

        Returns
        -------
        bool
            Tells if the cluster was matched
        """
        if len(cat1["mt_multi_self"][ind1]) == 0:
            return False
        for id2 in sorted(
            cat1.mt_input["share_mems"][ind1], key=cat1.mt_input["share_mems"][ind1].get, reverse=True
        ):
            shared_frac = cat1.mt_input["share_mems"][ind1][id2] / cat1.mt_input["nmem"][ind1]
            if shared_frac < minimum_share_fraction:
                return False
            # fills cat2 mt_other if empty or shared_frac is greater
            ind2 = cat2.id_dict[id2]
            share_mems2 = cat2.mt_input["share_mems"][ind2]
            id1, id1_replace = cat1["id"][ind1], cat2["mt_other"][ind2]
            if share_mems2[id1] >= share_mems2.get(id1_replace, 0):
                i1_replace = cat1.id_dict.get(id1_replace)
                if i1_replace is not None and (
                    share_mems2[id1] == share_mems2[id1_replace]
                    and cat1["mass"][ind1] <= cat1["mass"][i1_replace]
                ):
                    return False
                cat1["mt_self"][ind1] = id2
                cat1["mt_frac_self"][ind1] = shared_frac
                cat2["mt_other"][ind2] = id1
                cat2["mt_frac_other"][ind2] = shared_frac
                if i1_replace is not None:
                    cat1["mt_self"][i1_replace] = None
                    cat1["mt_frac_other"][i1_replace] = 0
                    self._match_sharepref(cat1, i1_replace, cat2, minimum_share_fraction)
                return True
        return False

    def _get_dist_mt(self, dat1, dat2, match_pref):
        """
        Get distance for matching preference

        Parameters
        ----------
        dat1: clevar.ClData
            Data of base catalog
        dat2: clevar.ClData
            Data of target catalog
        match_pref: str
            Matching preference, can be 'angular_proximity' or 'redshift_proximity'

        Return
        ------
        bool
            If there was a match
        """
        if match_pref == "angular_proximity":
            return dat1["SkyCoord"].separation(dat2["SkyCoord"]).value
        if match_pref == "redshift_proximity":
            return abs(dat1["z"] - dat2["z"])
        raise ValueError(
            "match_pref (={match_pref}) value not accepted."
            " It must be angular_proximity or redshift_proximity"
            )

    def cross_match(self, cat1):
        """Makes cross matches of catalog, requires unique matches to be done first.

        Parameters
        ----------
        cat1: clevar.ClCatalog
            Base catalog
        """
        print(f"Cross Matches ({cat1.name})")
        cat1.cross_match()
        print(f'* {cat1.get_matching_mask("cross").sum():,}/{cat1.size:,} objects matched.')

    def save_matches(self, cat1, cat2, out_dir, overwrite=False):
        """
        Saves the matching results

        Parameters
        ----------
        cat1: clevar.ClCatalog
            ClCatalog 1
        cat2: clevar.ClCatalog
            ClCatalog 2
        out_dir: str
            Path of directory to save output
        overwrite: bool
            Overwrite saved files
        """
        if not os.path.isdir(out_dir):
            os.system(f"mkdir {out_dir}")
        cat1.save_match(f"{out_dir}/match1.fits", overwrite=overwrite)
        cat2.save_match(f"{out_dir}/match2.fits", overwrite=overwrite)

    def load_matches(self, cat1, cat2, out_dir):
        """
        Load matching results to catalogs

        Parameters
        ----------
        cat1: clevar.ClCatalog
            ClCatalog 1
        cat2: clevar.ClCatalog
            ClCatalog 2
        out_dir: str
            Path of directory with saved match files
        """
        print(cat1.name)
        cat1.load_match(f"{out_dir}/match1.fits")
        print()
        print(cat2.name)
        cat2.load_match(f"{out_dir}/match2.fits")
