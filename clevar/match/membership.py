"""@file membership.py
The MembershipMatch class
"""
import pickle
import numpy as np

from .parent import Match
from .proximity import ProximityMatch
from ..catalog import ClData


class MembershipMatch(Match):
    """
    MembershipMatch Class

    Attributes
    ----------
    type : str
        Type of matching object. Set to "Membership"
    history : list
        Steps in the matching
    """

    # pylint: disable=abstract-method
    def __init__(self):
        Match.__init__(self)
        self.type = "Membership"
        self.matched_mems = None
        self._valid_unique_preference_vals += ["shared_member_fraction"]

    def multiple(self, cat1, cat2, minimum_share_fraction=0, verbose=True):
        """
        Make the one way multiple matching

        Parameters
        ----------
        cat1: clevar.ClCatalog
            Base catalog with members attribute.
        cat2: clevar.ClCatalog
            Target catalog with members attribute.
        minimum_share_fraction: float
            Parameter for `preference='shared_member_fraction'`.
            Minimum share fraction to consider in matches (default=0).
        verbose: bool
            Print result for individual matches.
        """
        # pylint: disable=arguments-renamed
        if cat1.mt_input is None:
            raise AttributeError("cat1.mt_input is None, run fill_shared_members first.")
        if cat2.mt_input is None:
            raise AttributeError("cat2.mt_input is None, run fill_shared_members first.")
        self._cat1_mmt = np.zeros(cat1.size, dtype=bool)  # To add flag in multi step matching
        print(f"Finding candidates ({cat1.name})")
        for ind1, (share_mems1, nmem1) in enumerate(
            zip(cat1.mt_input["share_mems"], cat1.mt_input["nmem"])
        ):
            for id2, num_shared_mem2 in share_mems1.items():
                ind2 = int(cat2.id_dict[id2])
                if num_shared_mem2 / nmem1 >= minimum_share_fraction:
                    cat1["mt_multi_self"][ind1].append(id2)
                    cat2["mt_multi_other"][ind2].append(cat1["id"][ind1])
                    self._cat1_mmt[ind1] = True
            if verbose:
                self._prt_cand_mt(cat1, ind1)
        hist = {"func": "multiple", "cats": f"{cat1.name}, {cat2.name}"}
        self._rm_dup_add_hist(cat1, cat2, hist)

    def fill_shared_members(self, cat1, cat2):
        """
        Adds shared members dicts and nmem to mt_input in catalogs.

        Parameters
        ----------
        cat1: clevar.ClCatalog
            Cluster catalog with members attribute.
        cat2: clevar.ClCatalog
            Cluster catalog with members attribute.
        """
        if self.matched_mems is None:
            raise AttributeError("Members not matched, run match_members before.")
        if "pmem" not in cat1.members.tags:
            cat1.members["pmem"] = 1.0
        if "pmem" not in cat2.members.tags:
            cat2.members["pmem"] = 1.0
        cat1.mt_input = ClData(
            {"share_mems": list(map(lambda x: {}, range(cat1.size))), "nmem": self._comp_nmem(cat1)}
        )
        cat2.mt_input = ClData(
            {"share_mems": list(map(lambda x: {}, range(cat2.size))), "nmem": self._comp_nmem(cat2)}
        )
        for ind1, ind2 in self.matched_mems:
            idcl1, pmem1 = cat1.members["id_cluster"][ind1], cat1.members["pmem"][ind1]
            idcl2, pmem2 = cat2.members["id_cluster"][ind2], cat2.members["pmem"][ind2]
            self._add_pmem(cat1.mt_input["share_mems"], cat1.id_dict[idcl1], idcl2, pmem1)
            self._add_pmem(cat2.mt_input["share_mems"], cat2.id_dict[idcl2], idcl1, pmem2)
        # sort order in dicts by mass
        cat1.mt_input["share_mems"] = [
            self._sort_share_mem_mass(share_mem1, cat2)
            for share_mem1 in cat1.mt_input["share_mems"]
        ]
        cat2.mt_input["share_mems"] = [
            self._sort_share_mem_mass(share_mem2, cat1)
            for share_mem2 in cat2.mt_input["share_mems"]
        ]

    def _sort_share_mem_mass(self, share_mem1, cat2):
        """
        Sorts members in dict by mass (decreasing).
        """
        if len(share_mem1) == 0:
            return {}
        ids2 = np.array(list(share_mem1.keys()))
        mass2 = np.array(cat2["mass"][cat2.ids2inds(ids2)])
        return {id2: share_mem1[id2] for id2 in ids2[mass2.argsort()[::-1]]}

    def _comp_nmem(self, cat):
        """
        Computes number of members for clusters (sum of pmem)

        Parameters
        ----------
        cat: clevar.ClCatalog
            Cluster catalog with members attribute.
        """
        out = np.zeros(cat.size)
        for ind, pmem in zip(cat.members["ind_cl"], cat.members["pmem"]):
            out[ind] += pmem
        return out

    def _add_pmem(self, cat1_share_mems, ind1, cat2_id, pmem1):
        """
        Adds pmem of specific cluster to mt_input['shared_mem'] of another cluster.

        Parameters
        ----------
        cat1_share_mems: list
            List with dictionaries of shared members (cat1.mt_input['share_mems']).
        ind1: int
            Index of cat1 to add pmem to
        cat2_id: str
            Id of catalog2 cluster to add pmem of
        pmem1: float
            Pmem of catalog1 galaxy
        """
        cat1_share_mems[ind1][cat2_id] = cat1_share_mems[ind1].get(cat2_id, 0) + pmem1

    def _set_unique_matching_function(self, preference, **kwargs):
        def set_unique(*args):
            return self._match_sharepref(*args, kwargs["minimum_share_fraction"])

        return set_unique

    def save_shared_members(self, cat1, cat2, fileprefix):
        """
        Saves dictionaries of shared members

        Parameters
        ----------
        cat1: clevar.ClCatalog
            Cluster catalog with members attribute.
        cat2: clevar.ClCatalog
            Cluster catalog with members attribute.
        fileprefix: str
            Prefix for name of files
        """
        if cat1.mt_input is None:
            raise AttributeError("cat1.mt_input is None cannot save it.")
        if cat2.mt_input is None:
            raise AttributeError("cat2.mt_input is None cannot save it.")
        with open(f"{fileprefix}.1.p", "wb") as handle:
            pickle.dump(
                {c: cat1.mt_input[c] for c in cat1.mt_input.colnames},
                handle,
            )
        with open(f"{fileprefix}.2.p", "wb") as handle:
            pickle.dump(
                {c: cat2.mt_input[c] for c in cat2.mt_input.colnames},
                handle,
            )

    def load_shared_members(self, cat1, cat2, fileprefix):
        """
        Load dictionaries of shared members

        Parameters
        ----------
        cat1: clevar.ClCatalog
            Base catalog
        cat2: clevar.ClCatalog
            Target catalog
        filename: str
            Prefix of files name
        """
        with open(f"{fileprefix}.1.p", "rb") as handle:
            cat1.mt_input = ClData(pickle.load(handle))
        with open(f"{fileprefix}.2.p", "rb") as handle:
            cat2.mt_input = ClData(pickle.load(handle))

    def match_members(self, mem1, mem2, method="id", radius=None, cosmo=None):
        """
        Match member catalogs.
        Adds array with indices of matched members `(ind1, ind2)` to self.matched_mems.

        Parameters
        ----------
        mem1: clevar.ClCatalog
            Members of base catalog
        mem2: clevar.ClCatalog
            Members of target catalog
        method: str
            Method for matching. Options are `id` or `angular_distance`.
        radius: str, None
            For `method='angular_distance'`. Radius for matching,
            with format `'value unit'` - used fixed value (ex: `1 arcsec`, `1 Mpc`).
        cosmo: clevar.Cosmology, None
            For `method='angular_distance'`. Cosmology object for when radius has physical units.
        """
        if mem1 is None:
            raise AttributeError("members of catalog 1 is None, add members to catalog 1 first.")
        if mem2 is None:
            raise AttributeError("members of catalog 2 is None, add members to catalog 2 first.")
        if method == "id":
            self._match_members_by_id(mem1, mem2)
        elif method == "angular_distance":
            self._match_members_by_ang(mem1, mem2, radius, cosmo)
        print(f"{self.matched_mems.size:,} members were matched.")
        # Add id_cluster if found in other catalog
        mem1["match"] = None
        mem2["match"] = None
        for i in range(mem1.size):
            mem1["match"][i] = []
        for i in range(mem2.size):
            mem2["match"][i] = []
        for ind1, ind2 in self.matched_mems:
            mem1["match"][ind1].append(mem2["id_cluster"][ind2])
            mem2["match"][ind2].append(mem1["id_cluster"][ind1])
        # self.history.append({
        #    'func':'match_members', 'cat': cat.name, 'method': method,
        #    'cosmo': cosmo if cosmo is None else cosmo.get_desc()})
        # cfg = {'func':'match_members', 'cats': (cat1.name, cat2.name), 'method': method,
        #       'radius': radius, 'cosmo': cosmo if cosmo is None else cosmo.get_desc()}
        # cat1.mt_hist.append(cfg)
        # cat2.mt_hist.append(cfg)

    def _match_members_by_id(self, mem1, mem2):
        """
        Match member catalogs by id.
        Adds array with indices of matched members `(ind1, ind2)` to self.matched_mems.

        Parameters
        ----------
        mem1: clevar.ClCatalog
            Members of base catalog
        mem2: clevar.ClCatalog
            Members of target catalog
        """
        self.matched_mems = np.array(
            [
                [ind1, ind2]
                for ind2, i in enumerate(mem2["id"])
                for ind1 in mem1.id_dict_list.get(i, [])
            ]
        )

    def _match_members_by_ang(self, mem1, mem2, radius, cosmo):
        """
        Match member catalogs.
        Adds array with indices of matched members `(ind1, ind2)` to self.matched_mems.

        Parameters
        ----------
        mem1: clevar.ClCatalog
            Members of base catalog
        mem2: clevar.ClCatalog
            Members of target catalog
        method: str
            Method for matching. Options are `id` or `angular_distance`.
        radius: str, None
            For `method='angular_distance'`. Radius for matching,
            with format `'value unit'` - used fixed value (ex: `1 arcsec`, `1 Mpc`).
        cosmo: clevar.Cosmology, None
            For `method='angular_distance'`. Cosmology object for when radius has physical units.
        """
        match_config = {
            "type": "cross",
            "which_radius": "max",
            "preference": "angular_proximity",
            "catalog1": {"delta_z": None, "match_radius": radius},
            "catalog2": {"delta_z": None, "match_radius": radius},
        }
        prox_mt = ProximityMatch()
        # pylint: disable=protected-access
        mem1._init_match_vals()
        mem2._init_match_vals()
        prox_mt.match_from_config(mem1, mem2, match_config, cosmo=cosmo)
        mask1 = mem1.get_matching_mask(match_config["type"])
        self.matched_mems = []
        for ind1, id2 in zip(
            np.arange(mem1.size, dtype=int)[mask1], mem1[f"mt_{match_config['type']}"][mask1]
        ):
            for ind2 in mem2.id_dict_list[id2]:
                self.matched_mems.append([ind1, ind2])
        self.matched_mems = np.array(self.matched_mems)

    def save_matched_members(self, filename):
        """
        Saves the matching results of members

        Parameters
        ----------
        filename: str
            Name of file
        overwrite: bool
            Overwrite saved files
        """
        if self.matched_mems is None:
            raise AttributeError("self.matched_mems is None cannot save it.")
        np.savetxt(filename, self.matched_mems, fmt="%d")

    def load_matched_members(self, filename):
        """
        Load matching results of members

        Parameters
        ----------
        filename: str
            Name of file with matching results
        """
        self.matched_mems = np.loadtxt(filename, dtype=int)

    def match_from_config(self, cat1, cat2, match_config, cosmo=None):
        """
        Make matching of catalogs based on a configuration dictionary

        Parameters
        ----------
        cat1: clevar.ClCatalog
            ClCatalog 1
        cat2: clevar.ClCatalog
            ClCatalog 2
        mem1: clevar.ClCatalog
            Members of base catalog
        mem2: clevar.ClCatalog
            Members of target catalog
        match_config: dict
            Dictionary with the matching configuration. Keys must be:

                * `type` -  type of matching, can be: `cat1`, `cat2`, `cross`.
                * `preference` -  Preference to set best match, can be: `more_massive`,
                  `angular_proximity`, `redshift_proximity`, `shared_member_fraction` (default).
                * `minimum_share_fraction1` -  Minimum share fraction of catalog 1 to consider
                  in matches (default=0). It is used for both multiple and unique matches.
                * `minimum_share_fraction2` -  Minimum share fraction of catalog 2 to consider
                  in matches (default=0). It is used for both multiple and unique matches.
                * `match_members` -  Match the members catalogs (default=`True`).
                * `match_members_kwargs` -  `kwargs` used in `match_members(mem1, mem2, **kwargs)`,
                  needed if `match_members=True`.
                * `match_members_save` -  saves file with matched members (default=`False`).
                * `match_members_load` -  load matched members (default=`False`), if `True` skips
                  matching (and save) of members.
                * `match_members_file` -  file to save matching of members, needed if
                  `match_members_save` or `match_members_load` is `True`.
                * `shared_members_fill` -  Adds shared members dicts and nmem to mt_input
                  in catalogs (default=`True`).
                * `shared_members_save` -  saves files with shared members (default=`False`).
                * `shared_members_load` -  load files with shared members (default=`False`), if
                  `True` skips matching (and save) of members and fill (and save) of shared members.
                * `shared_members_file` -  Prefix of file names to save shared members,
                  needed if `shared_members_save` or `shared_members_load` is `True`.
                * `verbose`: Print result for individual matches (default=`True`).
                * `minimum_share_fraction1_unique` (optional) -  Minimum share fraction of
                  catalog 1 to consider in unique matches only. It overwrites
                  `minimum_share_fraction1` in the unique matching step.
                * `minimum_share_fraction2_unique` (optional) -  Minimum share fraction of
                  catalog 2 to consider in unique matches only. It overwrites
                  `minimum_share_fraction2` in the unique matching step.


        """
        if match_config["type"] not in ("cat1", "cat2", "cross"):
            raise ValueError("config type must be cat1, cat2 or cross")

        # Match members
        load_mt_member = match_config.get("match_members_load", False)
        if match_config.get("match_members", True) and not load_mt_member:
            self.match_members(cat1.members, cat2.members, **match_config["match_members_kwargs"])
        if match_config.get("match_members_save", False) and not load_mt_member:
            self.save_matched_members(match_config["match_members_file"])
        if load_mt_member:
            self.load_matched_members(match_config["match_members_file"])

        # Fill shared members
        load_shared_member = match_config.get("shared_members_load", False)
        if match_config.get("shared_members_fill", True) and not load_shared_member:
            self.fill_shared_members(cat1, cat2)
        if match_config.get("shared_members_save", False) and not load_shared_member:
            self.save_shared_members(
                cat1,
                cat2,
                match_config["shared_members_file"],
            )
        if load_shared_member:
            self.load_shared_members(cat1, cat2, match_config["shared_members_file"])

        # Multiple match
        verbose = match_config.get("verbose", True)
        if match_config["type"] in ("cat1", "cross"):
            print("\n## Multiple match (catalog 1)")
            self.multiple(
                cat1, cat2, match_config.get("minimum_share_fraction1", 0), verbose=verbose
            )
        if match_config["type"] in ("cat2", "cross"):
            print("\n## Multiple match (catalog 2)")
            self.multiple(
                cat2, cat1, match_config.get("minimum_share_fraction2", 0), verbose=verbose
            )

        # Unique match
        preference = match_config.get("preference", "shared_member_fraction")
        if match_config["type"] in ("cat1", "cross"):
            print("\n## Finding unique matches of catalog 1")
            self.unique(
                cat1,
                cat2,
                preference,
                match_config.get(
                    "minimum_share_fraction1_unique", match_config.get("minimum_share_fraction1", 0)
                ),
            )
        if match_config["type"] in ("cat2", "cross"):
            print("\n## Finding unique matches of catalog 2")
            self.unique(
                cat2,
                cat1,
                preference,
                match_config.get(
                    "minimum_share_fraction2_unique", match_config.get("minimum_share_fraction2", 0)
                ),
            )

        # Cross match
        if match_config["type"] == "cross":
            self.cross_match(cat1)
            self.cross_match(cat2)
