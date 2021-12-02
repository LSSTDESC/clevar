import numpy as np
import pickle
from scipy.interpolate import InterpolatedUnivariateSpline as spline

from .parent import Match
from .proximity import ProximityMatch
from ..geometry import units_bank, convert_units
from ..catalog import ClData
from ..utils import veclen, str2dataunit


class MembershipMatch(Match):
    def __init__(self, ):
        self.type = 'Membership'
        self.matched_mems = None
    def multiple(self, cat1, cat2):
        """
        Make the one way multiple matching

        Parameters
        ----------
        cat1: clevar.ClCatalog
            Base catalog with members attribute.
        cat2: clevar.ClCatalog
            Target catalog with members attribute.
        """
        self.cat1_mmt = np.zeros(cat1.size, dtype=bool) # To add flag in multi step matching
        print(f'Finding candidates ({cat1.name})')
        for i, (share_mems1, nmem1) in enumerate(zip(cat1.mt_input['share_mems'], cat1.mt_input['nmem'])):
            for id2, share_mem in share_mems1.items():
                cat1['mt_multi_self'][i].append(id2)
                i2 = int(cat2.id_dict[id2])
                cat2['mt_multi_other'][i2].append(cat1['id'][i])
                self.cat1_mmt[i] = True
            print(f"  {i:,}({cat1.size:,}) - {len(cat1['mt_multi_self'][i]):,} candidates", end='\r')
        print(f'* {(veclen(cat1["mt_multi_self"])>0).sum():,}/{cat1.size:,} objects matched.')
        cat1.remove_multiple_duplicates()
        cat2.remove_multiple_duplicates()
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
            raise ValueError('Members not matched, run match_members before.')
        if 'pmem' in cat1.members.data.colnames:
            cat1.members['pmem'] = 1.
        if 'pmem' in cat2.members.data.colnames:
            cat2.members['pmem'] = 1.
        cat1.mt_input = ClData({'share_mems': [{} for i in range(cat1.size)],
                                'nmem': self._comp_nmem(cat1)})
        cat2.mt_input = ClData({'share_mems': [{} for i in range(cat2.size)],
                                'nmem': self._comp_nmem(cat2)})
        for mem1_, mem2_ in zip(cat1.members[self.matched_mems[:,0]],
                                cat2.members[self.matched_mems[:,1]]):
            self._add_pmem(cat1.mt_input['share_mems'], cat1.id_dict[mem1_['id_cluster']],
                           mem2_['id_cluster'], mem1_['pmem'])
            self._add_pmem(cat2.mt_input['share_mems'], cat2.id_dict[mem2_['id_cluster']],
                           mem1_['id_cluster'], mem2_['pmem'])
        # sort order in dicts by mass
        cat1.mt_input['share_mems'] = [self._sort_share_mem_mass(share_mem1, cat2)
                                        for share_mem1 in cat1.mt_input['share_mems']]
        cat2.mt_input['share_mems'] = [self._sort_share_mem_mass(share_mem2, cat1)
                                        for share_mem2 in cat2.mt_input['share_mems']]
    def _sort_share_mem_mass(self, share_mem1, cat2):
        """
        Sorts members in dict by mass (decreasing).
        """
        if len(share_mem1)==0:
            return {}
        ids2 = np.array(list(share_mem1.keys()))
        mass2 = np.array(cat2['mass'][cat2.ids2inds(ids2)])
        return {id2:share_mem1[id2] for id2 in ids2[mass2.argsort()[::-1]]}
    def _comp_nmem(self, cat):
        """
        Computes number of members for clusters (sum of pmem)

        Parameters
        ----------
        cat: clevar.ClCatalog
            Cluster catalog with members attribute.
        """
        return [cat.members['pmem'][cat.members['ind_cl']==i].sum()
                for i in range(cat.size)]
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
        cat1_share_mems[ind1][cat2_id] = cat1_share_mems[ind1].get(cat2_id, 0)+pmem1
    def save_shared_members(self, cat1, cat2, fileprefix, overwrite=False):
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
        overwrite: bool
            Overwrite saved files
        """
        pickle.dump({c:cat1.mt_input[c] for c in cat1.mt_input.colnames},
                    open(f'{fileprefix}.1.p', 'wb'))
        pickle.dump({c:cat2.mt_input[c] for c in cat2.mt_input.colnames},
                    open(f'{fileprefix}.2.p', 'wb'))
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
        cat1.mt_input = ClData(pickle.load(open(f'{fileprefix}.1.p', 'rb')))
        cat2.mt_input = ClData(pickle.load(open(f'{fileprefix}.2.p', 'rb')))
    def match_members(self, mem1, mem2, method='id', radius=None, cosmo=None):
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
        if method=='id':
            self._match_members_by_id(mem1, mem2)
        elif method=='angular_distance':
            self._match_members_by_ang(mem1, mem2, radius, cosmo)
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
        self.matched_mems = np.array([[ind1, ind2]
            for ind2, i in enumerate(mem2['id'])
                for ind1 in mem1.id_dict_list.get(i, [])
                ])
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
            'type': 'cross', # options are cross, cat1, cat2
            'which_radius': 'max', # Case of radius to be used, can be: cat1, cat2, min, max
            'preference': 'angular_proximity', # options are more_massive, angular_proximity or redshift_proximity
            'catalog1': {'delta_z':None, 'match_radius': radius},
            'catalog2': {'delta_z':None, 'match_radius': radius},
            }
        mt = ProximityMatch()
        mem1._init_match_vals()
        mem2._init_match_vals()
        mt.match_from_config(mem1, mem2, match_config, cosmo=cosmo)
        mask1 = mem1.get_matching_mask(match_config['type'])
        mask2 = mem2.ids2inds(mem1[mask1][f"mt_{match_config['type']}"])
        self.matched_mems = np.transpose([np.arange(mem1.size, dtype=int)[mask1],
                                          np.arange(mem2.size, dtype=int)[mask2]])
    def save_matched_members(self, filename, overwrite=False):
        """
        Saves the matching results of members

        Parameters
        ----------
        filename: str
            Name of file
        overwrite: bool
            Overwrite saved files
        """
        np.savetxt(filename, self.matched_mems, fmt='%d')
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
                `type` -  type of matching, can be: `cat1`, `cat2`, `cross`.
                `preference` -  Preference to set best match, can be: `more_massive`, `angular_proximity`, `redshift_proximity`, `shared_member_fraction` (default).
                `minimum_share_fraction` -  Minimum share fraction to consider in matches (default=0).
                `match_members` -  Match the members catalogs (default=`True`).
                `match_members_kwargs` -  `kwargs` used in `match_members(mem1, mem2, **kwargs)`, needed if `match_members=True`.
                `match_members_save` -  saves file with matched members (default=`False`).
                `match_members_load` -  load matched members (default=`False`), if `True` skips matching (and save) of members.
                `match_members_file` -  file to save matching of members, needed if `match_members_save` or `match_members_load` is `True`.
                `shared_members_fill` -  Adds shared members dicts and nmem to mt_input in catalogs (default=`True`).
                `shared_members_save` -  saves files with shared members (default=`False`).
                `shared_members_load` -  load files with shared members (default=`False`), if `True` skips matching (and save) of members and fill (and save) of shared members.
                `shared_members_file` -  Prefix of file names to save shared members, needed if `shared_members_save` or `shared_members_load` is `True`.
        """
        if match_config['type'] not in ('cat1', 'cat2', 'cross'):
            raise ValueError("config type must be cat1, cat2 or cross")

        load_mt_member = match_config.get('match_members_load', False)
        if match_config.get('match_members', True) and not load_mt_member:
            self.match_members(cat1.members, cat2.members, **match_config['match_members_kwargs'])
        if match_config.get('match_members_save', False) and not load_mt_member:
            self.save_matched_members(match_config['match_members_file'], overwrite=True)
        if load_mt_member:
            self.load_matched_members(match_config['match_members_file'])

        load_shared_member = match_config.get('shared_members_load', False)
        if match_config.get('shared_members_fill', True) and not load_shared_member:
            self.fill_shared_members(cat1, cat2)
        if match_config.get('shared_members_save', False) and not load_shared_member:
            self.save_shared_members(cat1, cat2, match_config['shared_members_file'], overwrite=True)
        if load_shared_member:
            self.load_shared_members(cat1, cat2, match_config['shared_members_file'])

        if match_config['type'] in ('cat1', 'cross'):
            print("\n## Multiple match (catalog 1)")
            self.multiple(cat1, cat2)
        if match_config['type'] in ('cat2', 'cross'):
            print("\n## Multiple match (catalog 2)")
            self.multiple(cat2, cat1)

        preference = match_config.get('preference', 'shared_member_fraction')
        minimum_share_fraction = match_config.get('minimum_share_fraction', 0)
        if match_config['type'] in ('cat1', 'cross'):
            print("\n## Finding unique matches of catalog 1")
            self.unique(cat1, cat2, preference, minimum_share_fraction)
        if match_config['type'] in ('cat2', 'cross'):
            print("\n## Finding unique matches of catalog 2")
            self.unique(cat2, cat1, preference, minimum_share_fraction)

        if match_config['type'] == 'cross':
            self.cross_match(cat1)
            self.cross_match(cat2)
