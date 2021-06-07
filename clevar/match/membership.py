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
    def multiple(self, cat1, cat2, minimum_share_fraction=0):
        """
        Make the one way multiple matching

        Parameters
        ----------
        cat1: clevar.ClCatalog
            Base catalog
        cat2: clevar.ClCatalog
            Target catalog
        minimum_share_fraction: float
            Minimum share fraction to consider in matches
        """
        self.cat1_mmt = np.zeros(cat1.size, dtype=bool) # To add flag in multi step matching
        print(f'Finding candidates ({cat1.name})')
        for i, (share_mems1, nmem1) in enumerate(zip(cat1.mt_input['share_mems'], cat1.mt_input['nmem'])):
            for id2, share_mem in share_mems1.items():
                if share_mem/nmem1 >= minimum_share_fraction:
                    cat1['mt_multi_self'][i].append(id2)
                    i2 = int(cat2.id_dict[id2])
                    cat2['mt_multi_other'][i2].append(cat1['id'][i])
                    self.cat1_mmt[i] = True
            print(f"  {i:,}({cat1.size:,}) - {len(cat1['mt_multi_self'][i]):,} candidates", end='\r')
        print(f'* {len(cat1[veclen(cat1["mt_multi_self"])>0]):,}/{cat1.size:,} objects matched.')
        cat1.remove_multiple_duplicates()
        cat2.remove_multiple_duplicates()
    def fill_shared_members(self, cat1, cat2, mem1, mem2):
        """
        Adds shared members dicts and nmem to mt_input in catalogs.

        Parameters
        ----------
        cat1: clevar.ClCatalog
            Base catalog
        cat2: clevar.ClCatalog
            Target catalog
        mem1: clevar.ClCatalog
            Members of base catalog
        mem2: clevar.ClCatalog
            Members of target catalog
        """
        if self.matched_mems is None:
            raise ValueError('Members not matched, run match_members before.')
        mem1['pmem'] = mem1['pmem'] if 'pmem' in mem1.data.colnames else np.ones(mem1.size)
        mem2['pmem'] = mem2['pmem'] if 'pmem' in mem2.data.colnames else np.ones(mem2.size)
        cat1.mt_input = {'share_mems': [{} for i in range(cat1.size)],
                         'nmem': self._comp_nmem(cat1, mem1)}
        cat2.mt_input = {'share_mems': [{} for i in range(cat2.size)],
                         'nmem': self._comp_nmem(cat2, mem2)}
        for mem1_, mem2_ in zip(mem1[self.matched_mems[:,0]], mem2[self.matched_mems[:,1]]):
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
    def _comp_nmem(self, cat, mem):
        """
        Computes number of members for clusters (sum of pmem)

        Parameters
        ----------
        cat: clevar.ClCatalog
            Cluster catalog
        mem: clevar.ClCatalog
            Members of cluster catalog
        """
        out = np.zeros(cat.size)
        for ID, pmem in zip(mem['id_cluster'], mem['pmem']):
            out[cat.id_dict[ID]] += pmem
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
        cat1_share_mems[ind1][cat2_id] = cat1_share_mems[ind1].get(cat2_id, 0)+pmem1
    def save_shared_members(self, cat1, cat2, fileprefix, overwrite=False):
        """
        Saves dictionaries of shared members

        Parameters
        ----------
        cat1: clevar.ClCatalog
            Base catalog
        cat2: clevar.ClCatalog
            Target catalog
        fileprefix: str
            Prefix for name of files
        overwrite: bool
            Overwrite saved files
        """
        pickle.dump(cat1.mt_input, open(f'{fileprefix}.1.p', 'wb'))
        pickle.dump(cat2.mt_input, open(f'{fileprefix}.2.p', 'wb'))
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
        cat1.mt_input = pickle.load(open(f'{fileprefix}.1.p', 'rb'))
        cat2.mt_input = pickle.load(open(f'{fileprefix}.2.p', 'rb'))
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
        self.matched_mems = np.loadtxt(filename)
