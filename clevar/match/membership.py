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
        radius_selection: str (optional)
            Case of radius to be used, can be: max, min, self, other.
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
    def fill_shared_members(self, cat1, cat2, mem1, mem2, radius_selection='max'):
        if self.matched_mems is None:
            raise ValueError('Members not matched, run match_members before.')
        mem1['pmem'] = mem1['pmem'] if 'pmem' in mem1.colnames else np.ones(mem1.size)
        mem2['pmem'] = mem2['pmem'] if 'pmem' in mem2.colnames else np.ones(mem2.size)
        cat1.mt_input = {'share_mems': [{} for i in range(cat1.size)],
                         'nmem': self._comp_nmem(cat1, mem1)}
        cat2.mt_input = {'share_mems': [{} for i in range(cat2.size)],
                         'nmem': self._comp_nmem(cat2, mem2)}
        for mem1_, mem2_ in zip(mem1[self.matched_mems[:,0]], mem2[self.matched_mems[:,1]]):
            self._add_pmem(cat1.mt_input['share_mems'], cat1.id_dict[mem1_['id_cluster']], 
                           mem2_['id_cluster'], mem1_['pmem'])
            self._add_pmem(cat2.mt_input['share_mems'], cat2.id_dict[mem2_['id_cluster']], 
                           mem1_['id_cluster'], mem2_['pmem'])
        '''
        pmem1 = mem1['pmem'] if 'pmem' in mem1.colnames else np.ones(mem1.size)
        pmem2 = mem2['pmem'] if 'pmem' in mem2.colnames else np.ones(mem2.size)
        for mem_ind1, mem_ind2 in self.matched_mems:
            cl_id1, cl_id2 = mem1[mem_ind1]['id_cluster'], mem2[mem_ind2]['id_cluster']
            ind1, ind2 = cat1.id_dict[cl_id1], cat2.id_dict[cl_id2]
            cat1.mt_input['share_mem'][ind1][cl_id2] = \
                cat1.mt_input['share_mem'][ind1].get(cl_id2, 0)+pmem2[mem_ind2]
            cat2.mt_input['share_mem'][ind2][cl_id1] = \
                cat2.mt_input['share_mem'][ind2].get(cl_id1, 0)+pmem1[mem_ind1]
        '''
    def _comp_nmem(self, cat, mem):
        out = np.zeros(cat.size)
        for ID, pmem in zip(mem['id_cluster'], mem['pmem']):
            out[cat.id_dict[ID]] += pmem
        return out
    def _add_pmem(self, cat1_share_mems, ind1, cat2_id, pmem1):
        cat1_share_mems[ind1][cat2_id] = cat1_share_mems[ind1].get(cat2_id, 0)+pmem1
    def save_shared_members(self, fileprefix, overwrite=False):
        """
        Saves dictionaries of shared members

        Parameters
        ----------
        fileprefix: str
            Prefix for name of files
        overwrite: bool
            Overwrite saved files
        """
        pickle.dump(self.mt1, open(f'{fileprefix}.1.p', 'wb'))
        pickle.dump(self.mt2, open(f'{fileprefix}.2.p', 'wb'))
    def load_shared_members(self, fileprefix):
        """
        Load dictionaries of shared members

        Parameters
        ----------
        filename: str
            Prefix of files name
        """
        self.mt1 = pickle.load(open(f'{fileprefix}.1.p', 'rb'))
        self.mt2 = pickle.load(open(f'{fileprefix}.2.p', 'rb'))
    def match_members(self, mem1, mem2, method='id', radius=None, cosmo=None):
        if method=='id':
            self._match_members_by_id(mem1, mem2)
        elif method=='angular_distance':
            self._match_members_by_ang(mem1, mem2, radius, cosmo)
    def _match_members_by_id(self, mem1, mem2):
        self.matched_mems = np.array([[mem1.id_dict[i], ind] for ind, i in enumerate(mem2['id'])
                                                                            if i in mem1.id_dict])
    def _match_members_by_ang(self, mem1, mem2, radius, cosmo):
        match_config = {
            'type': 'cross', # options are cross, cat1, cat2
            'which_radius': 'max', # Case of radius to be used, can be: cat1, cat2, min, max
            'preference': 'angular_proximity', # options are more_massive, angular_proximity or redshift_proximity
            'catalog1': {'delta_z':None, 'match_radius': radius},
            'catalog2': {'delta_z':None, 'match_radius': radius},
            }
        mt = ProximityMatch()
        mt.match_from_config(mem1, mem2, match_config, cosmo=cosmo)
        mask1 = cat1.get_matching_mask(match_config['type'])
        mask2 = cat2.ids2inds(cat1[mask1][f"mt_{match_config['type']}"])
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
