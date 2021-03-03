import numpy as np
import os
from ..catalog import ClData

class Match():
    """
    Matching Class
    """
    def __init__(self, ):
        self.type = None
    def _prep_for_match(self, config):
        raise NotImplementedError
    def multiple(self, cat1, cat2):
        """Makes multiple matchig

        Parameters
        ----------
        cat1: clevar.Catalog
            Base catalog
        cat2: clevar.Catalog
            Target catalog

        Note
        ----
            Not implemented in parent class
        """
        raise NotImplementedError
    def unique(self, cat1, cat2, preference):
        """Makes unique matchig, requires multiple matching to be made first

        Parameters
        ----------
        cat1: clevar.Catalog
            Base catalog
        cat2: clevar.Catalog
            Target catalog
        preference: str
            Preference to set best match
        """
        i_vals = range(cat1.size)
        if preference=='mproxy':
            set_unique = lambda cat1, i, cat2, i2: self._match_mpref(cat1, i, cat2, i2)
            i_vals = np.arange(cat1.size, dtype=int)[np.argsort(cat1.data['mass'])]
        elif preference=='ang':
            set_unique = lambda cat1, i, cat2, i2: self._match_apref(cat1, i, cat2, i2, 'ang')
        elif preference=='z':
            set_unique = lambda cat1, i, cat2, i2: self._match_apref(cat1, i, cat2, i2, 'z')
        for i in i_vals:
            inds2 = cat2.ids2inds(cat1.match['multi_self'][i])
            set_unique(cat1, i, cat2, inds2)
        print(f'* {len(cat1.match[cat1.match["self"]!=None]):,} objects matched.')
    def _match_mpref(self, cat1, i, cat2, inds2):
        """
        Make the unique match by mass preference

        Parameters
        ----------
        cat1: clevar.Catalog
            Base catalog
        i: int
            Index of the cluster from cat1 to be matched
        cat2: clevar.Catalog
            Target catalog
        inds2: list
            List of indices of objects in cat2 to be used
        """
        for i2 in inds2[np.argsort(cat2.data['mass'])[inds2]]:
            if cat2.match['other'][i2]=='':
                cat1.match['self'][i] = cat2.data['id'][i2]
                cat2.match['other'][i2] = cat1.data['id'][i]
                return
    def _match_apref(self, cat1, i, cat2, inds2, MATCH_PREF):
        """
        Make the unique match by angula (or redshift) distance preference

        Parameters
        ----------
        cat1: clevar.Catalog
            Base catalog
        i: int
            Index of the cluster from cat1 to be matched
        cat2: clevar.Catalog
            Target catalog
        inds2: list
            List of indices of objects in cat2 to be used
        MATCH_PREF: str
            Match preference (ang dist, redshift dist)
        """
        dists = self._get_dist_mt(cat1.data[i], cat2.data[inds2], MATCH_PREF)
        sort_d = np.argsort(dists)
        for dist, i2 in zip(dists[sort_d], inds2[sort_d]):
            i1_replace = cat1.id_dict[cat2.match['other'][i2]] if cat2.match['other'][i2] \
                            else None
            if i1_replace is None:
                cat1.match['self'][i] = cat2.data['id'][i2]
                cat2.match['other'][i2] = cat1.data['id'][i]
                return
            elif dist < self._get_dist_mt(cat1.data[i1_replace], cat2.data[i2], MATCH_PREF):
                cat1.match['self'][i] = cat2.data['id'][i2]
                cat2.match['other'][i2] = cat1.data['id'][i]
                self._match_apref(cat1, i1_replace, cat2, MATCH_PREF)
                return
    def _get_dist_mt(self, dat1, dat2, MATCH_PREF):
        """
        Get distance for matching preference

        Parameters
        ----------
        dat1: clevar.ClData
            Data of base catalog
        dat2: clevar.ClData
            Data of target catalog
        MATCH_PREF: str
            Type of distance, can be ang or z.

        Return
        ------
        bool
            If there was a match
        """
        if MATCH_PREF=='ang':
            return dat1['SkyCoord'].separation(
                dat2['SkyCoord']).value
        elif MATCH_PREF=='z':
            return abs(dat1['z']-dat2['z'])
        else:
            raise ValueError("MATCH_PREF in get_dist_mt must be 'ang' or 'z'")
    def save_matches(self, cat1, cat2, config):
        """
        Saves the matching results
        """
        if not os.path.isdir(config["out_dir"]):
            os.system(f'mkdir {config["out_dir"]}')
        self._save_match(cat1, f'{config["out_dir"]}/match1.fits')
        self._save_match(cat2, f'{config["out_dir"]}/match2.fits')
    def _save_match(self, cat, out_name):
        """
        Saves the matching results of one catalog
        """
        out = ClData()
        out['id'] = cat.data['id']
        for col in ('self', 'other'):
            out[col] = [c if c else '' for c in cat.match[col]]
        for col in ('multi_self', 'multi_other'):
            out[col] = [','.join(c) if c else '' for c in cat.match[col]]
        out.write(out_name, overwrite=True)
    def load_matches(self, cat1, cat2, config):
        """
        Load matching results to catalogs
        """
        mt1 = ClData.read(f'{config["out_dir"]}/match1.fits')
        self._load_match(cat1, mt1)
        del mt1
        #return out1
        mt2 = ClData.read(f'{config["out_dir"]}/match2.fits')
        self._load_match(cat2, mt2)
        del mt2
    def _load_match(self, cat, mt):
        """
        Load matching results of one catalog
        """
        for col in ('self', 'other'):
            cat.match[col] = np.array([c if c!='' else None for c in mt[col]], dtype=np.ndarray)
        for col in ('multi_self', 'multi_other'):
            cat.match[col] = np.array([None for c in mt[col]], dtype=np.ndarray)
            for i, c in enumerate(mt[col]):
                if len(c)>0:
                    cat.match[col][i] = c.split(',')
                else:
                    cat.match[col][i] = []
        cat.cross_match()
