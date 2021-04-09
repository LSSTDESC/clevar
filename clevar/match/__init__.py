from .proximity import ProximityMatch

class MatchedPairs():
    def __init__(self, cat1, cat2, matching_type):
        """
        Parameters
        ----------
        cat1: clevar.Catalog
            Catalog with matching information
        cat2: clevar.Catalog
            Catalog matched to
        matching_type: str
            Type of matching to be considered. Must be in:
            'cross', 'self', 'other'
        """
        #convert_mt = {'cross':'cross', 'cat1':'self', 'cat2':'other'}
        #is_matched = cat1.get_matching_mask(convert_mt[matching_type])
        is_matched = cat1.get_matching_mask(matching_type)
        self.data1 = cat1[is_matched]
        self.data2 = cat2[cat2.ids2inds(self.data1[f'mt_{matching_type}'])]
