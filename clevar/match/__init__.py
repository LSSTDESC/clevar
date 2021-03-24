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
        is_matched = cat1.get_matching_mask(matching_type)
        self.data1 = cat1.data[is_matched]
        self.data2 = cat2.data[cat2.ids2inds(cat1.match[matching_type][is_matched])]
