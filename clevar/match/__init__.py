from .proximity import ProximityMatch

class MatchedPairs():
    def __init__(self, cat1, cat2, matching_type):
        """
        Parameters
        ----------
        cat1: clevar.ClCatalog
            ClCatalog with matching information
        cat2: clevar.ClCatalog
            ClCatalog matched to
        matching_type: str
            Type of matching to be considered. Must be in:
            'cross', 'self', 'other'
        """
        # convert matching type to the values expected by get_matching_mask
        matching_type_conv = matching_type.replace('cat1', 'self').replace('cat2', 'other')
        self.data1 = cat1[cat1.get_matching_mask(matching_type_conv)]
        self.data2 = cat2[cat2.ids2inds(self.data1[f'mt_{matching_type_conv}'])]
