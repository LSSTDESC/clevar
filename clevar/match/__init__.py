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
        mask1, mask2 = self.matching_masks(cat1, cat2, matching_type)
        self.data1 = cat1[mask1]
        self.data2 = cat2[mask2]
    def matching_masks(self, cat1, cat2, matching_type):
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

        Returns
        -------
        mask1: ndarray
            Array of booleans to get catalog1 matched clusters
        mask2: ndarray
            Array of catalog2 indexes corresponding to catalog1 matched clusters
        """
        # convert matching type to the values expected by get_matching_mask
        matching_type_conv = matching_type.replace('cat1', 'self').replace('cat2', 'other')
        mask1 = cat1.get_matching_mask(matching_type_conv)
        mask2 = cat2.ids2inds(cat1[mask1][f'mt_{matching_type_conv}'])
        return mask1, mask2
