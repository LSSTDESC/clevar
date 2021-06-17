from .parent import Match
from .proximity import ProximityMatch
from .membership import MembershipMatch
import numpy as np

def get_matched_pairs(cat1, cat2, matching_type, mask1=None, mask2=None):
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
    mask1: array, None
        Mask for clusters 1 properties, must have size=cat1.size
    mask2: array, None
        Mask for clusters 2 properties, must have size=cat2.size
    """
    mt_mask1, mt_mask2 = get_matched_masks(cat1, cat2, matching_type)
    mask = np.ones(len(mt_mask2), dtype=bool) if mask1 is None else mask1[mt_mask1]
    mask = mask if mask2 is None else mask*mask2[mt_mask2]
    return cat1[mt_mask1][mask], cat2[mt_mask2][mask]
def get_matched_masks(cat1, cat2, matching_type):
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
