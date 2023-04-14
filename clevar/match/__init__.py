from ..catalog import ClData
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
        'cross', 'cat1', 'cat2'
    mask1: array, None
        Mask for clusters 1 properties, must have size=cat1.size
    mask2: array, None
        Mask for clusters 2 properties, must have size=cat2.size
    """
    mt_mask1, mt_mask2 = get_matched_masks(cat1, cat2, matching_type)
    mask = np.ones(len(mt_mask2), dtype=bool) if mask1 is None else mask1[mt_mask1]
    mask = mask if mask2 is None else mask * mask2[mt_mask2]
    mt_mask1[mt_mask1] *= mask
    mt_mask2 = mt_mask2[mask]
    mt1, mt2 = cat1[mt_mask1], cat2[mt_mask2]
    if mt1.members is not None:
        if "match" in mt1.members.data.colnames:
            mt1.members["in_mt_sample"] = [
                any(id2 in mt2.id_dict for id2 in mem1["match"]) for mem1 in mt1.members
            ]
    if mt2.members is not None:
        if "match" in mt2.members.data.colnames:
            mt2.members["in_mt_sample"] = [
                any(id1 in mt1.id_dict for id1 in mem2["match"]) for mem2 in mt2.members
            ]
    return mt1, mt2


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
        'cross', 'cat1', 'cat2'

    Returns
    -------
    mask1: ndarray
        Array of booleans to get catalog1 matched clusters
    mask2: ndarray
        Array of catalog2 indexes corresponding to catalog1 matched clusters
    """
    # convert matching type to the values expected by get_matching_mask
    matching_type_conv = matching_type.replace("cat1", "self").replace("cat2", "other")
    mask1 = cat1.get_matching_mask(matching_type_conv)
    mask2 = cat2.ids2inds(cat1[f"mt_{matching_type_conv}"][mask1])
    return mask1, mask2


def output_catalog_with_matching(file_in, file_out, catalog, overwrite=False):
    """Add matching information to original catalog.

    Parameters
    ----------
    file_in: str
        Name of input catalog file
    file_out: str
        Name of output catalog file
    catalog: clevar.ClCatalog
        ClCatalog with matching information
    overwrite: bool
        Overwrite output file
    """
    out = ClData.read(file_in)
    if len(out) != len(catalog):
        ValueError(
            f"Input file ({file_in}) size (={len(out)})"
            + f" differs from catalog size (={len(catalog)})."
        )
    for col in [c_ for c_ in catalog.data.colnames if c_[:3] in ("mt_", "ft_", "cf_")]:
        if col in ("mt_self", "mt_other", "mt_cross"):
            out[col] = [c if c else "" for c in catalog[col]]
        elif col in ("mt_multi_self", "mt_multi_other"):
            out[col] = [",".join(c) if c else "" for c in catalog[col]]
        else:
            out[col] = catalog[col]
    out.write(file_out, overwrite=overwrite)


def output_matched_catalog(
    file_in1, file_in2, file_out, cat1, cat2, matching_type="cross", overwrite=False
):
    """Output matched catalog with information of both inputs.

    Parameters
    ----------
    file_in1: str
        Name of input catalog file
    file_in2: str
        Name of input catalog file
    file_out: str
        Name of output catalog file
    cat1: clevar.ClCatalog
        ClCatalog with matching information corresponding to file_in1
    cat1: clevar.ClCatalog
        ClCatalog with matching information corresponding to file_in2
    matching_type: str
        Type of matching to be considered. Must be in:
        'cross', 'cat1', 'cat2'
    overwrite: bool
        Overwrite output file
    """
    c_matched = ClData()
    # match masks
    m1, m2 = get_matched_masks(cat1, cat2, matching_type)
    # add cat 1 info
    dat1_full = ClData.read(file_in1)
    if len(dat1_full) != len(cat1):
        raise ValueError(
            f"Input file ({file_in1}) size (={len(dat1_full)})"
            + f" differs from cat1 size (={len(cat1)})."
        )
    for col in dat1_full.colnames:
        c_matched[f"cat1_{col}"] = dat1_full[col][m1]
    del dat1_full
    if "mt_frac_self" in cat1.data.colnames:
        c_matched["cat1_mt_frac"] = cat1["mt_frac_self"][m1]
    # add cat 2 info
    dat2_full = ClData.read(file_in2)
    if len(dat2_full) != len(cat2):
        raise ValueError(
            f"Input file ({file_in2}) size (={len(dat2_full)})"
            + f" differs from cat2 size (={len(cat2)})."
        )
    for col in dat2_full.colnames:
        c_matched[f"cat2_{col}"] = dat2_full[col][m2]
    del dat2_full
    if "mt_frac_self" in cat2.data.colnames:
        c_matched["cat2_mt_frac"] = cat2["mt_frac_self"][m2]
    # Save
    c_matched.write(file_out, overwrite=overwrite)
