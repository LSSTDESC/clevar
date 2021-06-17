import numpy as np

veclen = np.vectorize(len)
def none_val(value, none_value):
    """
    Set default value to be returned if input is None

    Parameters
    ----------
    value:
        Input value
    none_value:
        Value to be asserted if input is None

    Returns
    -------
    type(value), type(none_value)
        Value if value is not None else none_value
    """
    return value if value is not None else none_value
def autobins(values, bins, log=False):
    """
    Get bin values automatically from bins, values

    Parameters
    ----------
    values: array
        Data
    bins: int, array
        Bins/Number of bins
    log: bool
        Logspaced bins (used if bins is int)

    Returns
    -------
    ndarray
        Bins based on values
    """
    if hasattr(bins, '__len__'):
        return np.array(bins)
    if log:
        logvals = np.log10(values)
        return np.logspace(logvals.min(), logvals.max(), bins+1)
    else:
        return np.linspace(values.min(), values.max(), bins+1)
def binmasks(values, bins):
    """
    Get corresponding masks for each bin. Last bin is inclusive.

    Parameters
    ----------
    values: array
        Data
    bins: array
        Bins

    Returns
    -------
    bin_masks: list
        List of masks for each bin
    """
    bin_masks = [(values>=b0)*(values<b1) for b0, b1 in zip(bins, bins[1:])]
    bin_masks[-1] += values==bins[-1]
    return bin_masks
def str2dataunit(input_str, units_bank, err_msg=''):
    """
    Convert a string to a float with unit.
    ex: '1mpc' -> (1, 'mpc')

    Parameters
    ----------
    input_str: str
        Input string
    unit_bank: list
        Bank of units available.
    """
    for unit in units_bank:
        if unit.lower() in input_str.lower():
            try:
                return float(input_str.lower().replace(unit.lower(), '')), unit.lower()
            except:
                pass
    raise ValueError(f"Unknown unit of '{input_str}', must be in {units_bank}. {err_msg}")
########################################################################
########## Monkeypatching healpy #######################################
########################################################################
import healpy as hp
def pix2map(nside, pixels, values, null):
    '''
    Convert from pixels, values to map

    Parameters
    ----------
    nside: int
        Healpix nside
    pixels: array
        Array of pixel indices
    values: Array
        Value of map in each pixel
    null: obj
        Value for pixels outside the map
    '''
    outmap = np.zeros(12*nside**2)+null
    outmap[pixels] = values
    return outmap
def neighbors_of_pixels(nside, pixels, nest=False):
    '''
    Get all neighbors of a pixel list

    Parameters
    ----------
    nside: int
        Healpix nside
    pixels: array
        Array of pixel indices
    nest: bool
        If ordering is nested

    Return
    ------
    array
        Neighbor pixels
    '''
    nbs = np.array(list(set(hp.get_all_neighbours(nside, pixels, nest=nest).flatten())))
    return nbs[nbs>-1]
hp.pix2map = pix2map
hp.neighbors_of_pixels = neighbors_of_pixels
