import numpy as np


########################################################################
########## Improved list and dict ######################################
########################################################################
class NameList(list):
    """
    List without case consideration in `in` function
    """
    def __contains__(self, item): # implements `in`
        if isinstance(item, str):
            return item.lower() in (n.lower() for n in self)
        else:
            return list.__contains__(self, item)


class LowerCaseDict(dict):
    """
    Dictionary with lowercase keys
    """
    @classmethod
    def _k(cls, key):
        return key.lower() if isinstance(key, str) else key
    def __init__(self, *args, **kwargs):
        super(LowerCaseDict, self).__init__(*args, **kwargs)
        self._convert_keys()
    def __getitem__(self, key):
        return super(LowerCaseDict, self).__getitem__(self.__class__._k(key))
    def __setitem__(self, key, value):
        super(LowerCaseDict, self).__setitem__(self.__class__._k(key), value)
    def __delitem__(self, key):
        return super(LowerCaseDict, self).__delitem__(self.__class__._k(key))
    def __contains__(self, key):
        return super(LowerCaseDict, self).__contains__(self.__class__._k(key))
    def has_key(self, key):
        return super(LowerCaseDict, self).has_key(self.__class__._k(key))
    def pop(self, key, *args, **kwargs):
        return super(LowerCaseDict, self).pop(self.__class__._k(key), *args, **kwargs)
    def get(self, key, *args, **kwargs):
        return super(LowerCaseDict, self).get(self.__class__._k(key), *args, **kwargs)
    def setdefault(self, key, *args, **kwargs):
        return super(LowerCaseDict, self).setdefault(self.__class__._k(key), *args, **kwargs)
    def update(self, E={}, **F):
        super(LowerCaseDict, self).update(self.__class__(E))
        super(LowerCaseDict, self).update(self.__class__(**F))
    def _convert_keys(self):
        for k in list(self.keys()):
            v = super(LowerCaseDict, self).pop(k)
            self.__setitem__(k, v)
########################################################################
########## Helpful functions ###########################################
########################################################################
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
        bins = np.array(bins)
    elif log:
        logvals = np.log10(values)
        bins = np.logspace(logvals.min(), logvals.max(), bins+1)
        bins[-1] *= 1.0001
    else:
        bins = np.linspace(values.min(), values.max(), bins+1)
        bins[-1] *= 1.0001
    return bins
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
def deep_update(dict_base, dict_update):
    """
    Update a multi-layer dictionary.

    Parameters
    ----------
    dict_base: dict
        Dictionary to be updated
    dict_update: dict
        Dictionary with the updates

    Returns
    -------
    dict_base: dict
        Updated dictionary (the input dict is also updated)
    """
    for k, v in dict_update.items():
        if isinstance(v, dict) and k in dict_base:
            deep_update(dict_base[k], v)
        else:
            dict_base[k] = dict_update[k]
    return dict_base
def gaussian(value, mean, std):
    """
    Gaussian function.

    Parameters
    ----------
    value: float, array
        Point(s) to compute the distribution.
    mean: float, array
        Mean ("centre") of the distribution.
    std: float, array
        Standard deviation (spread or "width") of the distribution. Must be non-negative.

    Returns
    -------
    float, array
        Value of the gaussian distribution at input `value`.
    """
    return np.exp(-0.5*(value-mean)**2/std**2)/np.sqrt(2*np.pi)/std
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
