"""General utility functions that are used in multiple modules"""
import numpy as np
from scipy.interpolate import interp1d
import healpy as hp


########################################################################
########## Improved list and dict ######################################
########################################################################


class NameList(list):
    """
    List without case consideration in `in` function
    """

    def __contains__(self, item):  # implements `in`
        if isinstance(item, str):
            return item.lower() in (n.lower() for n in self)
        return list.__contains__(self, item)


class LowerCaseDict(dict):
    """
    Dictionary with lowercase keys
    """

    # pylint: disable-msg=protected-access

    @classmethod
    def _k(cls, key):
        return key.lower() if isinstance(key, str) else key

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._convert_keys()

    def __getitem__(self, key):
        return super().__getitem__(self.__class__._k(key))

    def __setitem__(self, key, value):
        super().__setitem__(self.__class__._k(key), value)

    def __delitem__(self, key):
        return super().__delitem__(self.__class__._k(key))

    def __contains__(self, key):
        return super().__contains__(self.__class__._k(key))

    def pop(self, key, *args, **kwargs):
        return super().pop(self.__class__._k(key), *args, **kwargs)

    def get(self, key, *args, **kwargs):
        return super().get(self.__class__._k(key), *args, **kwargs)

    def setdefault(self, key, *args, **kwargs):
        return super().setdefault(self.__class__._k(key), *args, **kwargs)

    def update(self, E=None, **F):
        super().update(self.__class__({} if E is None else E))
        super().update(self.__class__(**F))

    def _convert_keys(self):
        for key in list(self.keys()):
            value = super().pop(key)
            self[key] = value


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
    if hasattr(bins, "__len__"):
        bins = np.array(bins)
    elif log:
        logvals = np.log10(values)
        bins = np.logspace(logvals.min(), logvals.max(), bins + 1)
        bins[-1] *= 1.0001
    else:
        bins = np.linspace(values.min(), values.max(), bins + 1)
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
    bin_masks = [(values >= b0) * (values < b1) for b0, b1 in zip(bins, bins[1:])]
    bin_masks[-1] += values == bins[-1]
    return bin_masks


def str2dataunit(input_str, units_bank, err_msg=""):
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
    # pylint: disable-msg=bare-except
    for unit in units_bank:
        if unit.lower() in input_str.lower():
            try:
                return float(input_str.lower().replace(unit.lower(), "")), unit.lower()
            except:
                pass
    raise ValueError(f"Unknown unit of '{input_str}', must be in {units_bank}. {err_msg}")


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
    return np.exp(-0.5 * (value - mean) ** 2 / std**2) / np.sqrt(2 * np.pi) / std


def pack_mt_col(col):
    """Convert match column for saving catalog"""
    return list(map(lambda c: c if c else "", col))


def pack_mmt_col(col):
    """Convert multiple match column for saving catalog"""
    return list(map(lambda c: ",".join(c) if c else "", col))


def unpack_mt_col(col):
    """Convert match column from saved catalog"""
    out = np.array(np.array(col, dtype=str), dtype=np.ndarray)
    out[out == ""] = None
    return out


def unpack_mmt_col(col):
    """Convert multiple match column from saving catalog"""
    out = np.full(col.size, None)
    for i, value in enumerate(np.array(col, dtype=str)):
        out[i] = value.split(",") if len(value) > 0 else []
    return out


########################################################################
### dict functions #####################################################
########################################################################
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
    for key, value in dict_update.items():
        if isinstance(value, dict) and key in dict_base:
            deep_update(dict_base[key], value)
        else:
            dict_base[key] = dict_update[key]
    return dict_base


def dict_with_none(dict_in):
    """
    Get dict replacing "None" with None.

    Parameters
    ----------
    dict_in : dict
        Input dictionary

    Returns
    -------
    dict
        Dictionary with None instead of "None".
    """
    return {k: (None if str(v) == "None" else v) for k, v in dict_in.items()}


def updated_dict(*dict_list):
    """
    Returns an dictionary with updated values if new dictionaries are not none

    Parameters
    ----------
    *dict_list: positional arguments
        Lists of dictionary with updated values

    Returns
    -------
    dict
        Updated dictionary
    """
    out = {}
    for update_dict in dict_list:
        updict = none_val(update_dict, {})
        if not isinstance(updict, dict):
            raise ValueError(
                f"all arguments of updated_dict must be dictionaries or None, got: {updict}"
            )
        out.update(updict)
    return out


def add_dicts_diff(dict1, dict2, pref="", diff_lines=None):
    """
    Adds the differences between dictionaries to a list

    Parameters
    ----------
    dict1, dict2: dict
        Dictionaies to be compared
    pref: str
        Prefix to be added in output
    diff_lines: list, None
        List where differences will be appended to. If None, it is a new list.
    """
    if diff_lines is None:
        diff_lines = []
    for key in set(k for d in (dict1, dict2) for k in d):
        if key not in dict1:
            diff_lines.append((f"{pref}[{key}]", "missing", "present"))
            return
        if key not in dict2:
            diff_lines.append((f"{pref}[{key}]", "present", "missing"))
            return
        if dict1[key] != dict2[key]:
            if isinstance(dict1[key], dict):
                add_dicts_diff(dict1[key], dict2[key], pref=f"{pref}[{key}]", diff_lines=diff_lines)
            else:
                diff_lines.append((f"{pref}[{key}]", str(dict1[key]), str(dict2[key])))


def get_dicts_diff(dict1, dict2, keys=None, header=("Name", "dict1", "dict2"), msg=""):
    """
    Get all the differences between dictionaries, accounting for nested dictionaries.
    If there are differences, a table with the information is printed.

    Parameters
    ----------
    dict1, dict2: dict
        Dictionaries to be compared
    keys: list, None
        List of keys to be compared. If None, all keys are compared
    header: tuple
        Header for differences table
    msg: str
        Message printed before the differences

    Returns
    -------
    diff_lines:
        List of dictionaries differences
    """
    diff_lines = [header]
    if keys is None:
        keys = set(list(dict1.keys()) + list(dict2.keys()))
    for key in keys:
        add_dicts_diff(
            dict1.get(key, {}), dict2.get(key, {}), pref=f"[{key}]", diff_lines=diff_lines
        )
    if len(diff_lines) > 1:
        diff_lines = np.array(diff_lines)
        max_sizes = [max(veclen(l)) for l in diff_lines.T]
        fmts = f"  %-{max_sizes[0]}s | %{max_sizes[1]}s | %{max_sizes[2]}s"
        print(msg)
        print(fmts % tuple(diff_lines[0]))
        print(f'  {"-"*max_sizes[0]}-|-{"-"*max_sizes[1]}-|-{"-"*max_sizes[2]}')
        for line in diff_lines[1:]:
            print(fmts % tuple(line))
    return diff_lines[1:]


########################################################################
########## Smooth Line #################################################
########################################################################


def smooth_loop(xvalues, yvalues, scheme=(1, 1)):
    """Loop for smooth line using pixar's algorithm.

    Parameters
    ----------
    xvalues: array
        x values.
    yvalues: array
        y values.
    scheme: tuple
        Scheme to be used for smoothening. Newton's binomial coefficients work better.

    Returns
    -------
    xsmooth, ysmooth: array
        Smoothened line

    Note
    ----
    Good description of the method can be found at
    https://www.youtube.com/watch?v=mX0NB9IyYpU&ab_channel=Numberphile
    """
    # add midpoints
    xmid = 0.5 * (xvalues[:-1] + xvalues[1:])
    ymid = interp1d(xvalues, yvalues, kind="linear")(xmid)
    xsmooth, ysmooth = np.zeros(len(xvalues) + len(xmid)), np.zeros(len(yvalues) + len(ymid))
    xsmooth[::2] = xvalues
    xsmooth[1::2] = xmid
    ysmooth[::2] = yvalues
    ysmooth[1::2] = ymid

    # move
    n_edge = int(len(scheme) / 2)
    ncrop = 2 * n_edge

    xmid_new = np.zeros(xsmooth.size - ncrop)
    ymid_new = np.zeros(ysmooth.size - ncrop)
    i = 0
    for weight in scheme:
        if i == len(scheme) / 2:
            i += 1
        xmid_new += weight * xsmooth[i : xsmooth.size - ncrop + i]
        ymid_new += weight * ysmooth[i : ysmooth.size - ncrop + i]
        i += 1

    xmid_new /= sum(scheme)
    ymid_new /= sum(scheme)
    xsmooth[n_edge:-n_edge] = xmid_new
    ysmooth[n_edge:-n_edge] = ymid_new
    return xsmooth, ysmooth


def smooth_line(xvalues, yvalues, n_increase=10, scheme=(1, 2, 1)):
    """Make smooth line using pixar's algorithm.

    Parameters
    ----------
    xvalues: array
        x values.
    yvalues: array
        y values.
    n_increase: int
        Number of loops for the algorithm.
    scheme: tuple
        Scheme to be used for smoothening. Newton's binomial coefficients work better.

    Returns
    -------
    xsmooth, ysmooth: array
        Smoothened line

    Note
    ----
    Good description of the method can be found at
    https://www.youtube.com/watch?v=mX0NB9IyYpU&ab_channel=Numberphile
    """
    if n_increase == 0:
        return xvalues, yvalues
    xsmooth, ysmooth = smooth_loop(xvalues, yvalues, scheme=scheme)
    for _ in range(1, n_increase):
        xsmooth, ysmooth = smooth_loop(xsmooth, ysmooth, scheme=scheme)
    return xsmooth, ysmooth


########################################################################
########## Monkeypatching healpy #######################################
########################################################################


def pix2map(nside, pixels, values, null):
    """
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
    """
    outmap = np.zeros(12 * nside**2) + null
    outmap[pixels] = values
    return outmap


def neighbors_of_pixels(nside, pixels, nest=False):
    """
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
    """
    nbs = np.array(list(set(hp.get_all_neighbours(nside, pixels, nest=nest).flatten())))
    return nbs[nbs > -1]


hp.pix2map = pix2map
hp.neighbors_of_pixels = neighbors_of_pixels
