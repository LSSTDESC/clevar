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
def logbins(values, nbins):
    """
    Make log10 spaced bins from data

    Parameters
    ----------
    values: array
        Data
    nbins: int
        Number of bins

    Returns
    -------
    ndarray
        Log10 spaced bins based on values
    """
    logvals = np.log10(values)
    return np.logspace(logvals.min(), logvals.max(), nbins)
########################################################################
########## Monkeypatching healpy #######################################
########################################################################
import healpy as hp
def pix2mask(nside, pixels):
    '''
    Create a mask from pixels

    Parameters
    ----------
    nside: int
        Healpix nside
    pixels: array
        Array of pixel indices

    Returns
    -------
    outmask: array
        Mask from pixels
    '''
    outmask = np.zeros(12*nside**2, dtype=bool)
    outmask[pixels] = True
    return outmask
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
hp.pix2mask = pix2mask
hp.pix2map = pix2map
